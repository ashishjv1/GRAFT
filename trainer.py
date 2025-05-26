import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import copy
import gc
from utils.pickler import pickler
from genindices import sample_selection
from torch.utils.data import Subset, DataLoader

class ModelTrainer:
    def __init__(self, config, model, trainloader, valloader, trainset, data3=None):
        self.config = config
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.trainset = trainset
        self.data3 = data3
        
        self.device = config.device
        self.setup_training()

    def setup_training(self):
        if self.config.dataset_name.lower() == 'boston':
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), 
                                      lr=self.config.lr,
                                      weight_decay=self.config.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                     lr=self.config.lr, 
                                     momentum=0.9,
                                     weight_decay=self.config.weight_decay)

        self.setup_scheduler()

    def setup_scheduler(self):
        if self.config.sched.lower() == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                self.config.lr,
                epochs=self.config.num_epochs,
                steps_per_epoch=len(self.trainloader))
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=200)

    def train(self):
        selection = 0
        best_acc = 0
        train_stats = {'losses': [], 'acc': []}
        val_stats = {'losses': [], 'acc': []}

        for epoch in range(self.config.num_epochs):
            self.model.train()
            
            if self.should_select_samples(epoch, selection):
                self.trainloader = self.sample_data(selection)
                selection += 1

            epoch_loss = self.train_epoch()
            
            if self.config.sched.lower() == "cosine":
                self.scheduler.step()

            train_acc, train_loss, val_acc, val_loss = self.evaluate()
            
            train_stats['losses'].append(train_loss)
            train_stats['acc'].append(train_acc)
            val_stats['losses'].append(val_loss) 
            val_stats['acc'].append(val_acc)

            best_acc = max(best_acc, val_acc)
            
            self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, best_acc)
            
            if epoch % 20 == 0:
                self.save_checkpoint(epoch)

        return train_stats, val_stats

    def should_select_samples(self, epoch, selection):
        if epoch % self.config.selection_iter == 0:
            if self.config.warm_start and selection == 0:
                return True
            elif not self.config.warm_start:
                return True
        return False

    def sample_data(self, selection):
        if self.config.warm_start and selection == 0:
            return self.trainloader
            
        train_model = self.model
        cached_state = copy.deepcopy(train_model.state_dict())

        indices = sample_selection(
            self.trainloader,
            self.data3,
            train_model,
            self.config.batch_size,
            self.config.fraction,
            self.config.selection_iter,
            self.config.num_epochs,
            self.device,
            self.config.dataset_name
        )

        self.model.load_state_dict(cached_state)

        subset = Subset(self.trainset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=1
        )

        del train_model
        gc.collect()
        torch.cuda.empty_cache()

        return loader

    def train_epoch(self):
        total_loss = 0
        for batch in tqdm(self.trainloader):
            loss = self.train_step(batch)
            total_loss += loss

            if self.config.sched.lower() == "onecycle":
                self.scheduler.step()

        return total_loss / len(self.trainloader)

    def train_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        loss.backward()
        
        if self.config.grad_clip:
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_clip)
            
        self.optimizer.step()

        return loss.item()

    def evaluate(self):
        self.model.eval()
        train_acc, train_loss = self.evaluate_split(self.trainloader)
        val_acc, val_loss = self.evaluate_split(self.valloader)
        return train_acc, train_loss, val_acc, val_loss

    def evaluate_split(self, dataloader):
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
        return correct/total, total_loss/len(dataloader)

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, best_acc):
        wandb.log({
            "Validation accuracy": best_acc,
            "Val Loss": val_loss,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc * 100,
            "Epoch": epoch
        })

        print(f"Epoch [{epoch+1}/{self.config.num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"Best Acc: {best_acc*100:.2f}%")

    def save_checkpoint(self, epoch):
        path = f"saved_models/{self.config.model_name}"
        if not os.path.exists(path):
            os.makedirs(path)
            
        prefix = "Full" if self.config.selection_iter > self.config.num_epochs else "Sampled"
        filename = f"{prefix}_{self.config.dataset_name}_sch{self.config.sched}_si{self.config.selection_iter}_f{self.config.fraction}"
        
        torch.save(self.model.state_dict(), f"{path}/{filename}.pth")
