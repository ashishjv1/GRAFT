# Standard library imports
import os
import pickle
import copy
import gc
import argparse

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Local imports
from utils.loader import loader
from utils.model_mapper import ModelMapper
from utils.imagenetselloader import imagenet_selloader
import utils.pickler as pickler
from decompositions import feature_sel
from genindices import sample_selection

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import eco2ai
    ECO2AI_AVAILABLE = True
except ImportError:
    ECO2AI_AVAILABLE = False


class TrainingConfig:
    def __init__(self, numEpochs, batch_size, device, net, 
                 model_name, dataset_name, trainloader, valloader, 
                 trainset, data3, optimizer_name, lr, weight_decay, 
                 grad_clip, fraction, selection_iter, warm_start, 
                 imgntselloader, sched="cosine", multi_checkpoint=False,
                 use_wandb=True):  # Add use_wandb parameter
        
        self.numEpochs = numEpochs
        self.batch_size = batch_size
        self.device = device
        self.net = net
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.trainloader = trainloader
        self.valloader = valloader
        self.trainset = trainset
        self.data3 = data3
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.fraction = fraction
        self.selection_iter = selection_iter
        self.warm_start = warm_start
        self.imgntselloader = imgntselloader
        self.sched = sched
        self.multi_checkpoint = multi_checkpoint
        self.use_wandb = use_wandb and WANDB_AVAILABLE

    @classmethod
    def from_args(cls, args):
        return cls(
            numEpochs=args.numEpochs, 
            batch_size=args.batch_size, 
            device=args.device, 
            net=None,  # Placeholder, will be set in the trainer
            model_name=args.model, 
            dataset_name=args.dataset, 
            trainloader=None,  # Placeholder, will be set in the trainer
            valloader=None,  # Placeholder, will be set in the trainer
            trainset=None,  # Placeholder, will be set in the trainer
            data3=None,  # Placeholder, will be set in the trainer
            optimizer_name=args.optimizer, 
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            grad_clip=args.grad_clip, 
            fraction=args.fraction, 
            selection_iter=args.select_iter, 
            warm_start=args.warm_start, 
            imgntselloader=None,  # Placeholder, will be set in the trainer
            sched="cosine", 
            multi_checkpoint=False,
            use_wandb=getattr(args, 'use_wandb', True)
        )


class ModelTrainer:
    def __init__(self, config, model, trainloader, valloader, trainset, data3):
        self.config = config
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.trainset = trainset
        self.data3 = data3
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.curr_high = 0
        self.total = 0
        self.correct = 0
        self.trn_losses = list()
        self.val_losses = list()
        self.trn_acc = list()
        self.val_acc = list()  
        self.selection = 0
        self.weight_decay = 1e-4

        self.dir_save = f"saved_models/{config.model_name}"
        self.save_dir = f"{self.dir_save}/multi_checkpoint"

        self._setup()

    def _setup(self):
        # Default to cross entropy loss unless specifically handling regression
        self.loss_fn = torch.nn.functional.cross_entropy

        # Create save directories
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.config.optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay = self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")

        if self.config.sched.lower() == "onecycle":
            self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, self.config.lr, epochs=self.config.numEpochs, 
                                                        steps_per_epoch=len(self.trainloader))
        else:
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        if self.config.use_wandb:
            if WANDB_AVAILABLE:
                wandb.login()
                if self.config.model_name.lower() == "efficientnet-b0":
                    model_name = "efficientnetb0"
                elif self.config.model_name.lower() == "efficientnet-b5":
                    model_name = "efficientnetb5"
                config = {"lr": self.config.lr, "batch_size": self.config.batch_size}
                config.update({"architecture": f'{self.model}'})
                wandb.init(project=f"Smart_Sampling_{self.config.model_name}_{self.config.dataset_name}", 
                          config=config)

        self.main_trainloader = self.trainloader

    def train(self):
        train_stats = {
            'losses': self.trn_losses,
            'accuracies': self.trn_acc,
            'best_acc': max(self.trn_acc) if self.trn_acc else 0
        }
        
        val_stats = {
            'losses': self.val_losses,
            'accuracies': self.val_acc,
            'best_acc': self.curr_high
        }
        
        for epoch in range(self.config.numEpochs):
            self.model.train()
            tot_train_loss = 0
            before_lr = self.optimizer.param_groups[0]["lr"]
            pruned_samples = 0
            total_samples = 0

                      
            if (epoch) % self.config.selection_iter == 0:
                if self.config.warm_start and self.selection == 0:
                    trainloader = self.trainloader
                    self.selection += 1
                else:
                    train_model = self.model
                    cached_state_dict = copy.deepcopy(train_model.state_dict())
                    clone_dict = copy.deepcopy(train_model.state_dict())
                    
                    # Skip selection if no data3 available (for tests)
                    if self.data3 is None:
                        continue
                        
                    if not self.config.imgntselloader:
                        indices = sample_selection(self.main_trainloader, self.data3, self.model, 
                                                  clone_dict, self.config.batch_size, self.config.fraction, 
                                                  self.config.selection_iter, self.config.numEpochs, 
                                                  self.config.device, self.config.dataset_name)
                    else:
                        
                        indices = sample_selection(self.config.imgntselloader, self.data3, self.model,
                                                 clone_dict, self.config.batch_size, self.config.fraction,
                                                 self.config.selection_iter, self.config.numEpochs, self.config.device, self.config.dataset_name)
                        
                    self.model.load_state_dict(cached_state_dict)

                    self.selection += 1

                    datasubset = Subset(self.trainset, indices)
                    new_trainloader = DataLoader(datasubset, batch_size=self.config.batch_size,
                                                  shuffle=True, pin_memory=False, num_workers=1)

                    self.trainloader = new_trainloader
                
                    del cached_state_dict
                    del clone_dict
                    del train_model
                    torch.cuda.empty_cache()    
                    gc.collect()
                    
            for _, (trainsamples, labels) in enumerate(tqdm(self.trainloader)):
                
                trainsamples = trainsamples.to(self.config.device)
                labels = labels.to(self.config.device)
              
                X = trainsamples
                Y = labels
                pred = self.model(X)

                    
    #             loss = torch.nn.functional.cross_entropy(pred, Y.to(device))
                loss = self.loss_fn(pred, Y.to(self.config.device))


                tot_train_loss += loss.item()

                self.optimizer.zero_grad()

                loss.backward()
                
                if self.config.grad_clip:
                    nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_clip)

                self.optimizer.step()

                # calculate accuracy
                _, predicted = torch.max(pred.cpu().data, 1)
                self.total += Y.size(0)

                self.correct += (predicted == Y.cpu()).sum().item()
                # accuracy = 100 * correct / total
                pruned_samples += len(trainsamples) - len(X)
                total_samples += len(trainsamples)
                
                if self.config.sched.lower() == "onecycle":
                    self.scheduler.step()

            if self.config.sched.lower() == "cosine":
                self.scheduler.step()

            after_lr = self.optimizer.param_groups[0]["lr"]

            print("Last Epoch [%d] -> Current Epoch [%d]: lr %.4f -> %.4f optimizer %s" % (epoch, epoch+1, before_lr, after_lr, self.config.optimizer_name))


            if epoch % 20 == 0:
                dir_parts = self.dir_save.split('/')
                current_dir = ''

                for part in dir_parts:
                    current_dir = os.path.join(current_dir, part)
                    if not os.path.exists(current_dir):
                        os.makedirs(current_dir)

                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                    
                if not os.path.exists(self.dir_save):
                    os.makedirs(self.dir_save)

                if self.config.selection_iter > self.config.numEpochs:
                    file_prefix = "Full"
                else:
                    file_prefix = "Sampled"

                if self.config.multi_checkpoint:
                    file_prefix += "_multi"

                filename = f"{file_prefix}_{self.config.dataset_name}_sch{self.config.sched}_si{self.config.selection_iter}_f{self.config.fraction}"
                if self.config.multi_checkpoint:
                    filename += f"_ep{epoch}"
                    torch.save(self.model.state_dict(), f"{self.save_dir}/{filename}.pth")
                else:
                    torch.save(self.model.state_dict(), f"{self.dir_save}/{filename}.pth")



            if (epoch+1) % 1 == 0:
                    trn_loss = 0
                    trn_correct = 0
                    trn_total = 0
                    val_loss = 0
                    val_correct = 0
                    val_total = 0
                    self.model.eval()
                    with torch.no_grad():
                        for _, (inputs, targets) in enumerate(self.trainloader):
                                inputs, targets = inputs.to(self.config.device), \
                                                targets.to(self.config.device, non_blocking=True)
                                outputs = self.model(inputs)
                                loss = self.loss_fn(outputs, targets)
                                trn_loss += loss.item()
                                _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()
                        self.trn_losses.append(trn_loss)
                        self.trn_acc.append(trn_correct / trn_total)
                    with torch.no_grad():        
                        for _, (inputs, targets) in enumerate(self.valloader):
                                inputs, targets = inputs.to(self.config.device), \
                                                  targets.to(self.config.device, non_blocking=True)
                                outputs = self.model(inputs)
                                loss = self.loss_fn(outputs, targets)
                                val_loss += loss.item()
                                _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                        self.val_losses.append(val_loss)
                        self.val_acc.append(val_correct / val_total)

                    if self.val_acc[-1] > self.curr_high:
                        self.curr_high = self.val_acc[-1]


                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "Validation accuracy": self.curr_high, 
                            "Val Loss": self.val_losses[-1]/100,
                            "loss": self.trn_losses[-1]/100, 
                            "Train Accuracy": self.trn_acc[-1]*100, 
                            "Epoch": epoch
                        })
                    
                    print("Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%".format(
                        epoch+1, 
                        self.config.numEpochs,
                        self.trn_losses[-1],
                        self.trn_acc[-1]*100
                    ))

                    print("Highest Accuracy:", self.curr_high)
                    print("Validation Accuracy:", self.val_acc[-1])
                    print("Validation Loss", self.val_losses[-1])
        
        return train_stats, val_stats


def get_model(args):
    arguments = type('', (), {'model': args.model.lower(), 'numClasses': args.numClasses, 
                              'device': args.device, 'in_chanls':args.inp_channels})()
    model_mapper = ModelMapper(arguments)
    return model_mapper.get_model()


def prepare_data(args, trainloader):
    if args.select_iter < args.numEpochs:
        imgntselloader = None
        pickle_dir = f"{args.dataset}_pickle"
        file = os.path.join(pickle_dir, f"V_{args.batch_size}.pkl")
        
        # Create pickle directory if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
            
        if os.path.exists(file):
            print("Loading existing pickle file")
            with open(file, 'rb') as f:
                data3 = pickle.load(f)
        else:
            print("Generating new pickle file")
            if args.dataset.lower() != "imagenet":
                V = feature_sel(trainloader, args.batch_size, device=args.device, decomp_type=args.decomp)
                data3 = V
                # Save pickle
                with open(file, 'wb') as f:
                    pickle.dump(V, f)
            else:
                imgntselloader = imagenet_selloader(args.dataset, dirs=args.dataset_dir, 
                                                       trn_batch_size=args.batch_size, 
                                                       val_batch_size=args.batch_size, 
                                                       tst_batch_size=1000, resize=32)
                
                V = feature_sel(imgntselloader, args.batch_size, device=args.device, decomp_type=args.decomp)
                data3 = V
                
                with open(file, 'wb') as f:
                    pickle.dump(V, f)
    else:
        data3 = None
    
    if args.dataset.lower() == "imagenet" and not imgntselloader:
        imgntselloader = imagenet_selloader(args.dataset, dirs=args.dataset_dir, 
                                            trn_batch_size=args.batch_size, 
                                            val_batch_size=args.batch_size, 
                                            tst_batch_size=1000, resize=32)
        
    return data3


def setup_tracker(args):
    if not ECO2AI_AVAILABLE:
        print("Warning: eco2ai not available, skipping emissions tracking")
        return None
        
    if args.warm_start:
        ttype = "warm"
    else:
        ttype = "nowarm"
        
    tracker = eco2ai.Tracker(
        project_name=f"{args.model}_dset-{args.dataset}_bs-{args.batch_size}", 
        experiment_description="training DEIM_IS model",
        file_name=f"emission_-{args.model}_dset-{args.dataset}_bs-{args.batch_size}_epochs-{args.numEpochs}_fraction-{args.fraction}_{args.optimizer}_{ttype}.csv"
    )
    return tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training with smart Sampling")
    parser.add_argument('--batch_size', default='128', type=int, required=True, help='(default=%(default)s)')
    parser.add_argument('--numEpochs', default='5', type=int, required=True, help='(default=%(default)s)')
    parser.add_argument('--numClasses', default='10', type=int, required=True, help='(default=%(default)s)')
    parser.add_argument('--lr', default='0.001', type=float, required=False, help='learning rate')
    parser.add_argument('--device', default='cuda', type=str, required=False, help='device to use for decompositions')
    parser.add_argument('--model', default='resnet50', type=str, required=False, help='model to train')
    parser.add_argument('--dataset', default="cifar10", type=str, required=False, help='Indicate the dataset')
    parser.add_argument('--dataset_dir', default="./cifar10", type=str, required=False, help='Imagenet folder')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use pretrained or not')
    parser.add_argument('--weight_decay', default=0.0001, type=float, required=False, help='Weight Decay to be used')
    parser.add_argument('--inp_channels', default="3", type=int, required=False, help='Number of input channels')
    parser.add_argument('--save_pickle', default=False,  action='store_true', help='to save or not to save U, S, V components')
    parser.add_argument('--decomp', default="numpy", type=str, required=False, help='To perform SVD using torch or numpy')
    parser.add_argument('--optimizer', default="sgd", type=str, required=True, help='Choice for optimizer')
    parser.add_argument('--select_iter', default="50", type=int, required=True, help='Data Selection Iteration')
    parser.add_argument('--fraction', default="0.50", type=float, required=True, help='fraction of data')
    parser.add_argument('--grad_clip', default=0.00, type=float, required=False, help='Gradient Clipping Value')
    parser.add_argument('--warm_start', default=False, action='store_true', help='Train with a warm-start')
    
    args = parser.parse_args()

    trainloader, valloader, trainset, valset = loader(dataset=args.dataset, dirs=args.dataset_dir, trn_batch_size=args.batch_size, val_batch_size=args.batch_size, tst_batch_size=1000)

    config = TrainingConfig.from_args(args)

    model = get_model(args)
    data3 = prepare_data(args, trainloader)
    
    trainer = ModelTrainer(config, model, trainloader, valloader, trainset, data3)
    
    tracker = setup_tracker(args)
    if tracker:
        tracker.start()
    
    train_stats, val_stats = trainer.train()
    
    if tracker:
        tracker.stop()


