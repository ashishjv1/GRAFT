import torch
from utils.loader import loader
import torch.optim as optim
from tqdm import tqdm
import copy
import torch.nn as nn
from decompositions import feature_sel
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import utils.pickler as pickler
import pickle
import wandb
from genindices import sample_selection
import os
import eco2ai
from utils.model_mapper import ModelMapper
from torch.utils.data import Subset, DataLoader
import gc
from utils.imagenetselloader import imagenet_selloader
# from timm.models import create_model
from scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from apex import amp
# torch.autograd.set_detect_anomaly(True)


def main(numEpochs, batch_size, device, net, 
         model_name, dataset_name, trainloader, valloader, 
         trainset, data3, optimizer_name, lr, weight_decay, 
         grad_clip, fraction, 
         selection_iter, warm_start, 
         imgntselloader, sched="cosine", warmup=True):
    
    
    total = 0
    correct = 0
    trn_losses = list()
    val_losses = list()
    trn_acc = list()
    val_acc = list()  
    selection = 0
    curr_high = 0
    # update_step = 1
    
    
    if dataset_name.lower() == 'boston':
        loss_fn = nn.MSELoss(reduction='mean')
    elif model_name.lower() == "swin":
#         loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss_fn = nn.CrossEntropyLoss()
    else:
#         loss_fn = nn.functional.cross_entropy
        loss_fn = nn.CrossEntropyLoss()
        
        
    dir_save = f"saved_models/{model_name}"
#     save_dir = f"{dir_save}/multi_checkpoint"
    
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    if warmup:
        warmup_steps = 300
        
        if sched == "cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=numEpochs * warmup_steps)
        else:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=warmup_steps * numEpochs)
            
    else:
        if sched.lower() == "onecycle":
            scheduler = lr_scheduler.OneCycleLR(optimizer, lr, epochs=numEpochs, 
                                                        steps_per_epoch=len(trainloader))
        elif sched.lower() == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
        else:
            lr = lr        

    wandb.login()
    
    if model_name.lower() == "swinb" or model_name.lower() == "swinl":
        model, optimizer = amp.initialize(models=net, optimizers=optimizer, opt_level='O2')
        amp._amp_state.loss_scalers[0]._loss_scale = 2**18
            

    config = {"lr": lr, "batch_size": batch_size}
    config.update({"architecture": f'{net}'})
    
    main_trainloader = trainloader
    

    
    for epoch in range(numEpochs):
        net.train()
        tot_train_loss = 0
        before_lr = optimizer.param_groups[0]["lr"]
        pruned_samples = 0
        total_samples = 0

        if (epoch) % selection_iter == 0:
            if warm_start and selection == 0:
                trainloader = trainloader
                selection += 1
            else:
                train_model = net
                cached_state_dict = copy.deepcopy(train_model.state_dict())
                clone_dict = copy.deepcopy(train_model.state_dict())

                if not imgntselloader:
                    indices = sample_selection(main_trainloader, data3, net, 
                                              clone_dict, batch_size, fraction, 
                                              selection_iter, numEpochs, device, dataset_name)
                else:
                    indices = sample_selection(imgntselloader, data3, net,
                                              clone_dict, batch_size, fraction,
                                              selection_iter, numEpochs, device, dataset_name)

                net.load_state_dict(cached_state_dict)
                selection += 1

                datasubset = Subset(trainset, indices)
                new_trainloader = DataLoader(datasubset, batch_size=batch_size,
                                             shuffle=True, pin_memory=False, num_workers=2)
                trainloader = new_trainloader

                del cached_state_dict
                del clone_dict
                del train_model
                torch.cuda.empty_cache()
                gc.collect()

        for _, (trainsamples, labels) in enumerate(tqdm(trainloader)):
                 
            if wandb.run is None:
                name = f"wd{weight_decay}_opt{optimizer_name}_bs{batch_size}_gclip{grad_clip}_lr{lr}_f{fraction}_siter{selection_iter}"
                wandb.init(project=f"Smart_Sampling_{model_name}_{dataset_name}", config=config, name=name)
            
            
            trainsamples = trainsamples.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            X = trainsamples
            Y = labels
            pred = net(X)

            loss = loss_fn(pred, Y.to(device))
            
            

            if model_name.lower() == 'swinb' or model_name.lower() == 'swinl':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     print(scaled_loss)
                    scaled_loss.backward()
            else:
                loss.backward()

            tot_train_loss += loss.item()

            if grad_clip:
                if model_name.lower() == 'swin':
                    nn.utils.clip_grad_norm_(amp.master_params(optimizer), grad_clip)
                else:
                    nn.utils.clip_grad_value_(net.parameters(), grad_clip)
                
            
                
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(pred.cpu().data, 1)
            total += Y.size(0)
            correct += (predicted == Y.cpu()).sum().item()
            pruned_samples += len(trainsamples) - len(X)
            total_samples += len(trainsamples)

            if sched.lower() == "onecycle":
                scheduler.step()


        if sched.lower() == "cosine":
            scheduler.step()
            
        after_lr = optimizer.param_groups[0]["lr"]
        print("Last Epoch [%d] -> Current Epoch [%d]: lr %.4f -> %.4f optimizer %s" % (epoch, epoch+1, before_lr, after_lr, optimizer_name))


        if epoch % 20 == 0:
            dir_parts = dir_save.split('/')
            current_dir = ''

            for part in dir_parts:
                current_dir = os.path.join(current_dir, part)
                if not os.path.exists(current_dir):
                    os.makedirs(current_dir)

#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
                
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)

            if selection_iter > numEpochs:
                file_prefix = "Full"
            else:
                file_prefix = "Sampled"

#             if multi_checkpoint:
#                 file_prefix += "_multi"

            filename = f"{file_prefix}_{dataset_name}_sch{sched}_si{selection_iter}_f{fraction}"
#             if multi_checkpoint:
#                 filename += f"_ep{epoch}"
#                 torch.save(net.state_dict(), f"{save_dir}/{filename}.pth")
#             else:
            torch.save(net.state_dict(), f"{dir_save}/{filename}.pth")



        if (epoch+1) % 1 == 0:
                trn_loss = 0
                trn_correct = 0
                trn_total = 0
                val_loss = 0
                val_correct = 0
                val_total = 0
                net.eval()
                with torch.no_grad():
                    for _, (inputs, targets) in enumerate(trainloader):
                            inputs, targets = inputs.to(device), \
                                              targets.to(device, non_blocking=True)
                            outputs = net(inputs)
                            loss = loss_fn(outputs, targets)
                            trn_loss += loss.item()
                            _, predicted = outputs.max(1)
                            trn_total += targets.size(0)
                            trn_correct += predicted.eq(targets).sum().item()
                    trn_losses.append(trn_loss)
                    trn_acc.append(trn_correct / trn_total)
                with torch.no_grad():        
                    for _, (inputs, targets) in enumerate(valloader):
                            inputs, targets = inputs.to(device), \
                                              targets.to(device, non_blocking=True)
                            outputs = net(inputs)
                            loss = loss_fn(outputs, targets)
                            val_loss += loss.item()
                            _, predicted = outputs.max(1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                    val_losses.append(val_loss)
                    val_acc.append(val_correct / val_total)

                if val_acc[-1] > curr_high:
                    curr_high = val_acc[-1]


                wandb.log({"Validation accuracy": val_acc[-1], "Val Loss":val_losses[-1]/100,
                        "loss": trn_losses[-1]/100, "Train Accuracy": trn_acc[-1]*100, "Epoch": epoch})
                
                wandb.log({"loss": trn_losses[-1]/100, "Train Accuracy": trn_acc[-1]*100, "Epoch": epoch})      

                print("Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%".format(epoch+1, numEpochs,
                                                                                    trn_losses[-1],
                                                                                    trn_acc[-1]*100))

                print("Highest Accuracy:", curr_high)
                print("Validation Accuracy:", val_acc[-1])
                print("Validation Loss", val_losses[-1])


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
    parser.add_argument('--warmup', default=True, action='store_true', help='Warm-up scheduler')
    parser.add_argument('--sched', default="cosine", type=str, required=False, help='Choice for scheduler for fixed learning rate use --sched="fixed" and assign lr using the --lr switch')
    
    
    args = parser.parse_args()

    trainloader, valloader, trainset, valset = loader(dataset=args.dataset, dirs=args.dataset_dir, trn_batch_size=args.batch_size, val_batch_size=args.batch_size, tst_batch_size=1000)

    arguments = type('', (), {'model': args.model.lower(), 'numClasses': args.numClasses, 
                              'device': args.device, 'in_chanls':args.inp_channels, 'pretrained':args.pretrained})()
    model_mapper = ModelMapper(arguments)
    net = model_mapper.get_model()
    
    
    if args.warm_start:
        ttype = "warm"
    else:
        ttype = "nowarm"
        
    tracker = eco2ai.Tracker(
    project_name=f"{args.model}_dset-{args.dataset}_bs-{args.batch_size}", 
    experiment_description="training BTG model",
    file_name=f"emission_-{args.model}_dset-{args.dataset}_bs-{args.batch_size}_epochs-{args.numEpochs}_fraction-{args.fraction}_{args.optimizer}_{ttype}.csv"
    )

    tracker.start()
    if args.select_iter < args.numEpochs:
        imgntselloader = None
        
        if args.save_pickle:
            ###Check if V for this batch already exists 
                    
            file = os.path.join(f"{args.dataset}" + "_pickle", f"V_{args.batch_size}.pkl")
            if os.path.exists(file):
                print("saved pickle exists")
                with open(file, 'rb') as f:
                    data3 = pickle.load(f)
            else:
                print("saving Pickle")
                if args.dataset.lower() != "imagenet":
                    V = feature_sel(trainloader, args.batch_size, device=args.device, decomp_type=args.decomp)
                    data3 = V
                else:
                    imgntselloader = imagenet_selloader(args.dataset, dirs=args.dataset_dir, 
                                                           trn_batch_size=args.batch_size, 
                                                           val_batch_size=args.batch_size, 
                                                           tst_batch_size=1000, resize=32)
                    
                    V = feature_sel(imgntselloader, args.batch_size, device=args.device, decomp_type=args.decomp)
                    data3 = V
                    
                    
                pickler.save_pickle(V, f"{args.dataset}", args.batch_size)
        else:
            # Construct the directory name based on the dataset name
            print("Expecting Pickle was saved Trying to load pickle for the provided batch size")

            data3 = pickler.load_pickle(args.dataset, args.batch_size)
            if data3 is not None:
                print("Pickle loaded successfully.")
            else:
                print("Failed to load pickle.")
    else:
        data3 = None
        imgntselloader = None
        
    if args.dataset.lower() == "imagenet" and not imgntselloader:
        imgntselloader = imagenet_selloader(args.dataset, dirs=args.dataset_dir, 
                                            trn_batch_size=args.batch_size, 
                                            val_batch_size=args.batch_size, 
                                            tst_batch_size=1000, resize=32)

    main(args.numEpochs, args.batch_size, args.device, net, args.model, args.dataset, trainloader, valloader, trainset, data3, 
         args.optimizer, args.lr, args.weight_decay, args.grad_clip, args.fraction, args.select_iter, args.warm_start, imgntselloader, 
         sched=args.sched, warmup=args.warmup)
    tracker.stop()
    

