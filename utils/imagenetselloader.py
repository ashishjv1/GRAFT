# from libauc.datasets import CheXpert
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os



def imagenet_selloader(dataset, dirs="./imagenet", trn_batch_size=64, val_batch_size=64, tst_batch_size=1000, resize=32):
    

    if dataset.lower() == "imagenet":
        # Define the data transforms
       
        traindir = os.path.join(dirs, 'train')
        valdir = os.path.join(dirs, 'val')
        

        fullset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        )

#         testset = datasets.ImageFolder(
#             valdir,
#             transforms.Compose([
#                 transforms.Resize(resize),
#                 transforms.CenterCrop(resize),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])
#         )
        

    

    
    # Creating the Data Loaders
    trainloader = torch.utils.data.DataLoader(fullset, batch_size=trn_batch_size,
                                              shuffle=False, pin_memory=True, num_workers=2)

#     valloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size,
#                                             shuffle=False, pin_memory=True, num_workers=2)

#     testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
#                                               shuffle=False, pin_memory=True, num_workers=1)
    
    return trainloader