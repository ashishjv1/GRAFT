import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import PIL.Image as Image

class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None, isreg=False):
        self.transform = transform
        self.isreg = isreg
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            if isreg:
                self.targets = target.float().to(device)
            else:
                self.targets = target.long().to(device)

        else:
            self.data = data.float()
            if isreg:
                self.targets = target.float()
            else:
                self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label)  # .astype('float32')

class standard_scaling:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit_transform(self, data):
        self.std = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data

    def transform(self, data):
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data
    
# class TinyImageNetDataset(Dataset):
#     def __init__(self, root_dir, split='train', transform=None):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#             split (string): Either 'train', 'val', or 'test'.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform
#         self.classes = os.listdir(os.path.join(root_dir, split))
#         self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
#         self.class_name = self._get_names()
#         self.images = self._load_images()

#     def _load_images(self):
#         images = []
#         for cls in self.classes:
#             cls_dir = os.path.join(self.root_dir, self.split, cls, 'images')
#             for image_file in os.listdir(cls_dir):
#                 image_path = os.path.join(cls_dir, image_file)
#                 images.append((image_path, self.class_to_idx[cls]))
#         return images

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path, label = self.images[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

#     def _get_names(self):
#         entity_dict = {}
#         # Open the text file
#         with open('tiny-imagenet-200/words.txt', 'r') as file:
#             # Read each line
#             for line in file:
#                 # Split the line into key and value using tab ('\t') as delimiter
#                 key, value = line.strip().split('\t')
        
#                 first = value.strip().split(',')
#                 # Add the key-value pair to the dictionary
#                 entity_dict[key] = first[0]
#                 # entity_dict.append(line)
#         return entity_dict

class TinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): Either 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root
#         self.root_dir = os.path.join(root, main_dir) 
        self.split = split
        self.transform = transform
        self.classes = []
        with open(os.path.join(self.root_dir, 'wnids.txt'), 'r') as f:
            self.classes = f.read().strip().split()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_name = self._get_names()
        self.images = self._load_images()        
        # self.repl_str = str_out

    def _load_images(self):
        images = []
        if self.split == 'train':
            for cls in self.classes:
                cls_dir = os.path.join(self.root_dir, self.split, cls, 'images')
                for image_file in os.listdir(cls_dir):
                    image_path = os.path.join(cls_dir, image_file)
                    images.append((image_path, self.class_to_idx[cls]))
                    
        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, self.split, 'images')
            image_to_cls = {}
            with open(os.path.join(self.root_dir, self.split, 'val_annotations.txt'), 'r') as f:
                for line in f.read().strip().split('\n'):
                    # print(line)
                    image_to_cls[line.split()[0].strip()] = line.split()[1].strip()                  
            for image_file in os.listdir(val_dir):
                # print(image_file)
                image_path = os.path.join(val_dir, image_file)
                images.append((image_path, self.class_to_idx[image_to_cls[image_file]]))
                
            # for cls in self.classes:
            #     cls_dir = os.path.join(self.root_dir, self.split, cls, 'images')
            #     for image_file in os.listdir(cls_dir):
            #         image_path = os.path.join(cls_dir, image_file)
            #         images.append((image_path, self.class_to_idx[cls]))
            
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def _get_names(self):
        entity_dict = {}
        # Open the text file
        with open(os.path.join(self.root_dir, 'words.txt'), 'r') as file:
            # Read each line
            for line in file:
                # Split the line into key and value using tab ('\t') as delimiter
                key, value = line.strip().split('\t')
                first = value.strip().split(',')
                # Add the key-value pair to the dictionary
                entity_dict[key] = first[0]
        return entity_dict


def loader(dataset, dirs="./cifar10", trn_batch_size=64, val_batch_size=64, tst_batch_size=1000):
    """Load and return data loaders for the specified dataset"""
    
    if dataset.lower() == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=trn_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=tst_batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader, trainset, testset
        
    elif dataset.lower() == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=trn_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=tst_batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader, trainset, testset
        
    elif dataset.lower() == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dir = os.path.join(dirs, 'train')
        val_dir = os.path.join(dirs, 'val')
        
        train_dataset = torchvision.datasets.ImageFolder(
            train_dir,
            train_transform
        )
        
        val_dataset = torchvision.datasets.ImageFolder(
            val_dir,
            val_transform
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=trn_batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader, train_dataset, val_dataset
    
    else:
        raise ValueError(f"Dataset {dataset} not supported")

if __name__ == "__main__":
    # Run some basic tests
    try:
        # Test CIFAR10
        train_l, test_l, train_s, test_s = loader("cifar10")
        # Test CIFAR100 
        train_l, test_l, train_s, test_s = loader("cifar100")
        print("✓ All test cases passed successfully!")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
