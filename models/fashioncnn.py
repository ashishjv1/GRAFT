import torch.nn as nn
import torch
from torch import Tensor


class FashionCNN(nn.Module):
    
    def __init__(self,in_channels, num_classes):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_classes)
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.drop(out)
#         out = self.fc2(out)
#         out = self.fc3(out)
        
#         return out
    
    def forward(self, x: Tensor, last=False, freeze=False) -> Tensor:
        # See note [TorchScript super()]
        if freeze:
            with torch.no_grad():
                self.eval()
                x = self.layer1(x)
                x = self.layer2(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.drop(x)
                x = self.fc2(x)
#                 x = self.fc3(x)
                features = torch.flatten(x, 1)
                self.train()
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.drop(x)
            x = self.fc2(x)
#             x = self.fc3(x)
            features = torch.flatten(x, 1)

        out = self.fc3(features)
        if last:
            return out, features
        else:
            return out