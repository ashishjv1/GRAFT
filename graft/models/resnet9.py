import torch.nn as nn
import torch
from torch import Tensor


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)
#         self.classifier = nn.Sequential(nn.MaxPool2d(4), 
#                                         nn.Flatten(), 
#                                         nn.Linear(512, num_classes))
        
#     def forward(self, xb):
#         out = self.conv1(xb)
#         out = self.conv2(out)
#         out = self.res1(out) + out
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.res2(out) + out
#         out = self.classifier(out)
#         return out
    
    def _forward_impl(self, x: Tensor, last=False, freeze=False) -> Tensor:
        # See note [TorchScript super()]
        if freeze:
            with torch.no_grad():
                self.eval()
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.res1(x) + x
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.res2(x) + x
                x = self.pool(x)
                features = torch.flatten(x, 1)
                self.train()
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.res1(x) + x
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.res2(x) + x
            x = self.pool(x)
            features = torch.flatten(x, 1)

        out = self.fc(features)
        if last:
            return out, features
        else:
            return out

    def forward(self, x: Tensor, last=False, freeze=False) -> Tensor:
        return self._forward_impl(x, last, freeze)