import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels = 3, out_channels =10, kernel_size=3)     
        self.conv_layer2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3)
        self.conv_layer3 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=20, out_channels=25, kernel_size=3)
        
        self.sig = nn.Sigmoid()
        self.leak = nn.LeakyReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.lin = nn.Linear(225, 10)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.leak(out)
        
        out = self.conv_layer2(out)
        out = self.leak(out)
        
        out = self.max_pool(out)
        
        out = self.conv_layer3(out)
        out = self.leak(out)
        
        out = self.conv_layer4(out)
        out = self.leak(out)
        
        out = self.max_pool(out)
                
        out = torch.flatten(out, 1)
        
        out = self.lin(out)
        return out