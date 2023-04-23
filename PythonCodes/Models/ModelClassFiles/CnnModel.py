# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels = 28*3, out_channels = 28*4, kernel_size=1)     
        self.conv_layer2 = nn.Conv2d(in_channels=28*4, out_channels=28*5, kernel_size=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=28*5, out_channels=28*4, kernel_size=1)
        self.conv_layer4 = nn.Conv2d(in_channels=28*4, out_channels=28*3, kernel_size=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.sig1 = nn.Sigmoid()
        self.fc1 = nn.Linear(5844, 10)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.sig1(out)
        out = self.fc1(out)
        return out
