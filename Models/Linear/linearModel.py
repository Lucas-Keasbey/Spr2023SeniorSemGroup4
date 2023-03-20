# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:46:39 2023

@author: Lucas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


learningRate = 0.01
targetVal = 5

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 5)
        self.fc5 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x

net = Net()
target = targetVal
criterion = nn.BCELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

output = net(input)