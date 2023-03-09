# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import numpy as np
import BasicModel

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##BATCH_SIZE = int(input("Enter Batch Size: "))
    ##learning_rate = float(input("Enter Learning Rate: "))
    ##num_epochs = int(input("Enter Number of Epochs: "))

    #for testing
    BATCH_SIZE = 64
    learning_rate = 0.01
    num_epochs = 2

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


    model = BasicModel.BasicModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) ##SGD
    
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))

    for images, labels in trainloader:
       print("Image batch dimensions:", images.shape)
       print("Image label dimensions:", labels.shape)
       break
    
    
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model.train() #set model to training mode

        for i, (images, labels) in enumerate(trainloader, 0):
            
            images = images.to(device)
            labels = labels.to(device)
    
            logits = model(images) #give model images to guess
            loss = criterion(logits, labels) #see how it did
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  
            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(logits, labels, BATCH_SIZE)
   
        model.eval()
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(epoch, train_running_loss / i, train_acc/i)) 
    
    
    PATH = "./Models/Test/BasicModel.pth"
    torch.save(model, PATH)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    #gets random image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()



main()