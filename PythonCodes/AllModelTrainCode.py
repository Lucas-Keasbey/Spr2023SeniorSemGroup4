# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 2023

@author: Jacob
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import numpy as np
import BasicModel
import linearModel
import time
import DataSaver


def main():
    ##started at 12pm

    #Edit these for training for now
    BATCH_SIZE = 64
    learning_rate = 0.05
    num_epochs = 25

    ##picking model
    modelType = selectModelType()
  
    print("Running %s Model with %d epochs, %d batch size, and %.4f learning rate\n"%(modelType, num_epochs,BATCH_SIZE,learning_rate))
    
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform) #our training set

    # use 30% of training data for validation, 70% for training
    trainSetSize = int(len(trainset) * 0.7)
    validSetSize = int(len(trainset) * 0.3)

    # giving the validloader a random 30% of the trainingset, 70% to the trainloader
    seed = torch.Generator().manual_seed(42) #for randomness
    trainset, validset = data.random_split(trainset,[trainSetSize,validSetSize],generator=seed)
    validloader = data.DataLoader(validset, batch_size=1, shuffle=True)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    printSetStats(device,trainloader)
    model = model.to(device)

    displayTrainSet(trainloader)

    #declaring loss fucntion and optimizer
    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) ##SGD

    print("Running %s Model with %d epochs, %d batch size, and %.4f learning rate\n"%(modelType, num_epochs,BATCH_SIZE,learning_rate))
    
    saver = DataSaver.dataSaver(num_epochs, learning_rate, BATCH_SIZE, modelType)
    saver.initialize()
    
    print("\nBegining Training and Validation...\n")
    startTime = time.time()
    trainAndValidate(lossFunc, optimizer, trainloader, validloader, saver)
    finishTime = time.time();
    print('\nTime elaspsed (seconds): %.2f'%(finishTime - startTime))
    print("To test model, run the tesing code and with the respective model trained. You need to do this in order to save it")
    

def trainAndValidate(lossFunc, optimizer, trainloader, validloader, saver):
    bestAcc = 0.0 
    bestValidLoss = 999.99
    for epoch in range(num_epochs):
        #training loop
        model.train() #set model to training mode
      
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            
            images = images.to(device)
            labels = labels.to(device)
    
            guess = model(images) #give model images to guess
            loss = criterion(guess, labels) #find loss
            optimizer.zero_grad() #clear gradients
            loss.backward() #calc new gradients
            optimizer.step() #update weights
  
            # Calculate Loss
            train_loss += loss.detach().item()
        
        #validation loop
        model.eval() #set model to evaluation mode
        valid_loss = 0.0
        valid_running_acc = 0.0
        total = 0 
        for i, (images, labels) in enumerate(validloader, 0):
            images = images.to(device)
            labels = labels.to(device)

            guess = model(images) #give model images to guess
            loss = lossFunc(guess, labels) #find loss
            validLoss += loss.detach().item()

            #calc acc
            _, predicted = torch.max(logits, 1) #grabs the largest probability outputted by the model
            valid_running_acc += (predicted == labels).sum().item() #add together the ones it got right
            total += labels.size(0)
            

        accuracy = (100 * valid_running_acc / total) #divied the total it got right by the total
        
#        if accuracy > bestAcc: 
#            path = './Models/ModelsForTesting/%sModelTest.pth'%(modelType)
#            torch.save(model.state_dict(),path) #saving the most accurate instance of the model for testing
#            bestAcc = accuracy
        
        if validLoss < bestValidLoss:
            path = './Models/ModelsForTesting/%sModelTest.pth'%(modelType)
            torch.save(model.state_dict(),path) #saving the model when valid loss stops decreasing
            bestValidLoss = validLoss

        print('Epoch:%d | TrainingLoss:%.4f  | Validation Loss:%.4f | Accuracy:%.2f'%(epoch, train_loss / len(trainloader), valid_loss / len(validloader), best_accuracy)) 
        saver.saveRunData(epoch,(trainLoss  / len(trainloader)), (validLoss / len(validloader)), (accuracy))


def printSetStats(device, trainloader):
    print("The model will be running on", device, "device\n") 

    for images, labels in trainloader:
       print("Image batch dimensions:", images.shape)
       print("Image label dimensions:", labels.shape)
       break

def displayTrainSet(trainloader):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))

def selectModelType():
    modelType = ""
    while(true):
        modelType = input("What Model would you like to use? (Basic, Linear, CNN)")
        if(modelType.__eq__("Basic")):
            model = BasicModel();
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training
            break
        elif(modelType.__eq__("Linear")):
            model = linearModel();
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training
            break
        elif(modelType.__eq__("CNN")):
            break
        else:
            print("Awnser not valid, please try again")

        return modelType

main()