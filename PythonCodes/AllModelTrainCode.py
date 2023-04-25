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
from Models.ModelClassFiles import BasicModel
from Models.ModelClassFiles import LinearModel
from Models.ModelClassFiles import CNN
import time
import DataSaver
import argparse


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help = "decides what model to train. Options are Basic, Linear, or CNN", type=str)
    parser.add_argument("--num_epochs", help = "decides the number of epochs. Default value is 30", type=int)
    parser.add_argument("--lr", help = "decides the learning rate for the model. Default is 0.01", type=float)
    args = parser.parse_args()
    #Edit these for training for now
    BATCH_SIZE = 64
    
    
    learningRate = args.lr if args.lr else 0.01
    numEpochs = args.num_epochs if args.num_epochs else 25

    ##picking model
    modelType = args.model if args.model else "Basic"
    model, transform = selectModelType(modelType)
  
    print("Running %s Model with %d epochs, %d batch size, and %.4f learning rate\n"%(modelType, numEpochs, BATCH_SIZE, learningRate))
    
    #For running on Visual Studio
    #trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform) #our training set
    
    #For running on Spyder
    trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=False, transform=transform) #our training set

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
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9) ##SGD

    
    saver = DataSaver.dataSaver(numEpochs, learningRate, BATCH_SIZE, modelType)
    #saver.initialize()
    
    print("\nBegining Training and Validation...\n")
    startTime = time.time()
    trainAndValidate(model, numEpochs, lossFunc, optimizer, trainloader, validloader, saver, device, modelType)
    finishTime = time.time();
    print('\nTime elaspsed (seconds): %.2f'%(finishTime - startTime))
    ##print("To test model, run the tesing code and with the respective model trained. You need to do this in order to save it")

    strAwns = input("Would you like to save the trained model (Y/N)?\n")
    PATH = ("./PythonCodes/Models/TrainedModels/%sModel.pth"%(modelType))
    if(strAwns == "Y"):
        torch.save(model, PATH)


    

def trainAndValidate(model, numEpochs, lossFunc, optimizer, trainloader, validloader, saver, device, modelType):
    """
    

    Parameters
    ----------
    model : nn.module
        The model being trained. Either:
            Basic
            Linear
            CNN
    numEpochs : int
        The number of epochs the model is being trained for
    lossFunc : nn.CrossEntropyLoss
        The function we use to calcualte loss. In this case, it is cross entropy
    optimizer : torch.optim.SGD
        The function we use to optimize the model. In this case, it is stochastic gradient descent
    trainloader : Data loader
        The loaded set of training images
    validloader : Data loader
        The loaded set of validation images
    saver : datasaver
        Saves the accuracy and loss at each epoch into a text file
    device : torch.device
        The drive that the training will be run on.
        Either a CPU or GPU
    modelType : string
        Describes what type the model is. Used in the data saver.

    Returns
    -------
    None.

    """
    bestAcc = 0.0 
    bestValidLoss = 999.99
    for epoch in range(numEpochs):
        #training loop
        model.train() #set model to training mode
      
        trainLoss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            
            images = images.to(device)
            labels = labels.to(device)
    
            guess = model(images) #give model images to guess
            loss = lossFunc(guess, labels) #find loss
            optimizer.zero_grad() #clear gradients
            loss.backward() #calc new gradients
            optimizer.step() #update weights
  
            # Calculate Loss
            trainLoss += loss.detach().item()
        
        #validation loop
        model.eval() #set model to evaluation mode
        validLoss = 0.0
        validRunAcc = 0.0
        total = 0 
        for i, (images, labels) in enumerate(validloader, 0):
            images = images.to(device)
            labels = labels.to(device)

            guess = model(images) #give model images to guess
            loss = lossFunc(guess, labels) #find loss
            validLoss += loss.detach().item()

            #calc acc
            value, predicted = torch.max(guess, 1) #grabs the largest probability outputted by the model
            validRunAcc += (predicted == labels).sum().item() #add together the ones it got right
            total += labels.size(0)

        accuracy = (100 * validRunAcc / total) #divied the total it got right by the total
        
#        if accuracy > bestAcc: 
#            path = './Models/ModelsForTesting/%sModelTest.pth'%(modelType)
#            torch.save(model.state_dict(),path) #saving the most accurate instance of the model for testing
#            bestAcc = accuracy
        
        #'./Models/ModelsForTesting/BasicModelTest.pth'
        if (validLoss / len(validloader)) < bestValidLoss:
            path = './PythonCodes/Models/ModelsForTesting/%sModelTest.pt'%(modelType)
            torch.save(model.state_dict(),path) #saving the model when valid loss stops decreasing
            bestValidLoss = validLoss

        print('Epoch:%d | TrainingLoss:%.4f  | ValidationLoss:%.4f | Accuracy:%.2f'%(epoch, trainLoss / len(trainloader), validLoss / len(validloader), accuracy)) 
        #saver.saveRunData(epoch,(trainLoss  / len(trainloader)), (validLoss / len(validloader)), (accuracy))


def printSetStats(device, trainloader):
    """
    Function
    ----------
    Prints the stats of the training as it begins.
    Displays what device the training will be run on as well as the size of the batches

    Parameters
    ----------
    device : torch.device
        The drive that the training will be run on.
        Either a CPU or GPU
    trainloader : Data loader
        The loaded set of training images

    Returns
    -------
    None.

    """
    print("The model will be running on", device, "device\n") 

    for images, labels in trainloader:
       print("Image batch dimensions:", images.shape)
       print("Image label dimensions:", labels.shape)
       break

def displayTrainSet(trainloader):
    """
    Function
    ----------
    Displays the set of data that is being trained. Mainly used to verify it was loaded correctly

    Parameters
    ----------
    trainloader : Data loader
        The loaded set of training images

    Returns
    -------
    None.

    """
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    #gets random image
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))

def selectModelType(modelType):
    """
    Function
    ----------
    Decides what model is being used. There are three options:
        Basic
        Linear
        CNN
    If none of the options are entered, it chooses Basic as the default
    
    Parameters
    ----------
    modelType : String
        

    Returns
    -------
    model : nn.Module
        The actual model that will be trained
    transform : transform
        The transform
    """
    if(modelType.__eq__("Basic")):
        model = BasicModel.BasicModel()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training
    elif(modelType.__eq__("Linear")):
        model = LinearModel.Net()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(1,1,1))]) #maniputlating the set to feed into the model for training
    elif(modelType.__eq__("CNN")):
        model = CNN.ConvModel();
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training
    else:
        print("Answer not valid, using Basic as the default")
        model = BasicModel.BasicModel()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training

    return model, transform

main()