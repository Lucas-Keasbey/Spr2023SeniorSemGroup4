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
from sklearn.metrics import f1_score
from Models.ModelClassFiles import BasicModel
from Models.ModelClassFiles import LinearModel
import time
import DataSaver

def main():
    print("Beginning Testing...\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    modelType, modelTest, transform = selectModelType()
    modelTest = modelTest.to(device)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    test_set_size = int(len(testset) * 0.3)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=True)
   
    #load model
    path = './PythonCodes/Models/ModelsForTesting/%sModelTest.pth'%(modelType)
    #C:\Users\starm\Source\Repos\Spr2023SeniorSemGroup4\PythonCodes\Models\ModelsForTesting\BasicModelTest.pth
    modelTest.load_state_dict(torch.load('./PythonCodes/Models/ModelsForTesting/BasicModelTest.pth'))
    print("\nTesting %s model\n"%(modelType))



    testAcc(testloader,modelTest,device)
    testClassAcc(testloader,modelTest,device)
    calcF1Score(testloader,modelTest,device)



    




def selectModelType():
    modelType = ""
    while(True):
        modelType = input("What Model would you like to Test? (Basic, Linear, CNN): ")
        if(modelType.__eq__("Basic")):
            model = BasicModel.BasicModel()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training
            break
        elif(modelType.__eq__("Linear")):
            model = LinearModel.Net()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(1,1,1))]) #maniputlating the set to feed into the model for training
            break
        elif(modelType.__eq__("CNN")):
            print("Not implemented yet, selct another")
        else:
            print("Awnser not valid, please try again")

    return modelType, model, transform

def testAcc(testloader, modelTest, device):
    BestTestAcc = 0.0
    runningAcc = 0.0
    total = 0 
    modelTest.eval()
    with torch.no_grad(): 
        for i, (images, labels) in enumerate(testloader, 0):

            images = images.to(device)
            labels = labels.to(device)
            
            labels = labels.to(torch.float32) 
            predicted_outputs = modelTest(images) 
            _, predicted = torch.max(predicted_outputs, 1) 
            total += labels.size(0) 
            runningAcc += (predicted == labels).sum().item() 

            #f1Score
            f1Score = f1_score(labels.data, predicted)

        bestTestAcc = (100 * runningAcc / total)
        print("TestingAccuracy: %.2f, F1 Score: %.2f"%(bestTestAcc,f1Score)) 

def testClassAcc(testloader, model, device):
    for images, labels in testloader:
        numLabels = len(labels.shape)
        break
    
    #confusionmatix.sipy
    classCorrect = list(0. for i in range(10))
    classTotal = list(0. for i in range(10))
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader, 0):
            images = images.to(device)
            labels = labels.to(device)

            guess = model(images)
            _, predicted = torch.max(guess, 1) ##grabs the highest probability
            correct = (predicted == labels).squeeze()
            for i in range(10):#64=batch size
                label = labels[i]
                classCorrect[label] += correct[i].item()
                classTotal[label] += 1

    for i in range(numberLabels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * classCorrect[i] / classTotal[i]))

main()