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
from Models.ModelClassFiles import CNN
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
import DataSaver

def main():
    modelType, model, transform = selectModelType()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
   
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    test_set_size = int(len(testset) * 0.3)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=True)
    images = None
    with torch.no_grad(): 
        for i, (images, labels) in enumerate(testloader, 0):

            images = images.to(device) #getting random testing data to give to the model
            labels = labels.to(device)


   
    path = ('./PythonCodes/Models/ModelsForTesting/%sModelTest.pth'%(modelType))
    #C:\Users\starm\Source\Repos\Spr2023SeniorSemGroup4\PythonCodes\Models\ModelsForTesting\BasicModelTest.pth
    model.load_state_dict(torch.load('./PythonCodes/Models/ModelsForTesting/%sModelTest.pth'%(modelType)))
    #print("\nTesting %s model\n"%(modelType))
    model.eval();

    # Generate some random noise
    X = torch.distributions.uniform.Uniform(-10000, 10000).sample((4, 2))

    # Generate the optimized model
    traced_script_module = torch.jit.trace(model,images)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)

    # Save the optimzied model
    traced_script_module_optimized._save_for_lite_interpreter("./PythonCodes/Models/TrainedModels/%sModelApp.pt"%(modelType))
    print("Done!")


def selectModelType():
    modelType = ""
    while(True):
        modelType = input("What Model would you like to reformat? (Basic, Linear, CNN): ")
        if(modelType.__eq__("Basic")):
            model = BasicModel.BasicModel()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training
            break
        elif(modelType.__eq__("Linear")):
            model = LinearModel.Net()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(1,1,1))]) #maniputlating the set to feed into the model for training
            break
        elif(modelType.__eq__("CNN")):
            model = CNN.ConvModel();
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training
            break
        else:
            print("Awnser not valid, please try again")

    return modelType, model, transform

main();
