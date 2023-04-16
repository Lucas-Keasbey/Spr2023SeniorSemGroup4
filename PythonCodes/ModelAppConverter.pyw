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
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
import DataSaver

def main():
    model = LinearModel.Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(1,1,1))])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    test_set_size = int(len(testset) * 0.3)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=True)
    images = None
    with torch.no_grad(): 
        for i, (images, labels) in enumerate(testloader, 0):

            images = images.to(device)
            labels = labels.to(device)


   
    path = './PythonCodes/Models/ModelsForTesting/LinearModelTest.pth'
    #C:\Users\starm\Source\Repos\Spr2023SeniorSemGroup4\PythonCodes\Models\ModelsForTesting\BasicModelTest.pth
    model.load_state_dict(torch.load('./PythonCodes/Models/ModelsForTesting/LinearModelTest.pth'))
    #print("\nTesting %s model\n"%(modelType))
    model.eval();

    # Generate some random noise
    X = torch.distributions.uniform.Uniform(-10000, 10000).sample((4, 2))

    # Generate the optimized model
    traced_script_module = torch.jit.trace(model,images)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)

     # Save the optimzied model
    traced_script_module_optimized._save_for_lite_interpreter("./PythonCodes/Models/TrainedModels/LinearModelApp.pt")

main();
