# -*- coding: utf-8 -*-
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
import time
import DataSaver

def main():

#   !!!This is the old training code, do not use!!!
#   !!!Parameters for various functions have been changed and it will not run correctly!!!
#   !!!Please use AllModelTrainCode.py!!!!
    
    while(true):
        awns = input("This is old training code, please run AllModelTrainCode.py. If you want to run anyway, type Y")
        if(awns == "Y"):
            break
    ##BATCH_SIZE = int(input("Enter Batch Size: "))
    ##learning_rate = float(input("Enter Learning Rate: "))
    ##num_epochs = int(input("Enter Number of Epochs: "))

    #for testing
    BATCH_SIZE = 64
    learning_rate = 0.05
    num_epochs = 25

    print("Running Model with %d epochs, %d batch size, and %.4f learning rate\n"%(num_epochs,BATCH_SIZE,learning_rate))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    
    # use 20% of training data for validation, 20% for testing
    train_set_size = int(len(trainset) * 0.6)
    valid_set_size = int(len(trainset) * 0.2)
    test_set_size = int(len(trainset) * 0.2)

    # giving the validloader a random 20% of the trainingset, 60% to the trainloader, 20% to testloader
    seed = torch.Generator().manual_seed(42)
    trainset, validset, testset = data.random_split(trainset,[train_set_size,valid_set_size, test_set_size],generator=seed)
    validloader = data.DataLoader(validset, batch_size=1, shuffle=True)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=True) ##test set in on a training set
    
    model = BasicModel.BasicModel()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("The model will be running on", device, "device\n") 
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
    
    saver = DataSaver.dataSaver(num_epochs, learning_rate, BATCH_SIZE)
    saver.initialize()
   
    print("\nBegining Training and Validation...\n")
    startTime = time.time()
    
    best_accuracy = 0.0 
    for epoch in range(num_epochs):
        total = 0 

        model.train() #set model to training mode
        
        ##training loop
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            
            images = images.to(device)
            labels = labels.to(device)
    
            logits = model(images) #give model images to guess
            loss = criterion(logits, labels) #find loss
            optimizer.zero_grad() #clear gradients
            loss.backward() #calc new gradients
            optimizer.step() #update weights
  
            # Calculate Loss
            train_loss += loss.detach().item()
            #train_acc += get_accuracy(logits, labels, BATCH_SIZE)
   
        model.eval()
        #print('Epoch:%d\t|TrainingLoss:%.4f\t|Train Accuracy:%.2f'%(epoch, train_loss / i, train_acc/i)) 
        

        valid_loss = 0.0
        valid_running_acc = 0.0
        for i, (images, labels) in enumerate(validloader, 0):
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass
            logits = model(images) 
            # Find the Loss
            loss = criterion(logits, labels) #find loss
            
            # Calculate Loss
            valid_loss += loss.detach().item()
            #valid_acc += get_accuracy(logits, labels, BATCH_SIZE)
            
            #calc acc
            _, predicted = torch.max(logits, 1)
            valid_running_acc += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = (100 * valid_running_acc / total)
        
        if accuracy > best_accuracy: 
            torch.save(model.state_dict(),'./Models/ModelsForTesting/BasicModelTest.pth') #saving model for testing
            best_accuracy = accuracy
        #inlcude saving when valid loss stops decreasing
        print('Epoch:%d | TrainingLoss:%.4f  | Validation Loss:%.4f | Accuracy:%.2f'%(epoch, train_loss / len(trainloader), valid_loss / len(validloader), best_accuracy)) 
        saver.saveRunData(epoch,(train_loss  / len(trainloader)), (valid_loss / len(validloader)), (accuracy))

       

        
    finishTime = time.time();
    print('\nTime elaspsed (seconds): %.2f'%(finishTime - startTime))

    print("\nBeginning Testing...\n")
    
    # Load the model that we saved at the end of the training loop  
    modelTest = BasicModel.BasicModel()
    modelTest.load_state_dict(torch.load("./Models/ModelsForTesting/BasicModelTest.pth")) 
     
    running_accuracy = 0.0
    total = 0 
 
    modelTest.eval()
    with torch.no_grad(): 
        for i, (images, labels) in enumerate(testloader, 0):

            images = images.to(device)
            labels = labels.to(device)
            
            labels = labels.to(torch.float32) 
            predicted_outputs = model(images) 
            _, predicted = torch.max(predicted_outputs, 1) 
            total += labels.size(0) 
            running_accuracy += (predicted == labels).sum().item() 

        bestTestAcc = (100 * running_accuracy / total)
        print("TestingAccuracy: %.2f"%(bestTestAcc)) 
        saver.saveTestAcc(bestTestAcc);



    strAwns = input("Would you like to save the trained model (Y/N)?\n")
    PATH = "./Models/TrainedModels/BasicModel.pth"
    if(strAwns == "Y"):
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