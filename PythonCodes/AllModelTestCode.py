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
import sys
import matplotlib
#matplotlib.use('Agg')
import numpy as np 
from sklearn.metrics import f1_score
from Models.ModelClassFiles import BasicModel
from Models.ModelClassFiles import LinearModel
from Models.ModelClassFiles import CNN
import time
import DataSaver
from sklearn import metrics
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help = "decides what model to train. Options are Basic, Linear, or CNN", type=str)
    args = parser.parse_args()
    
    modelType = modelType = args.model if args.model else "Basic"
    
    print("Beginning Testing...\n")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device= torch.device("cpu") #dont really need gpu for testing

    modelTest, transform = selectModelType(modelType)
    modelTest = modelTest.to(device)
    testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=False, transform=transform)
    test_set_size = int(len(testset) * 0.3) #give the test set 30% of the entire data set
    testloader = data.DataLoader(testset, batch_size=1, shuffle=True)
   
    #load model
    path = '../PythonCodes/Models/ModelsForTesting/%sModelTest.pth'%(modelType)
    
    modelTest.load_state_dict(torch.load(path, map_location=device))
    print("\nTesting %s model\n"%(modelType))



    testAcc(testloader,modelTest,device)



def selectModelType(modelType):
    """
    Function
    ----------
    Selects the model that is being tested.

    Parameters
    ----------
    modelType : String
        The model type being tested. Supplied by the command line argument

    Returns
    -------
    model : nn.module
        The model being tested. Either Basic, Linear or CNN
    transform : transform

    """
    modelType = ""
    while(True):
        modelType = input("What Model would you like to Test? (Basic, Linear, CNN): ")
        if(modelType.__eq__("Basic")):
            model = BasicModel.BasicModel()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for testing (turning data into 3 channels)
            break
        elif(modelType.__eq__("Linear")):
            model = LinearModel.Net()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(1,1,1))]) #maniputlating the set to feed into the model for testing
            break
        elif(modelType.__eq__("CNN")):
            model = CNN.ConvModel();
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))]) #maniputlating the set to feed into the model for training (turning data into 3 channels)
            break
        else:
            print("Awnser not valid, please try again")

    return model, transform

def testAcc(testloader, modelTest, device):
    """
    Function
    ----------
    Gives the model the test batch and sees how it does.

    Parameters
    ----------
    testloader : data loader
        the object with all the images for testing
    modelTest : nn.module
        The model being tested
    device : torch.device
        The drive that the training will be run on.
        Either a CPU or GPU

    Returns
    -------
    None.

    """
    
    modelTest.eval()
    totalTestAcc = 0.0
    total = 0
    predictions = []
    actuals = []
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    with torch.no_grad(): 
        for i, (images, labels) in enumerate(testloader, 0):

            images = images.to(device)
            labels = labels.to(device)
            
            labels = labels.to(torch.float32) #formatting for comparison
            predicted_outputs = modelTest(images) #ask the model to guess the image
            value, prediction = torch.max(predicted_outputs, 1) #grab the highest prediction
            actuals = np.append(actuals,labels) #add the actual label to the tensor
            predictions = np.append(predictions,prediction) #add the prediction to the tensor
            totalTestAcc += metrics.accuracy_score(labels,prediction) #determine the accuracy



            total += labels.size(0) #keep track of the total amount

        testAcc = (totalTestAcc / total) * 100    
               
        #Scores for all classes:
        #Macro scores
        macroPrecision = metrics.precision_score(actuals, predictions, average = 'macro')  
        macroRecall = metrics.recall_score(actuals, predictions, average = 'macro')  
        macroF1 = metrics.f1_score(actuals, predictions, average = 'macro')
        
        #Weighted scores
        weightedPrecision =  metrics.precision_score(actuals, predictions, average = 'weighted')
        weightedRecall = metrics.recall_score(actuals, predictions, average = 'weighted')
        weightedF1 = metrics.f1_score(actuals, predictions, average = 'weighted')


        print("Overall Accuracy: %.2f"%(testAcc))
        print("Macro Precision: %.4f, Macro Recall: %.4f, Macro F1Score: %.4f"%(macroPrecision,macroRecall,macroF1))
        print("Weighted Precision: %.4f, Weighted Recall: %.4f, Weighted F1Score: %.4f"%(weightedPrecision,weightedRecall,weightedF1))


        confusion_matrix = metrics.confusion_matrix(actuals, predictions)
        testClassAcc(confusion_matrix)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['0-T-shirt','1-Trouser','2-Pullover','3-Dress','4-Coat','5-Sandal','6-Shirt','7-Sneaker','8-Bag','9-Ankle Boot'])
        #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4,5,6,7,8,9]) #x-axis spacing is weird

        cm_display.plot()
        plt.show()

        plt.savefig(sys.stdout.buffer)

        #confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        #confusion_matrix.diagonal()

     

def testClassAcc(cm):
    """
    Function
    ----------
    Identifies stats from the training such as confusion matrices, and precision

    Parameters
    ----------
    cm : confusion matrix
        The entire list of guesses and answers to create the plot.

    Returns
    -------
    None.

    """
    per_class_accuracies = {}
    classes = [0,1,2,3,4,5,6,7,8,9]
    # Calculate the accuracy for each one of our classes
    #idx = class id, cls = class
    for idx, cls in enumerate(classes):
        print("\n----Class %d----"%(cls))
    
        # True positives are all the samples of our current GT class that were predicted as such
        tp = cm[idx,idx] #diagonal (correct ones)
        fn = np.sum(cm[idx]) - tp #the entire row minus the correct ones
        fp = np.sum(cm[:,idx]) - tp #the entire column minus the correct ones
        tn = np.sum(cm) - tp - fn - fp #everything else
        
        
        
        recall = tp / (tp + fn) #formula for recall
        precision = tp / (tp + fp) #formula for precision
        f1 = 2 * (precision * recall) / (precision + recall) #formula for f1

    
        # The accuracy for the current class is the ratio between correct predictions to all predictions
        per_class_accuracies[cls] = (tp + tn) / np.sum(cm)
        print("Overall Accuracry: %.2f"%(per_class_accuracies[cls] * 100))
        print("Recall: %.4f, Precision: %.4f, F1 Score: %.4f"%(recall,precision,f1))
        print("Confusion Matrix: (Actual by Predictied)")
        print("| TP:%d FN:%d |\n| FP:%d TN:%d |"%(tp,fn,fp,tn))
    

    

main()

