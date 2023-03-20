# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self, fileNames):
        self.fileNames = fileNames.copy()
        self.data = self.createData(fileNames)
    
#Sets the number of files. Was created then lost its need. Still here just in case though
    def setNumFiles(fileNames):
        num = 0
        for names in fileNames:
            num += 1
        return num

#Reads all of the data from the file names, then returns an array of a custom "data" object
    def createData(fileNames):
        d = []        
        
        #Looks at every file in the name list
        for name in fileNames:
            #variables for the "data" object
            trial, epochs, lr, loss, acc = -1, 0, 0.0, [], []
            f = open(name, "r")
            for line in f:
                #Each line is split up into pieces of data separated by tabs
                parts = line.split("\t")
                for p in parts:
                    #Each piece is then split by the colon into name of data and actual value.
                    #It is then recorded into the correct place
                    piece = p.split(":")
                    if piece[0] == "Trial":
                        trial = int(piece[1])
                    elif piece[0] == "Epochs":
                        epochs = int(piece[1])
                    elif piece[0] == "Learning Rate":
                        lr = float(piece[1])
                    elif piece[0] == "loss":
                        loss.append(float(piece[1]))
                    elif piece[0] == "acc":
                        acc.append(float(piece[1]))
                        
            #Adds the data object into the array                                            
            d.append(DataSet(trial, epochs, lr, loss, acc))
         
        #returns all of the data
        return d
    
#Plots the loss of each trial
    def plotLoss(self):
        for trial in self.data:
            name = "Trial:" + trial.trial +"\tLearning Rate:" + trial.lr
            plt.plot(trial.loss, label = name)
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss")
            plt.show()
                
#Plots the accuracy of each trial
    def plotAcc(self):
        for trial in self.data:
            name = "Trial:" + trial.trial +"\tLearning Rate:" + trial.lr
            plt.plot(trial.acc, label = name)
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Acc")
            plt.title("Accuracy")
            plt.show()    
            
class DataSet:
    def __init__(self, trial, epochs, lr, loss, acc):
        self.trial = trial
        self.epochs = epochs
        self.lr = lr
        self.loss = loss.copy()
        self.acc = acc.copy()
        