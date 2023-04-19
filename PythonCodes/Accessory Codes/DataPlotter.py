# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self, fileNames):
        self.fileNames = fileNames.copy()
        print("File is " + fileNames[0])
        self.data = self.createData(fileNames)
    
#Sets the number of files. Was created then lost its need. Still here just in case though
    def setNumFiles(fileNames):
        num = 0
        for names in fileNames:
            num += 1
        return num

#Reads all of the data from the file names, then returns an array of a custom "data" object
    def createData(self, fileNames):
        d = []        
        
        #Looks at every file in the name list
        for name in fileNames:
            #variables for the "data" object
            trial, epochs, lr, trainLoss, testLoss, acc, model = -1, 0, 0.0, [], [], [], "Basic"
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
                    elif piece[0] == "lr":
                        lr = float(piece[1])
                    elif piece[0] == "LearningRate":
                        lr = float(piece[1])
                    elif piece[0] == "Loss":
                        trainLoss.append(float(piece[1]))
                    elif piece[0] == "TrainingLoss":
                        trainLoss.append(float(piece[1]))
                    elif piece[0] == "ValidLoss":
                        testLoss.append(float(piece[1]))
                    elif piece[0] == "Accuracy":
                        acc.append(float(piece[1]))
                    elif piece[0] == "Accuaracy":
                        acc.append(float(piece[1]))
                    elif piece[0] == "Model":
                        model = piece[1]
                        
                        
            #Adds the data object into the array                                            
            d.append(DataSet(trial, epochs, lr, trainLoss, testLoss, acc))
         
        #returns all of the data
        return d
    
#Plots the loss of each trial
    def plotLoss(self, id):
        for trial in self.data:
            if(trial.trial==id):
                plt.plot(trial.trainLoss, label = "Training")
                plt.plot(trial.testLoss, label = "Testing")
                plt.title("Loss\nModel:%s Trial:%d Learning Rate:%.3f"%(trial.model, trial.trial, trial.lr))
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
                
#Plots the accuracy of each trial
    def plotAcc(self):
        for trial in self.data:
            name = "Model:%s Trial:%d LR:%.3f Epochs:%d"%(trial.model, trial.trial, trial.lr, trial.epochs)
            plt.plot(trial.acc, label = name)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.title("TrainingAccuracy")
        plt.show()    
            
class DataSet:
    def __init__(self, trial, epochs, lr, trainLoss, testLoss, acc, model):
        self.trial = trial
        self.epochs = epochs
        self.lr = lr
        self.trainLoss = trainLoss.copy()
        self.testLoss = testLoss.copy()
        self.acc = acc.copy()
        self.model = model
        