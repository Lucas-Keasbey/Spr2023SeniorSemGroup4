# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self, fileNames):
        self.fileNames = fileNames.copy()
        print("File is " + fileNames[0])
        self.data = self.createData(fileNames)

#Reads all of the data from the file names, then returns an array of a custom "data" object
    def createData(self):
        """
        Function
        -------
        Finds all the accuracies and losses for each trial using the file names

        Returns
        -------
        d : array[DataSet]
            array of each trials dataset

        """
        d = []        
        
        #Looks at every file in the name list
        for name in self.fileNames:
            #variables for the "data" object
            trial, epochs, lr, trainLoss, valLoss, acc, model = -1, 0, 0.0, [], [], [], "Basic"
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
                        valLoss.append(float(piece[1]))
                    elif piece[0] == "Accuracy":
                        acc.append(float(piece[1]))
                    elif piece[0] == "Accuaracy":
                        acc.append(float(piece[1]))
                    elif piece[0] == "Model":
                        model = piece[1]
                        
                        
            #Adds the data object into the array                                            
            d.append(DataSet(trial, epochs, lr, trainLoss, valLoss, acc, model))
         
        #returns all of the data
        return d
    
#Plots the loss of each trial
    def plotLoss(self, id):
        """
        Function
        ----------
        Plots the training and validation loss at each epoch of a specific trial

        Parameters
        ----------
        id : int
            The id of the trial. The function looks through all datasets until it finds the one with the correct id.

        Returns
        -------
        None.

        """
        for trial in self.data:
            if(trial.trial==id):
                plt.plot(trial.trainLoss, label = "Training")
                plt.plot(trial.valLoss, label = "Validation")
                plt.title("Loss\nModel:%s\nTrial:%d Learning Rate:%.3f"%(trial.model, trial.trial, trial.lr))
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
                
#Plots the accuracy of each trial
    def plotAcc(self):
        """
        Function
        -------
        Plots the accuracy at each epoch for each trial in the plotter

        Returns
        -------
        None.

        """
        for trial in self.data:
            name = "Model:%s Trial:%d LR:%.3f Epochs:%d"%(trial.model, trial.trial, trial.lr, trial.epochs)
            plt.plot(trial.acc, label = name)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.title("TrainingAccuracy")
        plt.show()    
            
class DataSet:
    def __init__(self, trial, epochs, lr, trainLoss, valLoss, acc, model):
        """
        Function
        ----------
        Creates a data set, which is a list of the accuracy, training loss, and validation loss of a trial at each epoch

        Parameters
        ----------
        trial : int
            Which trial number it is
        epochs : int
            Number of epochs the trial was run for
        lr : double
            learning rate the trial had
        trainLoss : array[double]
            array of training loss at each epoch
        valLoss : array[double]
            array of validation loss at each epoch
        acc : array[double]
            array of accuracy at each epoch
        model : string
            what model the trial was run using

        Returns
        -------
        None.

        """
        self.trial = trial
        self.epochs = epochs
        self.lr = lr
        self.trainLoss = trainLoss.copy()
        self.valLoss = valLoss.copy()
        self.acc = acc.copy()
        self.model = model
        