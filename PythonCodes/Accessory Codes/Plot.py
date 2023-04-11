# -*- coding: utf-8 -*-
from DataPlotter import DataPlotter

filePrefix = "..\\..\\TrainingData\\GraphingData\\"

files = [""] * 3

# Data from just training. LR: 0.0003, 0.010, 0.100, Epochs: 15
# for i in range(3):
#     files[i] = filePrefix + "trial%d_lr.txt"%(i+1)

# Data from training and testing. All data
# for i in range(9):
#     files[i] = filePrefix + "Trial%d.txt"%(i+139)

# Data from training and testing: LR 0.010, Epochs: 23, 25, 50
# files[0] = filePrefix + "Trial143.txt"
# files[1] = filePrefix + "Trial140.txt"
# files[2] = filePrefix + "Trial139.txt"

# Data from training and testing: LR: 0.007, 0.008, 0.009, Epochs: 20
files[0] = filePrefix + "Trial145.txt"
files[1] = filePrefix + "Trial146.txt"
files[2] = filePrefix + "Trial147.txt"

# Data from training and testing: LR: 0.01, 0.03, 0.05, Epochs: 25
# files[0] = filePrefix + "Trial140.txt"
# files[1] = filePrefix + "Trial141.txt"
# files[2] = filePrefix + "Trial149.txt"

print("Loading Plotter")

Plot = DataPlotter(files)

Plot.plotLoss(147)

#Plot.plotAcc()

