class dataSaver:
    #numEpochs = 0
    #learnRate = 0.0
    #batchSize = 0
    #trialFile = None
    #trialNum = 0
    #writeFile = None
    #writeFileName = ""

    def __init__(self, numEpochs, learnRate, batchSize, modelType):
        self.numEpochs = numEpochs
        self.learnRate = learnRate
        self.batchSize = batchSize
        self.modelType = modelType

    def initialize(self):
        ##get trialnum
        trialFile = open('./TrainingData/trialNum.txt', 'r')
        trialNumString = trialFile.read(3)
        trialNum = int(trialNumString)
        trialFile.close()

        ##update for trial num for next trial
        newtrailNum = (trialNum + 1)
        trialFile = open('./TrainingData/trialNum.txt', 'w')
        strNewTrialNum = ("%d"%(newtrailNum))
        trialFile.write(strNewTrialNum) 
        trialFile.close()

        ##First line of trial
        self.writeFileName = ('./TrainingData/Trial%d.txt'%(trialNum))
        writeFile = open(self.writeFileName,'w')
        strinit = "Trial:%d\tEpochs:%d\tLearningRate:%.3f\tBatchSize:%d\tModel:%s"%(trialNum, self.numEpochs, self.learnRate, self.batchSize, self.modelType)
        writeFile.write(strinit)
        writeFile.close()
    
    #saves the validation data from model
    def saveRunData(self, epoch, trainloss, validloss, acc):
        writeFile = open(self.writeFileName,'a')
        writeFile.write("Epoch:%d\tTrainingLoss:%.3f\tValidLoss:%.3f\tAccuracy:%.3f\n"% (epoch, trainloss, validloss, acc))
        writeFile.close()

    def saveTestAcc(self,acc):
        writeFile = open(self.writeFileName,'a')
        writeFile.write("TestingAccuracy:%.2f\n"% (acc))
        writeFile.close()





