class dataSaver:
    #numEpochs = 0
    #learnRate = 0.0
    #batchSize = 0
    #trialFile = None
    #trialNum = 0
    #writeFile = None
    #writeFileName = ""

    def __init__(self, numEpochs, learnRate, batchSize):
        self.numEpochs = numEpochs
        self.learnRate = learnRate
        self.batchSize = batchSize

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
        strinit = "Trial:%d\tEpochs:%d\tLearningRate:%.2f\tBatchSize:%d\n"%(trialNum, self.numEpochs, self.learnRate, self.batchSize)
        writeFile.write(strinit)
        writeFile.close()
    
    #saves the validation data from model
    def saveRunData(self, epoch, loss, acc):
        writeFile = open(self.writeFileName,'a')
        writeFile.write("Epoch:%d\tLoss:%.2f%\tAccuaracy:%.2f\n"% (epoch, loss, acc))
        writeFile.close()

    def saveTestAcc(self,acc):
        writeFile = open(self.writeFileName,'a')
        writeFile.write("Testing Accuracy:%d%\n"% (acc))
        writeFile.close()





