class dataSaver:
    ##to be deleted/merged
    numEpochs = 0
    learnRate = 0.0
    batchSize = 0
    trialFile = None
    trialNum = 0
    writeFile = None

    def __init__(self, numEpochs, learnRate, batchSize):
        self.numEpochs = numEpochs
        self.learnRate = learnRate
        self.batchSize = batchSize

    def initialize(self):
        ##get trialnum
        trialFile = open('./TrainingData/trials.txt', 'r')
        trialNumString = trialFile.read(100)
        trialNum = int(trialNumString)
        trialFile.close()

        ##update for trial num for next trial
        fileName = "Trial"
        trialFile = open('./TrainingData/trials.txt', 'w')
        trialFile.write(trialNum)
        trialFile.close()

        ##First line of trial
        writeFileName = ("%d",trialNum)
        writeFile = open(writeFileName,'w')
        writeFile.write("Trial: %d Epochs: %d LearningRate: %f BatchSize: %d", trialNum, numEpochs, learnRate, batchSize)
        writeFile.close()
    
    def saveRunData(epoch, loss, acc):
        writeFile = open(writeFileName,'a')
        writeFile.write("Epoch: %d Loss: %f Accuaracy: %f", epoch, loss, acc)
        writeFile.close()






