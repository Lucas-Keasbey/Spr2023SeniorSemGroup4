
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import numpy as np

#This code is a very primitive training code for inital testing and reference, please run AllModelTrainCode instead

def main():
    gpu_available = torch.cuda.is_available()
    print(f"Gpu Available? {gpu_available}")
    print("Using torch",torch.__version__)
    BATCH_SIZE = 64
    ## transformations
    transform = transforms.Compose([transforms.ToTensor()])
    
    
    ## download and load training dataset
    # This is the dataset that will be using 'root = where the dataset is', 
    #train if the data is pretrained, download if it is downloaded
    trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=False, transform=transform)
    
    #DataLoader(What is the dataset, what is the batch size, 
    #if you would like to shuffle the data(optional), num of cpu cores(optional))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # For more information about custom data with images: more information can be
    # Found https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # https://www.pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/
    # https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/
    
    ## functions to show an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        #gets random images
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ##plt.show(npimg)
    
    ## get some random training images
    #iterates through the dataset
    dataiter = iter(trainloader)
    #iterates the images and labels together 
    images, labels = next(dataiter)
    
    ## show images
    imshow(torchvision.utils.make_grid(images))
    
    for images, labels in trainloader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        break
    
    model = MyModel()
    for images, labels in trainloader:
        print("batch size:", images.shape)
        out = model(images)
        print(out.shape)
        break
    

    learning_rate = 0.001
    num_epochs = 3
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) ##gradient descent
    ##optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) ##SGD
    
    #1 epoch = training_data_size / batch_size
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0
    
        model = model.train()
    
        ## training step
        #Goes through a loop and with the images from 
        for i, (images, labels) in enumerate(trainloader):
            
            images = images.to(device)
            labels = labels.to(device)
    
            ## forward + backprop + loss
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
    
            ## update model params
            optimizer.step()
    
            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(logits, labels, BATCH_SIZE)
        
        model.eval()

        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(epoch, train_running_loss / i, train_acc/i)) 
    
    
    PATH = "./Models/Test/model.pth"
    torch.save(model.state_dict(), PATH)

    model.load_state_dict(torch.load(PATH)) 
    
    
## compute accuracy
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
    
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.d1 = nn.Linear(26 * 26 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)

        # logits => 32x10
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out
    
main()