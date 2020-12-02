#!/bin/python
"""
ECE695: HW3
author: Rahul Deshmukh
email: deshmuk5@purdue.edu
"""

import numpy as np

import torch 
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset

#from google.colab import drive
#drive.mount('/content/gdrive')

# initialize seeds
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class TemplateNet_SingleLayer(nn.Module):
  def __init__(self, no_padding=True):
    super().__init__()
    # valid convolution
    self.conv1 = nn.Conv2d(3, 128, 3)   #A
    self.conv2 = nn.Conv2d(128, 128, 3) #B
    self.input_size_of_denselayer = 128*15*15 # 28800
    self.fc1 = nn.Linear(self.input_size_of_denselayer, 1000)   #C

    self.pool = nn.MaxPool2d(2, 2)
    self.fc2 = nn.Linear(1000, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    # x = self.pool(F.relu(self.conv2(x))) #D
    x = x.view(-1, self.input_size_of_denselayer)                  #E
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class TemplateNet_TwoLayer(nn.Module):
  def __init__(self, no_padding=True):
    super().__init__()
    if no_padding:
      # same convolution
      """ size transition: 32 --Conv1--> 30 --MaxPool1--> 15 --Conv2--> 13 --MaxPool2--> 6 """
      self.conv1 = nn.Conv2d(3, 128, 3)   #A
      self.conv2 = nn.Conv2d(128, 128, 3)  #B
      self.input_size_of_denselayer = 128*6*6 #
      self.fc1 = nn.Linear(self.input_size_of_denselayer, 1000)           #C
    else:
      """ size transition: 32 --Conv1--> 32 --MaxPool1--> 16 --Conv2--> 14 --MaxPool2--> 7 """
      self.conv1 = nn.Conv2d(3, 128, 3, padding=1)   #A
      self.conv2 = nn.Conv2d(128, 128, 3)  #B
      self.input_size_of_denselayer = 128*7*7 #6272
      self.fc1 = nn.Linear(self.input_size_of_denselayer, 1000)           #C

    self.pool = nn.MaxPool2d(2, 2)
    self.fc2 = nn.Linear(1000, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x))) #D
    x = x.view(-1, self.input_size_of_denselayer)                  #E
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

def run_code_for_training(net, train_data_loader, device, epochs=10):
  net = net.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if (i+1)% 2000 == 0:
        if (i+1)==12000: print("[epoch:%d, batch:%d] loss: %.3f" %(epoch + 1, i + 1, running_loss / float(2000)))
        running_loss = 0.0

def run_code_for_testing(net, test_data_loader, device):
  confusion_matrix = torch.zeros(10,10)
  with torch.set_grad_enabled(False):
    net.eval()
    net.to(device)  
    for i,data in enumerate(test_data_loader):
      x, y_true = data
      x, y_true = x.to(device), y_true.to(device)
      y_pred = net(x)
      _, y_pred = torch.max(F.softmax(y_pred, dim=1), dim=1)
      for b in range(y_pred.shape[0]): confusion_matrix[y_true[b], y_pred[b]] += 1
  print(confusion_matrix)

# ----------- Main Code ------------------#
  
if __name__ == "__main__":
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    # load CIFAR-10 dataset 
    dataset_train = torchvision.datasets.CIFAR10("./data_train", train=True, transform=transform, download=True)
    dataset_test = torchvision.datasets.CIFAR10("./data_test", train=False, transform = transform, download=True)
    # make data loader
    batch_size= 4
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net1 = TemplateNet_SingleLayer()
    run_code_for_training(net1, dataloader_train, device, epochs=1)
    
    # Task 2
    net2 = TemplateNet_TwoLayer()
    run_code_for_training(net2, dataloader_train, device, epochs=1)
    
    # Task 3
    net3 = TemplateNet_TwoLayer(no_padding=False)
    run_code_for_training(net3, dataloader_train, device, epochs=1)
    
    # Taks 4: confusion matrix
    run_code_for_testing(net3, dataloader_test, device)
