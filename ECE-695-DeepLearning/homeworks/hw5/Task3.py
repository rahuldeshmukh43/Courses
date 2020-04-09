"""
# ECE695 DL HW5: Object Detection
# Task3
Author: Rahul Deshmukh
email: deshmuk5@purdue.edu

Sources:
1. DLStudio: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.1.0.html#DLStudio
2. For IOU this was helpful: https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py

In this task, I first train a noise classifier using all datasets. For the noise classifier, I have chosen a simple 4 layered network with 2 conv followd by 2 fully connected layers (The same network as HW3 submission). Since we are trying to estimate the amount of noise in the images, the noise classifier should essentially be able to compute the sum of all pixels in the image. So higher the noise, higher would be the sum which can be used as way to identify noise level. To get the sum of pixels in the image a shallow network should be good enough.

After training the noise classifier, I am not retraining new networks. Instead, I have modified the inference logic. In my testing code, I first predict the noise level using the noise classifier then I use the previously trained model corresponding to the predicted noise level to carry out inference. This can be thought of as an ensemble.

Upon, running the code for my inference logic I notice that the performance deteriorates. This might be because we have not trained the individual models for more than 2 epochs. 

NOTE: I have deviced a strategy to improve the accuracy in Task4.

The output of this task is:

+Task 3:
Noise Classification Accuracy: 100.00%
Noise Confusion Matrix:
[[1000.    0.    0.    0.]
 [   0. 1000.    0.    0.]
 [   0.    0. 1000.    0.]
 [   0.    0.    0. 1000.]]

Dataset00 Classification Accuracy:	 82.30%
Dataset00 Confusion Matrix:
[[160.   0.   3.  34.   3.]
 [  4. 196.   0.   0.   0.]
 [ 12.   2. 186.   0.   0.]
 [ 90.   0.   1. 109.   0.]
 [ 20.   2.   6.   0. 172.]]

Dataset20 Classification Accuracy:	 65.20%
Dataset20 Confusion Matrix:
[[141.  14.   0.  36.   9.]
 [  1. 198.   0.   1.   0.]
 [ 21. 121.  25.  23.  10.]
 [ 82.   3.   0. 112.   3.]
 [ 10.  14.   0.   0. 176.]]

Dataset50 Classification Accuracy:	 40.70%
Dataset50 Confusion Matrix:
[[ 80.  13.  20.  44.  43.]
 [ 28.  69.  47.   8.  48.]
 [ 33.  29.  67.  11.  60.]
 [109.   7.  12.  40.  32.]
 [ 18.  10.  15.   6. 151.]]

Dataset80 Classification Accuracy:	 33.80%
Dataset80 Confusion Matrix:
[[51. 17. 19. 82. 31.]
 [33. 66. 31. 36. 34.]
 [33. 32. 53. 38. 44.]
 [60.  6. 22. 96. 16.]
 [44. 26. 30. 28. 72.]]
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

import gzip, pickle

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class PurdueShapes5Dataset(Dataset):
  img_size = 32
  root_dir = './gdrive/My Drive/ECE695_DL/HW5/data/'
  def __init__(self, noise_lvl, train=True, transform= None, smoothing= None):
    super().__init__()
    self.noise_lvl = noise_lvl # a string- 00,20,50,80
    self.transform = transform
    if train:
      self.dataset_name = 'PurdueShapes5-10000-train-noise-'+noise_lvl+'.gz'
    else:
      self.dataset_name = 'PurdueShapes5-1000-test-noise-'+noise_lvl+'.gz'

    # smoothing is a dictionary {'kernel size':kernel_size, 'sigma':sigma}
    if smoothing is not None: 
      self.kernel = PurdueShapes5Dataset._smoothing_kernel(smoothing)
    else:
      self.kernel = None

    if os.path.exists('torch_saved_'+ self.dataset_name+'_data.pt'):
      self.dataset = torch.load('torch_saved_'+ self.dataset_name+'_data.pt')
      self.label_map = torch.load('torch_saved_'+ self.dataset_name+'_label_map.pt')
      # reverse the key-value pairs in the label dictionary:
      self.class_labels = dict(map(reversed, self.label_map.items()))
    else:
      print('Loading data for first time, this might take some time')
      f = gzip.open(self.root_dir + self.dataset_name, 'rb')
      dataset = f.read()
      if sys.version_info[0] == 3:
        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
      else:
        self.dataset, self.label_map = pickle.loads(dataset)
      torch.save(self.dataset, 'torch_saved_'+ self.dataset_name+'_data.pt')
      torch.save(self.label_map,'torch_saved_'+ self.dataset_name+'_label_map.pt')
      # reverse the key-value pairs in the label dictionary:
      self.class_labels = dict(map(reversed, self.label_map.items()))

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,idx):
    r = np.array( self.dataset[idx][0] )/255.
    g = np.array( self.dataset[idx][1] )/255.
    b = np.array( self.dataset[idx][2] )/255.
    R,G,B = r.reshape(self.img_size,self.img_size), g.reshape(self.img_size,self.img_size), b.reshape(self.img_size,self.img_size)
    if self.kernel is not None:
      R = np.clip(signal.convolve2d(R, self.kernel, mode= 'same'),0.,1.)
      G = np.clip(signal.convolve2d(G, self.kernel, mode= 'same'),0.,1.)
      B = np.clip(signal.convolve2d(B, self.kernel, mode = 'same'),0.,1.)
    im_tensor = torch.zeros(3,self.img_size,self.img_size, dtype=torch.float32)
    im_tensor[0,:,:] = torch.from_numpy(R)
    im_tensor[1,:,:] = torch.from_numpy(G)
    im_tensor[2,:,:] = torch.from_numpy(B)
    if self.transform:
      im_tensor = self.transform(im_tensor)
    sample = {'image' : im_tensor, 
              'bbox'  : torch.tensor(self.dataset[idx][3], dtype = torch.float32),                          
              'label' : torch.tensor(self.dataset[idx][4]) }
    return sample
  
  @staticmethod
  def _smoothing_kernel(smoothing):
    kernel_size = smoothing['kernel size']
    sigma = smoothing['sigma']
    var = sigma**2
    mean = (kernel_size-1)//2
    x_pos = np.arange(kernel_size)
    kernel_x = np.tile(x_pos,(kernel_size,1)) -mean
    kernel_y = kernel_x.T
    kernel = np.stack((kernel_x,kernel_y)).transpose(1,2,0)
    kernel = np.power(np.linalg.norm(kernel,axis=2),2)
    kernel = (1./(2.*np.pi*var))*np.exp(-(1./2.*var)*kernel)
    kernel = kernel/np.sum(kernel)      
    return kernel

def shapes_dataloader(train_dataset, test_dataset, batch_size=4, num_workers=4):
  train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle = True, num_workers=num_workers)
  test_dataloader = DataLoader(test_dataset, batch_size= batch_size, num_workers=num_workers)
  return [train_dataloader, test_dataloader]

##test dataset
#smoothing_params= {'kernel size': 7, 'sigma':2}
#train_dataset = PurdueShapes5Dataset('PurdueShapes5-10000-train-noise-20.gz',smoothing = smoothing_params)
##noise_train_dataset = PurdueShapes5Dataset('PurdueShapes5-10000-train-noise-20.gz')
#
#l = len(train_dataset)
#for i in np.random.randint(0,l, 10):
#  a = train_dataset[i]
#  #b = noise_train_dataset[i]
#  plt.imshow(a['image'].numpy().transpose(1,2,0).astype(np.int))
#             #b['image'].numpy().transpose(1,2,0).astype(np.int))))

"""## Model definition"""

# LOADNet2
class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super().__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        norm_layer = nn.BatchNorm2d
        self.bn = norm_layer(out_ch)
        if downsample:
            self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
    def forward(self, x):
        identity = x                                     
        out = self.convo(x)                              
        out = self.bn(out)                              
        out = torch.nn.functional.relu(out)
        if self.in_ch == self.out_ch:
            out = self.convo(out)                              
            out = self.bn(out)                              
            out = torch.nn.functional.relu(out)
        if self.downsample:
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if self.skip_connections:
            if self.in_ch == self.out_ch:
                out += identity                              
            else:
                out[:,:self.in_ch,:,:] += identity
                out[:,self.in_ch:,:,:] += identity
        return out

class LOADnet2(nn.Module):
  def __init__(self, skip_connections=True, depth=32):
      super().__init__()
      self.pool_count = 3
      self.depth = depth // 2
      self.conv = nn.Conv2d(3, 64, 3, padding=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.skip64 = SkipBlock(64, 64,skip_connections=skip_connections)
      self.skip64ds = SkipBlock(64, 64,downsample=True, skip_connections=skip_connections)
      self.skip64to128 = SkipBlock(64, 128,skip_connections=skip_connections )
      self.skip128 = SkipBlock(128, 128, skip_connections=skip_connections)
      self.skip128ds = SkipBlock(128,128, downsample=True, skip_connections=skip_connections)
      self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
      self.fc2 =  nn.Linear(1000, 5)
      ##  for regression
      self.conv_seqn = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True)
      )
      self.fc_seqn = nn.Sequential(
          nn.Linear(16384, 1024),
          nn.ReLU(inplace=True),
          nn.Linear(1024, 512),
          nn.ReLU(inplace=True),
          nn.Linear(512, 4)
      )

  def forward(self, x):
      x = self.pool(F.relu(self.conv(x)))          
      ## The labeling section:
      x1 = x.clone()
      for _ in range(self.depth // 4):
          x1 = self.skip64(x1)                                               
      x1 = self.skip64ds(x1)
      for _ in range(self.depth // 4):
          x1 = self.skip64(x1)                                               
      x1 = self.skip64to128(x1)
      for _ in range(self.depth // 4):
          x1 = self.skip128(x1)                                               
      x1 = self.skip128ds(x1)                                               
      for _ in range(self.depth // 4):
          x1 = self.skip128(x1)                                               
      x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
      x1 = torch.nn.functional.relu(self.fc1(x1))
      x1 = self.fc2(x1)
      ## The Bounding Box regression:
      x2 = self.conv_seqn(x)
      x2 = self.conv_seqn(x2)
      # flatten
      x2 = x2.view(x.size(0), -1)
      x2 = self.fc_seqn(x2)
      return x1,x2

# test network
#smoothing_params= {'kernel size': 7, 'sigma':2}
#train_dataset = PurdueShapes5Dataset('PurdueShapes5-10000-train-noise-20.gz',smoothing = smoothing_params)
#noise_train_dataset = PurdueShapes5Dataset('PurdueShapes5-10000-train-noise-20.gz')
#net = ObjDetNet60()
#a = train_dataset[0]
#x1,x2 = net(a['image'].unsqueeze(0))
#plt.imshow(a['image'].numpy().transpose(1,2,0).astype(np.int))
#print(a['bbox'], a['label'], train_dataset.class_labels)
#print(x1,x2)

"""## IOU Loss definition"""

class IOU_Loss(nn.Module):
  def __init__(self, eps=1e-10):
    super().__init__()
    self.eps = eps

  def forward(self, pred_bbx, gt_bbx):
    B = gt_bbx.shape[0]
    intersect = _intersection(pred_bbx, gt_bbx)
    union = _union(pred_bbx, gt_bbx)- intersect #+ self.eps
    return (1 - (intersect/union)).sum()/B
  
def _union(pred_bbx, gt_bbx):
  return _area(pred_bbx) + _area(gt_bbx)
  
def _area(bbx):
  delta_x = bbx[:,2] -bbx[:,0]
  delta_y = bbx[:,3] -bbx[:,1]
  area = delta_x*delta_y
  return area

def _intersection(pred_bbx, gt_bbx):
  lower_right_xy = torch.min( pred_bbx[:,2:], gt_bbx[:,2:] )
  upper_left_xy = torch.max( pred_bbx[:,:2], gt_bbx[:,:2] )
  delta_xy = torch.clamp( lower_right_xy - upper_left_xy, min=0 )
  int_area = delta_xy[:,0]*delta_xy[:,1]
  return int_area

## test IOU
#a = [1.,1,3,3]
#b = [2.,2,4,4]
#c  = [5.,5.,7.,7.]
#pred_bbx = torch.tensor([a,a], requires_grad = True)
#gt_bbx= torch.Tensor([a,b])
#crit = IOU_Loss()
#loss = crit(pred_bbx, gt_bbx)
#loss.backward()
#print(pred_bbx.grad)
#loss

"""### (2) Task 3: Training a Noise Classifier to improve classification accuracy"""

# new dataset to import all noise level images
class All_PurdueShapes5_Dataset(Dataset):
  img_size = 32
  root_dir = './gdrive/My Drive/ECE695_DL/HW5/data/'
  noise_lvls = ['00','20','50','80']
  def __init__(self, train=True, transform= None):
    super().__init__()
    self.train = train
    self.transform = transform
    self.noise_lvls = self.noise_lvls # a string- 00,20,50,80
    self.dataset, self.label_maps, self.class_labels = load_shape_data(self.root_dir, self.noise_lvls, self.train)
    self.dataset_lens = [len(dataset) for dataset in self.dataset]

  def __len__(self):
    return np.sum(self.dataset_lens)

  def __getitem__(self,global_idx):
    # compute which dataset the idx corresponds to
    cum_sum = np.cumsum(self.dataset_lens)
    class_idx = np.nonzero(global_idx < cum_sum )[0][0]
    cum_sum = list(cum_sum)
    cum_sum.reverse()
    cum_sum.append(0)
    cum_sum.reverse()
    idx = global_idx - cum_sum[class_idx]
    dataset_idx = self.dataset[class_idx][idx]

    r = np.array( dataset_idx[0] )/255.
    g = np.array( dataset_idx[1] )/255.
    b = np.array( dataset_idx[2] )/255.
    R,G,B = r.reshape(self.img_size,self.img_size), g.reshape(self.img_size,self.img_size), b.reshape(self.img_size,self.img_size)
    im_tensor = torch.zeros(3,self.img_size,self.img_size, dtype=torch.float32)
    im_tensor[0,:,:] = torch.from_numpy(R)
    im_tensor[1,:,:] = torch.from_numpy(G)
    im_tensor[2,:,:] = torch.from_numpy(B)
    if self.transform:
      im_tensor = self.transform(im_tensor)
    sample = {'image' : im_tensor, 
              'bbox'  : torch.tensor(dataset_idx[3], dtype = torch.float32),                          
              'label' : torch.tensor(dataset_idx[4]), 
              'noise' : torch.tensor(class_idx)}
    return sample

#helping function
def load_shape_data(root_dir, noise_lvls, train):
  all_dataset = []
  all_label_maps =[]
  all_class_labels = []
  for noise_lvl in noise_lvls:    
    if train:
        dataset_name = 'PurdueShapes5-10000-train-noise-'+noise_lvl+'.gz'
    else:
        dataset_name = 'PurdueShapes5-1000-test-noise-'+noise_lvl+'.gz'
    
    if os.path.exists('torch_saved_'+ dataset_name+'_data.pt'):
      dataset = torch.load('torch_saved_'+ dataset_name+'_data.pt')
      label_map = torch.load('torch_saved_'+ dataset_name+'_label_map.pt')
      # reverse the key-value pairs in the label dictionary:
      class_labels = dict(map(reversed, label_map.items()))
    else:
      print('Loading data for first time, this might take some time')
      f = gzip.open(root_dir + dataset_name, 'rb')
      dataset = f.read()
      if sys.version_info[0] == 3:
        dataset, label_map = pickle.loads(dataset, encoding='latin1')
      else:
        dataset, label_map = pickle.loads(dataset)
      torch.save(dataset, 'torch_saved_'+ dataset_name+'_data.pt')
      torch.save(label_map,'torch_saved_'+ dataset_name+'_label_map.pt')
      # reverse the key-value pairs in the label dictionary:
      class_labels = dict(map(reversed, label_map.items()))
    all_dataset.append(dataset)
    all_label_maps.append(label_map)
    all_class_labels.append(class_labels)
  return(all_dataset, all_label_maps, all_class_labels)                           

def all_shapes_dataloader(train_dataset, test_dataset, batch_size=4, num_workers=4):
  train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle = True, num_workers=num_workers)
  test_dataloader = DataLoader(test_dataset, batch_size= batch_size, num_workers=num_workers)
  return [train_dataloader, test_dataloader]

# noise classifier network: something shallow should work
# using network from hw2 
class Noise_Classifier(nn.Module):
  img_size = 32
  num_noise_lvls = 4
  out_features = 32
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, self.out_features, 3, padding=1)
    self.conv2 = nn.Conv2d(self.out_features, self.out_features, 3)
    self.input_size_of_denselayer =self.out_features*7*7 #1568
    self.fc1 = nn.Linear(self.input_size_of_denselayer, 1000)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc2 = nn.Linear(1000, self.num_noise_lvls)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x))) 
    x = x.view(-1, self.input_size_of_denselayer)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

def train_noise_classifier(net, train_data_loader, device, train_params, save_name):
  epochs = train_params['epochs']
  debug =  train_params['debug']

  net = net.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
      inputs, labels = data['image'], data['noise']
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if (i+1)% 2000 == 0:
        avg_loss = running_loss / 22000.0
        if debug: print("\n[epoch:%d, batch:%d] loss: %.2f" %(epoch + 1, i + 1,avg_loss ))
        running_loss = 0.0
  save_model(net,save_name)

def test_noise_classifier(net, test_data_loader, device, save_name):
  # load wts to net
  save_dir = r'./gdrive/My Drive/ECE695_DL/HW5/saved_models/'
  net.load_state_dict(torch.load(save_dir+ save_name))  
  confusion_matrix = np.zeros((4,4))
  with torch.set_grad_enabled(False):
    net.eval()
    net.to(device)  
    for i,data in enumerate(test_data_loader):
      x, y_true = data['image'], data['noise']
      x, y_true = x.to(device), y_true.to(device)
      y_pred = net(x)
      _, y_pred = torch.max(F.softmax(y_pred, dim=1), dim=1)
      for b in range(y_pred.shape[0]): confusion_matrix[y_true[b], y_pred[b]] += 1.
  accuracy = np.sum(confusion_matrix[np.arange(4), np.arange(4)])/np.sum(confusion_matrix)
  print('Noise Classification Accuracy: %0.2f%%'%(accuracy*100))
  print('Noise Confusion Matrix:')    
  print(confusion_matrix)

def save_model(model,name):
  save_dir = './gdrive/My Drive/ECE695_DL/HW5/saved_models/'  
  torch.save(model.state_dict(), save_dir + name)

#training the noise classifier
noise_net = Noise_Classifier()
transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = All_PurdueShapes5_Dataset(transform=transform)
test_dataset = All_PurdueShapes5_Dataset(train=False, transform=transform)

train_dataloader, test_dataloader = all_shapes_dataloader(train_dataset, test_dataset, batch_size = 16, num_workers=8)

train_params = {'epochs':10, 'debug': True}

train_noise_classifier(noise_net, train_dataloader, device, train_params, 'noise_net.pt')
test_noise_classifier(noise_net, test_dataloader, device, 'noise_net.pt')

# inferencing on different noise level test datasets using trained noise classifier to decide which trained network to use 
def noise_classsifier_augmented_testing(noise_lvl, test_dataloader, device, label_dict, debug=True):
  #make a list of trained models
  save_dir = r'./gdrive/My Drive/ECE695_DL/HW5/saved_models/'
  noise_lvls = ['00','20','50','80']
  net_list = []
  for i_noise_lvl in noise_lvls:
    net = LOADnet2()
    save_name = 'LOADnet' + i_noise_lvl + '_original.pt'
    net.load_state_dict(torch.load(save_dir+ save_name))
    net_list.append(net)
  # load the trained noise nclassifier
  noise_net = Noise_Classifier()
  noise_net.load_state_dict(torch.load(save_dir+ 'noise_net.pt'))

  # carry out inference
  accuracy_label=0
  accuracy_bbx = 0
  count=0
  num_class = len(label_dict)
  conf_mat = np.zeros((num_class,num_class))
  crit_bbx = IOU_Loss()
  with torch.set_grad_enabled(False):
    #set rained networks to eval mode and send to device
    for net in net_list: 
      net.eval()
      net.to(device)
    noise_net.eval()
    noise_net.to(device)  
    # inference on samples
    for i,data in enumerate(test_dataloader):
      x, gt_label, gt_bbx = data['image'], data['label'], data['bbox']
      x, gt_label, gt_bbx = x.to(device), gt_label.to(device), gt_bbx.to(device)
      pred_noise_lvl = noise_net(x)
      _,pred_noise_lvl = torch.max(pred_noise_lvl, axis=1)
      if debug: print(pred_noise_lvl)
      batch_size = x.shape[0]
      pred_label = torch.zeros(batch_size, num_class, dtype = torch.float32)
      pred_bbx = torch.zeros_like(gt_bbx,dtype = torch.float32 )
      for ib in range(batch_size):
        inet = net_list[pred_noise_lvl[ib]]
        ipred_label, ipred_bbx = inet(x[ib,:,:,:].unsqueeze(0))
        pred_label[ib,:] = ipred_label
        pred_bbx[ib,:] = ipred_bbx
      accuracy_bbx += 1. - crit_bbx(pred_bbx, gt_bbx)
      _, pred_label = torch.max(pred_label,axis=1)
      for ib in range(pred_label.shape[0]): conf_mat[gt_label[ib].item(), pred_label[ib].item()] += 1.
      count += pred_label.shape[0]
  accuracy_label = np.sum(conf_mat[np.arange(num_class),np.arange(num_class)])/np.sum(conf_mat)
  accuracy_label = accuracy_label*100
  accuracy_bbx = (accuracy_bbx/count)*100
  # print stuff
  print('Dataset'+noise_lvl+' Classification Accuracy:\t %.2f%%'%(accuracy_label))
  print('Dataset'+noise_lvl+' IOU:\t %.2f%%\n'%(accuracy_bbx))
  print('Dataset'+noise_lvl+' Confusion Matrix:')
  print(conf_mat)

# noise: 0%
noise_lvl = '00'
transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = PurdueShapes5Dataset(noise_lvl,train=False,transform= transform)
label_dict = test_dataset.class_labels
test_dataloader = DataLoader(test_dataset, batch_size= 8, num_workers=4)
noise_classsifier_augmented_testing(noise_lvl, test_dataloader, device, label_dict, debug=False)

# noise: 20%
noise_lvl ='20'
test_dataset = PurdueShapes5Dataset(noise_lvl,train=False,transform= transform)
label_dict = test_dataset.class_labels
test_dataloader = DataLoader(test_dataset, batch_size= 8, num_workers=4)
noise_classsifier_augmented_testing(noise_lvl, test_dataloader, device, label_dict,debug=False)

# noise: 50%
noise_lvl ='50'
test_dataset = PurdueShapes5Dataset(noise_lvl,train=False,transform= transform)
label_dict = test_dataset.class_labels
test_dataloader = DataLoader(test_dataset, batch_size= 8, num_workers=4)
noise_classsifier_augmented_testing(noise_lvl, test_dataloader, device, label_dict, debug=False)

# noise: 80%
noise_lvl ='80'
test_dataset = PurdueShapes5Dataset(noise_lvl,train=False,transform= transform)
label_dict = test_dataset.class_labels
test_dataloader = DataLoader(test_dataset, batch_size= 8, num_workers=4)
noise_classsifier_augmented_testing(noise_lvl, test_dataloader, device, label_dict, debug=False)
