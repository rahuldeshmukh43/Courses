"""
# ECE695 DL HW5: Object Detection
# Task4
Author: Rahul Deshmukh
email: deshmuk5@purdue.edu

Sources:
1. DLStudio: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.1.0.html#DLStudio
2. For IOU this was helpful: https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
3. For definition of bottle neck layer(ResNet paper): https://arxiv.org/pdf/1512.03385.pdf 

In this task, I am using a different network for training and testing of images with 50% noise. 

I am using a 60 layer network with bottleneck layers as defined in ResNet paper. 

I am also using both MSE loss and IOU loss for training. For initial epochs I choose MSE loss as the predicted bbox will rarely match ground truth bbox which makes IOU unable to learn. After achieveing a small MSE loss, I switch to IOU loss for remaining epochs. 

I am also using the smoothing kernel for 50% noise images from task2. 

I trained the new model for 20 epochs and achieve a classification accuracy of 75.30%

Conclusion: The new network with smoothing operation and more number of epochs for training help in increasing the accuracy.
Accuracy increases from 57.30% in Task2 -> 75.30% in Task4

The output for this task:

+Task 4:
Dataset50 Classification Accuracy:	 75.30%
Dataset50 Confusion Matrix:
[[ 52.  10.  12.  99.  27.]
 [  8. 185.   7.   0.   0.]
 [  3.   2. 188.   5.   2.]
 [ 34.   0.   7. 152.   7.]
 [  6.   3.   4.  11. 176.]]
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

"""## Training and Testing definition"""

def run_code_for_training(net,train_dataloader, device, params, label_dict,save_name):
  lr = params['lr']
  momentum= params['momentum']
  num_epochs = params['num epochs']
  debug = params['debug']
  freq= params['freq']

  net.to(device)
  crit_CE = nn.CrossEntropyLoss()
  crit_bbx = nn.MSELoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  epoch_loss_label=[]
  epoch_loss_bbx=[]
  for epoch in range(num_epochs):  
      print("\n")
      running_loss_label = 0.0
      running_epoch_loss_label=0.0
      running_loss_bbx = 0.0
      running_epoch_loss_bbx=0.0
      count = 0
      if epoch > 5:
        if avg_loss_label< 5.0:
          if debug: print('#### Using IOU Loss for bbox ###') 
          crit_bbx = IOU_Loss()
      else:
        if debug: print('Using MSELoss for bbox ') 
        crit_bbx = nn.MSELoss()
      for i, data in enumerate(train_dataloader):
          x, gt_label, gt_bbx = data['image'], data['label'], data['bbox']
          x, gt_label, gt_bbx = x.to(device), gt_label.to(device), gt_bbx.to(device)
          optimizer.zero_grad()
          # Make the predictions with the model:
          pred_label, pred_bbx = net(x)
          loss_label = crit_CE(pred_label, gt_label)
          loss_bbx = crit_bbx(pred_bbx, gt_bbx)
          loss_label.backward(retain_graph = True)
          loss_bbx.backward()
          optimizer.step()
          running_loss_label += loss_label.item()
          running_epoch_loss_label += loss_label.item()
          running_loss_bbx += loss_bbx.item()
          running_epoch_loss_bbx += loss_bbx.item()
          if (i+1)%freq == 0:    
              avg_loss_label = running_loss_label / float(freq)
              avg_loss_bbx = running_loss_bbx/float(freq)
              #print("[epoch:%d, batch:%5d] labeling loss: %.3f bbox loss: %.3f" % (epoch + 1, i + 1, avg_loss_label, avg_loss_bbx))
              running_loss_label = 0.0
              running_loss_bbx = 0.0
              if debug:
                print("[epoch:%d, batch:%5d] labeling loss: %.3f bbox loss: %.3f" % (epoch + 1, i + 1, avg_loss_label, avg_loss_bbx))
                # show gt and predicted bbx
                display_gt_and_pred(x, gt_label,gt_bbx, pred_label, pred_bbx, label_dict)
          count += x.shape[0]
      epoch_loss_label.append(running_epoch_loss_label/count)
      epoch_loss_bbx.append(running_epoch_loss_bbx/count)
      running_epoch_loss_label=0.0
      running_epoch_loss_bbx=0.0
  print("\nFinished Training\n")
  save_model(net, save_name)
  #return(epoch_loss_label, epoch_loss_bbx)

def display_gt_and_pred(x, gt_label,gt_bbx, pred_label, pred_bbx, label_dict):
  b,c,h,w = x.shape
  x_copy = x.detach().clone().cpu()  
  pred_bbx = pred_bbx.detach().clone()
  pred_bbx[pred_bbx<0] = 0
  pred_bbx[pred_bbx>31] = 31
  pred_bbx[torch.isnan(pred_bbx)] = 0
  _, pred_label = torch.max(pred_label.data,axis=1)
  # print labels
  print('#----------- debug prints----------------#')
  print('ground truth labels: \t' + ''.join('%10s'%label_dict[gt_label[ib].item()] for ib in range(b)))
  print('predicted labels: \t' + ''.join('%10s'%label_dict[pred_label[ib].item()] for ib in range(b)))
  for idx in range(b):
      i1 = int(gt_bbx[idx][1])
      i2 = int(gt_bbx[idx][3])
      j1 = int(gt_bbx[idx][0])
      j2 = int(gt_bbx[idx][2])
      k1 = int(pred_bbx[idx][1])
      k2 = int(pred_bbx[idx][3])
      l1 = int(pred_bbx[idx][0])
      l2 = int(pred_bbx[idx][2])
      print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
      print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
      x_copy[idx,0,i1:i2,j1] = 1.
      x_copy[idx,0,i1:i2,j2] = 1.
      x_copy[idx,0,i1,j1:j2] = 1.
      x_copy[idx,0,i2,j1:j2] = 1.
      x_copy[idx,2,k1:k2,l1] = 1.                      
      x_copy[idx,2,k1:k2,l2] = 1.
      x_copy[idx,2,k1,l1:l2] = 1.
      x_copy[idx,2,k2,l1:l2] = 1.
  img= torchvision.utils.make_grid(x_copy)#, normalize=True)
  img = img/2.0 + 0.5 
  npimg = img.numpy()
  plt.imshow(npimg.transpose(1, 2, 0))
  plt.show()
  print('#--------------------------------------#')

def run_code_for_testing(net, test_dataloader, device, label_dict, save_name, noise_lvl):
  save_dir = r'./gdrive/My Drive/ECE695_DL/HW5/saved_models/'
  net.load_state_dict(torch.load(save_dir+ save_name))
  accuracy_label=0
  accuracy_bbx = 0
  count=0
  num_class = len(label_dict)
  conf_mat = np.zeros((num_class,num_class))
  crit_bbx = IOU_Loss()
  with torch.set_grad_enabled(False):
    net.eval()
    net.to(device)  
    for i,data in enumerate(test_dataloader):
      x, gt_label, gt_bbx = data['image'], data['label'], data['bbox']
      x, gt_label, gt_bbx = x.to(device), gt_label.to(device), gt_bbx.to(device)
      pred_label, pred_bbx = net(x)
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
  ## percentages conf mat
  # print_conf_mat = ((((conf_mat.T)/np.sum(conf_mat,axis = 1)).T)*100).astype(np.int)
  # print('%10s'%(' '),end='')
  # for i in range(num_class): print('%10s'%(label_dict[i]),end='')
  # print('')
  # for i in range(num_class):
  #   print('%10s'%(label_dict[i]+':'),end='')
  #   for j in range(num_class): print('%10s'%(str(print_conf_mat[i,j])),end='')
  #   print('')
  # print("\nFinished Testing\n")
  ##return accuracy_label, accuracy_bbx, conf_mat

def save_model(model,name):
  save_dir = './gdrive/My Drive/ECE695_DL/HW5/saved_models/'  
  torch.save(model.state_dict(), save_dir + name)


"""### (3) Task 4: Improving accuracy for images with noise levels of 50%"""

class BottleNeck_Block(nn.Module):
  """ 3-conv layer bottleneck block"""
  def __init__(self,in_channels,mid_channels,out_channels, downsample=False, skip_connections=True):
    super().__init__()
    self.in_channels = in_channels
    self.mid_channels = mid_channels
    self.out_channels = out_channels
    self.downsample = downsample
    self.skip_connections = skip_connections

    norm_layer = nn.BatchNorm2d
    self.bn_in = norm_layer(in_channels)
    self.bn_mid = norm_layer(mid_channels)
    self.bn_out = norm_layer(out_channels)    
    if downsample:
      self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=2)
      self.downsampler = nn.Conv2d(in_channels,in_channels, 1, stride=2)
    else:
      self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
    self.conv2 = nn.Conv2d(mid_channels,mid_channels,3,padding=1)
    self.conv3 = nn.Conv2d(mid_channels,out_channels, 1)    

  def forward(self,x):
    identity = x.clone()
    # first conv layer
    out = self.conv1(x)
    out = self.bn_mid(out)
    out = F.relu(out)
    # second conv layer
    out = self.conv2(out)
    out = self.bn_mid(out)
    out = F.relu(out)
    # third conv layer
    out = self.conv3(out)
    out = self.bn_out(out)
    out = F.relu(out)
    if self.downsample:
      identity = self.downsampler(identity)
      identity = self.bn_in(identity)      
    if self.skip_connections:
      if self.in_channels == self.out_channels:
          out += identity                              
      else:
        for num in range(self.out_channels//self.in_channels):
          out[:,num*self.in_channels:(num+1)*self.in_channels,:,:] += identity
    return(out)

class ObjDetNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.conv = nn.Conv2d(3,64,3,padding=1) 
    self.pool = nn.MaxPool2d(3,stride=2, padding=1)
    #classification  
    self.btlnck_blk_64_256 = BottleNeck_Block(64,64,256)
    self.btlnck_blk256 = ObjDetNet._make_module_list(BottleNeck_Block(256,64,256), self.config[0])
    self.btlnck_blk_256_512_ds = BottleNeck_Block(256,128,512,downsample=True)
    self.btlnck_blk512 = ObjDetNet._make_module_list(BottleNeck_Block(512,128,512), self.config[1])
    self.btlnck_blk_512_1024_ds = BottleNeck_Block(512,256,1024,downsample=True)
    self.btlnck_blk1024 = ObjDetNet._make_module_list(BottleNeck_Block(1024,256,1024), self.config[2])
    self.av_pool = nn.AvgPool2d(4,1)
    self.fc1 =  nn.Linear(1024, 256)
    self.fc2 = nn.Linear(256,5)
    # for bbx
    self.conv_seqn = ObjDetNet._make_module_list(nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)),2)
    self.fc_seqn = nn.Sequential(
      nn.Linear(16*16*64, 1024),
      nn.ReLU(inplace=True),
      nn.Linear(1024, 512),
      nn.ReLU(inplace=True)
      )
    self.fc3 = nn.Linear(512, 4)

  def forward(self,x):
    x = self.pool(F.relu(self.conv(x)))
    #labeling section
    x1 = self.btlnck_blk_64_256(x)    
    for btlnck_blk256 in self.btlnck_blk256:
      x1 = btlnck_blk256(x1)                                               
    x1 = self.btlnck_blk_256_512_ds(x1)
    for btlnck_blk512 in self.btlnck_blk512:
      x1 = btlnck_blk512(x1)                                               
    x1 = self.btlnck_blk_512_1024_ds(x1)
    for btlnck_blk1024 in self.btlnck_blk1024:
      x1 = btlnck_blk1024(x1)
    x1 = self.av_pool(x1)
    x1 = x1.view(-1, 1024)
    x1 = F.relu(self.fc1(x1))
    x1 = self.fc2(x1)
    # bbx prediction    
    x2 = self.conv_seqn[0](x)
    x2 = self.conv_seqn[1](x2)
    x2 = x2.view(x.size(0),-1)
    x2 = self.fc_seqn(x2)
    x2 = self.fc3(x2)
    return x1, x2

  @staticmethod
  def _make_module_list(module, rep):
    mod_list = nn.ModuleList()
    for _ in range(rep): mod_list.append(module) 
    return(mod_list)

class ObjDetNet30(ObjDetNet):
 config = np.array([2,2,2]) 
 def __init__(self):
   super().__init__(self.config)

class ObjDetNet60(ObjDetNet):
 config = np.array([4,8,4]) 
 def __init__(self):
   super().__init__(self.config)

# 50% noise:
smoothing_params= {'kernel size': 5, 'sigma':0.4} 
noise_lvl ='50'
train_dataset = PurdueShapes5Dataset(noise_lvl,transform= transform, smoothing=smoothing_params)
test_dataset = PurdueShapes5Dataset(noise_lvl,train=False,transform= transform, smoothing=smoothing_params)
label_dict = train_dataset.class_labels
train_dataloader, test_dataloader = shapes_dataloader(train_dataset, test_dataset, batch_size = 8)
net = ObjDetNet60()
train_params = {'lr':1e-5,
                'momentum': 0.8,
                'num epochs': 20,
                'debug': True,
                'freq':1000}

run_code_for_training(net, train_dataloader, device, train_params, label_dict, 'ObjDetNet60_noise'+noise_lvl+'.pt')

model = ObjDetNet60()
run_code_for_testing(model, test_dataloader, device, label_dict,'ObjDetNet60_noise'+noise_lvl+'.pt', noise_lvl)
