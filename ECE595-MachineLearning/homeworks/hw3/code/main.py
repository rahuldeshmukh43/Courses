"""
ECE 595: ML-1
HW-3: main file
@author- Rahul Deshmukh
"""
#%% Import libraries
import numpy as np
import functions as myfun
import cv2 
#%% 1 Read data from files
read_path = '../data/'
save_path = '../result/'
#-----------------------------Exercise-2:CAT Classification-------------------#
#(a)
#(i)
train_cat= np.matrix(np.loadtxt(read_path+'train_cat.txt',delimiter=','))
train_grass= np.matrix(np.loadtxt(read_path+'train_grass.txt',delimiter=','))
N_cat= np.shape(train_cat)[1]
N_grass= np.shape(train_grass)[1]
N_total =N_cat+N_grass
#(ii)
# calculate the mean
u_cat = np.mean(train_cat,axis=1)
u_grass = np.mean(train_grass,axis = 1)
# calculate the variance
sigma_cat = myfun.find_covariance(train_cat,u_cat)
sigma_grass = myfun.find_covariance(train_grass,u_grass)
#priors
pi_cat= N_cat/(N_total)
pi_grass= N_grass/N_total
#%% (b)
#(i)
cat_img = cv2.imread(read_path+'cat_grass.jpg',0) # read in grayscale
cat_img = cat_img/255.0 # normalizing
#cat_img = np.flip(cat_img,axis=1)

#(ii)
# prepare lists for classification
u = [u_grass,u_cat]
sigma = [sigma_grass,sigma_cat]
prior = [pi_grass,pi_cat]
user_def_label = [0,1]

overlapping_labels = myfun.classify_overlapping_patches(cat_img,u,sigma,prior,user_def_label)
cv2.imwrite(save_path+'overlapping_patches.jpg',255*overlapping_labels)

#(iii) non-overlapping patches 
non_overlapping_labels = myfun.classify_non_overlapping_patches(cat_img,u,sigma,prior,user_def_label)
cv2.imwrite(save_path+'non_overlapping_patches.jpg',255*non_overlapping_labels)

#(iv)
#read ground truth
ground_truth = cv2.imread(read_path+'truth.png',0)/255
# find mean abs errors
error1 = myfun.mean_abs_error(overlapping_labels,ground_truth)
error2 = myfun.mean_abs_error(non_overlapping_labels,ground_truth)
print('Error for overlapping: '+str(100*error1)+'%\nand for non-overlapping:'+str(100*error2)+'%')

#(v) Read image of cheetah on grass
cheetah_img= cv2.imread(read_path+'cheetah3.jpg',0)

cheetah_overlapping_labels = myfun.classify_overlapping_patches(cheetah_img,u,sigma,prior,user_def_label)
cv2.imwrite(save_path+'cheetah_overlapping_patches.jpg',255*cheetah_overlapping_labels)

cheetah_non_overlapping_labels = myfun.classify_non_overlapping_patches(cheetah_img,u,sigma,prior,user_def_label)
cv2.imwrite(save_path+'cheetah_non_overlapping_patches.jpg',255*cheetah_non_overlapping_labels)