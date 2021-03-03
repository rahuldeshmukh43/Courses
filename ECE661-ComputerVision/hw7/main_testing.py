"""
ECE661: hw7 main file
@author: Rahul Deshmukh
email: deshmuk5@purdue.edu
"""

#import libraries
import cv2
import os
import numpy as np
import scipy.io as sio
import sys
sys.path.append('../../')
import MyCVModule as MyCV

# define training and testing path
mainpath='../imagesDatabaseHW7/'

# read training data from pickle file
import pickle
with open('training.pickle','rb') as f:
    P,R,class_names,class_sizes,class_LBPs=pickle.load(f)

#-----------------Testing----------------------#
# define testing path
testing_path=mainpath+'testing/'

test_img_list=os.listdir(testing_path)

#define number of nearest neighbors for NNclassifier
n=5
test_vec_store=np.zeros((len(test_img_list),P+2),dtype=int)
#loop over each image
for i in range(len(test_img_list)):
    print(test_img_list[i]) # testing image name
    # read image
    img=cv2.imread(testing_path+test_img_list[i],0)#gray scale image
    # find LBP vector of the testing image
    test_vec=MyCV.LBP(img,P,R)
    test_vec_store[i,:]=test_vec
#    test_vec=test_vec_store_norm[i,:]
    print(test_vec)
    # find predicted class using NNclassifier
    identified_class=MyCV.NNClassifier(test_vec,class_LBPs,
                                       class_names,
                                       class_sizes,n)
    print('identified class was '+identified_class)
    print('-------------------------')    
#----------------------------------------------------#
np.save('test_vec_store.npy',test_vec_store)