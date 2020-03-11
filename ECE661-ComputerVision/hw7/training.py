"""
ECE661: hw7 main file
@author: Rahul Deshmukh
email: deshmuk5@purdue.edu
PUID: 0030004932
"""

#import libraries
import cv2
import os
import numpy as np
import sys
sys.path.append('../../')
import MyCVModule as MyCV

# define training and testing path
mainpath='../imagesDatabaseHW7/'

# constants
P=8 # neighbors
R=1 # radius

#-------------------Training-------------------------#
training_path=mainpath+'training/'
# find class names from folder names
class_names=os.listdir(training_path) # list of names
class_sizes={}
class_imgNames={}
for i in range(len(class_names)):
    temp=os.listdir(training_path+class_names[i]+'/')
    class_sizes[class_names[i]]=len(temp)
    class_imgNames[class_names[i]]=temp
# find LBP vector for all classes  and images
class_LBPs={}
for i in range(len(class_names)):
    print(class_names[i])
    LBP_vec=np.zeros((class_sizes[class_names[i]],P+2),dtype=int)
    for j in range(class_sizes[class_names[i]]):
        print(class_imgNames[class_names[i]][j])
        # find image name
        img_path=training_path+class_names[i]+'/'+\
                    class_imgNames[class_names[i]][j]
        # read image
        img=cv2.imread(img_path,0) # gray scale image
        # find LBP vector
        LBP_vec[j,:]=MyCV.LBP(img,P,R)
        print(LBP_vec[j,:])
        print('-------------------------------')
    # put LBP_vec into the class_LBPs dictionary
    class_LBPs[class_names[i]]=LBP_vec # rows as LBP of jth image
    print('#########################################')

#save data to pickle file
import pickle
with open('training.pickle','wb') as f:
    pickle.dump((P,R,class_names,class_sizes,class_LBPs),f)

#----------- Training ends------------------------------#

    

