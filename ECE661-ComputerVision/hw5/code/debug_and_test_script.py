"""
ECE661: hw5 debug and testing
@author: rahul deshmukh
email: deshmuk5@purdue.edu
PUID: 00 
"""

#import libraries
import numpy as np
import cv2
import sys

sys.path.append('../../')
import MyCVModule as MyCV
#define path
readpath='../images/myfountain_raw/' # path of images to be read
savepath='../images/myfountain/' #path for saving results of images

#savename='test'
#MyCV.Panaroma(savepath,savepath,savename)
#H=MyCV.AutoHomoCalc(img1,img2)

#resize images
h=500
for i in range(1,6):    
    img=cv2.imread(readpath+str(i)+'.jpg')
    im_w=np.shape(img)[1]
    im_h=np.shape(img)[0]
    a=im_h/im_w
    w=int(np.ceil(h/a))
    temp=cv2.resize(img,(w,h))
    cv2.imwrite(savepath+str(i)+'.jpg',temp)



