"""
Created on Sat Sep  8 15:27:50 2018

@author: rahul deshmukh 
HW3 ECE 661: Computer Vision
email: deshmuk5@purdue.edu
"""
#import py libraries
import numpy as np
import cv2
# import my functions
import sys
sys.path.append('../')
import MyCVModule as MyCV

readpath='./given/HW3Pics/'
imgname='2'
img1=cv2.imread(readpath+imgname+'.jpg')
imgcopy=cv2.imread(readpath+imgname+'.jpg') # to be used for plotting purposes


#---------- Correction using Point to Point correspondence---------#
savepath='./result/point_to_point/'
#define points in distorted image
img1P=[246,70]
img1Q=[326,83]
img1R=[245,269]
img1S=[323,265]

#define points in world coordinates
width=40;
height=80;

worldP=[0,0]
worldQ=[width,0]
worldR=[0,height]
worldS=[width,height]
#annotate portion of image with box
img1a=np.copy(imgcopy)
pt_names=['P','Q','S','R']
pts_temp=np.array([img1P,img1Q,img1S,img1R,img1P])
#font = cv2.FONT_HERSHEY_SIMPLEX
font = cv2.FONT_HERSHEY_PLAIN
for i in range(4):
    cv2.putText(img1a,pt_names[i],tuple(pts_temp[i]), font,5,(0,255,255),3,cv2.LINE_AA)
    cv2.line(img1a,tuple(pts_temp[i]),tuple(pts_temp[i+1]),(0,0,255),10)

cv2.imwrite(savepath+imgname+'_annotated.jpg',img1a)

#creating a blank canvas for destinaion image
pts1=np.array([img1P,img1Q,img1S,img1R])
ptsw=np.array([worldP,worldQ,worldS,worldR])
H_P2P=MyCV.HMat_Pts(ptsw,pts1) # gives H*ptsw=pts1 ie push fwd
#defining the points of source image as the whole image
img1P=[0,0]
img1Q=[np.shape(img1)[1]-1,0]
img1R=[0,np.shape(img1)[0]-1]
img1S=[np.shape(img1)[1]-1,np.shape(img1)[0]-1]
pts1=np.array([img1P,img1Q,img1S,img1R])
canvas=np.zeros(np.shape(img1))
#ar=int(np.shape(img1)[0]/np.shape(img1)[0])
#canvas=np.zeros((200,ar*200,3))
distype='L2'
imgP2P=MyCV.mapFitToCanvas(img1,pts1,canvas,H_P2P,distype)
cv2.imwrite(savepath+imgname+'_P2P.jpg',imgP2P)    
#-------------------------------------------------------------------#

#---------- Correction using two_step method------------------------#
savepath='./result/two_step/'

img1P=[246,70]
img1Q=[326,83]
img1R=[245,269]
img1S=[323,265]

img1T=[246,72]
img1U=[325,83]
img1V=[245,169]
img1W=[325,174]

img1A=[58,134]
img1B=[126,138]
img1C=[58,231]
img1D=[125,230]

img1E=[404,160]
img1F=[437,163]
img1G=[404,221]
img1H=[438,220]

#annotate the image
img1a=np.copy(imgcopy)
pt_names=['P','Q','S','R']
pts_temp=np.array([img1P,img1Q,img1S,img1R,img1P])
font = cv2.FONT_HERSHEY_PLAIN
for i in range(4):
    cv2.putText(img1a,pt_names[i],tuple(pts_temp[i]), font,3,(0,255,255),2,cv2.LINE_AA)
    cv2.line(img1a,tuple(pts_temp[i]),tuple(pts_temp[i+1]),(0,0,255),5)

cv2.imwrite(savepath+imgname+'_a_PC.jpg',img1a)

#Step-1: Projective correction
pts1=np.array([img1P,img1Q,img1R,img1S,
               img1P,img1R,img1Q,img1S])

#find Projective correction homography
H_PC=MyCV.Hmat_ProjectiveCorrection(pts1) #gives inverse mapping ie H_PC*image_pts= affine_pts
"""
for img1 using PQRS2
The Vanishing Line is: [ -8.27945524e-04  -1.16746051e-04   1.00000000e+00]
for image1 using TUVW2
The Vanishing Line is: [ -6.78756414e-04   1.02674768e-04   1.00000000e+00]
"""
#pts1=np.array([img1P,img1Q,img1Q,img1S, 
#               img1T,img1W,img1U,img1V])
    
pts1=np.array([img1P,img1Q, img1Q,img1S, img1P,img1R, img1P,img1Q, img1P,img1R,img1R,img1S, img1R,img1S, img1S,img1Q,
               img1T,img1W, img1U,img1V,
           img1A,img1B, img1B,img1D, img1A,img1C, img1C,img1D, img1A,img1C, img1A,img1B, img1B,img1D, img1C,img1D,
           img1E,img1F, img1E,img1G, img1E,img1F, img1F,img1H, img1F,img1H, img1H,img1G, img1G,img1H, img1E,img1G])
    
#plot points on image and lines connecting those points
img1a=np.copy(imgcopy)
font = cv2.FONT_HERSHEY_PLAIN
pt_names=['P','Q','R','S','T','U','V','W']
pts_temp=[img1P,img1Q,img1R,img1S,img1T,img1U,img1V,img1W]
#plot all points
for i in range(len(pt_names)):
    cv2.putText(img1a,pt_names[i],tuple(pts_temp[i]),font,3,(0,255,255),2,cv2.LINE_AA)
#plot lines
for i in range(int(np.shape(pts1)[0]/2)):
    cv2.line(img1a,tuple(pts1[2*i]),tuple(pts1[2*i+1]),(255,255,0),2)    

cv2.imwrite(savepath+imgname+'_a_AC.jpg',img1a)  


#Step-2: Affine Correction
#find Affine correction Homography  
H_AC=MyCV.Hmat_AffineCorrection(pts1,H_PC)

#map to canvas with Projective Correction
#defining the points of source image as the whole image
img1P=[0,0]
img1Q=[np.shape(img1)[1]-1,0]
img1R=[0,np.shape(img1)[0]-1]
img1S=[np.shape(img1)[1]-1,np.shape(img1)[0]-1]
pts1=np.array([img1P,img1Q,img1S,img1R])
canvas=np.zeros(np.shape(img1))
#ar=int(np.shape(img1)[0]/np.shape(img1)[0])
#canvas=np.zeros((200,ar*200,3))
distype='L2'
img_PC=MyCV.mapFitToCanvas(img1,pts1,canvas,np.linalg.inv(H_PC),distype)
cv2.imwrite(savepath+imgname+'_PC.jpg',img_PC)  
#map to canvas with Projective and affine correction
img_AC=MyCV.mapFitToCanvas(img1,pts1,canvas,np.dot(np.linalg.inv(H_PC),H_AC),distype)
cv2.imwrite(savepath+imgname+'_AC.jpg',img_AC)  
#-------------------------------------------------------------------#

#---------- Correction using one_step method------------------------#
savepath='./result/one_step/'

#define points
#pts for image 2
img1P=[246,70]
img1Q=[326,83]
img1R=[245,269]
img1S=[323,265]

img1T=[246,72]
img1U=[325,83]
img1V=[245,169]
img1W=[325,174]

img1A=[58,134]
img1B=[126,138]
img1C=[58,231]
img1D=[125,230]

img1E=[404,160]
img1F=[437,163]
img1G=[404,221]
img1H=[438,220]
    
#for image 2
pts1=np.array([img1P,img1Q, img1Q,img1S, img1P,img1R, img1P,img1Q,
               img1P,img1R, img1R,img1S, img1R,img1S, img1S,img1Q,
               img1T,img1W, img1U,img1V,
               img1A,img1B, img1B,img1D, img1A,img1C, img1C,img1D,
               img1A,img1C, img1A,img1B, img1B,img1D, img1C,img1D,
               img1E,img1F, img1E,img1G, img1E,img1F, img1F,img1H,
               img1F,img1H, img1H,img1G, img1G,img1H, img1E,img1G])#[  9.49939244e-01   3.56259041e-01   8.82813635e-06]

#plot points on image and lines connecting those points
img1a=np.copy(imgcopy)
font = cv2.FONT_HERSHEY_PLAIN
pt_names=['P','Q','R','S','T','U','V','W','A','B','C','D','E','F','G','H']
pts_temp=[img1P,img1Q,img1R,img1S,img1T,img1U,img1V,img1W,img1A,img1B,img1C,img1D,img1E,img1F,img1G,img1H]
#plot all points
for i in range(len(pt_names)):
    cv2.putText(img1a,pt_names[i],tuple(pts_temp[i]),font,3,(0,255,255),2,cv2.LINE_AA)
#plot lines
for i in range(int(np.shape(pts1)[0]/2)):
    cv2.line(img1a,tuple(pts1[2*i]),tuple(pts1[2*i+1]),(255,0,255),2)
    
cv2.imwrite(savepath+imgname+'_a_onestep.jpg',img1a)      
    
# find H
H_onestep=MyCV.HMat_OneStep(pts1)  #world to distorted: push fwd
    
#map to canvas
img1P=[0,0]
img1Q=[np.shape(img1)[1]-1,0]
img1R=[0,np.shape(img1)[0]-1]
img1S=[np.shape(img1)[1]-1,np.shape(img1)[0]-1]
pts1=np.array([img1P,img1Q,img1S,img1R])
canvas=np.zeros(np.shape(img1))
#ar=int(np.shape(img1)[0]/np.shape(img1)[0])
#canvas=np.zeros((200,ar*200,3))
distype='L2'
img_onestep=MyCV.mapFitToCanvas(img1,pts1,canvas,H_onestep,distype)
cv2.imwrite(savepath+imgname+'_onestep'+'.jpg',img_onestep)  
#-------------------------------------------------------------------# 