"""
ECE661: hw6 main file
@author: Rahul Deshmukh
email: deshmuk5@purdue.edu
PUID: 0030004932
"""
#import libraries
import cv2
import sys
import numpy as np
sys.path.append('../../')
import MyCVModule as MyCV
#define path
readpath='../images/' # path of images to be read
savepath='../results/' #path for saving results of images

# define case 0,1,2
case=0

# define readnames
if case==0:
    readname='lighthouse'
elif case==1:
    readname='baby'
elif case==2:
    readname='ski'
savepath=savepath+readname+'/'
# read color image
imgC=cv2.imread(readpath+readname+'.jpg')
img_gray=cv2.imread(readpath+readname+'.jpg',0)
m=imgC.shape[0]
n=imgC.shape[1]

#%%
# define number of otsu iterations for the RGB channels
if case==0:
    N=[2,3,2]#R,G,B
    imgR=imgC[:,:,2]
    imgG=imgC[:,:,1]
    imgB=imgC[:,:,0]
elif case==1:
    N=[1,1,1]
    imgR=MyCV.invert_img(imgC[:,:,2])
    imgG=MyCV.invert_img(imgC[:,:,1])
    imgB=MyCV.invert_img(imgC[:,:,0])
elif case==2:
    N=[4,4,2]
    imgR=MyCV.invert_img(imgC[:,:,2])
    imgG=MyCV.invert_img(imgC[:,:,1])
    imgB=MyCV.invert_img(imgC[:,:,0])   
L=256 # number of levels
# find k_star from otsu using color channels
print('Red')
k_R=MyCV.Otsu(imgR,L,N[0])
print('Green')
k_G=MyCV.Otsu(imgG,L,N[1])
print('Blue')
k_B=MyCV.Otsu(imgB,L,N[2])
# find masks for the color channels
maskR=MyCV.img_mask(imgR,L,k_R)
maskG=MyCV.img_mask(imgG,L,k_G)
maskB=MyCV.img_mask(imgB,L,k_B)
# save binary images
cv2.imwrite(savepath+readname+'_maskR.jpg',255*maskR)
cv2.imwrite(savepath+readname+'_maskG.jpg',255*maskG)
cv2.imwrite(savepath+readname+'_maskB.jpg',255*maskB)
# make combined image
if case==0:
    color_seg_img = maskR*MyCV.not_img(maskG)*MyCV.not_img(maskB)
    option=0;sigma=2.5;N=5       
elif case==1:
#    color_seg_img =MyCV.not_img(MyCV.or_img(MyCV.or_img(maskR,maskG),maskB))   
    color_seg_img =maskR*maskG*maskB   
    option=0;sigma=2.5;N=5 
elif case==2:
    color_seg_img = MyCV.or_img(maskB,maskG)#*MyCV.not_img(maskR)
    option=0;sigma=2.5;N=10 
#save color based segmented image
cv2.imwrite(savepath+readname+'_color_seg.jpg',255*color_seg_img)
# contour extraction of color segmented image
cont_img=MyCV.contour_img(color_seg_img,option,sigma,N)
#save contour
cv2.imwrite(savepath+readname+'_color_contour.jpg',255*cont_img)
#%%
#------------texture based segmentation--------------------#
N=[3,5,7]
text_img=MyCV.textured_img(img_gray,N)
# save textured image
cv2.imwrite(savepath+readname+'_text_img.jpg',text_img)
# otsu's segmentation on textured image
# define number of otsu iterations for the RGB channels
if case==0:
    N=[3,3,3]#R,G,B
    imgR=MyCV.invert_img(text_img[:,:,2])
    imgG=MyCV.invert_img(text_img[:,:,1])
    imgB=MyCV.invert_img(text_img[:,:,0])
elif case==1:
    N=[3,3,3]
    imgR=MyCV.invert_img(text_img[:,:,2])
    imgG=MyCV.invert_img(text_img[:,:,1])
    imgB=MyCV.invert_img(text_img[:,:,0])
elif case==2:
    N=[4,4,4]
    imgR=MyCV.invert_img(text_img[:,:,2])
    imgG=MyCV.invert_img(text_img[:,:,1])
    imgB=MyCV.invert_img(text_img[:,:,0]) 
L=256 # number of levels
# find k_star from otsu using color channels
k_R=MyCV.Otsu(imgR,L,N[0])
k_G=MyCV.Otsu(imgG,L,N[1])
k_B=MyCV.Otsu(imgB,L,N[2])
# find masks for the color channels
maskR=MyCV.not_img(MyCV.img_mask(imgR,L,k_R))
maskG=MyCV.not_img(MyCV.img_mask(imgG,L,k_G))
maskB=MyCV.not_img(MyCV.img_mask(imgB,L,k_B))
# save binary images
cv2.imwrite(savepath+readname+'_mask1.jpg',255*maskR)
cv2.imwrite(savepath+readname+'_mask2.jpg',255*maskG)
cv2.imwrite(savepath+readname+'_mask3.jpg',255*maskB)
# make combined image
if case==0:
    text_seg_img=maskR*maskG*maskB
    option=0;sigma=2.5;N=5      
elif case==1:
#    text_seg_img=MyCV.not_img(maskR)*MyCV.not_img(maskG)*MyCV.not_img(maskB)
    text_seg_img=maskR*maskG*maskB       
    option=0;sigma=1.5;N=3   
elif case==2:
    text_seg_img=MyCV.or_img(MyCV.or_img(maskB,maskG),maskR)#*MyCV.not_img(maskR)
    option=1;sigma=4;N=10   
#save color based segmented image
cv2.imwrite(savepath+readname+'_text_seg.jpg',255*text_seg_img)
# contour extraction of texture segmented image
cont_img=MyCV.contour_img(text_seg_img,option,sigma,N)
#save contour
cv2.imwrite(savepath+readname+'_texture_contour.jpg',255*cont_img)