"""
ECE661: hw4
@author: rahul deshmukh
email: deshmuk5@purdue.edu
"""

#import libraries
import numpy as np
import cv2
import sys
sys.path.append('../../')
import MyCVModule as MyCV
#define path
readpath='../HW4Pics/' # path of images to be read
savepath='../results/' #path for saving results of images
imgname='6'
#read image in grayscale
img1=cv2.imread(readpath+imgname+'scene1.jpg',0)
img2=cv2.imread(readpath+imgname+'scene2.jpg',0)
img1color=cv2.imread(readpath+imgname+'scene1.jpg',-1)
img2color=cv2.imread(readpath+imgname+'scene2.jpg',-1)
#----------Harris corner detection---------------------##
scale_list=[1,2,3,4]
#scale_list=[1]
for iscale in range(len(scale_list)):
    print('scale='+str(scale_list[iscale]))
    scale=scale_list[iscale]
    print('Harris corners')
    pts1,Ix1_g,Iy1_g=MyCV.Harris_Corners(img1,scale); print('pts1 done');print(len(pts1))
    pts2,Ix2_g,Iy2_g=MyCV.Harris_Corners(img2,scale); print('pts2 done');print(len(pts2))
    print('NCC')
#    pt_pairs_NCC=MyCV.NCC(img1,pts1,img2,pts2); print('pt_pairs_NCC done');print(len(pt_pairs_NCC))     
#    NCC_img=MyCV.plot_matching_points(img1color,pts1,img2color,pts2,pt_pairs_NCC); print('combined images') 
#    cv2.imwrite(savepath+imgname+'_'+str(iscale)+'_NCC.jpg',NCC_img); print('image saved')  

    print('SSD')
    pt_pairs_SSD=MyCV.SSD(img1,pts1,img2,pts2);print(len(pt_pairs_SSD)) 
    SSD_img=MyCV.plot_matching_points(img1color,pts1,img2color,pts2,pt_pairs_SSD)
    cv2.imwrite(savepath+imgname+'_'+str(iscale)+'_SSD.jpg',SSD_img)
    
    
    
#print('SIFT')
###sift key points
#sift=cv2.xfeatures2d.SIFT_create()
#sift_pts1,sift_des1=sift.detectAndCompute(img1,None);print('Sift pts1 found')
#sift_pts2,sift_des2=sift.detectAndCompute(img2,None);print('Sift pts2 found')
###convert to my format for using NCC and SSD
#sift_mypts1=MyCV.convert_SIFT_myFormat(sift_pts1)    
#sift_mypts2=MyCV.convert_SIFT_myFormat(sift_pts2)
#
###sift correlation
##print('SIFT_NCC')
##pt_pairs_sift_NCC=MyCV.NCC(img1,sift_mypts1,img2,sift_mypts2); print('pt_pairs_NCC done');print(len(pt_pairs_sift_NCC))     
##NCC_img=MyCV.plot_matching_points(img1color,sift_mypts1,img2color,sift_mypts2,pt_pairs_sift_NCC); print('combined images') 
##cv2.imwrite(savepath+imgname+'_SIFT_NCC.jpg',NCC_img); print('image saved')  
#
#print('SIFT Euclidean')
#pt_pairs_sift_eu=MyCV.Euclidean_sift_match(sift_mypts1,sift_des1,sift_mypts2,sift_des2)
#new_img=MyCV.plot_matching_points(img1color,sift_mypts1,img2color,sift_mypts2,pt_pairs_sift_eu)
#cv2.imwrite(savepath+imgname+'_SIFT_EU.jpg',new_img)



