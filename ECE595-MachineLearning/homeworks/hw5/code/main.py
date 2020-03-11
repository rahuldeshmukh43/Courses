#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE 595 ML-1: HW5
Attack on Classifiers
@author: rahul
"""

# import libraries
import numpy as np, matplotlib.pyplot as plt, cv2
from functions import *
# declare paths
read_path = '../data/'
save_path = '../results/'
ex2_save_path = save_path+'ex2/'
ex3_save_path = save_path+'ex3/'
#-----------------------------------------------------------------------------#
# find the Quadratic classifier
# read data
train_cat= np.matrix(np.loadtxt(read_path+'train_cat.txt',delimiter=','))
train_grass= np.matrix(np.loadtxt(read_path+'train_grass.txt',delimiter=','))
N_cat= np.shape(train_cat)[1]
N_grass= np.shape(train_grass)[1]
N_total =N_cat+N_grass
# calculate the mean
u_cat = np.mean(train_cat,axis=1)
u_grass = np.mean(train_grass,axis = 1)
# calculate the variance
sigma_cat = np.cov(train_cat)
sigma_grass = np.cov(train_grass) 
#priors
pi_cat= N_cat/(N_total)
pi_grass= N_grass/N_total
# make classifier
W_cat = np.linalg.inv(sigma_cat)
W_grass = np.linalg.inv(sigma_grass)

w_cat = -W_cat@u_cat
w_grass = -W_grass@u_grass

w0_cat = (1/2)*u_cat.T@W_cat@u_cat + (1/2)*np.log(np.linalg.det(sigma_cat)) - np.log(pi_cat)
w0_grass= (1/2)*u_grass.T@W_grass@u_grass + (1/2)*np.log(np.linalg.det(sigma_grass)) - np.log(pi_grass)

# 
W_cat *= -1
W_grass *= -1
w_cat *= -1
w_grass *= -1
w0_cat *= -1
w0_grass *= -1
#-----------------------------------------------------------------------------#


##%%-------------------------Exercise 2----------------------------------------#
#lam_list = [1,5,10]
#alpha = 0.0001
#non_overlap_test_img = TestImgClass()
#non_overlap_test_img.Load_img(read_path+'cat_grass.jpg')
#non_overlap_test_img.get_non_overlap_patches()
#non_overlap_norms_record =[]
#for i in range(len(lam_list)):
#    lam = lam_list[i]
#    non_overlap_test_img.get_attack_image_from_non_overlap_patches([W_grass,W_cat],\
#                                                                   [w_grass,w_cat],\
#                                                                   [w0_grass,w0_cat],\
#                                                                   lam,alpha)
#    perturbation_img = np.abs(non_overlap_test_img.attacked_image - non_overlap_test_img.img)
#    i_norm = np.linalg.norm(perturbation_img)
#    non_overlap_norms_record.append(i_norm)
#    # print stuff
#    print('The Frobenius Norm of Pertubation is: '+str(i_norm)+'\n')   
#    
#    #save images and masks
#    cv2.imwrite(ex2_save_path+'attacked_img_lam'+str(i)+'.jpg',non_overlap_test_img.attacked_image*255)
#    cv2.imwrite(ex2_save_path+'pertubation_lam'+str(i)+'.jpg',perturbation_img*255)
#    cv2.imwrite(ex2_save_path+'classified_attack_lam'+str(i)+'_.jpg',\
#                non_overlap_test_img.attacked_image_mask*255)
#    cv2.imwrite(ex2_save_path+'original_classification_lam'+str(i)+'.jpg',\
#                non_overlap_test_img.original_mask*255)
#plt.figure()
#plt.grid()
#plt.plot(lam_list,non_overlap_norms_record)
#plt.xlabel('$\lambda$')
#plt.ylabel('Frobenius norm of perturbation')
#plt.show()    
##-----------------------------------------------------------------------------#

#%%-------------------------Exercise 3----------------------------------------#
lam_list = [0.5,1,5]
freqs = [50,50,50]
alpha = 0.00001
overlap_test_img = TestImgClass()
overlap_test_img.Load_img(read_path+'cat_grass.jpg')
overlap_norms_record =[]
for i in range(len(lam_list)):
    lam = lam_list[i]
    print('Solving for lam='+str(lam)+' ......')
    freq=freqs[i]
    overlap_test_img.get_attack_image_from_overlap_patches([W_grass,W_cat],\
                                                       [w_grass,w_cat],\
                                                       [w0_grass,w0_cat],\
                                                       lam,alpha,freq=freq)
    
    perturbation_img = np.abs(overlap_test_img.attacked_image - overlap_test_img.img)
    i_norm = np.linalg.norm(perturbation_img)
    overlap_norms_record.append(i_norm)
    # print stuff
    print('The Frobenius Norm of Pertubation is: '+str(i_norm)+'\n')
    
    attacked_image_mask = classify_perturbed_overlap_image(overlap_test_img.img,
                                                           [W_grass,W_cat],\
                                                      [w_grass,w_cat],\
                                                      [w0_grass,w0_cat])
    
    cv2.imwrite(ex3_save_path+'attacked_img_lam'+str(i)+'.jpg',\
            overlap_test_img.attacked_image*255)
    cv2.imwrite(ex3_save_path+'pertubation_lam'+str(i)+'.jpg',perturbation_img*255)
    cv2.imwrite(ex3_save_path+'classified_attack_lam'+str(i)+'.jpg',\
            attacked_image_mask*255)
    try :
for j in range(len(overlap_test_img.x_k4freqs)):
    temp = classify_perturbed_overlap_image(overlap_test_img.x_k4freqs[j],\
                                            [W_grass,W_cat],\
                                            [w_grass,w_cat],\
                                            [w0_grass,w0_cat])
    cv2.imwrite(ex3_save_path+'int_med_lam_'+str(i)+'_freq_'+str(j)+'.jpg',
            temp*255)
    except AttributeError: 
        print('for lam ='+str(lam)+' no intermediate data for the given frequency') 
        pass
    print('=====================================================================\n')
    
plt.figure()
plt.plot(lam_list,overlap_norms_record)
plt.xlabel('$\lambda$')
plt.ylabel('Frobenius norm of perturbation')
plt.show()   
#-----------------------------------------------------------------------------#