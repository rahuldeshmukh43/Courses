#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE 595 ML-1: HW5
Attack on Classifiers
Functions
@author: rahul
"""
# import libraries
import numpy as np
import cv2, time
#%%-----------------------------------------------------------------------------------#
class TestImgClass:
    """
    Class for Test sample
    """
    def Load_img(self,path):
        """
        Load the image from given path
        """
        self.img = cv2.imread(path,0)/255.0
        
    def get_non_overlap_patches(self):
        """
        Function for obtaining all non-overlapping patches
        of size 8,8 and made as vector of size 64
        """
        try: 
            m,n = np.shape(self.img)         
        except AttributeError: 
            raise Exception('Test Image not loaded')      
        patches = [] # for storing the patches
        for i in range(m//8):
            for j in range(n//8):
                temp = self.img[8*i:8*i+8,8*j:8*j+8]
                patches.append(np.reshape(temp,(64,1)))
        self.non_overlap_patches = patches
        
#    def get_overlap_patches(self):
#        """
#        Function for obtaining all overlapping patches
#        of size 8,8 and made as vector of size 64
#        """
#        try: 
#            m,n = np.shape(self.img)         
#        except AttributeError: 
#            raise Exception('Test Image not loaded')      
#        patches = [] # for storing the patches
#        projections = [] # for storing P_i matrices
#        idx_mat = np.reshape(np.arange(m*n),(m,n))
#        
#        for i in range(m-8):
#            for j in range(n-8):
#                temp = self.img[i:i+8,j:j+8]
#                patches.append(np.reshape(temp,(64,1)))
##                temp_idx = idx_mat[i:i+8,j:j+8]
##                temp_projection = get_projection_mat(temp_idx,m,n)
##                projections.append(temp_projection)
#                
#        self.overlap_patches = patches
#        self.projections = projections
        
    def get_attack_image_from_non_overlap_patches(self,W,w,w0,lam,alpha,\
                                                  Max_iter=300,tol=0.001):
        """
        lam: regularization coefficient
        alpha = step size
        W,w,w0: parameters for Quadratic classifier,is a list each for diff classes
        labels = 0:grass,1:cat
        Max_iter: maximum iterations for CW attack 
        tol: tolerance on frobenius norm for attack 
        """
        m,n = np.shape(self.img)
        attacked_image = np.zeros((m,n))
        attacked_image_mask = np.zeros((m,n))
        original_mask = np.zeros((m,n))
        attacked_patches = []
        for i in range(len(self.non_overlap_patches)):
            i_patch = self.non_overlap_patches[i]# sample
            i_predicted_label = classify_sample(i_patch,W,w,w0)
            i_attacked_patch = CW_attack(i_patch,i_predicted_label,W,w,w0,\
                                         lam,alpha,Max_iter,tol)
            
            attacked_patches.append(i_attacked_patch)
            row= i//(n//8)
            col= i - row*(n//8)          
            attacked_image[8*row:8*row+8,8*col:8*col+8] = np.reshape(i_attacked_patch,(8,8))
            attacked_image_mask[8*row:8*row+8,8*col:8*col+8] = classify_sample(i_attacked_patch,W,w,w0)
            original_mask[8*row:8*row+8,8*col:8*col+8] = i_predicted_label
            
        self.attacked_patches = attacked_patches
        self.attacked_image = attacked_image
        self.attacked_image_mask = attacked_image_mask
        self.original_mask = original_mask
        
    def get_attack_image_from_overlap_patches(self,W,w,w0,lam,alpha,\
                                                  Max_iter=300,tol=0.01,freq=0):
        """
        lam: regularization coefficient
        alpha = step size
        W,w,w0: parameters for Quadratic classifier,is a list each for diff classes
        labels = 0:grass,1:cat
        Max_iter: maximum iterations for CW attack 
        tol: tolerance on frobenius norm for attack 
        """
        m,n = np.shape(self.img)
        idx = np.reshape(np.arange(m*n),(m,n)) # matrix for indices 
        x0 = np.reshape(self.img,(m*n,1))
        x_k4freqs=[]
        # CW attack loop
        do=True;    k=0;    x_k = np.copy(x0);
        while do:
#            print('iteration#'+str(k)+'\n')
#            time_in = time.time()
            x_new = x_k - alpha*CW_grad_overlap(x_k,x0,lam,W,w,w0,idx)
            x_new = np.clip(x_new,0.0,1.0)
            k+=1        
            # update do
            change = np.linalg.norm(x_new-x_k)#/np.linalg.norm(x_k)
            if k>Max_iter or change<=tol:
                do =False
            x_k = x_new
            if freq!=0 :
                if k%freq==0: x_k4freqs.append(np.reshape(x_k,(m,n)))
#            print(change)
#            print(str(time.time()-time_in)+'sec\n')            
#            print('#------------------------#\n')
        print('For lam = '+str(lam)+ ' took total iterations = '+str(k))
        self.attacked_image = np.reshape(x_k,(m,n))
        if freq!=0: self.x_k4freqs = x_k4freqs
        
def classify_perturbed_overlap_image(img,W,w,w0):
    """
    classification using non-overlap patches    
    """
    m,n = np.shape(img)
    attacked_image_mask = np.zeros((m,n))
    for i in range(m-8):
        for j in range(n-8):
            x = img[i:i+8,j:j+8]
            x = np.reshape(x,(64,1))
            attacked_image_mask[i,j] = classify_sample(x,W,w,w0)                
    return(attacked_image_mask)
#%%---------------------------------------------------------------------------------#        
def CW_grad_overlap(x,x0,lam,W,w,w0,idx):
    """
    gradient of CW attack for overlapping patches      
    x: x_k iterate
    x0: original vectorized image
    lam: user defined value
    W,w,w0: quadratic classifier params
    idx: index matrix for original image
    """
    grad= np.zeros_like(x0)
    m,n= np.shape(idx)
    targets = [1,0]
    # loop over overlapping patches
    for i in range(m-8):
        for j in range(n-8):          
            # for ith patch x_i
            i_idx = idx[i:i+8,j:j+8]
            i_idx = np.reshape(i_idx,(64))            
            x_i = np.copy(x[i_idx])
            x0_i = np.copy(x0[i_idx])
            # get original & target label for this patch
            original_label = classify_sample(x0_i,W,w,w0)
            target_label= targets[original_label]
            g_j = ((1/2)*x_i.T@W[original_label]@x_i + w[original_label].T@x_i +w0[original_label]) 
            g_t = ((1/2)*x_i.T@W[target_label]@x_i + w[target_label].T@x_i +w0[target_label])
            s = g_j - g_t   
            if s>0:
                multiplier = (W[original_label]@x_i + w[original_label])  - (W[target_label]@x_i + w[target_label])
                grad[i_idx] += multiplier
    grad = 2*(x-x0) + lam*grad
    return(grad)

#def get_projection_mat(idx,M,N):
#    """
#    Input:
#        idx: matrix of element indices
#        m,n: size of original image
#    output:
#        PMat: matrix of size 64,(m*n) with ones in a column position using idx
#    """
#    m,n = np.shape(idx)
#    PMat = np.zeros(((m*n),(M*N)))
#    idx_vec = np.reshape(idx,(m*n,1))
#    PMat[:,idx_vec] = 1
#    return(PMat)

#%%---------------------------------------------------------------------------------#        
    
def CW_attack(x0,label,W,w,w0,lam,alpha,Max_iter,tol):
    """
    x0: input patch
    label: label of the x0 patch using classifier
    W,w,w0: parameters for Quadratic classifier,is a list each for diff classes
    lam: regularization coefficient
    alpha = step size
    labels = 0:grass,1:cat
    """
    targets = [1,0]
    target_label = targets[label]
    do=True;    k=0;    x_k = x0;
    while do:
        x_new = x_k - alpha*CW_grad(x_k,x0,lam,W,w,w0,label,target_label)
        x_new = np.clip(x_new,0.0,1.0)
        k+=1        
        # update do
        if k>Max_iter or np.linalg.norm(x_new-x_k)<=tol:
            do =False
        x_k = x_new
#    print('For lam = '+str(lam)+ ' took total iterations = '+str(k))
#    print('==============================\n')
    return(x_k)
        
def CW_grad(x,x0,lam,W,w,w0,i_star,target_label):
    """
    gradient of CW attack        
    """
    g_istar = ((1/2)*x.T@W[i_star]@x + w[i_star].T@x +w0[i_star]) 
    g_t = ((1/2)*x.T@W[target_label]@x + w[target_label].T@x +w0[target_label])
    s = g_istar - g_t    
    if s>0:
        multiplier = (W[i_star]@x + w[i_star]) - (W[target_label]@x + w[target_label])
        return(2*(x-x0)+lam*multiplier)
    else:
        return(2*(x-x0))
            

def classify_sample(x,W,w,w0):
    """
    x: test sample
    W,w,w0: parameters for Quadratic classifier,is a list each for diff classes
    labels = 0:grass,1:cat
    """
    temp =(1/2)*x.T@(W[1]-W[0])@x + (w[1]-w[0]).T@x + (w0[1]-w0[0])
    if temp>=0:
        return(1)
    else: 
        return(0)
#-----------------------------------------------------------------------------------#        