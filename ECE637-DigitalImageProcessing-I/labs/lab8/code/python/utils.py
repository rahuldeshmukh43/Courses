#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul
ECE637 DIP-1: lab 8 Halftoning
utility file for image metrics
"""

import numpy as np
from scipy.signal import convolve2d

def conv2d(X_img, conv_filter):
    "apply conv with zero padding"
    filter_size,_ = conv_filter.shape
    filter_half_wd = filter_size//2
    theta = conv_filter.flatten()
    ht,wd = X_img.shape
    padded_img = np.zeros((ht+2*filter_half_wd, wd+2*filter_half_wd))
    padded_img[filter_half_wd:-filter_half_wd, filter_half_wd:-filter_half_wd] = X_img
    out_img = np.zeros((ht,wd))
    for row in range(ht):
        for col in range(wd):
            x = padded_img[row : row + 2*filter_half_wd + 1, 
                           col : col + 2*filter_half_wd + 1]
            out_img[row,col] = np.dot(x.flatten(), theta)
    return out_img

def Un_gamma_correct(img, gamma):
    lin =  255.*((img/255.)**gamma)
    return lin.astype(np.uint8)

def Scale_to_equal_brightness(img):
    return Un_gamma_correct(img, 1./3.)

def LPF(img, sigma=np.sqrt(2), size=7):
    "pass image through low-pass filter"
    #make the filter
    h = np.zeros((size,size))
    for r in range(size):
        for c in range(size):
            i = r - (size//2)
            j = c - (size//2)
            h[r,c] = np.exp(-1.0*(i**2+j**2)/(2.0*(sigma**2)))
    h /= np.sum(h) #normalize
    #convolve filter
    out_img = convolve2d(img, h, mode='same')
    #out_img = conv2d(img, h)
    #rescale to 0-255
    out_img -= out_img.min()
    out_img *= (255./out_img.max())
    return out_img.astype(np.uint8)

def RMSE(original_img, binary_image):
    "compute rmse error betweeen two images"
    h,w = original_img.shape
    assert (h,w)== binary_image.shape
    rmse = (1./(h*w))*np.sum((original_img.astype(np.float32) -
                              binary_image.astype(np.float32))**2)
    return np.sqrt(rmse)

def Fidelity(original_img, binary_img):
    "Compute image fidelity metric"
    gamma = 2.2
    # un-gamma correct the images
    original_lin = Un_gamma_correct(original_img, gamma)    
    binary_lin = Un_gamma_correct(binary_img, gamma)    
    #print(np.linalg.norm(binary_lin-binary_img))
    # LPF
    original_lin_lpf = LPF(original_lin)
    binary_lin_lpf = LPF(binary_lin)
    assert original_lin_lpf.shape==original_img.shape
    # scale to equal brightness (cube-root)
    f_tilde = Scale_to_equal_brightness(original_lin_lpf)
    b_tilde = Scale_to_equal_brightness(binary_lin_lpf)
    #rmse
    fidelity = RMSE(f_tilde, b_tilde)
    return fidelity