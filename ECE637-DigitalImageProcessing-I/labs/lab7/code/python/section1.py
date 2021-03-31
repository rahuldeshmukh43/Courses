#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:58:00 2021

@author: rahul
lab 7: Image restoration Section-1
ece637-DIP-1
"""
import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import convolve2d as conv2d

filter_size = 7
sampling_frequency = 20

def get_base(filename):
    base = os.path.basename(filename).split('.')[0]
    return base

def conv(X_img, conv_filter):
    "apply conv with zero padding"
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

def main(X_img,Y_img, filter_size):
    filter_half_wd = filter_size//2
    ht, wd = X_img.shape
    assert (ht,wd) == Y_img.shape    
    #make Z
    Z_sparse=[]; Y_sparse=[]; N=0
    for i in range(ht//sampling_frequency):
        for j in range(wd//sampling_frequency):
            row = (i + 1)*sampling_frequency
            col = (j + 1)*sampling_frequency
            zs = X_img[row - filter_half_wd : row + filter_half_wd + 1, 
                       col - filter_half_wd : col + filter_half_wd + 1]
            ys = Y_img[row, col]
            Z_sparse.append(list(zs.reshape(filter_size*filter_size)))
            Y_sparse.append(ys)
            N+=1    
    Z_sparse = np.array(Z_sparse)
    Y_sparse = np.array(Y_sparse)

    #compute R_zz, r_zy, theta
    R_zz = np.dot(Z_sparse.T, Z_sparse)/N
    r_zy = np.dot(Z_sparse.T, Y_sparse)/N
    theta = np.dot(np.linalg.inv(R_zz), r_zy)
    
    #apply theta filter to X to get Y_hat
    theta = np.reshape(theta, (filter_size, filter_size))
    print('MSME filter: ')
    print(theta)
    #Y_hat_img = conv2d(X_img, theta)
    Y_hat_img = conv(X_img, theta)
    
    #rescale to 0-255
    Y_hat_img -= Y_hat_img.min()
    Y_hat_img *= (255./Y_hat_img.max())
    return Y_hat_img.astype(np.uint8)
    
if __name__=="__main__":
    original_img_name = sys.argv[1]
    noisy_img_name = sys.argv[2]
    output_name = get_base(noisy_img_name) + '_predicted.pdf'
    print(output_name)
    
    X_img = np.array(Image.open(noisy_img_name))
    Y_img = np.array(Image.open(original_img_name))
    
    Y_hat_img = main(X_img, Y_img, filter_size)
    
    gray = cm.get_cmap('gray',256)
    plt.figure(frameon=False)
    plt.imshow(Y_hat_img, cmap=gray)
    plt.axis('off')
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    
    #im = Image.fromarray(Y_hat_img)
    #im.save(temp_name)