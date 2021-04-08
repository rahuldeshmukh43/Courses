#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul
ECE637 DIP-1: lab 8 Halftoning
section 4
"""

import numpy as np
from PIL import Image
import os, sys
import matplotlib.pyplot as plt
from matplotlib import cm

from utils import RMSE, Fidelity

def get_base(filename):
    base = os.path.basename(filename).split('.')[0]
    return base

def Dither_mat(N):
    if N==2: 
        return np.array([[1,2],[3,0]])
    else:
        return np.block([[4*Dither_mat(N//2) + 1, 4*Dither_mat(N//2) + 2 ],
                         [4*Dither_mat(N//2) + 3, 4*Dither_mat(N//2)     ]])
    
def Threshold_mat(dither_mat):
    N,_=dither_mat.shape
    T = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            T[i,j] = 255.0*((dither_mat[i,j] + 0.5)/N**2)
    return T

def main(input_img, dither_size, gamma):
    h,w = input_img.shape
    linear_img = 255.*(input_img/255.)**gamma
    dither_mat = Dither_mat(dither_size)
    threshold_mat = Threshold_mat(dither_mat)

    out_img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            r = i%dither_size; c = j%dither_size
            if linear_img[i,j] > threshold_mat[r,c]: 
                out_img[i,j] = 255
            else: 
                out_img[i,j] = 0
    return out_img.astype(np.uint8)
    
if __name__=="__main__":
    input_img_name = sys.argv[1]
    dither_size = np.int(sys.argv[2])
    output_img_name = get_base(input_img_name)+ '_dither'+str(dither_size)
    gamma=2.2
    print(output_img_name)
    
    im = Image.open(input_img_name)
    input_img = np.array(im)
    
    out_img= main(input_img, dither_size, gamma)
    #compute rmse
    rmse = RMSE(input_img, out_img)
    fidelity = Fidelity(input_img, out_img)
    #print info
    print("RMSE:%0.3f \t Fidelity:%0.3f"%(rmse, fidelity))
    
    #save plots and images
    gray = cm.get_cmap('gray',256)
    plt.figure(frameon=False)
    plt.imshow(out_img, cmap=gray, interpolation='none')
    plt.axis('off')
    plt.savefig(output_img_name+'.pdf', bbox_inches='tight', pad_inches=0)
    
    out_im = Image.fromarray(out_img)
    out_im.save(output_img_name+'.tif')