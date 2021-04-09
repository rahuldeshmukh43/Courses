#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul
ECE637 DIP-1: lab 8 Halftoning
section 5
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
    
def main(input_img, gamma, Threshold):
    h,w = input_img.shape
    linear_img = 255.*(input_img/255.)**gamma
    out_img = np.zeros((h,w))
    H = np.array([7., 3., 5., 1.])/16.
    K = [0, 1, 1, 1]
    L = [1, -1, 0, 1]
    for i in range(h):
        for j in range(w):
            old_pxl = linear_img[i,j]
            q = 255. if old_pxl>Threshold else 0.
            out_img[i, j] = q
            e = old_pxl - q
            for h_kl,k,l in zip(H,K,L):
                r = i+k; c = j+l
                if r<h and r>=0 and c<w and c>=0: 
                    linear_img[r,c]+= e*h_kl
    return out_img.astype(np.uint8)
    
if __name__=="__main__":
    input_img_name = sys.argv[1]
    output_img_name = get_base(input_img_name)+ '_error_diff'
    gamma=2.2
    Threshold = 127.
    print(output_img_name)
    
    im = Image.open(input_img_name)
    input_img = np.array(im)
    
    out_img= main(input_img, gamma, Threshold)
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