#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul 
course: ECE637-DIP-I
lab4- section 2 histogram eq
"""
import sys,os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

Nbins=256

def equalize(X,L=256):
    h,w = X.shape
    x_hist,_ = np.histogram(X.flatten(),bins=np.arange(L+1))
    cumsum_norm = np.cumsum(x_hist).astype(np.float32)
    cumsum_norm /= cumsum_norm[-1]
    Y = np.zeros_like(X).astype(np.float32)
    for i in range(h):
        for j in range(w):
            Y[i][j]  = cumsum_norm[X[i][j]]            
    #normalize
    y_max = np.max(Y)
    y_min = np.min(Y)
    Z = (L-1.0)*(Y-y_min)/(y_max-y_min)
    return Z,cumsum_norm

def main(filename):
    basename = os.path.basename(filename).split('.')[0]
    im = Image.open(filename)
    img = np.array(im)
    img_hist_eq, cumsum_norm = equalize(img,L=Nbins)
    
    #save image
    im_save = Image.fromarray(img_hist_eq)
    im_save.save(basename+'_hist_eq.tif')
    
    #plot
    gray = mpl.cm.get_cmap('gray',256)
    plt.figure(1)
    plt.imshow(img_hist_eq,cmap=gray)
    plt.savefig(basename+'_hist_eq.eps',format='eps',
                bbox_inches='tight', pad_inches = 0)
   
    plt.figure(2)
    plt.plot(np.arange(Nbins),cumsum_norm)
    plt.xlabel('pixel intensity')
    plt.ylabel('normalized cumulative summed histogram')
    plt.xlim([0,Nbins])
    plt.savefig(basename+'_cumsum_norm.eps',format='eps',
                bbox_inches='tight', pad_inches = 0)

if __name__=="__main__":
    filename = sys.argv[1]
    main(filename)