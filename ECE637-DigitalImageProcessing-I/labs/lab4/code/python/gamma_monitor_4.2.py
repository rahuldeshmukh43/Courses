#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul 
course: ECE637-DIP-I
lab4- section 4.2 Gamma of monitor
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

img_size=256
stripe_ht=16
chkr_blk = np.array(([[255,255,0,0],
                      [255,255,0,0],
                      [0,0,255,255],
                      [0,0,255,255]]))

def compute_gamma(g):
    return np.log(0.5)/np.log(g/255.0)

def main(gray):
    img = gray*np.ones((img_size,img_size))
    h,w = chkr_blk.shape
    for row in range(img_size//(2*stripe_ht)):
        pivot_row = row*(2*stripe_ht)
        for i in range(stripe_ht//h):
            for j in range(img_size//w):
                img[pivot_row+i*h:pivot_row+(i+1)*h, j*w:(j+1)*w] = chkr_blk

    gamma = compute_gamma(gray)
    gamma_str = "{:.3f}".format(gamma)
    gamma_str = gamma_str.replace('.','_')
    #print computed gamma
    print("Computed gamma: %0.6f"%(gamma))
    
    #display image
    plt.imshow(img,cmap=mpl.cm.gray)
    plt.show(block=False)
    
    #ask if want to save
    print("Do you want to print the image?(y/n): ",end='')
    print_it=input();
    if(print_it=='y'): 
        plt.savefig('Array_pattern_gamma_'+gamma_str+'.eps', format='eps')
    
if __name__=="__main__":
    gray = np.int(sys.argv[1])
    main(gray)
