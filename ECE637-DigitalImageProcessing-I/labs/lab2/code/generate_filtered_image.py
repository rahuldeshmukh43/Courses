#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul
course: ECE637-DIP-I
lab2 2-D Random Process
"""

from PIL import Image as pil
import numpy as np
import sys
import matplotlib.pyplot as plt

A=3.0
B=0.99
C=-0.981

def IIR_filter(x):
    y=np.zeros_like(x)
    h,w = x.shape
    for m in range(1,h):
        for n in range(1,w):
            y[m][n] = A*x[m][n] +B*(y[m-1][n]+y[m][n-1]) +C*y[m-1][n-1]
    return y            

def main(fname):
    #create random image
    x = np.random.rand(512,512)
    x -= 0.5 # now in [-0.5,0.5]
    #display scaled image
    x_scaled = 255*(x+0.5)
    plt.imshow(x_scaled.astype(np.uint8))
    img_out = pil.fromarray(x_scaled.astype(np.uint8))
    img_out.save('Random_image.tif')
    #filter image
    y = IIR_filter(x)
    plt.imshow((y+127).astype(np.uint8))
    img_out = pil.fromarray((y+127).astype(np.uint8))
    img_out.save(fname) 
    
if __name__=="__main__":
    fname = sys.argv[1]
    main(fname)
