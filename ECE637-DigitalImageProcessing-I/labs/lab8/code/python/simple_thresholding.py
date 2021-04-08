#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul
ECE637 DIP-1: lab 8 Halftoning
section 3
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

def main(input_img, Threshold):
    #compute binary image using threshold
    out_img = (255*(input_img>Threshold).astype(np.int)).astype(np.uint8)
    #compute rmse
    rmse = RMSE(input_img, out_img)
    fidelity = Fidelity(input_img, out_img)
    return out_img, rmse, fidelity
    
if __name__=="__main__":
    input_img_name = sys.argv[1]
    output_img_name = get_base(input_img_name)+ '_threshold'
    Threshold = 127.0
    
    im = Image.open(input_img_name)
    input_img = np.array(im)
    
    out_img, rmse, fidelity = main(input_img, Threshold)
    
    #print info
    print(output_img_name)
    print("RMSE:%0.3f \t Fidelity:%0.3f"%(rmse, fidelity))
    
    #save plots and images
    gray = cm.get_cmap('gray',256)
    plt.figure(frameon=False)
    plt.imshow(out_img, cmap=gray, interpolation='none')
    plt.axis('off')
    plt.savefig(output_img_name+'.pdf', bbox_inches='tight', pad_inches=0)
    
    out_im = Image.fromarray(out_img)
    out_im.save(output_img_name+'.tif')