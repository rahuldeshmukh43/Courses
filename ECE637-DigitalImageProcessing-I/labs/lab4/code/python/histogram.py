#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul 
course: ECE637-DIP-I
lab4- section 1 histogram
"""
import sys,os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(filename):
    basename = os.path.basename(filename).split('.')[0]
    im = Image.open(filename)
    img = np.array(im)
    gray = mpl.cm.get_cmap('gray',256)
    plt.figure(1)
    plt.imshow(img,cmap=gray)
    plt.savefig(basename+'_gray.eps',format='eps',
                bbox_inches='tight', pad_inches = 0)
    
    plt.figure(2)
    plt.hist(img.flatten(),bins=np.linspace(0,255,256))
    plt.xlim([0,255])
    plt.xlabel('pixel intensity')
    plt.ylabel('number of pixels')
    plt.savefig(basename+'_histogram.eps',format='eps',
                bbox_inches='tight', pad_inches = 0)

if __name__=="__main__":
    filename = sys.argv[1]
    main(filename)