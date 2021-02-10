#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul 
course: ECE637-DIP-I
lab3- Connected Components
"""
import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mp

np.random.seed(seed=637)

def main(filename):
    basename = os.path.basename(filename).split('.')[0]
    threshold = basename.split('_')[1]
    im = Image.open(filename)
    img = np.array(im)
    NumRegions = np.max(img)
    cmap = mp.colors.ListedColormap(np.random.rand(NumRegions,3))
    plt.imshow(img,cmap=cmap)
    plt.title('Segmentation Image: Threshold='
              +str(threshold)+' NumRegions='+str(NumRegions))
    plt.colorbar(shrink=0.5, aspect=5)
    plt.savefig(basename+'.eps',bbox_inches='tight', pad_inches = 0, format='eps')
    
if __name__=="__main__":
    filename = sys.argv[1]
    main(filename)