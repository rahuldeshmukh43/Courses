#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:32:39 2021
ECE637
Prof. Charles A. Bouman
Image Processing Laboratory: Eigenimages and Principal Component Analysis

Description:

This is a Matlab script that reads in a set of training images into
the Matlab workspace.  The images are sets of English letters written
in various fonts.  Each image is reshaped and placed into a column
of a data matrix, "X".
@author: Wenrui Li
"""

import sys
import numpy as np
from PIL import Image    
import matplotlib.pyplot as plt

# The following are strings used to assemble the data file names
datadir='../../training_data/'    # directory where the data files reside
dataset=['arial','bookman_old_style','century','comic_sans_ms','courier_new',
  'fixed_sys','georgia','microsoft_sans_serif','palatino_linotype',
  'shruti','tahoma','times_new_roman']
datachar='abcdefghijklmnopqrstuvwxyz'

def read_data():
    """
        Read in all these training images into columns of a single matrix X.
    
        Returns:
            X: Image column matrix.
    
    """
    Rows=64    # all images are 64x64
    Cols=64
    n=len(dataset)*len(datachar)  # total number of images
    p=Rows*Cols   # number of pixels

    X=np.zeros((p,n))  # images arranged in columns of X
    k=0
    for dset in dataset:
        for ch in datachar:
            fname='/'.join([datadir,dset,ch])+'.tif'
            im=Image.open(fname)
            img = np.array(im)
            X[:,k]=np.reshape(img,(1,p))
            k+=1
    return X

# display samples of the training data
def display_samples(X,ch):
    """
    Display samples.

    Args:
    X (ndarray) : Image column matrix.
    ch (char) : A char 'a'~'z'.

    Returns:

    """
    ind = ord(ch)-ord('a')
    fig, axs = plt.subplots(3, 4)
    for k in range(len(dataset)):
        img=np.reshape(X[:,26*(k-1)+ind],(64,64))

        axs[k//4,k%4].imshow(img,cmap=plt.cm.gray, interpolation='none') 
        axs[k//4,k%4].set_title(dataset[k])
        
    plt.show()

if __name__ == "__main__":
    ch = sys.argv[1]
    X = read_data()
    display_samples(X,ch)