#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul 
course: ECE637-DIP-I
lab4- section 3 contrast stretching
"""
import sys,os
from PIL import Image
import numpy as np

Nbins=256

def stretch(X,T1,T2):
    h,w = X.shape    
    slope = (255.)/(T2-T1)
    Y = np.zeros_like(X).astype(np.float32)
    for i in range(h):
        for j in range(w):
            if(X[i][j]<=T1): 
                Y[i][j]=0.
            elif(X[i][j]>=T2):
                Y[i][j]=255.
            else:
                Y[i][j] = slope*(X[i][j]-T1)    
    return Y

def find_T1_T2(X):
    "T1 and T2 such that o/p histogram spans 0-255"
    T1 = X.min()
    T2 = X.max()
    return T1,T2    

def main(filename,T1,T2):
    basename = os.path.basename(filename).split('.')[0]
    im = Image.open(filename)
    img = np.array(im)
    # find T1 and T2
    if(T1<0 or T2<0 or T2<T1): 
        T1,T2 = find_T1_T2(img)
    print('T1='+str(T1)+' T2='+str(T2))
    img_cont_st = stretch(img,T1,T2)    
    #save image
    im_save = Image.fromarray(img_cont_st)
    im_save.save(basename+'_cont_st.tif')

if __name__=="__main__":
    filename = sys.argv[1]
    T1 = np.float32(sys.argv[2])
    T2 = np.float32(sys.argv[3])
    main(filename,T1,T2)