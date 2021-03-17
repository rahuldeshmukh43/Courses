#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul 
course: ECE637-DIP-I
lab4- section 4.2 Gamma of monitor
"""
import argparse
import os 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

gamma_monitor=1.709511 # hard coded

def gamma_correct_from_linear(img,gamma_out):
    out_img = np.zeros_like(img)
    h,w,ch = img.shape
    for k in range(ch):
        for i in range(h):
            for j in range(w):
                #inverse of eq (5)    
                out_img[i,j,k] = 255.0*np.exp((1.0/gamma_out)*np.log(img[i,j,k]/255.0))
    return out_img
    
def gamma_correct_from_gamma(img,gamma_out,gamma_in):
    out_img = np.zeros_like(img)
    h,w,ch = img.shape
    for k in range(ch):
        for i in range(h):
            for j in range(w):
                x = 255.0*(img[i,j,k]/255.0)**gamma_in
                out_img[i,j,k] = 255.0*np.exp((1.0/gamma_out)*np.log(x/255.0))    
    return out_img
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file",type=str, help="path to input image file")
    parser.add_argument("-l","--linear", help="Linear scaled input flag",action="store_true")
    parser.add_argument("-gin","--gamma_input",type=np.float, help="gamma value of input")
    parser.add_argument("-gout","--gamma_output",type=np.float, help="gamma value of output")
    
    args = parser.parse_args()
    
    filename = args.image_file
    
    gamma_out = args.gamma_output  if args.gamma_output else gamma_monitor
    if args.gamma_input: gamma_in = args.gamma_input
    
    basename = os.path.basename(filename).split('.')[0]
    im = Image.open(filename)
    img = np.array(im)
    
    #display input image
    plt.imshow(img,cmap=mpl.cm.gray)
    plt.savefig(basename+'.pdf')
    plt.close()
    
    if args.linear: 
        #print('gamma_output: '+str(gamma_out))
        out_img = gamma_correct_from_linear(img,gamma_out)
    else:
        print("gamma_input: "+str(gamma_in))
        print("gamma_output: "+str(gamma_out))
        out_img = gamma_correct_from_gamma(img,gamma_out,gamma_in)
    
    #save output image
    plt.imshow(out_img,cmap=mpl.cm.gray)
    plt.savefig(basename+'_gamma_corrected.pdf')
    plt.close()
    
