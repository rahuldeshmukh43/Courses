#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:41:42 2021

@author: rahul
Course: ECE637-DIP-I
lab-1
"""

import sys,os
import matplotlib.pyplot as plt
import gdal

def main(tiff_name, eps_dir_name):
    tiff_img = gdal.Open(tiff_name).ReadAsArray().transpose((1,2,0))
    fname = os.path.basename(tiff_name).split('.')[0]
    plt.imsave(eps_dir_name + fname + '.eps',tiff_img)

if __name__ == '__main__':
    tiff_name=sys.argv[1]
    eps_dir_name= sys.argv[2]
    #tiff_name='/home/rahul/Desktop/MyGitRepo/Courses/ECE637-DigitalImageProcessing-I/labs/lab1/imgblur.tif'
    #eps_name='/home/rahul/Desktop/MyGitRepo/Courses/ECE637-DigitalImageProcessing-I/labs/lab1/pix/imgblur.eps'
    print(tiff_name+'  '+eps_dir_name)
    main(tiff_name, eps_dir_name)
    print(tiff_name+ ' converted to eps')