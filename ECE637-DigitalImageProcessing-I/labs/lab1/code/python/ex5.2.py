#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:41:15 2021

@author: rahul
course: ECE637-DIP-I
lab1- 5.2
"""

import numpy as np
import cv2

img = np.zeros((256,256))
img[127][127] = 255*100

cv2.imwrite('lab1_5.2.jpg',img) #original image

img2 = np.zeros((256,256))

for i in range(256):
    for j in range(256):
        if((i>0) and (j>0)):
            img2[i][j] = (0.01*img[i][j] + 0.9*(img2[i-1][j] + img2[i][j-1]) 
                          - 0.81*(img2[i-1][j-1]))           
            
cv2.imwrite('lab1_5.2_IIR.jpg',img2)