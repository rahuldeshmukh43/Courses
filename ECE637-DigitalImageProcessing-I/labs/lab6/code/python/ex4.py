#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:26:23 2021

@author: rahul
ECE 637 DIP-1
Lab 6: Intro to colorimetry Ex 4
"""
import sys
import numpy as np
from PIL import Image

lam = np.linspace(400,700,31)
A_inv = np.array([[ 0.2430,  0.8560, -0.0440],
                  [-0.3910,  1.1650,  0.0870],
                  [ 0.0100, -0.0080,  0.5630]])

RGB_cie = np.array([[0.73467, 0.26533, 0.0],
                    [0.27376, 0.71741, 0.00883],
                    [0.16658, 0.00886, 0.82456]])

RGB_709 = np.array([[0.640, 0.330, 0.030],
                    [0.300, 0.600, 0.100],
                    [0.150, 0.060, 0.790]])
name=['R','G','B']

D_65_wp = np.array([0.3127, 0.3290, 0.3583])
EE_wp = 0.3333*np.ones(3)

def main(data, reflect, source, source_name):
    X = data['x'][0]
    Y = data['y'][0]
    Z = data['z'][0]    
    
    m,n,p = reflect.shape
    I = np.zeros_like(reflect)
    for i in range(m):
        for j in range(n): I[i,j,:] = reflect[i,j,:]*source
    
    XYZ = np.zeros((m,n,3))
    for i in range(m):
        for j in range(n): XYZ[i,j,:] = np.dot(np.vstack((X,Y,Z)),I[i,j,:])
    
    D_65_wp_scaled = D_65_wp/D_65_wp[1]
    scaling_coefs = np.dot(np.linalg.inv(RGB_709.T),D_65_wp_scaled)
    M = np.dot(RGB_709.T, np.diag(scaling_coefs))
    M_inv = np.linalg.inv(M)
    print('M709_d65:')
    print(M)
     
    rgb = np.zeros((m,n,3))
    for i in range(m):
        for j in range(n): rgb[i,j,:] = np.dot(M_inv,XYZ[i,j,:])
    rgb = np.clip(rgb,0,1)
    #print(rgb.shape)
    im = Image.fromarray((rgb*255).astype(np.uint8))
    im.save('rgb_'+source_name+'.tif','tiff')
    
    
if __name__=="__main__":
    data_path = sys.argv[1]    
    reflection_path = sys.argv[2]
    source_name = sys.argv[3]
    
    data = np.load(data_path,allow_pickle=True)[()]
    reflect = np.load(reflection_path,allow_pickle=True)[()]
    reflect = reflect['R']
    if source_name=='d65': source = data['illum1'][0]
    elif source_name=='ee': source = data['illum2'][0]

    main(data, reflect, source, source_name)
    