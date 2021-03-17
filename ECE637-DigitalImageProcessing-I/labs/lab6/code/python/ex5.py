#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:00:46 2021

@author: rahul
ECE 637 DIP-1
Lab 6: Intro to colorimetry Ex 4
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

RGB_709 = np.array([[0.640, 0.330, 0.030],
                    [0.300, 0.600, 0.100],
                    [0.150, 0.060, 0.790]])
name=['R','G','B']
D_65_wp = np.array([0.3127, 0.3290, 0.3583])

def main(step, gamma, x_data, y_data):    
    u = np.arange(0,1,step)
    N = u.shape[0]
    x,y = np.meshgrid(u,u)
    z= 1-x-y
    
    #D_65_wp_scaled = D_65_wp/D_65_wp[1]
    #scaling_coefs = np.dot(np.linalg.inv(RGB_709),D_65_wp_scaled)
    scaling_coefs = np.ones(3)
    M = np.dot(RGB_709.T, np.diag(scaling_coefs))
    M_inv = np.linalg.inv(M)
    rgb = np.zeros((N,N,3))
    for i in range(N):
        for j in range(N): 
            rgb[i,j,:] = np.dot(M_inv,np.array([x[i,j],
                                                y[i,j],
                                                z[i,j]]))
            if(np.any(rgb[i,j,:]<0)): rgb[i,j,:] = 1
            
    for k in range(3):
        for i in range(N):
            for j in range(N):
                rgb[i,j,k] = np.exp((1.0/gamma)*np.log(rgb[i,j,k]))

    plt.figure()
    plt.imshow(rgb, extent=[0,1,0,1],origin='lower')
    plt.plot(x_data,y_data,'-ob',label='pure spectral source')
    plt.plot(RGB_709[:2,0], RGB_709[:2,1],'r', label='Rec 709')
    plt.plot(RGB_709[1:3,0], RGB_709[1:3,1],'r')
    plt.plot(RGB_709[[0,2],0], RGB_709[[0,2],1],'r')
    for i in range(3): plt.text(RGB_709[i,0], RGB_709[i,1], name[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('chrom_plot.pdf')

if __name__=="__main__":
    step = 0.005
    gamma= 2.2
    data_path = sys.argv[1]    
    data = np.load(data_path,allow_pickle=True)[()]
    X = data['x'][0]
    Y = data['y'][0]
    Z = data['z'][0]
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)    
    main(step, gamma, x, y)