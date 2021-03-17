#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:26:23 2021

@author: rahul
ECE 637 DIP-1
Lab 6: Intro to colorimetry Ex 3
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

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

def main(data):
    X = data['x'][0]
    Y = data['y'][0]
    Z = data['z'][0]
    
    #plot chromacity (x,y)
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    
    plt.figure()
    plt.plot(x,y,'-ob',label='pure spectral source')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(RGB_cie[:2,0], RGB_cie[:2,1],'k',label = 'CIE 1931')
    plt.plot(RGB_cie[1:3,0], RGB_cie[1:3,1],'k')
    plt.plot(RGB_cie[[0,2],0], RGB_cie[[0,2],1],'k')
    for i in range(3): plt.text(RGB_cie[i,0], RGB_cie[i,1], name[i])
    
    plt.plot(RGB_709[:2,0], RGB_709[:2,1],'r', label='Rec 709')
    plt.plot(RGB_709[1:3,0], RGB_709[1:3,1],'r')
    plt.plot(RGB_709[[0,2],0], RGB_709[[0,2],1],'r')
    for i in range(3): plt.text(RGB_709[i,0], RGB_709[i,1], name[i])
    
    plt.plot(D_65_wp[0],D_65_wp[1], 'x',label='$D_{65}$')
    plt.plot(EE_wp[0],EE_wp[1], '*', label='EE')
    
    plt.legend()
    plt.savefig('chrom_diag.pdf')
    
    
if __name__=="__main__":
    data_path = sys.argv[1]    
    data = np.load(data_path,allow_pickle=True)[()]
    main(data)
    