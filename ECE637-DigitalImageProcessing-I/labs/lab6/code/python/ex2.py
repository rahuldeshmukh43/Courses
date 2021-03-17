#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:26:23 2021

@author: rahul
ECE 637 DIP-1
Lab 6: Intro to colorimetry Ex 2
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

lam = np.linspace(400,700,31)
A_inv = np.array([[ 0.2430,  0.8560, -0.0440],
                  [-0.3910,  1.1650,  0.0870],
                  [ 0.0100, -0.0080,  0.5630]])


def main(data):
    x = data['x'][0]
    y = data['y'][0]
    z = data['z'][0]
    # plot of x,y,z_0 wrt lam
    plt.subplots()
    plt.plot(lam, x,'-ob',label='$x_0(\lambda)$')
    plt.plot(lam, y,'-or',label='$y_0(\lambda)$')
    plt.plot(lam, z,'-oc',label='$z_0(\lambda)$')
    plt.xlim([lam[0], lam[-1]])
    plt.xlabel('$\lambda$')
    plt.legend()
    plt.savefig('xyz.pdf')
    plt.close() 

    #plot of l,m,s wrt lam
    lms = np.dot(A_inv,np.vstack((x,y,z)))
    plt.subplots()
    plt.plot(lam, lms[0,:],'-ob',label='$l_0(\lambda)$')
    plt.plot(lam, lms[1,:],'-or',label='$m_0(\lambda)$')
    plt.plot(lam, lms[2,:],'-oc',label='$s_0(\lambda)$')
    plt.xlim([lam[0], lam[-1]])
    plt.xlabel('$\lambda$')
    plt.legend()
    plt.savefig('lms.pdf')
    plt.close()

    #plot of D65 and fluor illum
    plt.figure()
    plt.plot(lam, data['illum1'][0],'-ob',label='$D_{65}$')
    plt.plot(lam, data['illum2'][0],'-or',label='fluorescent light')
    plt.xlim([lam[0], lam[-1]])
    plt.xlabel('$\lambda$')
    plt.legend()
    plt.savefig('illums_spectrum.pdf')
    plt.close()

if __name__=="__main__":
    data_path = sys.argv[1]    
    data = np.load(data_path,allow_pickle=True)[()]
    data.keys()
    main(data)
