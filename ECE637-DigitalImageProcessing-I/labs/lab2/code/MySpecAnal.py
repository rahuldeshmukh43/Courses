#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahul 
course: ECE637-DIP-I
lab2- 2D Random Process
"""

import numpy as np                 # Numpy is a library support computation of large, multi-dimensional arrays and matrices.
from PIL import Image              # Python Imaging Library (abbreviated as PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
import matplotlib.pyplot as plt    # Matplotlib is a plotting library for the Python programming language.
import sys,os


def SpecAnal(x,cx,cy,block_size,base_name):
    i = cx
    j = cy
    N = block_size
    
    z = x[i:N+i, j:N+j]
    
    # Compute the power spectrum for the NxN region.
    Z = (1/N**2)*np.abs(np.fft.fft2(z))**2
    
    # Use fftshift to move the zero frequencies to the center of the plot.
    Z = np.fft.fftshift(Z)
    
    # Compute the logarithm of the Power Spectrum.
    Zabs = np.log(Z)
    
    # Plot the result using a 3-D mesh plot and label the x and y axises properly. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, num = N)
    X, Y = np.meshgrid(a, b)
    
    surf = ax.plot_surface(X, Y, Zabs, cmap=plt.cm.coolwarm)
    
    ax.set_xlabel('$\mu$ axis')
    ax.set_ylabel('$\\nu$ axis')
    ax.set_zlabel('Z Label')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig('SpecAnal_'+base_name+'_'+str(block_size)+'.eps',format='eps')
    
def BetterSpecAnal(x,base_name):
    block_size=64
    h,w = x.shape
    cx=w//2; cy=h//2
    or_x = cx-5*(block_size//2)
    or_y = cy-5*(block_size//2)
    windows = []
    for i in range(5):
        for j in range(5):
            ul_x = or_x +i*block_size
            ul_y = or_y +j*block_size
            windows.append(x[ul_y:ul_y+block_size,ul_x:ul_x+block_size])
    W = np.hamming(block_size)
    W = np.outer(W,W)
    #multiply 2D hamming window
    windows = [w*W for w in windows]
    #compute squred DFT magnitude
    Z = [ (1/(block_size**2)*np.abs(np.fft.fft2(w))**2) for w in windows]
    Z = [np.fft.fftshift(z) for z in Z]
    Zabs = [np.log(z) for z in Z]
    #compute average
    av = np.zeros((block_size,block_size))
    for z in Zabs: av+=z
    av/=25
    
    # Plot the result using a 3-D mesh plot and label the x and y axises properly. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, num = block_size)
    X, Y = np.meshgrid(a, b)
    
    surf = ax.plot_surface(X, Y, av, cmap=plt.cm.coolwarm)
    ax.set_xlim(-1*np.pi,1*np.pi)
    ax.set_ylim(-1*np.pi,1*np.pi)
    ax.autoscale(enable=True,axis='z',tight=True)
    ax.set_xlabel('$\mu$ axis')
    ax.set_ylabel('$\\nu$ axis')
    ax.set_zlabel('Z Label')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('BetterSpecAnal_'+base_name+'.eps',format='eps')
            
    

if __name__=="__main__":
    # Read in a gray scale TIFF image.
    img_name=sys.argv[1]
    base_name=os.path.basename(img_name).split('.')[0]
    flag_betterspecanal = int(sys.argv[2])
    if(flag_betterspecanal==0):
        print('Enter BlockSize: ')
        block_size=int(input())
    
    im = Image.open(img_name)
    print('Read '+img_name)
    print('Image size: ', im.size)
    
    # Display image object by PIL.
    im.show(title='image')
    
    # Import Image Data into Numpy array.
    # The matrix x contains a 2-D array of 8-bit gray scale values. 
    x = np.array(im)
    print('Data type: ', x.dtype)
    
    # Display numpy array by matplotlib.
    plt.imshow(x, cmap=plt.cm.gray)
    plt.title('Image')
    
    # Set colorbar location. [left, bottom, width, height].
    cax =plt.axes([0.9, 0.15, 0.04, 0.7]) 
    plt.colorbar(cax=cax)
    plt.show()
    
    x = np.double(x)/255.0
    if flag_betterspecanal==0:
        SpecAnal(x,99,99,block_size,base_name)
    else:
        BetterSpecAnal(x,base_name)