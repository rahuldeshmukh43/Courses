#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:19:53 2021

@author: rahul
course: ece637 DIP-1
lab: 5 section 4
"""

import numpy as np
import matplotlib.pyplot as plt
from read_data import read_data

Ht=64
Wd=64
m = [1,5,10,15,20,30]

def display_eigen_images(U,Lam):
    fig, axs = plt.subplots(3, 4)
    for k in range(12):
        img=np.reshape(U[:,k],(Wd,Ht))
        axs[k//4,k%4].imshow(img,cmap=plt.cm.gray, interpolation='none') 
        axs[k//4,k%4].set_title('eigenimg '+str(k+1))
        axs[k//4,k%4].axis('off')
    plt.savefig('eigen_images.pdf')
    plt.close()
    
def plot_projections(Y):
    c=10
    x = np.arange(1,c+1,1)
    plt.figure()
    plt.xticks(x)
    plt.plot(x,Y[:c,0],'-ob',label='a')
    plt.plot(x,Y[:c,1],'-or',label='b')
    plt.plot(x,Y[:c,2],'-oc',label='c')
    plt.plot(x,Y[:c,3],'-ok',label='d')
    plt.xlabel('Eigen value index')
    plt.ylabel('Projection value')
    plt.legend()
    plt.savefig('projections.pdf')
    

def main():
    X = read_data()
    p,n = X.shape
    mu_hat = np.sum(X,axis=1)/n
    X = (X.T - mu_hat).T
    Z = X/np.sqrt(n)
    U,S,Vt = np.linalg.svd(Z,full_matrices=False)
    Lam = S**2
    display_eigen_images(U,Lam)
    
    #projection of images
    Y = np.dot(U.T,X[:,:4])
    plot_projections(Y)
    
    #reconstruction
    recons = np.zeros((p,len(m)))
    for i,im in enumerate(m):
        recons[:,i] = np.dot(U[:,:im],Y[:im,0])
    recons = (recons.T + mu_hat).T
    fig,axs=plt.subplots(3,2)
    for k in range(6):
        img=np.reshape(recons[:,k],(Wd,Ht))
        axs[k//2,k%2].imshow(img,cmap=plt.cm.gray, interpolation='none') 
        axs[k//2,k%2].set_title('m='+str(m[k]))
        axs[k//2,k%2].axis('off')
    plt.savefig('reconstruction.pdf')
    plt.close()

if __name__=="__main__":
    main()

