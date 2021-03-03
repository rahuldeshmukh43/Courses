#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:46:46 2021

@author: rahul
course: ece637 DIP-1
lab5: 2.1 and 2.2
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=637)
Rx = np.array([[2.0,-1.2],[-1.2,1.0]])

def plot_pts(W,name):
    plt.figure()
    plt.plot(W[0,:],W[1,:],'.')
    plt.axis('equal')
    plt.savefig(name+'.eps',format='eps')
    plt.close()
    
def estimate_mean_cov(X):
    "unbiased estimate of mean and cov (MLE)"
    dim,num_pts= X.shape
    mu_hat = X.sum(axis=1)/num_pts
    R_hat= np.zeros((dim,dim))
    for i in range(num_pts): R_hat+= np.outer(X[:,i]-mu_hat,X[:,i]-mu_hat)
    R_hat /= num_pts - 1.0
    return mu_hat, R_hat
    

def main(num_pts):
    #Section 2.1
    size,_=Rx.shape
    I = np.eye(size)
    mu= np.zeros(size)
    Lam,E = np.linalg.eig(Rx)
    W = np.random.multivariate_normal(mu,I,num_pts).T
    X_scaled = np.dot(np.diag(np.sqrt(Lam)),W)
    X = np.dot(E,X_scaled)
    plot_pts(W,'W')
    plot_pts(X_scaled,'X_scaled')
    plot_pts(X,'X')
    
    #Section 2.2
    mu_hat, R_hat = estimate_mean_cov(X)
    print('Input covariance(R): ')
    print(Rx);print()
    print('Estimated covariance(R_hat): ')
    print(R_hat);print()
    print('Difference in covariances(R_hat-R): ')
    print(R_hat-Rx);print()
    
    Lam_hat,E_hat = np.linalg.eig(R_hat)
    X_scaled_hat = np.dot(E_hat.T,X)
    W_hat = np.dot(np.diag(np.sqrt(1/Lam_hat)), X_scaled_hat)
    mu_W_hat, R_W_hat = estimate_mean_cov(W_hat)
    print('Estimated covariance of R_W_hat: ')
    print(R_W_hat);print()
    
    plot_pts(X_scaled_hat,'X_scaled_hat')
    plot_pts(W_hat,'W_hat')
    
if __name__=="__main__":
    num_pts = int(sys.argv[1])
    main(num_pts)
