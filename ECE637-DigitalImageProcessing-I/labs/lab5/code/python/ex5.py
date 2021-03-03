#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 00:48:48 2021

@author: rahul
course: ece637 DIP-1
lab-5 section 5
"""
import glob,os,sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from read_data import read_data

Ht=64; Wd=64;
eig_size=10
test_dir='../../test_data/'

def option014(Y,option):
    params = []
    for k in range(26):
        samples = Y[:,k::26]
        _,num_samples = samples.shape
        mu = samples.sum(axis=1)/(1.0*num_samples)
        if option!=4:
            cov = np.zeros((eig_size,eig_size))
            for i in range(num_samples):
                cov += np.outer(samples[:,i]-mu,samples[:,i]-mu)
            cov /= num_samples-1.0            
            if option==1:
                cov = np.diag(np.diagonal(cov)) #only diagonal elements
        else:
            cov = np.eye(eig_size)
        dic = {'mean':mu,'cov':cov}
        params.append(dic)
    return params
    
def option23(Y,option):
    params = []
    for k in range(26):
        samples = Y[:,k::26]
        _,num_samples = samples.shape
        mu = samples.sum(axis=1)/(1.0*num_samples)
        cov = np.zeros((eig_size,eig_size))
        for i in range(num_samples):
            cov += np.outer(samples[:,i]-mu,samples[:,i]-mu)
        cov /= num_samples-1.0            
        dic = {'mean':mu,'cov':cov}
        params.append(dic)
    R_wc = np.zeros((eig_size,eig_size))
    for k in range(26): R_wc += params[k]['cov']
    R_wc /= 26
    if option==2:
        for k in range(26): params[k]['cov'] = R_wc
    elif option==3:
        Lam = np.diag(np.diagonal(R_wc))
        for k in range(26): params[k]['cov'] = Lam
    return params

def training(option):
    X = read_data()
    p,n = X.shape
    mu_x = np.sum(X,axis=1)/n
    X = (X.T - mu_x).T
    Z = X/np.sqrt(n)
    U,S,Vt = np.linalg.svd(Z,full_matrices=False)
    A = U[:,:eig_size]
    Y = np.dot(A.T,X)
    
    if (np.any(np.array([0,1,4])==option)): params = option014(Y,option)
    elif(np.any(np.array([2,3])==option)): params = option23(Y,option)
    
    return params, A, mu_x

def testing(params, A, mu_x):
    #read images
    filenames = glob.glob(test_dir+'*/*.tif')
    true_labels = []
    for i,f in enumerate(filenames):
        class_label = os.path.basename(f).split('.')[0]
        true_labels.append(class_label)
    #classify data
    predicted_labels = []
    for i,f in enumerate(filenames):
        im = Image.open(f)
        img = np.array(im)
        x = np.reshape(img,Ht*Wd) - mu_x
        y = np.dot(A.T,x)
        class_scores = np.zeros(26)
        for k in range(26):
            mu_k = params[k]['mean']
            cov_k = params[k]['cov']
            class_scores[k] = (np.dot((y-mu_k),np.dot(np.linalg.inv(cov_k),(y-mu_k))) +
                               np.log(np.abs(np.linalg.det(cov_k))) )
        label = np.argmin(class_scores)
        char = chr(ord('a') + label)
        predicted_labels.append(char)
    return true_labels, predicted_labels    

if __name__=="__main__":
    option = int(sys.argv[1])
    params, A, mu_x = training(option)
    true_labels, predicted_labels = testing(params, A, mu_x)
    # print results
    print('True\t Predicted\t')
    count = 0.0
    for i in range(len(true_labels)):
        flag_correct = (true_labels[i]==predicted_labels[i])
        if flag_correct: count+=1
        print("%s\t %s"%(true_labels[i],predicted_labels[i]),end='\t')
        if not flag_correct:
            print('x',end='')
        print('')
    acc = (count/len(true_labels))*100
    print('Accuracy: %0.3f'%(acc))
