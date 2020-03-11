# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:56:16 2019

@author: rahul
"""
import numpy as np

N1=100
N2=200

x1 = np.random.rand(2,N1)
x2 = np.random.rand(2,N2)

u1=np.mean(x1,axis=1)
u2=np.mean(x2,axis=1)

sigma = np.zeros((2,2))
for i in range(N1):
    sigma+= np.outer((x1[:,i]-u1),(x1[:,i]-u1))
    
for i in range(N2):
    sigma+= np.outer((x2[:,i]-u2),(x2[:,i]-u2))
    
S = np.outer((u1-u2),(u1-u2))

print(sigma/S)
    