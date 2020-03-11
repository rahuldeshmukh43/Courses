#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE 595 ML-1: HW 6
@author: rahul
"""
import numpy as np, matplotlib.pyplot as plt

#%% ---------------------Exercise -1 -----------------------------------------#
Num_coins = 1000
Num_toss = 10
Num_iter = 100000

v1 = np.zeros(Num_iter); vrand = np.zeros(Num_iter); vmin =np.zeros(Num_iter);

#(b)
for i in range(Num_iter):
    # generate random experiments 1=head, 0=tails
    i_exp = np.random.randint(0,2,(Num_toss,Num_coins))
    #find c1,c_rand ,c_min and store v1,vrand, vmin
    v1[i] = np.sum(i_exp[:,0])/Num_toss
    c_rand = np.random.randint(0,Num_coins)
    vrand[i] = np.sum(i_exp[:,c_rand])/Num_toss
    c_min = np.argmin(np.sum(i_exp,axis = 0))
    vmin[i] = np.sum(i_exp[:,c_min])/Num_toss

u1 = 0.5; urand = 0.5; umin = 0.5;

plt.figure(1)
plt.hist(v1,color='blue',bins=10)
plt.xlim(0,1)
plt.title('Histogram for V1')
plt.xlabel('fraction of heads')
plt.ylabel('Number of samples')

plt.figure(2)
plt.hist(vrand,color='green',bins=10)
plt.title('Histogram for Vrand')
plt.xlim(0,1)
plt.xlabel('fraction of heads')
plt.ylabel('Number of samples')

plt.figure(3)
plt.hist(vmin,color='black',bins=10)
plt.title('Histogram for Vmin')
plt.xlim(0,1)
plt.xlabel('fraction of heads')
plt.ylabel('Number of samples')
#(c)
eps=np.arange(0,0.55,0.05)
num_eps = len(eps)

P1=np.zeros(num_eps); Prand=np.zeros(num_eps); Pmin=np.zeros(num_eps);
hf_bound = np.zeros(num_eps)

for i,e in enumerate(eps):
    P1[i] = np.sum(np.where(np.abs(v1-u1)>e,1,0))/Num_iter
    Prand[i] = np.sum(np.where(np.abs(vrand-urand)>e,1,0))/Num_iter
    Pmin[i] = np.sum(np.where(np.abs(vmin-umin)>e,1,0))/Num_iter
    hf_bound[i] = 2*np.exp(-2*(e**2)*Num_toss)

plt.figure(4)
plt.plot(eps,P1,c='blue',marker='*')
plt.plot(eps,Prand,c='green',marker='o')
plt.plot(eps,Pmin,c='black',marker='^')
plt.plot(eps,hf_bound,color='red')
plt.legend(['v1','vrand','vmin','Hoeffding bound'])
plt.ylabel('P(|v-$\mu$|>$\epsilon$)')
plt.xlabel('eps')
plt.show()