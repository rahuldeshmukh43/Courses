#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:18:31 2021

@author: rahul
course: ECE637-DIP-I, Spring 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def FIR_LP1(u):
    a =1.0
    for i in range(4):
        a += 2.0*np.cos((i+1)*u)
    return a/9.0

def FIR_LP(u,v):
    return FIR_LP1(u)*FIR_LP1(v)

def FIR_1(u):
    a = 1.0 +2*(np.cos(u) + np.cos(2*u))
    return a/5.0

def FIR(u,v): return FIR_1(u)*FIR_1(v)

def FIR_unsharp(u,v,lam=1.5):
    return 1.0 + lam*(1-FIR(u,v))

def IIR(u,v):
    return np.abs(0.01/(0.9*(np.exp(-1j*u) + np.exp(-1j*v)) 
                        -0.81*np.exp(-1j*u)*np.exp(-1j*v)))
def grid(U,V,func):
    z = np.array(func(np.ravel(U),np.ravel(V)))
    Z = z.reshape(U.shape)  
    return Z

def plot_fig(X,Y,Z,fig_name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,cmap=cm.viridis)
    ax.set_xlim(-1*np.pi,1*np.pi)
    ax.set_ylim(-1*np.pi,1*np.pi)
    plt.xlabel('u')
    plt.ylabel('v')
    ax.set_zlabel('$|H(e^{ju},e^{jv})|$')
    ax.autoscale(enable=True,axis='z',tight=True)
    plt.savefig(fig_name+'.eps',format='eps')
    
    fig2, ax2 = plt.subplots(figsize=(6,6))
    cf = ax2.contourf(X,Y,Z)
    fig2.colorbar(cf, ax=ax2)
    ax2.set_xlim(-1*np.pi,1*np.pi)
    ax2.set_ylim(-1*np.pi,1*np.pi)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.savefig(fig_name+'_contour.eps',format='eps')
    
def main():
    pi = np.pi
    step = 0.1
    u = np.arange(-1*pi,pi,step)
    v = np.arange(-1*pi,pi,step)
    U,V = np.meshgrid(u,v)

    Z1 = grid(U,V,FIR_LP)
    Z2 = grid(U,V,FIR)
    Z3 = grid(U,V,FIR_unsharp)
    Z4 = grid(U,V,IIR)
    
    plot_fig(U,V,Z1,'fig1')
    plot_fig(U,V,Z2,'fig2')
    plot_fig(U,V,Z3,'fig3')
    plot_fig(U,V,Z4,'fig4')
    
    return

if __name__=="__main__": 
    main() 
    