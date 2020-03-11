#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:07:13 2019

@author: rahul
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
#%% SVM 
# SVM Soft Margin
def SVM_soft_L1(samples,labels,C):
    """
    Input:
    sample: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
    labels: np array [y1,y2,....,yn]
    """
    t_in = time.time()
    dim,Nsamples = samples.shape
    theta = cvx.Variable(dim) # declaring dimension of variable    
    lam = 1/C
    Jump_term = cvx.max_elemwise(0,1-cvx.mul_elemwise(labels,(theta.T@samples).T))
    Jump_term = cvx.sum_entries(Jump_term)
    obj_expr = Jump_term +(lam/2)*(cvx.sum_squares(theta[:-1]))
    obj = cvx.Minimize(obj_expr)
    prob = cvx.Problem(obj)
    prob.solve(solver = cvx.ECOS)
    theta_star = theta.value
    theta_star = theta_star.tolist()
    theta_star= [x[0] for x in theta_star]
    print('SVM soft time '+str(time.time() - t_in)+'\n')
    return(theta_star)

# SVM Hard Margin
def SVM_hard(samples,labels):
    """
    Input:
    sample: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
    labels: np array [y1,y2,....,yn]
    """
    t_in = time.time()
    dim,Nsamples = samples.shape
    theta = cvx.Variable(dim) # declaring dimension of variable    
    obj = cvx.Minimize(cvx.sum_squares(theta[:-1]))
    temp = cvx.mul_elemwise(labels,(theta.T@samples).T)
    const = [ temp >= 1 ]
    prob = cvx.Problem(obj,const)
    prob.solve(solver = cvx.ECOS)
    theta_star = theta.value
    #format change
    theta_star = theta_star.tolist()
    theta_star= [x[0] for x in theta_star]
    print('SVM hard time '+str(time.time() - t_in)+'\n')
    return(theta_star)


#%% Perceptron Method
def  percept_batch_tangent(theta,samples,true_labels):
    """
    Input:
        theta: decision boundary
        sample: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
        labels: np array [y1,y2,....,yn]
    Out: J: Jacobian    
    """    
    # predict samples labels using theta
    gx = theta.T@samples
    ygx = np.multiply(true_labels,gx)
    miss_idx = np.where(ygx<0)[0]
    N_miss = miss_idx.size
    all_miss_labels = true_labels[miss_idx]
    all_miss_x = samples[:,miss_idx]    
    J = np.sum(all_miss_labels*all_miss_x,axis=1)
    return(J,N_miss)

def  percept_online_tangent(theta,samples,true_labels):
    """
    Input:
        theta: decision boundary
        sample: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
        labels: np array [y1,y2,....,yn]
    Out: J: Jacobian    
    """    
    # predict samples labels using theta
    gx = theta.T@samples
    ygx = np.multiply(true_labels,gx)
    miss_idx = np.where(ygx<0)[0]
    N_miss = miss_idx.size
    if N_miss>0:
        picked_idx = miss_idx[np.random.permutation(N_miss)]
        picked_idx = picked_idx[0]
        J = true_labels[picked_idx]*samples[:,picked_idx]
    else:
        J = np.zeros_like(samples[:,0]) #just to pass some value
    return(J,N_miss)

def perceptron(samples,labels,rate,Max_iter,convplot,online = True):
    """
    Input:
        samples: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
        labels: np array [y1,y2,....,yn]
        rate: learning rate or the step length
        Max_iter: for the gradient descent
        online: if online mode then True(default) else False
    """
    t_in = time.time()
    xdim,Nsamples= samples.shape
    theta_k = np.ones(xdim) # initial guess
    theta_store = []
    N_store = []
    theta_store.append(np.copy(theta_k))
    if online:        
        for k in range(Max_iter):
            grad_k,N_miss = percept_online_tangent(theta_k,samples,labels)
            if N_miss == 0:
                break
            theta_k += rate*grad_k
            theta_store.append(np.copy(theta_k))   
            N_store.append(N_miss)
        print('online percep time '+str(time.time() - t_in)+'\n')
    else:
        for k in range(Max_iter):
            grad_k,N_miss = percept_batch_tangent(theta_k,samples,labels) 
            if N_miss == 0:
                break
            theta_k += rate*grad_k
            theta_store.append(np.copy(theta_k))
            N_store.append(N_miss)
        print('batch percep time '+str(time.time() - t_in)+'\n')
    if convplot==1:
        plt.plot(np.arange(len(N_store)),N_store,'*--')
        plt.xlabel('Iters')
        plt.ylabel('Number of missclassified samples')
        plt.show()
    
    return(theta_k,theta_store)

#%% function for plotting data
def plotdata_single(samples,labels,theta_star,*argv):
    """
    Input:
        sample: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
        labels: np array [y1,y2,....,yn]
        theta_star: decision boundary params
        N: frequency of plots for decision boundary
    """
    
    plt.figure()
    #plot the training set
    for i in range(np.shape(samples)[1]):
        if labels[i] == 1:
            plt.scatter(samples[0,i],samples[1,i],c='red',marker='*')
        else:
            plt.scatter(samples[0,i],samples[1,i],c='blue',marker='^')
    
    # plot the decision boundary 
    x_min = min(samples[0,:])
    x_max = max(samples[0,:])
    y_min = min(samples[1,:])
    y_max = max(samples[1,:])
    x_line = np.linspace(x_min,x_max,10)
    y_line = -(theta_star[0]*x_line +theta_star[2])/theta_star[1]
    plt.plot(x_line,y_line,'k-')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    if argv!=None:
        plt.title('C='+str(argv[0]))
    plt.show()
    return()
    
def plotdata(samples,labels,theta_store,N,*argv):
    """
    Input:
        sample: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
        labels: np array [y1,y2,....,yn]
        theta_store: decision boundary params
        N: frequency of plots for decision boundary
    """
    
    plt.figure()
    #plot the training set
    for i in range(np.shape(samples)[1]):
        if labels[i] == 1:
            plt.scatter(samples[0,i],samples[1,i],c='red',marker='*')
        else:
            plt.scatter(samples[0,i],samples[1,i],c='blue',marker='^')
    
    # plot the decision boundary 
    x_min = min(samples[0,:])
    x_max = max(samples[0,:])
    y_min = min(samples[1,:])
    y_max = max(samples[1,:])
    x_line = np.linspace(x_min,x_max,10)
    count=0
    while count<len(theta_store):
        theta_k = theta_store[count]
        y_line = -(theta_k[0]*x_line +theta_k[2])/theta_k[1]
        plt.plot(x_line,y_line,'k--',alpha=0.5)
        count+=N
    theta_k = theta_store[-1]
    y_line = -(theta_k[0]*x_line +theta_k[2])/theta_k[1]
    plt.plot(x_line,y_line,'k-')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    if argv!=None:
        plt.title('M='+str(argv[0]))
    plt.show()
    return()

#%% Logistic regression function
# tangent for logistic function
def logistic_tangent(theta,samples,labels):
    """
    Input:
        theta: decision boundary
        sample: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
        labels: np array [y1,y2,....,yn]
    Out: J: Jacobian
    """
    thetaTx  = theta.T@samples
    h_theta_x = 1/(1+np.exp(-thetaTx))
    temp = (h_theta_x - labels)*samples    
    J  = np.sum(temp,axis=1)
    return(J)
# main logistic function
def logistic(samples,labels,rate,Max_iter):
    """
    Input:
        samples: np array in the format [[x1],[x2],....,[xn]] xi as col vectors
        labels: np array [y1,y2,....,yn]
        rate: learninig rate or the step length
        Max_iter: for the gradient descent
    """
    t_in = time.time()
    xdim,Nsamples= samples.shape
    theta_k = np.zeros(xdim) # initial guess
    theta_store = []
    theta_store.append(np.copy(theta_k))
    for k in range(Max_iter):
        theta_k -= rate*logistic_tangent(theta_k,samples,labels)
        theta_store.append(np.copy(theta_k))
    print('SVM logistic time '+str(time.time() - t_in)+'\n')
    return(theta_k,theta_store)
