#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:47:50 2019
@author: rahul
"""
# import libraries
import numpy as np , csv
from functions import *
import matplotlib.pyplot as plt
#%% Read Data
#(a)
read_path = '../data/'
label_filename = 'hw04_labels.csv'
sample_filename = 'hw04_sample_vectors.csv'

samples = []
with open(read_path+sample_filename) as csvfile:
    readcsv = csv.reader(csvfile,delimiter=',')
    for row in readcsv:
        vals = list(map(float,row))
        samples.append(vals)
samples_copy = np.array(samples)
samples = np.ones((np.shape(samples_copy)[0],np.shape(samples_copy)[1]+1))
samples[:,:-1]=samples_copy
samples = samples.T

labels = []
with open(read_path+label_filename) as csvfile:
    readcsv = csv.reader(csvfile,delimiter=',')
    for row in readcsv:
        vals = float(row[0])
        labels.append(vals)    
labels = np.array(labels)       

scaled_labels = 2*labels-1 #percep labels in range -1 to +1 
# Declare constants

rate = 0.7
#%%
#(a) Logistic regression based classification
log_star_store=[]
Ms= [10,50,100,200,1000,2000,3000]

for M in Ms:
    t_in= time.time()
    [log_theta_star,log_theta_store] = logistic(samples,labels,rate,M)
    log_star_store.append(np.linalg.norm(log_theta_star))
    freq = int(0.2*M)
#    plotdata(samples,labels,log_theta_store,freq,M)
#    plotdata_single(samples,labels,log_theta_star,M)
    log_gamma = min(scaled_labels*(log_theta_star.T@samples)/np.linalg.norm(log_theta_star))
#    print('gamma signed for logistic regression is '+str(log_gamma))
   
#plt.figure()
#plt.plot(Ms,log_star_store)
#plt.xlabel('Max Iters')
#plt.ylabel('||$\Theta^*$||')
#plt.title('Non-Convergence of Logistic Regression')
#plt.show()    

#%%(b) Perceptron Online mode
M = 100 # max number of iters
freq = int(0.2*M)
scaled_labels = 2*labels-1 #percep labels in range -1 to +1
##(i)
[ol_percp_theta_star,ol_percp_theta_store]= perceptron(samples,scaled_labels,rate,M,convplot=0)
plotdata(samples,labels,ol_percp_theta_store,freq,[M])
ol_gamma = min((scaled_labels*(ol_percp_theta_star.T@samples))/np.linalg.norm(ol_percp_theta_star))
#print('gamma signed for online perceptron is '+str(ol_gamma))

##(ii)
[bt_percp_theta_star,bt_percp_theta_store]= perceptron(samples,scaled_labels,rate,M,convplot=0,online=False)
#plotdata(samples,labels,bt_percp_theta_store,freq)
bt_gamma = min(scaled_labels*(bt_percp_theta_star.T@samples)/np.linalg.norm(bt_percp_theta_star))
"""
comparison of online and batch mode:
    convergence is faster for batch mode
    number of iterations is less for batch mode
    accuracy:both methods are doing a good job for the given data 

"""
#%%(c) SVM 
#(i) Hard Margin
svm_hard_theta_star = SVM_hard(samples,scaled_labels)
#plotdata_single(samples,labels,svm_hard_theta_star,M)
hard_gamma = min((scaled_labels*(np.array(svm_hard_theta_star).T@samples))/np.linalg.norm(svm_hard_theta_star))

##(ii) Soft Margin
C = 1
#for C in [10**(-2),10**(-1),1,10]:
svm_soft_theta_star = SVM_soft_L1(samples,scaled_labels,C)
#plotdata_single(samples,labels,svm_soft_theta_star,C)
soft_gamma = min((scaled_labels*(np.array(svm_soft_theta_star).T@samples))/np.linalg.norm(svm_soft_theta_star))

#plt.figure()
#plt.bar(['Logistic','Online Percep','Batch Percep','SVM Hard','SVM Soft'],\
#        [log_gamma,ol_gamma,bt_gamma,hard_gamma,soft_gamma])
#plt.ylabel('$\gamma_{unsigned}$')
#plt.title('Robustness')
#plt.show()