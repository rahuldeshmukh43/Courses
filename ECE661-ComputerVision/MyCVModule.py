"""
MyCVModule: all the functions created in ECE661: Computer Vision
Taught By: Prof. Avinash Kak

Created on Sun Sep  9 21:05:39 2018

Author: Rahul Deshmukh
PUID: 0030004932
email: deshmuk5@purdue.edu
"""
# all the functions created in homeworks will be called from this central library 
# functions added in chronological order with latest hw first
#----make sure to add documentation in each function and also link to the home works---#
#%%========================== Import Libraries ===============================#
import numpy as np
import cv2
from scipy.optimize import least_squares
import os
#import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import time
#%%
#========================= V Imp functions ===================================#
# most called functions 
#Take note of the format of input/output while using them 

#function to convert the Physical coordinates into HC coordinates
def convert2HC(pts):
    """
    Input: pts: list of list/np.array of coordinate [[x,y],...]
    Output: pts_hc: HC coordinates of pts [[x,y,1],....] np.array
    """
    pts_hc = np.ones((np.shape(pts)[0],np.shape(pts)[1]+1))
    pts_hc[:,:-1]=pts
    return(pts_hc)
#function for converting the HC coordinates to Physical coordinates
def convert2phy(pts_hc):
    """
    Input: pts_hc: HC coordinates format: np.array !!! format: [[x,y,1]^T,...] !!!
    Output: pts: array of pts format: [[x,y],..]
    """
    pts = np.linalg.inv(np.diag(pts_hc[-1,:]))@pts_hc.T
    pts= pts[:,:-1] 
    return(pts)
#=============================================================================#


#%%================================HW10=======================================#
#%%------------------------------ADABoost-------------------------------------#
#%%----------------------------- Testing -------------------------------------#
# function for running cascades of adaboost for testing
def Cascade_Testing(all_features,all_labels,Strong_classifiers):
    """
    Input: all_features:  matrix of all test features with single row as a feature
                          vector for one image
           known_labels: lables of all test images, a array of 1,0
           Strong_classifiers: Learned Classifier in the format
           a list of list with one element in the list as list of 5 elements:-
           1)h_t: index of the classifier to be used
           2)theta_t: threhold value for the classifer
           3)p_t: polarity of the classifier
           4)alpha_t: list of trust factors
           5)Cx_threshold: threshold for the strong classifiers
    Output: FP_measure,FN_measure: list of stage-wise values of FP and FN
    for the test data    
    """
    # find number of stages, N_pos and N_neg for test images
    Num_stages= len(Strong_classifiers)
    N_pos = np.sum(all_labels)
    N_neg = len(all_labels)-N_pos
    original_idx = np.arange(N_pos+N_neg)
    
    # initialize counts for FP and FN
    FP_measure = []
    FN_measure = []
    FP_count = 0
    TN_count = 0
    
    known_label=all_labels
    this_stage_idx = original_idx
    print('FP\tFN')    
    for s in range(Num_stages):
        # get the Strong classifier parameters for this stage
        h_t,theta_t,p_t,alpha_t,Cx_threshold = Strong_classifiers[s]
        # find number of weak classifiers
        T = len(alpha_t)
        # predict the class of the testing data for this stage
        # make features and known_labels
        known_label = all_labels[this_stage_idx]
        features = all_features[this_stage_idx,:]
        sN_pos = np.sum(known_label)
        sN_neg = len(known_label)-sN_pos
        predicted_label = AdaBoost_Testing(features,known_label,T,sN_pos,sN_neg,
                          h_t,theta_t,p_t,alpha_t)        
        # calculate FP and FN 
        num_correctly_classified_positives = np.sum(np.multiply(known_label,predicted_label))
        num_misclassified_positives = sN_pos - num_correctly_classified_positives
        FP_count+=num_misclassified_positives
        
        num_correctly_classified_negatives = np.sum(np.multiply(1-predicted_label,1-known_label))
        TN_count += num_correctly_classified_negatives
        
        # store FP and TN for this stage
        sFP = (N_neg-TN_count)/N_neg
        FP_measure.append(sFP)
        
        sFN = FP_count/N_pos
        FN_measure.append(sFN)
        print('-------------------- Stage '+str(s+1)+'-----------------------')
        print(sFP,sFN)
        # update data: send only those positive images which were correctly classified
        # and those negative images which where wrongly classified for the next stage
        this_stage_idx = np.where(predicted_label>0)[0]    
    return(FP_measure,FN_measure)    
# function for predicting class for one stage of adaboost testing
def AdaBoost_Testing(features,known_label,T,N_pos,N_neg,
                     h_t,theta_t,p_t,alpha_t):
    """
    Input: features: matrix of test features with single row as a feature
                          vector for one image
           known_label: labels of test samples
           T: number of weak classifiers in this stage
           h_t: index of the classifier to be used
           theta_t: threhold value for the classifer
           p_t: polarity of the classifier
           alpha_t: list of trust factors
    Output: predicted_label: array of 1,0 1:predicted as pos 0:predicted as neg
    """
    # initialize
    predicted_label = np.zeros(N_pos+N_neg) 
    
    h_q = np.zeros((N_pos+N_neg,T)) # array for storing value of weak classifier
    # for the query data
    
    # loop over weak classifier
    for t in range(T):
        fi = features[:,h_t[t]]
        for i in range(N_pos+N_neg):
            if p_t[t]*fi[i]<p_t[t]*theta_t[t]:
                h_q[i,t]= 1

    # find sum h_t*a_t for all query data
    Cx_q = h_q@np.array(alpha_t)    
    Cx_threshold= 0.5*np.sum(alpha_t)
    
    # find predicted labels using the threshold
    predicted_label = np.where(Cx_q>=Cx_threshold,1,0)
    
    return(predicted_label)
#%%-----------------------------Training--------------------------------------#
#function for performing one stage of cascade
def do_cascades(all_features,all_labels,
               Num_classifier,Smax,
               Tmax,TP_crit,FP_crit):
    """
    Input:Smax: maximum number of cascades allowed
          all_features: feature vector matrix with feature vec for one image 
          all_labels: list of all labels
          stacked as a row vector          
          ------ criterias for Adaboost-------
          Tmax: max num of adaboost iters
          TP_crit,FP_crit: true positive and false positive criterias (0-1)
          ------------------------------------
    Output:Strong_classifiers: list of strong classifiers
           a list of list with one element in the list as list of 5 elements:-
           1)h_t: index of the classifier to be used
           2)theta_t: threhold value for the classifer
           3)p_t: polarity of the classifier
           4)alpha_t: list of trust factors
           5)Cx_threshold: threshold for the strong classifiers
           TP_s: list of TP for each stage
           FP_s: list of FP for each stage
    """
    # initialize storing structures
    Strong_classifiers = []
    TP_s=[]
    FP_s=[]
    
    N_pos = np.sum(all_labels)
    N_neg = len(all_labels)-N_pos
    
    original_idx = np.arange(N_pos+N_neg)
    this_stage_idx = original_idx
    
    for s in range(Smax):
        # make feature matrix, label, N_pos,N_neg for adbaboost call
        known_label = all_labels[this_stage_idx]
        features = all_features[this_stage_idx,:]
        N_pos = np.sum(known_label)
        N_neg = len(known_label)-N_pos
        # check if all negative samples are exhausted
        if N_neg==0:
            print('Converged at Stage '+str(s))
            break;
        print('----------- Stage:  '+str(s+1)+'-------------')            
        print('Total images: '+str(N_pos+N_neg)+' Pos:'+str(N_pos)+' Neg:'+str(N_neg))
        #call adaboost
        h_t,theta_t,p_t,alpha_t,Cx_threshold,samples_classified_pos_idx,TP_final,FP_final=\
        Do_AdaBoost(features,Tmax,TP_crit,FP_crit,N_pos,N_neg,Num_classifier,known_label)
        #store this strong classifier
        Strong_classifiers.append([h_t,theta_t,p_t,alpha_t,Cx_threshold])
        TP_s.append(TP_final)
        FP_s.append(FP_final)
        # update indices for next stage of adaboost
        this_stage_idx = this_stage_idx[samples_classified_pos_idx]        
    return(Strong_classifiers,TP_s,FP_s)

# function for carrying out AdaBoost Learning from Training data set
def Do_AdaBoost(features,Tmax,TP_crit,FP_crit,
                   N_pos,N_neg,Num_classifier,known_label):
    """
    Input: features: feature vectors stacked as rows and one row for on image
           TP_crit,FP_crit: true positive and false positive criterias(0-1)
           Tmax: max time iterations to be carried by AdaBoost
           N_pos, N_neg: Number of positive and negative samples
           Num_classifier: number of classfiers in the library
           known_label: true labels of the images
    Output: Strong classifer data
            h_t: index of the classifier to be used
            theta_t: threhold value for the classifer
            p_t: polarity of the classifier
            alpha_t: list of trust factors
            samples_classified_pos_idx : indices of samples
                which will go for the next stage
            TP_final: last value of the measured TP
            FP_final: last value of the measured FP
    """
    
    #initialize storing stuctures for storing the classifiers ht
    h_t=[] 
    theta_t=[] 
    p_t=[]
    h_t_predicted_labels=[]
    alpha_t =[] #alpha: trust factors
    f_t = [] #list for storing the feature values 
    
    # initialize weights for samples
    wt= np.hstack((1/(2*N_pos)*np.ones(N_pos),1/(2*N_neg)*np.ones(N_neg)))
    
    # tracking varibles
    TP_measure=[]
    FP_measure=[]
    
    for t in range(Tmax):
        # normalize weights
        wt= wt/np.sum(wt)
        # find best weak classifier
        h_star,theta_star,p_star,error_star,predicted_label,f_star=\
        find_best_weak_classifier(Num_classifier,known_label,
                                  wt,features,N_pos,N_neg)
        # compute alpha
        beta = error_star/(1-error_star)
        a_t = np.log(1/beta)
        #store t'th best weak classifer
        alpha_t.append(a_t)
        h_t.append(h_star); 
        theta_t.append(theta_star);
        p_t.append(p_star);
        h_t_predicted_labels.append(predicted_label)
        f_t.append(f_star)
        #update wts
        wt = np.multiply(wt,np.power(beta,1-np.logical_xor(predicted_label,
                                                    known_label).astype(int)))
        # build current Strong classifier and find its acuracy
        Cx = np.array(h_t_predicted_labels).T@np.array(alpha_t) # this is just the value of the classifier
        # we need to compare it to a threshold value
        # instead of choosing the threshol as (1/2)*np.sum(a_t) 
        # we choose the threshold such that all the positive images get classified as 1
        # such a threshold will be the minimum of Cx slice corresponding to positive images
        Cx_threshold = np.min(Cx[:N_pos])
        # find predicted labels for the strong classifer
        Cx_predicted_label = np.where(Cx>=Cx_threshold,1,0)
        # find accuracy of current strong classifier
        # find true positive accuracy
        TP_count = np.sum(Cx_predicted_label[:N_pos])/N_pos
        TP_measure.append(TP_count)
        # find false positive accuracy
        FP_count = np.sum(Cx_predicted_label[N_pos:])/N_neg
        FP_measure.append(FP_count)
        print(TP_count,FP_count)
        # check for convergence
        if TP_count>=TP_crit and FP_count<=FP_crit:
            print('AdaBoost Early Termination: reached convergence in '+
                  str(t+1)+' iterations')
            break
    if t==Tmax-1: 
        print('!! Adaboost did not meet TP and FP criterias !!')   
    print('TP:'+str(TP_measure[-1]))
    print('FP:'+str(FP_measure[-1]))
    TP_final = TP_measure[-1]
    FP_final = FP_measure[-1]
    # find number the negative example wrongly predicted as positive and all
    # positive images
    samples_classified_pos_idx = np.where(Cx_predicted_label>0)[0]
    return(h_t,theta_t,p_t,alpha_t,Cx_threshold,samples_classified_pos_idx,
           TP_final,FP_final)
    
# function for finding the best weak classifier
def find_best_weak_classifier(Num_classifier,label,
                              wt,features,N_pos,N_neg):
    """
    Input: Num_classifier: number of classifiers
           label: true labels for all training images np.array
           wt: t'th iteration weights for s'th iter samples np.array                     
    Output: h_star: index number of the best classifier
            theta_star: threshold for the best classifier
            p_star: polarity of the best classifier
            f_star: array of feature vlaues for the best classifier
            error_star: weighted error for the best classifier
            predicted_label: binary list, classification of the images using the 
            best classifier
    """    
    # calculate T+ and T-
    T_plus = np.sum(wt[:N_pos])
    T_plus = T_plus*np.ones(N_pos+N_neg)
    T_minus= np.sum(wt[N_pos:])
    T_minus = T_minus*np.ones(N_pos+N_neg)
    # loop over all calssifiers to find the best classifier with min error
    error_star = np.inf # initialization    
    for i in range(Num_classifier):
        fi = features[:,i]
        # sort the feature vector in increasing order
        sorted_idx = np.argsort(fi)
        sorted_label = label[sorted_idx]
        sorted_wt = wt[sorted_idx]     
        # Calculate S+ and S-: for all values of threshold
        S_plus= np.multiply(sorted_wt,sorted_label)
        S_plus = np.cumsum(S_plus)
        S_minus = np.cumsum(sorted_wt)-S_plus
        #find the two types of error
        error1 = S_plus+(T_minus-S_minus)
        error2 = S_minus+(T_plus-S_plus)
        # find min error 
        element_wise_min = np.minimum(error1,error2)
        min_id = np.argmin(element_wise_min)
        min_error = element_wise_min[min_id]      
        # update the best ever if current_error is less than error_star
        if min_error<error_star:
            error_star=min_error
            # find polarity and prediction by the classifier
            predicted_label = np.zeros(N_pos+N_neg)
            if error1[min_id]<=error2[min_id]:
                p=-1
                predicted_label[min_id:] = 1
            else:
                p=1
                predicted_label[:min_id] = 1
            predicted_label[sorted_idx]=np.copy(predicted_label) # binary classification 1 or 0
            # store best polarity, classifier index, threshold value
            p_star= p
            h_star=i
            theta_star= fi[sorted_idx[min_id]]
            f_star = fi 
    return(h_star,theta_star,p_star,error_star,predicted_label,f_star)

#%% helper function for adaboost to read images 
def Read_img4AdaBoost(train_path):
    """
    Input: train_path: path for training dataset
    Output: S: is a list of integral images
            label: list of 1 or 0 1: positive 0:negative
    """
    pos_path = train_path+'positive/'
    neg_path = train_path +'negative/'
    S=[];
    # read postive images
    pos_img_list = os.listdir(pos_path)
    pos_img_list.sort(key = lambda x: int(x.split('.')[0]))
    N_pos =len(pos_img_list)
    for i in range(N_pos):
        iI = cv2.imread(pos_path+pos_img_list[i],0) # gray image
        iI = iI.astype(float)
        iS = Integrate_Img(iI)
        S.append(iS)
    # read negative images
    neg_img_list = os.listdir(neg_path)
    neg_img_list.sort(key = lambda x: int(x.split('.')[0]))
    N_neg=len(neg_img_list)
    for i in range(N_neg):
        iI = cv2.imread(neg_path+neg_img_list[i],0) # gray image
        iI = iI.astype(float)
        iS = Integrate_Img(iI)
        S.append(iS)
    label = np.hstack((np.ones(N_pos,dtype=int),np.zeros(N_neg,dtype=int)))
    return(S,label,N_pos,N_neg)
#%%function for finding the Haar like operators
def get_feature(S):
    """
    Input:S: integral Image with left and top padding
    Output: features
    """
    m,n=np.shape(S);
    m=m-1; n=n-1;
    feature=[]
#---------------------All Classifiers-----------------------------------------#
# for 20x40 image we will have a total of 312260 classifiers which is huge!!
#    # type 1: horizontal operator [0|1]
#    for h in range(1,m+1):
#        for w in range(1,n//2+1):
#            #operator is of size h,2*w 
#            # shift x & y position of upper left corner of operator
#            for x in range(n-2*w+1): 
#                for y in range(m-h+1):
#                    feature.append(type1_haar(S,x,y,h,w))

#    # type 2: vertical operator [0|1]^T
#    for h in range(1,m//2+1):
#        for w in range(1,n+1):
#            #operator is of size 2*h,w 
#            # shift x & y position of upper left corner of operator
#            for x in range(n-w+1): 
#                for y in range(m-2*h+1):
#                    feature.append(type2_haar(S,x,y,h,w))

#    # type 3: x derivative type operator [0|1|0]
#    for h in range(1,m+1):
#        for w in range(1,n//3+1):
#            #operator is of size h,3*w 
#            # shift x & y position of upper left corner of operator
#            for x in range(n-3*w+1): 
#                for y in range(m-h+1):
#                    feature.append(type3X_haar(S,x,y,h,w))

#    # type 3_2: Y derivative type operator [0|1|0]
#    for h in range(1,m//3+1):
#        for w in range(1,n+1):
#            #operator is of size h,3*w 
#            # shift x & y position of upper left corner of operator
#            for x in range(n-w+1): 
#                for y in range(m-3*h+1):
#                    feature.append(type3Y_haar(S,x,y,h,w))

#    # type 4: diagonal operator [[0,1][1,0]]
#    for h in range(1,m//2+1):
#        for w in range(1,n//2+1):
#            #operator is of size 2*h,2*w 
#            # shift x & y position of upper left corner of operator
#            for x in range(n-2*w+1): 
#                for y in range(m-2*h+1):
#                    feature.append(type4_haar(S,x,y,h,w))
#-----------------------------------------------------------------------------#
# to reduce computational cost i am picking a subset of the all classifiers 
# such that i still have a mix of all types
    k=1
    # type 1: horizontal operator [0|1]
    for h in [k]:
        for w in range(1,n//2+1):
            #operator is of size h,2*w 
            # shift x & y position of upper left corner of operator
            for x in range(n-2*w+1): 
                for y in range(m-h+1):
                    feature.append(type1_haar(S,x,y,h,w))
    # type 2: vertical operator [0|1]^T
    for h in range(1,m//2+1):
        for w in [k]:
            #operator is of size 2*h,w 
            # shift x & y position of upper left corner of operator
            for x in range(n-w+1): 
                for y in range(m-2*h+1):
                    feature.append(type2_haar(S,x,y,h,w))
    # type 3: x derivative type operator [0|1|0]
    for h in [k]:
        for w in range(1,n//3+1):
            #operator is of size h,3*w 
            # shift x & y position of upper left corner of operator
            for x in range(n-3*w+1): 
                for y in range(m-h+1):
                    feature.append(type3X_haar(S,x,y,h,w))
    # type 3_2: Y derivative type operator [0|1|0]^T
    for h in range(1,m//3+1):
        for w in [k]:
            #operator is of size h,3*w 
            # shift x & y position of upper left corner of operator
            for x in range(n-w+1): 
                for y in range(m-3*h+1):
                    feature.append(type3Y_haar(S,x,y,h,w))
    # type 4: diagonal operator [[0,1][1,0]]
    for h in [k]:
        for w in [k]:
            #operator is of size 2*h,2*w 
            # shift x & y position of upper left corner of operator
            for x in range(n-2*w+1): 
                for y in range(m-2*h+1):
                    feature.append(type4_haar(S,x,y,h,w))
    return(feature)
#function for finding feature value of type4 haar
def type4_haar(S,x,y,h,w):
    """
    Input:  S: integral image with left and top padding
            x,y: coordinates of A 
            h,w: half-height and half-width of the operator
    Output: f:feature value
     A------B-----C
     |  0   |  1  |
     D------E-----F
     |  1   |  0  |
     G------H-----I
    """
    A = S[y,x]
    B = S[y,x+w]
    C = S[y,x+2*w]
    D = S[y+h,x]
    E = S[y+h,x+w]
    F = S[y+h,x+2*w]
    G = S[y+2*h,x]
    H = S[y+2*h,x+w]
    I = S[y+2*h,x+2*w]
    f = (F-C-E+B)+(H-E-G+D)-(I-F-H+E)-(E-B-D+A)
    return(f)
#function for finding feature value of type3 haar
def type3X_haar(S,x,y,h,w):
    """
    X double derivative
    Input:  S: integral image with left and top padding
            x,y: coordinates of A 
            h,w:height and 1/3rd-width of the operator
    Output: f:feature value
     A------B-----C-----D
     |  0   |  1  |  0  |
     E------F-----G-----H
    """
    A = S[y,x]
    B = S[y,x+w]
    C = S[y,x+2*w]
    D = S[y,x+3*w]
    E = S[y+h,x]
    F = S[y+h,x+w]
    G = S[y+h,x+2*w]
    H = S[y+h,x+3*w]
    f = (G-C-F+B)-(F-B-E+A)-(H-D-G+C)
    return(f)
#function for finding feature value of type3_2 haar
def type3Y_haar(S,x,y,h,w):
    """
    Y double derivative
    Input:  S: integral image with left and top padding
            x,y: coordinates of A 
            h,w: 1/3rd-height and width of the operator
    Output: f:feature value
     A------E
     |  0   |
     B------F
     |  1   |
     C------G
     |  0   |
     D------H
    """
    A = S[y,x]
    B = S[y+h,x]
    C = S[y+2*h,x]
    D = S[y+3*h,x]
    E = S[y,x+w]
    F = S[y+h,x+w]
    G = S[y+2*h,x+w]
    H = S[y+3*h,x+w]
    f = (G-C-F+B)-(F-B-E+A)-(H-D-G+C)
    return(f)
#function for finding feature value of type2 haar
def type2_haar(S,x,y,h,w):
    """
    Input:  S: integral image with left and top padding
            x,y: coordinates of A 
            h,w: half-height and width of the operator
    Output: f:feature value
     A------D
     |  0   |
     B------E
     |  1   |
     C------F
    """
    A = S[y,x]
    B = S[y+h,x]
    C = S[y+2*h,x]
    D = S[y,x+w]
    E = S[y+h,x+w]
    F = S[y+2*h,x+w]
    f = (F-E-C+B)-(E-B-D+A)
    return(f)
#function for finding feature value of type1 haar
def type1_haar(S,x,y,h,w):
    """
    Input:  S: integral image with left and top padding
            x,y: coordinates of A 
            h,w:height and half-width of the operator
    Output: f:feature value
     A------B-----C
     |  0   |  1  |
     D------E-----F
    """
    A = S[y,x]
    B = S[y,x+w]
    C = S[y,x+2*w]
    D = S[y+h,x]
    E = S[y+h,x+w]
    F = S[y+h,x+2*w]
    f = (F-C-E+B)-(E-B-D+A)
    return(f)
#%% function for finding intergral representation of image
def Integrate_Img(I):
    """
    using summed table approach
    Input: I: grayscale image mxn 
    Output: S: integral image, m+1xn+1 array with left and top padding
    """
    S= np.zeros((np.shape(I)[0]+1,np.shape(I)[1]+1)) # padded with zeros
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            S[y+1,x+1]=I[y,x]+S[y+1,x]+S[y,x+1]-S[y,x]
    return(S)
#--------------------------------ADABoost-------------------------------------#
#%%--------------------------------PCA & LDA----------------------------------#
# function for finding the eigen vectors for the reduced dimensional space
def eig_rankdef(X,ascending=1):
    """
    Input: X a matrix of
           ascending: option 1 or 0(default=1) if 0 then output of
           indices is in descending order
    Output: w:all eigen vectors
            sorted_lam: list of eigen values indices in descending order
    using the computational trick by first finding the eigne vectors u of
    X.TX and then converting to w
    """
    # find eigen vectors of XTX
    lam,u = np.linalg.eig(X.T@X)
    sorted_lam = np.argsort(lam)
    # reverse the list of sorted _lam to make it in descending order
    if ascending == 0:
        sorted_lam= sorted_lam[::-1]
    # convert u to w 
    w = X@u
    # normalize w
    w = w/np.linalg.norm(w,axis=0)    
    return(w,lam,sorted_lam)
#----------------------------------PCA & LDA----------------------------------#
#==================================HW10=======================================#
#==================================HW9========================================#
#%%
#function for writing a ASCII PLY file for the 3D points for visualization
def write_ply_file(pts,filename):
    """
    Input: pts: physical cooordinates of 3D pt np.array[[X,Y,Z],..]
           filename: full filename wo file extension 
    Ouput: asc file will be printed with the coordinates
    """
    file = open(filename+'.asc','w')
    # write in asc format
    file.write('ply\n')
    file.write('format ascii 1.0\n')
    file.write('comment Rahul Deshmukh generated\n')
    file.write('element vertex '+str(np.shape(pts)[0])+'\n')
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('element face 0\n')
    file.write('property list uchar int vertex_indices\n')
    file.write('end_header\n')    
    for i in range(np.shape(pts)[0]):
        file.write(str(pts[i,0])+'\t'+str(pts[i,1])+'\t'+str(pts[i,2])+'\n')
    return()
#%% ---------------------Interest Pt Pairs on Rectfied Images-------------%
# pts detected using sift and mathcing is done using descriptor vector
# scanning done on rectified image row by row with a user def tolerance 

#function for finding pairs
def find_pairs_rectifed_img(img1,pts1,des1,img2,pts2,des2,H1,H2,scan_tol):
    """
    Input: img1,img2: image matrices mxnx3
            pts1,pts2: list of list of coordinates of N interest pts [[x,y],[x,y],..]
            des1,des2: descriptor vectors stacked as row vec in a np.array Nx128
            H1,H2: Homography for rectification of image1 and image 2
            scan_tol: row tolerance for finding pairs
    Output: pt_pairs: list of list of pt pairs [ [[x,y] , [x,y]] ,...]
    """
    # find m,n,corners of transformed second image
    tr_m2,tr_n2,cor2 = get_size_after_homo(img2,H2)
    # convert pts to HC
    pts1_hc = convert2HC(pts1) #[[x,y,1],..]
    pts2_hc = convert2HC(pts2)
    # find transformed sift pts using H1 and H2: both HC and phy
    tr_pts1_hc = H1@pts1_hc.T # hc vec as cols
    tr_pts1 = convert2phy(tr_pts1_hc)
    tr_pts2_hc = H2@pts2_hc.T
    tr_pts2 = convert2phy(tr_pts2_hc)
    # loop over pts
    pt_pairs = []
    record= np.zeros((np.shape(tr_pts2)[0],1)) # stores 1 or 0 if 1 then used
    for i in range(len(pts1)):
        ipt = tr_pts1[i,:]
        ides1 = des1[i,:]
        # find window of search for row scans in the second image
        irow1 = ipt[1]
        iwin = give_window(irow1,scan_tol,(tr_m2,tr_n2))
        # find pts in the second transfromed image which lie in  this window
        ind = pts_in_win(tr_pts2,iwin,record)
        if len(ind)>0:
            # make temp descriptor-vector packet for these indices
            temp_des = []
            for j in range(len(ind)): temp_des.append(des2[ind[j],:])
            # find euclidean distance of ides1 with temp_des
            d = np.array(temp_des)-ides1
            d = np.linalg.norm(d,axis=1)
            imin = np.argmin(d)
            i_pair = ind[imin]
            pt_pairs.append([pts1[i],pts2[i_pair]])
            #update record
            record[i_pair]=1
    return(pt_pairs)
#function for finding pts in a row window
def pts_in_win(tr_pts2,win,record):
    """
    Input: tr_pts2: Nx2 array of physical coordinates of interest pts in transformed img2
            format: [[x,y],...]
            win: size of row scan window [lb,ub]: indices of rows
            record: vector os size N, has 1 or 0 
            1: element is used dont send index
    Output: ind: indices of pts in this window
    """
    lb_ind = list(np.where(tr_pts2[:,1]>win[0])[0])
    ub_ind = list(np.where(tr_pts2[:,1]<win[1])[0])
    usable = list(np.where(record<1)[0])
    ind = list(set(lb_ind)&set(ub_ind)&set(usable))
    return(ind)
# function for finding window using scan_tol
def give_window(irow1,scan_tol,img_size):
    """
    Input:  irow1: row number of current sift pt on left image
            scan_tol: user defined scan tol on row
            img_size: tuple m,n: row col of transformed image: using corners
    Output: window: [lb,ub] lower bound and upper bounds for row index
    """
    win = int(irow1)*np.ones(2,dtype=int)
    win = win +scan_tol*np.array([-1,1])
    # check if win is not outside img_size
    if win[0]<0:
        win[0]=0
    if win[1]>=img_size[1]:
        win[1]=img_size[1]-1 # row number index
    return(win)
#%% -------------- Image Rectification-------------------------%
# pg 302 of text section 11.12
#function for plotting image with interest points of two scenes and lines joining matching points
def plot_rectify_img(img1,img2,H1,H2):
    """
    Input: img1,img2: original images of same sizes
            H1,H2: homography for the two images respectively st H1@img1 = rectified img
    OutPut: img3: image matrix with both scenes placed side by side horizontally
    """
    m=np.shape(img1)[0];n=np.shape(img1)[1];#n2=np.shape(img2)[1]
    # find transformed corners of both the images to determine the size of the 
    # cnavas
    tr_m1,tr_n1,cor1 = get_size_after_homo(img1,H1)
    tr_m2,tr_n2,cor2 = get_size_after_homo(img2,H2)
    # for canvas
    canvas_y = np.max([tr_m1,tr_m2])
    ind = np.argmax([tr_m1,tr_m2]) # which image had bigger size in y
    canvas = np.zeros((int(canvas_y),int(tr_n1+tr_n2),3))
    
    # get transformed images
    distype = 'BiLinear'
    rec_img1 = mapFitToCanvas(img1,cor1,[],np.linalg.inv(H1),distype)
    rec_img2 = mapFitToCanvas(img2,cor2,[],np.linalg.inv(H2),distype)
    # now assign image pixels to canvas
    if ind == 0:
        # left image is bigger
        # assign left part as rec_img1
        canvas[:,:int(tr_n1),:]=rec_img1
        # for right image we need to displace its y such that it comes into the midddle 
        displacement = canvas_y - (tr_m2); displacement = displacement/2
        canvas[int(displacement):int(displacement)+rec_img2.shape[0],int(tr_n1):,:]=rec_img2
    else:
        # right image is bigger
        # assign right part as rec_img1
        canvas[:,int(tr_n1):,:]=rec_img2
        # for left image we need to displace its y such that it comes into the midddle 
        displacement = canvas_y - (tr_m1); displacement = displacement/2
        canvas[int(displacement):int(displacement)+rec_img1.shape[0],:int(tr_n1),:]=rec_img1
    return(canvas)
# function for finding transfromed image size
def get_size_after_homo(img,H):
    """
    Input:img: image mat mxnx3
            H: Homography by which this image will transform 
            ie H*img = tr_img
    Output: m,n: row, col new image size using information of corners
            cor: original corners of image
    """
    # find corners:
    cor = [[0,0],
          [img.shape[1]-1,0],
          [img.shape[1]-1,img.shape[0]-1],
          [0,img.shape[0]-1]]
    cor_hc = convert2HC(cor)
    tr_cor_hc = H@cor_hc.T
    tr_cor = convert2phy(tr_cor_hc)
    x_min = min(tr_cor[:,0]); x_max = max(tr_cor[:,0]);
    y_min = min(tr_cor[:,1]); y_max = max(tr_cor[:,1]);
    m = y_max-y_min # height of image
    n = x_max-x_min # width of image
    return(m,n,cor)
#%%
# function for finding H and H' for image rectification
def get_rectify_homo(e,P1,P2,M,img1,img2,pts1,pts2):
    """
    Input:  e: right image epipole
            M: rotation part of P2 = [M|t]
            img1,img2: image matrices
            pts1,pts2: Matching points
    Output: H1 and H2: image rectification Matrices
    """
    # convert pts to HC
    pts1_hc = convert2HC(pts1) #[[x,y,1],...]
    pts2_hc = convert2HC(pts2) #[[x,y,1],...]
    # find image sizes for second image
    m=img2.shape[0]; n=img2.shape[1]; 
    # get T,R,G to make H2
    T=get_T(m,n)
    T2 = get_T(-1*m,-1*n) 
    R,f=get_Rot(e,m,n)
    G=get_G(f)
    H2=T2@G@R@T
    # find M=P'P+
    P1_psinv = P1.T@(np.linalg.inv(P1@P1.T)) # right psuedo inv
    M = P2@P1_psinv
    # Now find Matching H1    
    H0= H2@M
    x2_hat_hc = H2@pts2_hc.T
    x1_hat_hc = H0@pts1_hc.T
    # find physical coordinates of x1_hat and x2_hat
    x2_hat = convert2phy(x2_hat_hc) #[[x,y],..]
    x1_hat = convert2phy(x1_hat_hc) #[[x,y],..]
    # Find Ha 
    Ha = get_Ha(x1_hat,x2_hat)
    H1= Ha@H0
    return(H1,H2)
# function for solving Ha for first image
def get_Ha(x1_hat,x2_hat):
    """
    Input: x1_hat, x2_hat: transformed physical coordinates 
    Output: Ha homography using Linear solution 
    """
    A = np.ones((x1_hat.shape[0],3))
    A[:,:-1]= x1_hat
    b = x2_hat[:,0] 
    # solve Ax-b = 0 using left Psuedo inverse
    a = (np.linalg.inv((A.T)@A)@A.T)@b
    Ha = np.eye(3)
    Ha[0,:]=a
    return(Ha)
#function for getting G matrix for image rectification
def get_G(f):
    """
    Input: f: x coordinate of transformed epipole
    Output: G:matrix that take epipole from (f,0,1) to (f,0,0)
    """
    G= np.array([[1   ,0,0],
                 [0   ,1,0],
                 [-1/f,0,1]])
    return(G)
# function for finding Rotation matrix for epipolar lines
def get_Rot(e,m,n):
    """
    Input: e: the epipole e_prime for right image in HC
    m,n= image size row,col
    
    Output: R:Rotation MAtrix  , f: new epipole location
    Finds the Rotation Matrix which takes the epipole from
    ((ex-x0),(ey-y0),1) to (f,0,1)
    --------------------------------------------
    given:
    R@[(ex-x0),(ey-y0),1]^T =[f,0,1]
    Also, The rotation matrix(in-plane rot) is given by:
    R=[[cos(t),-sint(t),0],
       [sin(t),cos(t),0],
       [0   , 0,       1]]
    Which gives:
        (ex-x0)cos(t)-(ey-y0)sin(t) = f   &
        (ex-x0)sin(t)+(ey-y0)cos(t) = 0 
        => tan(t) = -(ey-y0)/(ex-x0)
        and then f can be found
    --------------------------------------------    
    """
    x0=n/2; y0=m/2;
    e = e/e[-1]
    ex= e[0]; ey=e[1];
    t = np.arctan(-(ey-y0)/(ex-x0)) # theta
    R=np.array([[np.cos(t),-1*np.sin(t),    0],
                [np.sin(t),   np.cos(t),    0],
                [0        ,   0        ,    1]])
    f=(ex-x0)*np.cos(t)-(ey-y0)*np.sin(t)    
    return(R,f)
# function for Translating image origin to image center
def get_T(m,n):
    """
    Input: m,n size of image as mxn 
      ie center is at x0=n/2 y0=m/2
    Output: T 3x3 transformation matrix
    """
    x0=n/2; y0=m/2;
    T= np.array([[1.0,0.0,-x0],
                 [0.0,1.0,-y0],
                 [0.0,0.0,1.0]])
    return(T)
#%%
# function for LM call for refinement of F
def Refine_F(F0,pts1,pts2):
    """
    Using Gold Standard Algorithm
    Input: F0: initial solution for LM 3x3 matrix
            pts1,pts2: lis of list of coordinates [[x1,y1],[x2,y2],...]
    Output: F_ref,P1_ref,P2_ref : refined properties
    """
    # make P2:[M|t] and Xi from F 
    F0=F0/F0[-1,-1]
    P1,P2=get_Proj_mat(F0)
    M=P2[:,:-1]; t=P2[:,-1];
    # find HC rep of pts
    pts1_hc=convert2HC(pts1)
    pts2_hc=convert2HC(pts2)
    # find world pt
    world_pt_hc =[]
    for i in range(len(pts1)):
        world_pt_hc.append(Triangulate(pts1_hc[i,:],pts2_hc[i,:],P1,P2)) #will give HC 
    world_pt_hc=np.array(world_pt_hc)
    # get physical coordinate of world pt
    world_pt = convert2phy(world_pt_hc.T) #format [[X,Y,Z],..]
    # make parameter vector p using M, t, X
    p0=np.zeros(3*len(pts1)+12)
    p0[:9]=np.reshape(M,(9))
    p0[9:12]=t
    p0[12:]=np.reshape(world_pt,(3*len(pts1)))
    # call to LM
    optim=least_squares(error_fun_F,p0,
                        method='lm',args=(pts1,pts2))
    p_star = optim['x']
    # get back P2,F
    M_star=p_star[:9]
    M_star= np.reshape(M_star,(3,3))
    t_star = p_star[9:12]    
    tx= cross_rep_mat(t_star)
    F_ref = tx@M
    F_ref = F_ref/F_ref[-1,-1]
    P2_ref = np.zeros((3,4))
    P2_ref[:,:-1]=M_star
    P2_ref[:,-1]=t_star
    P1_ref=P1
    return(F_ref,P1_ref,P2_ref,M_star)
#%%
# function for defining cost function for LM 
# optimization of Fundamental Matrix
def error_fun_F(p,pts1,pts2):
    """
    Gold-Standard Method cost function
    Input: 
    -------------------------------------------------------------------
        p: parameter vector of size 3n+12 
            3n: 3D physical coordinates of World_pts
            12: for Projection Matrix
            p=[M,t,X1,X2...,Xn]
            M=[M11,M12,M13,M21,...]
            t=[tx,ty,tz]
            Xi=[xi,yi,zi]
    -------------------------------------------------------------------
        pts1,pts2: list of coordinates of pts in format [[x,y],..]
        
    Output: cost: geometric distance vector
    """
    # Construct P1,P2
    P1=np.hstack((np.eye(3),np.zeros((3,1))))
    M=p[:9]
    M=np.reshape(M,(3,3))
    t=p[9:12]    
    P2=np.zeros((3,4))
    P2[:,:-1]=M ; P2[:,-1]=t;
    # construct World_pt from p
    world_pt = []
    for i in range(len(pts1)): world_pt.append(p[12+3*i:12+3*(i+1)])
    world_pt = np.array(world_pt) # rows as physical coord
    #make world pt HC
    world_pt_hc = convert2HC(world_pt) # [[x,y,z,1],....]
    # get extimates of world point using P1 and P2
    x1_hat_hc = P1@world_pt_hc.T
    x2_hat_hc = P2@world_pt_hc.T
    # convert to physical coordinates
    x1_hat = convert2phy(x1_hat_hc)
    x2_hat = convert2phy(x2_hat_hc)
    # find difference between physical coordinates
    diff1 = pts1-x1_hat
    diff2 = pts2-x2_hat
    error = np.hstack((diff1[:,0],diff1[:,1],diff2[:,0],diff2[:,1]))
    return(error)
#function for triangulating the world pt
def Triangulate(x1,x2,P1,P2):
    """
    Input: x1: Hc coordinate in left image
           x2: HC coordinate in right image
           P1,P2: Projection matrix for left and right resp    
    Output: world_pt: the cooresponding world point in HC
    """
    x1= x1/x1[-1]; x2= x2/x2[-1]
    A=np.zeros((4,4))
    A[0,:]=x1[0]*P1[2,:]-P1[0,:]
    A[1,:]=x1[1]*P1[2,:]-P1[1,:]
    A[2,:]=x2[0]*P2[2,:]-P2[0,:]
    A[3,:]=x2[1]*P2[2,:]-P2[1,:]
    # solve AX =0 using svd
    u,d,vt=np.linalg.svd(A)
    i=np.argmin(np.abs(d))
    world_pt = vt[i,:]
    world_pt = world_pt/world_pt[-1]
    return(world_pt)
#function for obtaining Projection Matrics from F
def get_Proj_mat(F):
    """
    Input: F:3x3 matrix
    Output: P1=[I|0] P2=[exF|e]
    """
    F=F/F[-1,-1]
    P1= np.hstack((np.eye(3),np.zeros((3,1))))
    e = get_left_nullvec(F)
    ex = cross_rep_mat(e)
    P2= np.zeros((3,4))
    P2[:,:-1]=ex@F
    P2[:,-1]=e
    return(P1,P2)
# function for finding the right Null vector
def get_right_nullvec(F):
    """
    Input: F: 3x3 matrix
    Output: e: right Null vector of F
    by solving Fe=0
    """
    u,d,vt = np.linalg.svd(F)
    i_null= np.argmin(np.abs(d))
    e=vt[i_null,:]
    return(e)
# function for finding the left Null vector 
def get_left_nullvec(F):
    """
    Input: F: 3x3 matrix
    Output: e: left Null vector of F
    by solving e_pr^T F=0 or FT e_pr = 0
    """
    u,d,vt = np.linalg.svd(F.T)
    i_null= np.argmin(np.abs(d))
    e=vt[i_null,:]
    return(e)
# function for getting cross-representation MAtrix
def cross_rep_mat(w):
    """
    Input:w 3x1 vector
    Output: W 3x3 matrix
    """
    # make Wx from w 
    Wx=np.array([[0     , -1*w[2],    w[1]],
                [w[2]   , 0      , -1*w[0]],
                [-1*w[1], w[0]   ,      0]]) 
    return(Wx)
#%%
#function for finding F using RANSAC
def F_Ransac(pts1,pts2):
    """
    Input: pts1,pts2: list of list of coordinates of interest pts
    Output: in_indices_store: indices of inlier points of image  1 and image 2 respectively
                            format:[index1,index2,....] list of index numbers
    """
    #----define RANSAC parameters:-----#
    p=0.95  # probability that at least one of the N trials will be free of outliers
    n=8     # minimal set of correspondences chosen randomly for constructing F
    delta=4 # decision threshold to construct inlier set    
    e=0.5   #worst case probability that a correspondence is an outlier
    N=int(np.ceil(np.log(1-p)/np.log(1-(1-e)**n))) #number of iterations needed
    #----------------begin-------------------#
    n_total=len(pts1) # total number of correspondences
    # convert pts lists to array and into homogeneous form
    pts1_hc=convert2HC(pts1)
    pts2_hc=convert2HC(pts2)
    
    size_old=0 
    count=0    
    while N>count:
        #randomly sample n points from pts1 and pts2
        rand_indices=np.random.randint(n_total,size=n)
        ipts1=pts1_hc[rand_indices,:]
        ipts2=pts2_hc[rand_indices,:]
        # find Fundamental Matrix using these points: linear least sqaures
        iF = Funda_using_pts(ipts1,ipts2)
        P1,P2 = get_Proj_mat(iF)
        # find world pt
        world_pt_hc =[]
        for i in range(len(pts1)):
             world_pt_hc.append(Triangulate(pts1_hc[i,:],pts2_hc[i,:],P1,P2)) #will give HC 
        world_pt_hc = np.array(world_pt_hc) #[[X,Y,Z,1],..]
        # find estimated pts
        est_pts1_hc=P1@world_pt_hc.T
        est_pts2_hc=P2@world_pt_hc.T
        # get physical coord of estimate pts
        est_pts1 = convert2phy(est_pts1_hc)#[[x,y],..]
        est_pts2 = convert2phy(est_pts2_hc) #[[x,y],..]
        # find error distance
        error1=pts1_hc[:,:-1]-est_pts1
        error2=pts2_hc[:,:-1]-est_pts2
        distance1=np.linalg.norm(error1,axis=1)
        distance2=np.linalg.norm(error2,axis=1)
        # consensus
        in1 = np.where(distance1<=delta)[0]
        in2 = np.where(distance2<=delta)[0]
        in_indices=list(set(in1)&set(in2))
        size_new=len(in_indices) #size of ith inlier set
        #store the set of indices with largest size
        if size_new>size_old:
            in_indices_store=in_indices 
            size_old=size_new            
            e = 1-size_old/n_total
            N=int(np.ceil(np.log(1-p)/np.log(1-(1-e)**n)))            
        count+=1
    print('RANSAC max inlier set was'+str(len(in_indices_store)))
    return(in_indices_store)
#%%
#function for finding Fundamental matrix using point correspondences: Linear Least sqaures
def Funda_using_pts(pts1_hc,pts2_hc):
    """
    Input: pts1_hc,pts2_hc: list of list of correponding points: size Nx3 in HC
            [[X,Y,1],...]
    Output: F: 3x3 array, Linear Least squre estimate
    F needs to be Conditioned to Rank 2
    """
    # convert points to physical coord
    pts1 = convert2phy(pts1_hc.T) #[[x,y],..]
    pts2 = convert2phy(pts2_hc.T) #[[x,y],..]
    # get Normalization transformation Matrix
    T1=Norm_Transform(pts1)
    T2=Norm_Transform(pts2)
    # Normalize Hc coordinates
    pts1_hc=(T1@pts1_hc.T).T
    pts2_hc=(T2@pts2_hc.T).T
    # make A matrix for Af=0
    A1=np.hstack((pts1_hc,pts1_hc,pts1_hc))
    A2=np.kron(pts2_hc,np.ones((1,3)))
    A=np.multiply(A1,A2)
    #solve for f using SVD 
    u,d,vt=np.linalg.svd(A)
    f=vt[-1,:]
    F=np.reshape(f,(3,3))
    # apply Rank=2 constraint
    F=make_detF_0(F)
    # de-normalized F
    F=T2.T@F@T1
    F=F/F[-1,-1]
    return(F) 
#function to make det(F)=0
def make_detF_0(F):
    """ 
    Input: F 3x3 matrix
    Output: F0: F with det(F)=0
    """
    # apply Rank=2 constraint
    U,D,VT=np.linalg.svd(F)
    d_prime=D
    d_prime[np.argmin(D)]=0
    F0=U@np.diag(d_prime)@VT
    return(F0)
# function for finding Transformation for normalization
def Norm_Transform(pts):
    """
    This just shifts the image origin to centroid of pts
    Input: pts: list of list of corrdinates [[x1,y1],[x2,y2],...]
    Output: Transformation T for normalization
    """
    pts=np.array(pts)
    # find centroid coordinataes
    x_mean= np.mean(pts[:,0])
    y_mean= np.mean(pts[:,1])
    # shift the origin to centroid pt
    shifted_pts=np.zeros((pts.shape),dtype=np.float32)
    shifted_pts[:,:]=pts
    shifted_pts[:,0]+= -1*x_mean
    shifted_pts[:,1]+= -1*y_mean
    # find mean distance of all points from new origin
    mean_d = np.mean(np.linalg.norm(shifted_pts,axis=1))
    # scale mean distance to sqrt(2)
    scale= np.sqrt(2)/mean_d
    T = np.array([[scale,0.0,-1*scale*x_mean],
                  [0.0,scale,-1*scale*y_mean],
                  [0.0, 0.0, 1.0]])
    return(T)
#=====================================HW9=====================================#
#%%
#=====================================HW8=====================================#
#%%
# function for reprojecting the world points onto the image
def ReprojectPoints(img,world_coord,Corners,K,R,t):
    """
    Input: img: colored image
           world_coord: list of list of coordinates [[x1,y1],[x2,y2],...]
           corners: list of list of original coordinates of corners [[x1,y1],[x2,y2],...]
           K: Intrinsic parameter matrix 3x3
           R: Rotation matrix for this image 3x3
           t: translation vector for this image 3x1
    Output: rep_img: img with reprojected points color image
            mean_e mean of error using Euclidean distance
            var_e: variance of error using Euclidena distance
    """
    # convert world_coord to HC
    X_hc= np.ones((len(world_coord),3))
    X_hc[:,:-1]=np.array(world_coord)
    X_hc=X_hc.T # hc coordinates as col vectors
    # make camera projection matrix P
    P= np.array([R[:,0].T,R[:,1].T,t.T])
    P=K@P.T
    #find reprojected points
    rep_pt_hc= P@X_hc
    # convert to physical coordinates for plotting
    rep_pt= np.linalg.inv(np.diag(rep_pt_hc[-1,:]))@rep_pt_hc.T
    rep_pt=rep_pt.T # physical coordinates as col vectors
    rep_pt=rep_pt[:-1,:]
    # find Euclidean distance error, mean and var
    e=np.array(Corners).T-rep_pt
    e=np.linalg.norm(e,axis=0)
    mean_e=np.mean(e)
    var_e=np.var(e)
    # plot corners on image
    rep_img=np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(world_coord)):
        rep_img=cv2.circle(img,(int(rep_pt[0,i]),int(rep_pt[1,i])),2,(0,0,255),-1)
        rep_img=cv2.circle(img,(int(Corners[i][0]),int(Corners[i][1])),2,(0,255,0),-1)
        rep_img=cv2.putText(img,str(i+1),(int(rep_pt[0,i]),int(rep_pt[1,i])), font,0.5,(255,0,0),1,cv2.LINE_AA)
    return(rep_img,mean_e,var_e)
#%%
# function for getting the cost function for camera calibration
# with radial distortion
def CostFun_cam_cal_radial(p,x,x_m):
    """
    Input: p: parameters of the cost function, is of size 5+N*6 
           format: [K,w1,t1,w2,t2,...wn,tn,k1,k2] where 
           K=a_x,a_y,s,x0,y0
           wi=wx,wy,wz
           ti=tx,ty,tz
           all contatenated into a single vector
           x: real coordinates of the corners in i images 
           x_m: model coordinates in real world     
    Output: cost function value: a scalar value of sum of squares of error    
    """
    # make K: intrinsic matrix
    a_x=p[0]; a_y=p[1]; s=p[2]
    x0=p[3]; y0=p[4]; k1=p[-2]; k2=p[-1]
    K=np.array([[a_x,s,x0],
                [0,a_y,y0],
                [0,0,1]])
    # make Rotation matrices R
    num_img=int((len(p)-7)/6)
    # double loop for finding dgeom**2
    N=len(x_m)
    cost=np.zeros(2*num_img*N)
    for i in range(num_img):
        iw=p[6*i+5:6*i+8]
        it=p[6*i+8:6*i+11]
        iR=Rot_vec2mat(iw)
        est_map=np.array([iR[:,0].T,iR[:,1].T,it.T])
        est_map=K@(est_map.T) # mapping function for finding estimate
        xij=np.array(x[i]); xij=xij.T # coordinates as col vectors
        x_m_hc=np.ones((len(x_m),3)); x_m_hc[:,:-1]=np.array(x_m)
        x_m_hc=x_m_hc.T # coordinates as col vectors
        # find estimate using pinhole model
        x_hat_hc=est_map@x_m_hc
        x_hat=np.linalg.inv(np.diag(x_hat_hc[-1,:]))@x_hat_hc.T 
        x_hat=x_hat.T; x_hat=x_hat[:-1,:] # physical coordinate
        # find coordinates with radial distortion
        diff=x_hat-(np.kron(np.array([x0,y0]),np.ones((N,1)))).T  # differnces as col vectors
        r_2=np.sum(np.square(diff),axis=0) #sqauring and summing
        m=k1*r_2+k2*np.square(r_2)
        m=np.vstack((m,m))
        x_hat_rad = x_hat + np.multiply(m,diff)        
        temp= xij-x_hat_rad
        cost[i*2*N:(i+1)*2*N]=np.hstack((temp[0,:],temp[1,:]))  
    return(cost)
# function for getting the cost function for camera calibration
# without radial distortion
def CostFun_cam_cal_linear(p,x,x_m):
    """
    Input: p: parameters of the cost function, is of size 5+N*6 : pinhole model
           format: [K,w1,t1,w2,t2,...wn,tn] where 
           K=a_x,a_y,s,x0,y0
           wi=wx,wy,wz
           ti=tx,ty,tz
           all contatenated into a single vector
           x: real coordinates of the corners in i images in the format:
              [corners1,corners2,.....cornersN] , where corners1 is another list of the form
              [[x1,y1],[x2,y2],.....,[xn,yn]]: n= number of corners in our calibration pattern
           x_m: model coordinates in real world in the format of list of list of coordinates
               [[x1,y1],[x2,y2],.....,[xn,yn]], note same will be repeated N times for all N images
    Output: cost function value: a scalar value of sum of squares of error    
    """
    # make K: intrinsic matrix
    a_x=p[0]; a_y=p[1]; s=p[2]
    x0=p[3]; y0=p[4]
    K=np.array([[a_x,s,x0],
                [0,a_y,y0],
                [0,0,1]])
    # make Rotation matrices R
    num_img=int((len(p)-5)/6)    
    # double loop for finding dgeom**2
    N=len(x_m)
    cost=np.zeros(2*num_img*N)
    for i in range(num_img):
        iw=p[6*i+5:6*i+8]
        it=p[6*i+8:6*i+11]
        iR=Rot_vec2mat(iw)
        est_map=np.array([iR[:,0].T,iR[:,1].T,it.T])
        est_map=K@(est_map.T) # mapping function for finding estimate
        xij=np.array(x[i]); xij=xij.T # coordinates as col vectors
        x_m_hc=np.ones((len(x_m),3)); x_m_hc[:,:-1]=np.array(x_m)
        x_m_hc=x_m_hc.T # coordinates as col vectors
        # find estimate using pinhole model
        x_hat_hc=est_map@x_m_hc #estimate
        x_hat=np.linalg.inv(np.diag(x_hat_hc[-1,:]))@x_hat_hc.T 
        x_hat=x_hat.T; x_hat=x_hat[:-1,:] # physical coordinate
        temp= xij-x_hat
        cost[i*2*N:(i+1)*2*N]=np.hstack((temp[0,:],temp[1,:]))
    return(cost)
#%%
# function for obtaining Rotation matrix from w vector
def Rot_vec2mat(w):
    """
    Input: w: col vector of size 3: three parameters of rotation [wx,wy,wz]
    Output: R: np.array of size 3x3: The rotation matrix
    Using Rodriguez Rotation Formula    
    """
    # make Wx from w 
    Wx=np.array([[0,-1*w[2],w[1]],
                [w[2],0,-1*w[0]],
                [-1*w[1],w[0],0]])    
    phi=np.linalg.norm(w)
    R=np.eye(3) + (np.sin(phi)/phi)*(Wx) + ((1-np.cos(phi))/phi**2)*(Wx@Wx)
    return(R)
# function for obtaining w vector from Rotation matrix
def Rot_mat2vec(R):
    """
    Input: R: Rotation matrix, format 3x3 np.array
    Output: w: size 3 vector, format np.array [wx,wy,wz]
    """
    phi=np.arccos((np.trace(R)-1)/2)
    w=(phi/(2*np.sin(phi)))*np.array([(R[2,1]-R[1,2]),
                                      (R[0,2]-R[2,0]),
                                      (R[1,0]-R[0,1])])
    return(w)
#%%
# -----------Zhang's Algo I-------------------------#
# function for finding V from H for one image
def Zhang_V(iH):
    """
    Input: iH: Homography for ith image: size 3x3
    Output: V: matrix for solving for abs conic size 2x6
    """
    # build iV matrix for w: image of abs conic
    iv_12=Zhang_Vij(iH,0,1)
    iv_11=Zhang_Vij(iH,0,0)
    iv_22=Zhang_Vij(iH,1,1)
    iV=np.array([iv_12.T,(iv_11-iv_22).T])
    return(iV)
#----------------------------------------------------#
# function for making V from H for Zhang's
def Zhang_Vij(H,i,j):
    """
    Input: H: 3x3 homography st H*world=img
            i,j: indices 
    Output: v: vector for zhang's algorithm   
    """
    # b=[w11 w12 w22 w13 w23 w33]
    v=np.zeros(6)
    v[0]=H[0,i]*H[0,j]
    v[1]=H[0,i]*H[1,j]+H[1,i]*H[0,j]
    v[2]=H[1,i]*H[1,j]
    v[3]=H[2,i]*H[0,j]+H[0,i]*H[2,j]
    v[4]=H[2,i]*H[1,j]+H[1,i]*H[2,j]
    v[5]=H[2,i]*H[2,j]
    return(v)
#%% 
# function for refining corners
def refine_corners(img,corners,window):
    """
    Input:img: gray scale image 
           corners: list of list of corners coordinates
           window: size of neigborhood 
    Output: ref_corner: list of refined corners
    """
    # convert corner list to numpy array for cornersubpix
    corner_array=np.zeros((len(corners),1,2),dtype=np.float32)
    for i in range(len(corners)): corner_array[i,0,:]=corners[i]
    #define criteria for subpix
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,0.01)
    ref_corners=cv2.cornerSubPix(img,corner_array,(window,window),(-1,-1),criteria)
#    ref_corners=corner_array
    ref_corner_list=[]
    for i in range(ref_corners.shape[0]): ref_corner_list.append(ref_corners[i,0,:])
    return(ref_corner_list)

#%%
# function for finding the corners from lines
def findCorners(lines,img_size,pattern_dim):
    """
    Input: lines: hough lines (N,1,2) has rho=lines[i,0,1] and theta=lines[i,0,1]
           img_size: (m,n) tuple gives size of image st
           m=num row n= num col
           pattern_dim: (a,b) tuple: defining the number of boxes pattern
           a=num box along y b= numbox along x
    Output:Corners: list of corner coordinates [[x1,y1],[x2,y2],....]
    """
    m,n=img_size
    # store all angles in a list
    thetas=[lines[i,0,1] for i in range(lines.shape[0])]
    thetas=np.array(thetas) # thetas obtained from hough tr will always be positive
    thetas+=-np.pi/2 
    # categorize lines into hor and vert based on thetas
    i_hor=np.where(np.abs(thetas)<np.pi/4)[0]  # hor line if less than 45 deg
    hor_lines=[lines[i,0,:] for i in i_hor]
    i_ver=np.where(np.abs(thetas)>=np.pi/4)[0] # ver line if more than 45 deg
    ver_lines=[lines[i,0,:] for i in i_ver]

    #refine lines
    hor_lines=RefineLines(hor_lines,img_size,2*pattern_dim[0])
    ver_lines=RefineLines(ver_lines,img_size,2*pattern_dim[1])
  
    # sort hor and ver lines based on the intercept values
    # y_int:hor ~ rho*sin(t), x_int:ver ~ rho*cos(t)
    hor_lines=sorted(hor_lines,key=lambda x:x[0]*np.sin(x[1]))
    ver_lines=sorted(ver_lines,key=lambda x:x[0]*np.cos(x[1]))
    
    n_hor=len(hor_lines)
    n_ver=len(ver_lines)
    Corners=[]
    # eq of lines is xcos(t)+ysin(t)-rho: using this to make HC
    for i in range(n_hor):
        i_hor_hc=np.array([np.cos(hor_lines[i][1]),np.sin(hor_lines[i][1]),-1*hor_lines[i][0]]) 
        for j in range(n_ver):
            j_ver_hc=np.array([np.cos(ver_lines[j][1]),np.sin(ver_lines[j][1]),-1*ver_lines[j][0]])
            corner=np.cross(i_hor_hc,j_ver_hc)
            corner=corner[:-1]/corner[-1]
            Corners.append(corner)    
    return(Corners,hor_lines,ver_lines)
#%% 
#function for removing closely resembling lines
def RefineLines(lines,img_size,final_size):
    """
    Refines lines based on intersection of the lines
    Assuming we have only one category of lines ie hor or ver in the set
    Input: lines: list of list of rho and theta values
           img_size: (m,n) tuple serves as a window in this fn
           final_size:
    Output: ref_lines: list of list of rho and theta of refined lines
    """
    rho=np.array([lines[i][0] for i in range(len(lines))])
    theta=np.array([lines[i][1] for i in range(len(lines))])
    HC_lines= np.ones((len(lines),3))
    # eq of lines is xcos(t)+ysin(t)-rho: using this to make HC
    HC_lines[:,0]=np.cos(theta)
    HC_lines[:,1]=np.sin(theta)
    HC_lines[:,-1]=-1*rho
    # loop to find intersections within set
    count=np.zeros(len(lines),dtype=int)
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            pt=np.cross(HC_lines[i,:],HC_lines[j,:])
            if pt[-1]!=0: 
                # when not parallel
                pt=pt[:-1]/pt[-1]
                if pt[0]>=0 and pt[0]<img_size[0] and pt[1]>=0 and pt[1]<img_size[1]:
                    count[j]=1
            else:
                # when lines are parallel
                rho1=lines[i][0];  rho2=lines[j][0]
                if abs((rho1-rho2)/rho1)<10**-2:
                    count[j]=1
#                c1=HC_lines[i,-1]/HC_lines[i,0]
#                c2=HC_lines[j,-1]/HC_lines[j,0]
#                if abs((c1-c2)/c1)<10**-2:
#                    count[j]=1
    ind=np.where(count==0)[0]
    if len(ind)>final_size:
        #--------------kmeans-------------------#
        """ 
         now we need to remove the false positive lines
         which would have interesected in the image region 
         but were just a little bit shy: we can do this using kmeans
         kmeans on rho values of lines
         """
        temp_lines=[]
        for i in range(len(ind)): temp_lines.append(lines[ind[i]])
        N=len(temp_lines)
        observations=np.array(temp_lines)
        observations=observations[:,0]# only rho
        centroids=kmeans(observations,final_size) # returns a tuple
    #    plt.scatter(observations,np.ones(len(observations)),c='b')
    #    plt.scatter(centroids[0],np.ones(len(centroids[0])),c='r')
    #    plt.show()
        # find rho values in lines closest to centroids
        store=np.zeros(final_size,dtype=int)
        for i in range(final_size): 
            temp=np.abs(observations-centroids[0][i]*np.ones(N))
            store[i]=np.argmin(temp)
        ref_lines=[]
        for i in range(final_size): ref_lines.append(temp_lines[store[i]]) # only single values
        #--------------kmeans-------------------#
    else:
        ref_lines=[]
        for i in range(final_size): ref_lines.append(lines[ind[i]])
    return(ref_lines)
#%%
# function for drawing Line on the calibration pattern
def drawLine(color_img,lines):
    """
    Input: color_img: colored image mxnx3
           lines: list of arrays with [rho,theta] values
           slices 1,2 have values of rho and theta
    Output: draw_img: m,n,3 image with line drawn
    using lines    
    """
    # find size of image
    m,n=color_img.shape[:-1]
    #find maximum edge length that can fit in image
    L=np.sqrt(m**2+n**2)
    # make copy of color_img
    draw_img=np.copy(color_img)
    # for every line
    for i in range(len(lines)):
        # find rho and theta
        rho,theta=lines[i]
        a=np.cos(theta);  b=np.sin(theta);
        x0=rho*a; y0= rho*b; # perpendicular point
        # find pts on line which will lie outside
        # image bounds or on borders
        x1=int(x0+L*(-b))
        y1=int(y0+L*(a))
        x2=int(x0-L*(-b))
        y2=int(y0-L*(a))
        # draw line joining the two points
        cv2.line(draw_img,(x1,y1),(x2,y2),(0,255,0),1)
    return(draw_img)
"""
V0: was used to draw raw hough lines
"""
# function for drawing Line on the calibration pattern
def drawLine_old(color_img,lines):
    """
    Input: color_img: colored image mxnx3
           lines: size N,1,2 N is the number of lines
           slices 1,2 have values of rho and theta
    Output: draw_img: m,n,3 image with line drawn
    using lines    
    """
    # find size of image
    m,n=color_img.shape[:-1]
    #find maximum edge length that can fit in image
    L=np.sqrt(m**2+n**2)
    # make copy of color_img
    draw_img=np.copy(color_img)
    # for every line
    for i in range(lines.shape[0]):
        # find rho and theta
        rho,theta=lines[i,0,:]
        a=np.cos(theta);  b=np.sin(theta);
        x0=rho*a; y0= rho*b; # perpendicular point
        # find pts on line which will lie outside
        # image bounds or on borders
        x1=int(x0+L*(-b))
        y1=int(y0+L*(a))
        x2=int(x0-L*(-b))
        y2=int(y0-L*(a))
        # draw line joining the two points
        cv2.line(draw_img,(x1,y1),(x2,y2),(0,255,0),1)
    return(draw_img)
#%%
#function for finding corners
def findLines(img,minVal,maxVal,HoughThresh):
    """
    Input: img: mxn gray scale image
           minVal,maxVal: hysterisis thresholds for canny operator
           HoughThresh: threshoold for hough transform for finding the peak
    Output: edges: binary image with only canny edges 
            lines: (rho,theta) format
    """
    # find canny edges for the image
    edges=cv2.Canny(img,minVal,maxVal)
    # find lines using hough transform
    lines = cv2.HoughLines(edges,1,np.pi/180,HoughThresh)
    return(edges,lines)
#======================================HW8====================================#
#%%
#======================================HW7====================================#
#%%
# function for Nearest Neighbor classifier
def NNClassifier(test_vec,class_LBPs,class_names,class_sizes,n):
    """
    Input: class_names= dictionary of class names
           class_LBPs: dictiory of LBPs for all classes
           class_LBP[class_names[i]]= np array of size j,P+2: j is size of image
           class_sizes: dictionary of number of training images in one class
           test_vec: LBP vector for testing image
           n: number of nearest neighbors to be found
    Output: identfied_class: name of identified class (string format)
    """
    P=len(test_vec)-2
    numClass=len(class_names)
    # make a vector containing all class_LBPs using vstack
    all_vec=np.zeros((1,P+2))
    for i in range(numClass):
        all_vec=np.vstack((all_vec,class_LBPs[class_names[i]]))
    all_vec=all_vec[1:,:] # remove the first row of zeros
    # find euclidean distance
    d= all_vec-np.kron(np.array(test_vec),np.ones((all_vec.shape[0],1))) # difference
    d=np.square(d) # element wise squaring
    d= d@np.ones((P+2,1))# sum all cols: squared distances
    # find smallest n entries in d
    sorted_d=np.sort(d[:,0])
    count=np.zeros(numClass,dtype=int)
    # make cumulative size array: to be used for comparison
    cum_size=np.zeros(numClass)
    cum_size[0]=class_sizes[class_names[0]]
    for i in range(1,numClass):
        cum_size[i]=cum_size[i-1]+class_sizes[class_names[i]]
    #loop over number of nearest neighbors n
    for i in range(n):
        # find ith smallest distance
        dmin=sorted_d[i]
        # find index of dmin in d
        i_dmin=np.where(d[:,0]==dmin)[0][0]
        # find class type for this index 
        for k in range(numClass):
            if i_dmin<cum_size[k]:
                count[k]+=1
                break        
    # find the class which has max count
    i_max=np.argmax(count)
    identified_class=class_names[i_max] # string output
    return(identified_class)
#%%
#function for making the lBP from given image
def LBP(img,P,R):
    """
    Input: img: grayscale image mxn
            P: number of points in the circular neighbour for making the 
            bit vector at any point
            R: radius of the neighbour circle
    Output: hist:  Feature vector from histogram of LBP of size P+2
    """
    m=img.shape[0];n=img.shape[1]
    hist=np.zeros(P+2,dtype=int)#LBP histogram with P+2 bins:0-P+1
    # loop over image pixels removing one row and col on both sides
    for i in range(m-2):
        for j in range(n-2):
            # find coordinate of current pixel: shifted by one unit 
            x=np.array([j+1,i+1]) # row vector
            # find P neighbours at a radius of R
            p=np.arange(P)
            x_nbor=np.kron(x,np.ones((P,1)))
            x_nbor=x_nbor.T+R*np.array([np.cos(2*np.pi*p/P),np.sin(2*np.pi*p/P)]) #stacked as col vectors
            # find pixel values at the neighbors using Bilinear
            p_nbor=[]
            for k in range(P): 
                    p_nbor.append(Bilinear_gray(x_nbor[:,k],img))
            # convert neighboring pixel values to binary using center threshold
            binvec=binary_vec(p_nbor,img[x[1],x[0]])
            # convert binary vector to minIntval: to make Rotation Invariant
            min_bv=minbv(binvec) # is a string of len P
            #find encoding of minintvalbinary pattern 
            encoding=encoding_LBP(min_bv)
            #update corresponding histogram range
            hist[encoding]+=1
    return(hist)
#function for finding the encoding of minintval pattern as developed by
# creators of LBP
def encoding_LBP(x):
    """
    Input: x: is a string of pattern made of 1 and 0, will be of size P
    Output: op: integer value for the pattern x using rules of LBP creators
    """
    P=len(x)
    #initialize runs 
    runs=1
    # find runs for the pattern
    old_str=x[0] # first string in the pattern
    for i in range(P-1):
        if x[i+1]!=old_str:
            old_str=x[i+1]
            runs+=1        
    # assign value for output to op
    if runs>2:
        op=P+1
    elif runs==1 and old_str=='1':
        op=P
    elif runs==1 and old_str=='0':
        op=0
    else:
        #runs=2; count the number 1s in the pattern
        op=0
        for i in range(P):op=op+int(x[i])        
#        op=0        
#        while True:
#            if x[P-1-op]=='1':
#                op+=1
#            else:
#                break
    return(op)
# function for finding the minimum binary vector
def minbv(x):
    """
    Input: x: array of 0 or 1s
    output: min_bv: str format minimum binary vector ...
    from all circular rotations
    """
    n=len(x)
    # make repeated str list from x for cicular rotations
    whole_str=''
    for i in range(n): whole_str=whole_str+str(x[i])
    repeat_str=whole_str+whole_str
    str_store=[]
    num_store=[]
    for i in range(n):
        # find circular rotation string slice
        str_store.append(repeat_str[i:i+n])
        num_store.append(int(str_store[i],2)) #integer value of the circular rotated string
    # find minintval
    imin=np.argmin(num_store)
    min_bv=str_store[imin] #string format
    return(min_bv)   

#function to make binary vector from neighboring pixel values list/array
def binary_vec(x,c):
    """
    Input: x: array of pixel values(scalars) around the neighbors
           c: center pixel values
    Output: binvec: array of size(x) st any element in x if greater than c
    gets 1 else 0
    """
    n=len(x)        
    binvec=np.zeros(n,dtype=int)
    for i in range(n):
        if x[i]>=c:
            binvec[i]=1
    return(binvec)

#function for Bilinear Interpolation of Gray Levels
def Bilinear_gray(x,img):
    """
    Input: x: [x,y] coordinates of pixel:np.array format 
        img: gray scale image of size mxn
    Output:p: interpolated pixel value at X    
    """
    # find nearest four pts
    A=[int(np.floor(x[0])),int(np.floor(x[1]))]
    B=[int(np.ceil(x[0])),int(np.floor(x[1]))]    
    C=[int(np.floor(x[0])),int(np.ceil(x[1]))]
    D=[int(np.ceil(x[0])),int(np.ceil(x[1]))]
    # find pixel values at these pts
    a=img[A[1],A[0]]
    b=img[B[1],B[0]]
    c=img[C[1],C[0]]
    d=img[D[1],D[0]]
    #use formula for bilinear
    del_l=x[0]-A[0]
    del_k=x[1]-A[1]
    p=(1-del_k)*(1-del_l)*a + (1-del_k)*del_l*b + del_k*(1-del_l)*c + del_k*del_l*d    
    p=round(p,3) #roundiing to three digits will take care of cases when del_k,de_l are close to 0 or 1
    return(p)
#=====================================HW7=====================================#

#=====================================HW6=====================================#
#%%
# function for contour extraction from binary mask
def contour_img(img,option,sigma,N):
    """
    Using 8-neighbours for defining foreground
    Input: img= binary image mxn
            option= 0 or 1 for gaussian smoothing of binary image
            prior to extraction
            sigma= std dev for gaussian filter
            N: size of gaussain filter
    Output: cont_img= contour image (binary image)
    """
    m=img.shape[0];n=img.shape[1]
    # do gaussian smoothing based on option value
    if option==1:
        img = cv2.GaussianBlur(img,(5,5),sigma)
    # create empty image
    cont_img=np.zeros((m,n))
    N=3# size of neighbours window 
    pad_img=np.zeros((m+2*N//2,n+2*N//2))
    pad_img[N//2:-N//2,N//2:-N//2]=img
    # loop over img xy coordinates
    for i in range(m):
        for j in range(n):
            # make NxN window 
            window=pad_img[i:i+2*N//2,j:j+2*N//2]
            # window center should be 1 for contour point
            if window[1,1]==1:
                # if all 1s in window then not a contour point
                if np.sum(window)!=N**2:
                    cont_img[i,j]=1                
    return(cont_img)
#%%
# function for texture based segmentation
def textured_img(img,N):
    """
    Input: img: grayscale image matrix mxn
            N: list of 3x1 specifying sizes of window 
            all N_i are assumed to be odd numbers
    Output: text_img: mxnx3 image matrix with window variances as pixel 
                        values.
    """
    m=img.shape[0];n=img.shape[1]
    #define empty texture image
    text_img=np.zeros((m,n,3))
    #loop over pixel pages in text_img
    for i in range(3):
        #pad the original image using Ni
        pad = N[i]//2
        pad_img=cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
#        pad_img=np.zeros((m+2*(N[i]//2),n+2*(N[i]//2)))
#        pad_img[(N[i]//2):-(N[i]//2),(N[i]//2):-(N[i]//2)]=img
        #loop over xy coordinates of text_img
        for j in range(m):
            for k in range(n):
                window=pad_img[j:j+2*(N[i]//2),k:k+2*(N[i]//2)]
                text_img[j,k,i]=np.var(window)
        #convert text_img pixels to the range of 0-255
        min_x=np.min(text_img[:,:,i])
        max_x=np.max(text_img[:,:,i])
        text_img[:,:,i]=(255/(max_x-min_x))*(text_img[:,:,i]-min_x*np.ones((m,n)))
    #convert text_img pixel values to integers
    text_img=text_img.astype(int)
    return(text_img)
#%%
# function for inverting a biinary image
def or_img(img1,img2):
    """
    Input: img1,2: binary images
    output: logical OR of image
    """
    op=(img1+img2-img1*img2)
    return(op)
#%%
# function for inverting a biinary image
def invert_img(img):
    """
    Input: img: single color channel image
    output: inverted color matrix
    """
    op=255*np.ones((img.shape[0],img.shape[1]))-img
    return(op)
#%%
#function for finding not a binary image
def not_img(mat):
    """
    Input: mat:2d matrix with only ones and zeros
    Output: logical not of mat
    """
    op=np.ones((mat.shape[0],mat.shape[1]))
    op=op-mat
    return(op)
#%%
# function for generating binary mask
def img_mask(img,L,k):
    """
    Input:img: image matrix mxn
            L: number of levels in histogram
            k: separation threshold using Otsu
    Output:mask: binary image size mxn
    """
    m=np.shape(img)[0];n=np.shape(img)[1]
    mask=np.zeros((m,n))
    nbins=256/L
    for i in range(m):
        for j in range(n):
            if img[i,j]//nbins>=k:
                mask[i,j]=1
    return(mask)
#%%
def Otsu(img,L,N):
    """
    Input: img: image matrix mxn
            L: number of levels in histogram
            N: number of iterations to be performed
    Outut: k_star: threshold that separates foreground and background
    """
    # get initial PDF of image pixels
    h=img_histogram(img,L)
    p=h/np.sum(h)
    p_ini=p
    # find u_t total mean
    ip=np.arange(1,L+1)
    u_t=ip@p
    # loop over number of iterations
    k_star=[0]
    ipnew=ip
    for iN in range(N):
        # initialize omega,u sigma as zeros        
        omega=np.zeros((L-k_star[iN],1))
        u=np.zeros((L-k_star[iN],1))
        sigma2_b=np.zeros((L-k_star[iN],1))
        for k in range(L-k_star[iN]):
            omega[k]=np.sum(p[:k])
            u[k]=ipnew[:k]@p[:k]
            if omega[k]>0 and omega[k]<1:
                sigma2_b[k]= ((u_t*omega[k]-u[k])**2)/(omega[k]*(1-omega[k])) 
        # find k_star at max sigma2_b value
        k_star.append(k_star[-1]+np.argmax(sigma2_b))
        # find image with pixel values greater than k_star only
        # find new probability in the region k_star-L
        hnew=h[k_star[-1]:L]
        ipnew=np.arange(k_star[-1]+1,L+1)
        p=hnew/np.sum(hnew)        
        u_t=ipnew@p
    # make plots
    plt.plot(ip,p_ini)
    plt.plot([k_star,k_star],[0,np.max(p_ini)])
    plt.title('PDF for img with '+str(L)+' levels')
    plt.xlabel(str(L)+' levels')
    plt.ylabel('probability')
    plt.show()
    return(k_star[-1])
#%%
# function for generating histogram of image pixels
def img_histogram(img,L):
    """
    Input: img: image matrix mxn
            L: number of levels in histogram
    Outut: h:Lx1 vector with histogram values ni/N
    """
    nbins=256/L
    h=np.zeros((L,1))
    # raster scan of image to read pixel values
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            #update ith bin in p by 1, where i depends on img[i,j]
                h[int(img[i,j]//nbins)]+=1
    return(h)
#=====================================HW6=====================================#

#=====================================HW5=====================================#
#%%
#function for panoramic image generation
def Panorama(readpath,savepath,savename,plot):
    """
    Input:  readpath: string type path of a folder containing all images ...
            named sequentially as 1.jpg,2,3.... folder should have only images
            assuming all images are of the same sizes m,n
            savepath: saving location of the panoramic image, string
            savename: name of the saved image, string
            plot: 1 or 0: 1= plot pairwise images with inliers an outliers
    Output: saved image at savepath location , empty return
    """
    print('panorama begins')
    img_names=os.listdir(readpath)
    print(img_names)
    N=len(img_names)#total number of images
    # read all images in the sequence 1,2,3,4....
    img=[];color_img=[]
    for i in range(N):
        temp=cv2.imread(readpath+img_names[i],0)
        temp2=cv2.imread(readpath+img_names[i],-1)
        img.append(temp)
        color_img.append(temp2)
    m=np.shape(img[0])[0];n=np.shape(img[0])[1]
    print('Images read')
    # find homographies sequentially ie H12, H23, H34, ...
    H=[]
    for i in range(N-1):
        temp=AutoHomoCalc(img[i],img[i+1],(plot,savepath+savename+str(i)+str(i+1)),color_img[i],color_img[i+1])
        H.append(temp)
    print('Individual Homographies calculated')
    # find middle image: 
    if N%2==0:
        mid=int(N/2)
    else:
        mid=int((N+1)/2)
    #find homographies to middle image ie projecting to mid-1 image: numbering from 0-(N-1)
    H_to_mid=[]
    for i in range(N-1):
        temp=np.identity(3)
        if i<mid-1:
            for j in range((mid-1)-i):
                temp=np.array(H[i+j])@temp
            H_to_mid.append(temp)
        else:
            for j in range(i-(mid-1)+1):
                temp=np.array(H[(mid-1)+j])@temp
            H_to_mid.append(np.linalg.inv(temp))
    # add H=I for center image
    if N%2==0:
        H_to_mid.insert(mid,np.identity(3))
    else:
        H_to_mid.insert(mid-1,np.identity(3))
    # find transformed corners of all images in the plane of mid
    corners=np.array([[0,0,1],
                      [n,0,1],
                      [n,m,1],
                      [0,m,1]])
    all_corners=np.kron(np.ones((N,1)),corners)# col vector 4Nx3
    all_corners=all_corners.T # 3x4N vector last row all 1
    all_H=np.zeros((3,3*N)) #row vector 3x3N: need as row vector only
    for i in range(N): all_H[:,3*i:3*i+3]=H_to_mid[i] 
    all_H_inv=np.zeros((3,3*N))#row vector 3x3N: need as row vector only
    for i in range(N): all_H_inv[:,3*i:3*i+3]=np.linalg.inv(H_to_mid[i]) 
    
    temp=np.zeros((3*N,4*N))
    for i in range(N): temp[3*i:3*i+3,4*i:4*i+4]=all_corners[:,4*i:4*i+4]
    tr_corners=all_H@temp# tarnsformed corners 3x4N last row all 1
    # scale tr_corners to real coordinates
    tr_corners=tr_corners.T # 4Nx3 
    tr_corners=np.linalg.inv(np.diag(tr_corners[:,-1]))@tr_corners    
    #find canvas size
    xmin=int(np.floor(np.min(tr_corners[:,0])))
    xmax=int(np.ceil(np.max(tr_corners[:,0])))
    deltax=xmax-xmin
    ymin=int(np.floor(np.min(tr_corners[:,1])))
    ymax=int(np.ceil(np.max(tr_corners[:,1])))
    deltay=ymax-ymin
    #create empty canvas 
    canvas=np.zeros(((ymax-ymin),(xmax-xmin),3)) #filan image of this size, no scaling
    print('canvas size:'+str(ymax-ymin)+'x'+str(xmax-xmin))
    # loop over canvas indices to assign pixel values
    distype='RoundDown'
    for i in range(deltay):
        for j in range(deltax):
            #transform [j,i] on canvas to the block of interest
            x_tr=xmin+j
            y_tr=ymin+i
            hc_mp=[x_tr,y_tr,1] # hc coordinate in mid plane
            #find transformed back hc coordinate
            hc_op=all_H_inv@(np.kron(np.identity(N),hc_mp)).T # hc own plane 3xN
            #scale to convert to real coordinates
            hc_op=hc_op.T # Nx3
            hc_op=np.linalg.inv(np.diag(hc_op[:,-1]))@hc_op
            x=hc_op[:,:-1] # only x and y coordinates in all planes            
            #find if coordinates lie in the image region of the specific plane
            count=0
            p=np.zeros((1,3))#pixel values
            for k in range(N):
                temp_x=x[k,:]
                if temp_x[0]>0 and temp_x[0]<n and temp_x[1]>0 and temp_x[1]<m:
                    p=p+InterpolatePixel(temp_x,color_img[k],distype)
                    count+=1
                    break
#             take average of summed pixel values and make integer
#            if count>0:
#                p=p/count
#            p=np.floor(p)
#            p=p.astype(int)
            #assign pixel value
            canvas[i,j,:]=p
    # write image
    cv2.imwrite(savepath+savename+'.jpg',canvas)
    return()
#%%
# function for automatic calculation of homography between two images
# using SIFT interest points, RANSAC, and Non-Linear Least squares Levenberg marqaudt
def AutoHomoCalc(img1,img2,options,img1c,img2c):
    """
    Input: img1,img2: grayscale image matrices
    img1c,img2c: colored images only for plotting purposes    
    options=tuple(plot,name)        
    plot: 0 or 1, 1= make a plot of inliers
    name: name of saved image, full name with path            
    Output: H: homograhy
    """
    # find sift points of images
    pts1,des1=SIFT_detector(img1)
    pts2,des2=SIFT_detector(img2)
    # find euclidean matches between images
    match_pts1,match_pts2=Euclidean_sift_match(pts1,des1,pts2,des2) # returns list of lists
    # find inlier points using RANSAC    
    in_indices=RANSAC(match_pts1,match_pts2)  # returns array of indices
    #store inliers and outlier separately
    in_pts1=np.array(match_pts1)
    in_pts1=in_pts1[in_indices,:]
    out_pts1=match_pts1
    for i in sorted(in_indices,reverse=True): del out_pts1[i]
    out_pts1=np.array(out_pts1)
    
    in_pts2=np.array(match_pts2)
    in_pts2=in_pts2[in_indices,:]
    out_pts2=match_pts2
    for i in sorted(in_indices,reverse=True): del out_pts2[i]
    out_pts2=np.array(out_pts2)
    ## plot inliers(green) and outliers(red) on one image and save image
    if options[0]==1:
        plot_inliers(img1c,img2c,in_pts1,out_pts1,in_pts2,out_pts2,options[1])
        
    # find Initial homography using Linear Least sqaures with all inliers
    H=HMat_Pts(in_pts1,in_pts2)
    # Non-Linear Least Squares fit to find better Homography
    # find intial solution for LM
    p0=np.reshape(H,np.shape(H)[0]*np.shape(H)[1]) # p0=[h11,h12,h13,h21,h22,h23,h31,h32,h33]
    # call Least squares with cost function as argument
    optim_results=least_squares(cost_fun,p0,method='lm',args=(in_pts1,in_pts2))
#    print('---------------------LM results-------------------------')
#    print('solution is :'+str(optim_results['x']))
#    print('min function value is:'+str(optim_results['cost']))
#    print('gradient is:'+str(optim_results['grad']))
#    print('number of function evalutaions were:'+str(optim_results['nfev']))
#    print('termination status:'+str(optim_results['status']))
#    print('termination message:'+str(optim_results['message']))
    
    p=optim_results['x']
    H=np.reshape(p,(3,3))
    H=H/H[-1,-1]
    return(H)
#%%
# cost function for Non-Linear Least Squares Fit
def cost_fun(p,in_pts1,in_pts2):
    """
    We are estimating homography such that H*in_pts1=in_pts2
    Input: p is an array of size 9 in the format np.array([h11,h12,h13,h21,h22,h23,h31,h32,h33])
           in_pts1,in_pts2: inlier points as np.array([[x1,y1],[x2,y2],...])
    Output: scalar cost function value: sum of sqaures of error 
    """
    #reshape p to form H
    H=np.reshape(p,(3,3))    
    #convert in_pts1 to hc representation
    in_pts1_hc=np.ones((np.shape(in_pts1)[0],3))
    in_pts1_hc[:,:-1]=in_pts1
    # find estimated point2 using homography
    est_hc= H@in_pts1_hc.T#estimated point as stacked col vectors
    est_hc=est_hc.T #points as row vectors
    #scale with 3rd ordinate to get real point
    est=np.linalg.inv(np.diag(est_hc[:,-1]))@est_hc
    est=est[:,:-1] # removing third column
    # define error and cost function
    X=np.reshape(in_pts2,np.shape(in_pts2)[0]*np.shape(in_pts2)[1]) #col vector
    f=np.reshape(est,np.shape(est)[0]*np.shape(est)[1]) #col vector
    error=X-f #col vector
    cost=error# Least sqaures function with lm option requires a M dimensional vector as objective function
    #cost= error.T@error
    return(cost)

#%%
def plot_inliers(img1,img2,in_pts1,out_pts1,in_pts2,out_pts2,name):
    """
    Input: img1,img2: colored images of the same sizes mxn
           in_pts1,in_pts2: np array [[x,y],..] inliers to be plotted in green circle
           out_pts1,out_pts2: np array [[x,y],..] outliers to be plotted in red circles
           name: filename of image with full path
    Output:  save the image, empty return
    """
    m=np.shape(img1)[0];n=np.shape(img1)[1]
    #convert in_pts and out_pts to integers
    in_pts1=in_pts1.astype(int)
    in_pts2=in_pts2.astype(int)
    out_pts1=out_pts1.astype(int)
    out_pts2=out_pts2.astype(int)
    # make empty canvas of twice width
    img=np.zeros((m,2*n,3))
    #copy original images onto the two halves of canvas
    img[:,:n,:]=img1
    img[:,n:,:]=img2
    #plot inliers in green
    r=4
    for i in range(len(in_pts1)):
        cv2.circle(img,tuple(in_pts1[i,:]),r,(0,255,0),-1)# left image
        cv2.circle(img,(n+in_pts2[i,0],in_pts2[i,1]),r,(0,255,0),-1)# right image
        cv2.line(img,tuple(in_pts1[i,:]),(n+in_pts2[i,0],in_pts2[i,1]),(0,255,0),1)#line joining inliers   
    #plot outliers in red
    for i in range(len(out_pts1)):
        cv2.circle(img,tuple(out_pts1[i]),r,(0,0,255),-1)# left image
        cv2.circle(img,(n+out_pts2[i,0],out_pts2[i,1]),r,(0,0,255),-1)# right image
    #save image
    cv2.imwrite(name+'.jpg',img)
    return()
#%%
# function for Random sampling and consensus
def RANSAC(pts1,pts2):
    """
    Input: pts1,pts2: matching points with inliers and outliers 
                      format:[[x1,y1],[x2,y2],....] list of lists of points 
                      
    Output: in_indices_store: indices of inlier points of image  1 and image 2 respectively
                            format:[index1,index2,....] list of index numbers
    """
    #----define RANSAC parameters:-----#
    p=0.99  # probability that at least one of the N trials will be free of outliers
    n=6     # minimal set of correspondences chosen randomly for constructing homography
    delta=2 # decision threshold to construct inlier set    
    e=0.1   # probability that a correspondence is an outlier
    N=int(np.ceil(np.log(1-p)/np.log(1-(1-e)**n))) #number of iterations needed
    #----------------begin-------------------#
    n_total=len(pts1) # total number of correspondences
    # convert pts lists to array and into homogeneous form
    pts1_hc=np.ones((n_total,3))
    pts1_hc[:,:-1]=pts1
    pts2_hc=np.ones((n_total,3))
    pts2_hc[:,:-1]=pts2   

    badinliers=True
    while badinliers:
        count=0
        size_old=0        
        while count<N:
            #randomly sample n points from pts1 and pts2
            rand_indices=np.random.randint(n_total,size=n)
            ipts1=pts1_hc[rand_indices,:]
            ipts2=pts2_hc[rand_indices,:]
            # find homography between these points: linear least sqaures
            iH=HMat_Pts(ipts1[:,:-1],ipts2[:,:-1])#gives H12*ipts1=ipts2: push fwd
            # find estimated pts2 and error distance
            est_pts2=iH@(pts1_hc.T) # est pts as col vectors
            est_pts2=est_pts2.T # est pts as row vectors
            # scale est_pts2 to get third col as 1
            temp=est_pts2[:,-1]
            est_pts2=np.linalg.inv(np.diag(temp))@est_pts2
            error=pts2_hc[:,:-1]-est_pts2[:,:-1]
            distance=np.power(error,2)
            distance=distance@np.ones((2,1))
            distance=np.sqrt(distance)
            # consensus
            in_indices=np.where(distance<=delta)[0] # inlier indices
            size_new=len(in_indices) #size of ith inlier set           
            if size_new>size_old:
                in_indices_store=in_indices #store the set of indices with largest size
                size_old=size_new       
            count+=1
        if len(in_indices_store)>(1-e)*n_total:
            badinliers=False
        else:
            e=e*2 #bad dataset than what was assumed
            #update N ie more iterations of RANSAC
            N=int(np.ceil(np.log(1-p)/np.log(1-(1-e)**n))) 
    print('RANSAC max inlier set was'+str(len(in_indices_store)))
    return(in_indices_store)
#%%
#function for Euclidean match for SIFT points 
#V1: for HW5 script: make sure V0 is commented out!!
def Euclidean_sift_match(pts1,des1,pts2,des2,tr):
    """
    Input: pts1,pts2= [[x,y],[],..] list of lists of interest points
           des1,des2 = vector of 128 size, sift descriptor stacked as row vectors
           tr:  threshold
    Output: match_pts1,match_pts2= [[x1,y1],[x2,y2],....] list of lists of points 
                                    both have same sizes
            Euclidean distance is less than a dynamic threshold
    """
    match_pts1=[]
    match_pts2=[]
    eu=np.zeros((len(pts1),len(pts2)))    
    for i in range(len(pts1)):
        for j in range(len(pts2)):
            eu[i,j]=np.linalg.norm(des1[i,:]-des2[j,:])
    
    eu=eu/np.min(eu)
    dy_threshold=tr
    for i in range(len(pts1)):            
        eu_min=np.min(eu[i,:])
        if eu_min<dy_threshold:
            j_min=np.argmin(eu[i,:])
            match_pts1.append(pts1[i])
            match_pts2.append(pts2[j_min])
    print('size of Euclidean match was:'+str(len(match_pts1)))
    return(match_pts1,match_pts2)   
#%%
#function for SIFT interest points detection
def SIFT_detector(img):
    """
    detects SIFT interest points using openCV function and returns pts and des
    in my particular format.
    Input: img: image matrix
    Output: pts: [[x,y],...] : list of lists of pt coordinates as row vectors
            des: 128 bit  SIFT descriptor matrix arranged as stacked row vector 
            =[d1,d2,d3,...] di is row vector of size 128
    """
    #sift key points
    sift=cv2.xfeatures2d.SIFT_create() #defining structure
    sift_pts,des=sift.detectAndCompute(img,None)
    pts=convert_SIFT_myFormat(sift_pts)
    return(pts,des)
#%%
#function for converting sift key points into my format 
#V1: to be used for HW5: make sure V0 is commented out!!
def convert_SIFT_myFormat(kp):
    """
    Input: kp: keypoint structure given by SIFT, point coordinates in kp.pt
    Output: [[x,y],...] : list of lists of pt coordinates as row vectors
    """
    mykp=[]
    for ikp in kp: 
        mykp.append([ikp.pt[0],ikp.pt[1]])
    return(mykp)
#=============================================================================#
#%%
#======================================HW4====================================#

##function for Euclidean match for SIFT points 
##V0: for HW4 script
#def Euclidean_sift_match(pts1,des1,pts2,des2):
#    """
#    Input: pts1,pts2= [(x,y),(),..] list of tuples of interest points
#           des1,des2 = vector of 128 size, sift descriptor
#    Output: pt_pairs= [[(x1,y1),(x2,y2)],....] list of list of point pair tuples such that
#            Euclidean distance is less than a dynamic threshold
#    """
#    pt_pairs=[]
#    sq_eu=np.zeros((len(pts1),len(pts2)))    
#    for i in range(len(pts1)):
#        for j in range(len(pts2)):
#            sq_eu[i,j]=np.linalg.norm(des1[i,:]-des2[j,:])
#    
#    sq_eu=sq_eu/np.min(sq_eu)
#    dy_threshold=2
#    print(dy_threshold)
#    for i in range(len(pts1)):            
#        sq_eu_min=np.min(sq_eu[i,:])
#        if sq_eu_min<dy_threshold:
#            j_min=np.argmin(sq_eu[i,:])
#            pt_pairs.append([pts1[i],pts2[j_min]])
#    return(pt_pairs)            

##function for converting sift key points into my format 
##V0: was used for HW3 script
#def convert_SIFT_myFormat(kp):
#    """
#    Input: kp: keypoint structure given by SIFT, point coordinates in kp.pt
#    Output: [(x,y),...] : list of tuples of coorinates
#    """
#    mykp=[]
#    for ikp in kp: 
#        #convert points to floored integer
#        mykp.append((int(ikp.pt[0]),int(ikp.pt[1])))
#    return(mykp)

#function for Squared Sum Differences for Point correspondences
def SSD(img1,pts1,img2,pts2):
    """
    Input: img1, img2 = grayscale images of a scene from different viewpoints, sizes are identical
            pts1,pt2 =[(x,y),...] list of tuples interest points detected for img1 and img2 respectively
    Output: pt_pairs= [[(x1,y1),(x2,y2)],....] list of list of point pair tuples which pass a certain
            confidence threshold
    """
    M=51 # window is of size M
    pad = int((M-1)/2)
    #pad img1 and img2
    img1_pad=cv2.copyMakeBorder(img1,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    img2_pad=cv2.copyMakeBorder(img2,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    # loop over length of pts1 and pts2: exhaustive search
    pt_pairs=[]
    SSD=np.zeros((len(pts1),len(pts2)))
    i=0
    for ipts1 in pts1:
        j=0
        for ipts2 in pts2:
            #make window of size MxM at both the point locations 
            win1=img1_pad[ipts1[1]+pad-pad:ipts1[1]+pad+pad+1,ipts1[0]+pad-pad:ipts1[0]+pad+pad+1]
            win2=img2_pad[ipts2[1]+pad-pad:ipts2[1]+pad+pad+1,ipts2[0]+pad-pad:ipts2[0]+pad+pad+1]
            #find SSD
            SSD[i,j]=np.sum(np.power((win1-win2),2))
            j+=1
        i+=1
    #using a dynamic threshold as criteria for matching points
    #converting SSD to 0-1 range and then using a dynamic threshold
    SSD=(SSD-np.min(SSD)*np.ones(np.shape(SSD)))/(np.max(SSD)-np.min(SSD))
    dy_threshold=0.05
    i=0
    for ipts1 in pts1:        
        #check if SSD_max is less than dy_threshold
        SSD_min=np.min(SSD[i,:])
        if SSD_min<dy_threshold:
            j_min=np.argmin(SSD[i,:])
            pt_pairs.append([ipts1,pts2[j_min]])
        i+=1
    return(pt_pairs)


#function for plotting image with interest points of two scenes and lines joining matching points
def plot_matching_points(img1,pts1,img2,pts2,pt_pairs):
    """
    Input: img1,img2: colored image matrices of same size of the same scene from diff viewpoints
            pts1,pts2: [(x,y),..] list of tuples of intresting points in img1 and img2 respectively
            pt_paris:[ [(x1,y1), (x2,y2)],... ] list of list of tuples of point pairs with matching metric
    OutPut: img3: image matrix with both scenes placed side by side horizontally,
            with marked points of interests in both the images and lines joining elements of pt_pairs
    """
    m=np.shape(img1)[0];n1=np.shape(img1)[1];n2=np.shape(img2)[1]
    img3=np.zeros((m,n1+n2,3))
    #assign img1 and img2 pixels to img3
    img3[:,:n1,:]=img1
    img3[:,n1:,:]=img2
    #draw interest points on both images
    for ipts1 in pts1: cv2.circle(img3,(int(ipts1[0]),int(ipts1[1])),2,(0,0,255),-1)    
    for ipts2 in pts2:
        x=n1+ipts2[0]
        cv2.circle(img3,(int(x),int(ipts2[1])),5,(0,0,255),-1)    
    #draw line joining matching points
    for i in range(len(pt_pairs)):
        cv2.line(img3,(int(pt_pairs[i][0][0]),int(pt_pairs[i][0][1])),
                    (int(pt_pairs[i][1][0])+n1,int(pt_pairs[i][1][1])),(0,255,0),2)    
    return(img3)

#function for Normalized Cross Correlation for point correspondences
def NCC(img1,pts1,img2,pts2):
    """
    Input: img1, img2 = grayscale images of a scene from different viewpoints, sizes are identical
            pts1,pt2 =[(x,y),...] list of tuples interest points detected for img1 and img2 respectively
    Output: pt_pairs= [[(x1,y1),(x2,y2)],....] list of list of point pair tuples which pass a certain
            confidence threshold
    """
    M=31 # window is of size M
    pad = int((M-1)/2)
    #pad img1 and img2
    img1_pad=cv2.copyMakeBorder(img1,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    img2_pad=cv2.copyMakeBorder(img2,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    # loop over length of pts1 and pts2: exhaustive search
    threshold=0.95
    pt_pairs=[]
    
    for ipts1 in pts1:
        NCC=np.zeros((len(pts2),1))
        k=0
        for ipts2 in pts2:
            #make window of size MxM at both the point locations 
            win1=img1_pad[ipts1[1]+pad-pad:ipts1[1]+pad+pad+1,ipts1[0]+pad-pad:ipts1[0]+pad+pad+1]
            win2=img2_pad[ipts2[1]+pad-pad:ipts2[1]+pad+pad+1,ipts2[0]+pad-pad:ipts2[0]+pad+pad+1]
            #find means for both the window
            mean1=np.mean(win1)
            mean2=np.mean(win2)
            #find NCC 
            num=np.sum(np.multiply( (win1-mean1*np.ones((M,M))),(win2-mean2*np.ones((M,M)))) )
            deno=np.sum(np.power((win1-mean1*np.ones((M,M))),2))
            deno=deno*(np.sum(np.power((win2-mean2*np.ones((M,M))),2)))
            deno=np.sqrt(deno)
            NCC[k]=num/deno
            k+=1
        #check if NCC_max is greater than threshold
#        NCC=(NCC-np.min(NCC)*np.ones(np.shape(NCC)))/(np.max(NCC)-np.min(NCC))
        NCC_max=np.max(NCC)
        if NCC_max>threshold:
            i_max=np.argmax(NCC)
            pt_pairs.append([ipts1,pts2[i_max]])
            #drop pts2[i_max] from pts2 list
#            del pts2[i_max]            
    return(pt_pairs)

#harris corner detection
def Harris_Corners(img,scale):
    """
    Input: img= image object for which corners are to be detected
        scale= smoothing scale for image 
    Output: pts=[[x,y],[],[]..] List of corner coordinate
        Ix: xderivative Image
        Iy: y derivative image
    """
    #find derivatives of image dx and dy
    dx,dy=Haar_filter(scale)
    #convolve dx dy to get Ix Iy
    Ix=cv2.filter2D(img,-1,dx) #by default pixelExtrapolation at borders
    Iy=cv2.filter2D(img,-1,dy)
    #pad Ix and Iy for next step
    #find pad
    window_size=int(np.ceil(5*scale))
    if window_size%2==0: window_size+=1
    pad=int((window_size-1)/2.0)
    Ix_pad=cv2.copyMakeBorder(Ix,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    Iy_pad=cv2.copyMakeBorder(Iy,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    R=np.zeros(np.shape(img))#stores info of eigenvalue ratios
    #loop over pixels and update R if a corner
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            #take slice of Ix Iy with window of 2*pad+1 size
            Ix_window=Ix_pad[(i+pad)-pad:(i+pad)+pad+1,(j+pad)-pad:(j+pad)+pad+1]
            Iy_window=Iy_pad[(i+pad)-pad:(i+pad)+pad+1,(j+pad)-pad:(j+pad)+pad+1]
            #find Ix**2 Iy**2 IxIy
            Ix2=np.sum(np.multiply(Ix_window,Ix_window))
            Iy2=np.sum(np.multiply(Iy_window,Iy_window))
            Ixy=np.sum(np.multiply(Ix_window,Iy_window))
            #now C=[[Ix2,Ixy],[Ixy,Iy2]] check if a corner
            #using determinant of C as check for rank2
            detC=Ix2*Iy2-(Ixy**2)
            if detC>0:
                R[i,j]=detC/((Ix2+Iy2)**2)                
    #remove points on the edges of image upto a pixel wdith of p pixels
    p=10
    R[:p,:]=0*R[:p,:]#top border
    R[-p:,:]=0*R[-p:,:]#bottom border
    R[:,:p]=0*R[:,:p]#left border
    R[:,-p:]=0*R[:,-p:]#right border
    #Suppression of non-maxium points in vicinity of a maxima
    #so that we dont get a cluster of points at a single location
    non_max_window=31 #pixels, should be odd
    threshold=np.mean(R)  
    pad = int((non_max_window-1)/2)
    R_pad=cv2.copyMakeBorder(R,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    pts=[] 
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if R[i,j]>0:
                R_window=R_pad[(i+pad)-pad:(i+pad)+pad+1,(j+pad)-pad:(j+pad)+pad+1]
                if R[i,j]==np.max(R_window) and R[i,j]>threshold:
                     pts.append((j,i))# x is j col y is i row
    return(pts,Ix,Iy)

#function for haar filter
def Haar_filter(scale):
    """
    Input: scale= defines the sigma for the filter and also the size of filter
    Output: dx, dy haar filter kernels
    """
    N=int(np.ceil(4*scale)) # size of kernel is smallest even integer greater than 4*sigma
    if N%2!=0:
        N=N+1 #if N was odd: add 1 to it
    dx=np.ones((N,N))
    dy=np.ones((N,N))
    dx[:,:int(N/2)]=-1*dx[:,:int(N/2)]
    dy[:int(N/2),:]=-1*dy[:int(N/2),:]
    return(dx,dy)  

#=============================================================================#

#%%
#=======================================HW3===================================#
# H using one step using SVD
def HMat_OneStep(imPts):
    """
    Input:
        imPts: 20x2 [P1,P2,P3,P4,...] array with information of 20 points in image plane
        can be more than 20 also but always in multiples of 4
        pts ordered such that line P1P2 and line P3P4 is prependicular
        therefore we have atleast 5 sets of prependicular lines
    Output:
        H: 3x3 homography matrix which will correct both 
        affine and projective distortion in one step
        H*world_pt = image_pt : push fwd
    """
    N=int(np.shape(imPts)[0]/4)# number of line pairs
    #convert imPts to HC
    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
    im_pts_hc[:,:-1]=imPts
    # find lines
    L=np.zeros((N,3));M=np.zeros((N,3))
    for i in range(N):
        L[i,:]=np.cross(im_pts_hc[4*i,:],im_pts_hc[4*i+1,:])
        M[i,:]=np.cross(im_pts_hc[4*i+2,:],im_pts_hc[4*i+3,:])
    #normalize lines: not really needed  and experimentation confirmed it: also it increases the condition number of A
#    L=np.linalg.inv(np.diag(L[:,-1]))@L
#    M=np.linalg.inv(np.diag(M[:,-1]))@M
    M_reshape=np.reshape(M,(N*3))
    # fill out A 
    A=np.zeros((N,6))
    temp=np.multiply(L,M)
    A[:,:3]=temp
    #permutation matrices
    P1=[[0,1,0],[1,0,0],[0,0,0]]
    P2=[[0,0,1],[0,0,0],[1,0,0]]
    P3=[[0,0,0],[0,0,1],[0,1,0]]
    A[:,3]= L@np.kron(np.ones(N),P1)@M_reshape  #l1m2+l2m1
    A[:,4]= L@np.kron(np.ones(N),P2)@M_reshape  #l1m3+l3m1
    A[:,5]= L@np.kron(np.ones(N),P3)@M_reshape  #l2m3+l3m2
    #solve for C using SVD 
    uc,dc,vct=np.linalg.svd(A,full_matrices=True)
    C=vct[-1,:]  #C=[a,c,f,b/2,d/2,e/2]
#    C=C/C[2] #make f as 1
    C=np.array([[C[0],C[3],C[4]],
       [C[3],C[1],C[5]],
       [C[4],C[5],C[2]]])
#   #find H from C using SVD
    u,d,v=np.linalg.svd(C,full_matrices=True)
    print(d)
    H=u
    return(H)



## H using one step: w/o SVD
#def HMat_OneStep(imPts):
#    """
#    Input:
#        imPts: 20x2 [P1,P2,P3,P4,...] array with information of 20 points in image plane
#        can be more than 20 also but always in multiples of 4
#        pts ordered such that line P1P2 and line P3P4 is prependicular
#        therefore we have atleast 5 sets of prependicular lines
#    Output:
#        H: 3x3 homography matrix which will correct both 
#        affine and projective distortion in one step
#        H*world_pt = image_pt : push fwd
#    """
#    N=int(np.shape(imPts)[0]/4)# number of line pairs
#    #convert imPts to HC
#    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
#    im_pts_hc[:,:-1]=imPts
#    # find lines
#    L=np.zeros((N,3));M=np.zeros((N,3))
#    for i in range(N):
#        L[i,:]=np.cross(im_pts_hc[4*i,:],im_pts_hc[4*i+1,:])
#        M[i,:]=np.cross(im_pts_hc[4*i+2,:],im_pts_hc[4*i+3,:])
#    #normalize lines: not really needed  and experimentation confirmed it: also it increases the condition number of A
##    L=np.linalg.inv(np.diag(L[:,-1]))@L
##    M=np.linalg.inv(np.diag(M[:,-1]))@M
#    print('L is \n')
#    print(L)
#    print('M is\n')
#    print(M)
#    M_reshape=np.reshape(M,(N*3))
#    # fill out A and b
#    A=np.zeros((N,5))
#    b=np.zeros((N,1))
#    temp=np.multiply(L,M)
#    A[:,:2]=temp[:,:-1]  #first two columns with l1m1, l2m2
#    b=-1*temp[:,-1] #RHS = -l3m3
#    #permutation matrices
#    P1=[[0,1,0],[1,0,0],[0,0,0]]
#    P2=[[0,0,1],[0,0,0],[1,0,0]]
#    P3=[[0,0,0],[0,0,1],[0,1,0]]
#    A[:,2]= L@np.kron(np.ones(N),P1)@M_reshape  #l1m2+l2m1
#    A[:,3]= L@np.kron(np.ones(N),P2)@M_reshape  #l1m3+l3m1
#    A[:,4]= L@np.kron(np.ones(N),P3)@M_reshape  #l2m3+l3m2
#    #solve for coeffs using psuedo inverse
#    C=(np.linalg.inv((A.T)@A)@A.T)@b  #C=[a,c,b/2,d/2,e/2]
##    C=C/max(np.abs(C)) #not needed
#    C=np.array([[C[0],C[2],C[3]],
#       [C[2],C[1],C[4]],
#       [C[3],C[4],1]])
##    #find A and v
##    S=C[:2,:2]
##    U,D,V=np.linalg.svd(S,full_matrices=True)
##    A=V@np.diag(np.sqrt(D))@V.T
##    v=np.linalg.inv(A)@C[:2,-1]
##    # construct H
##    H=np.identity(3)
##    H[:2,:2]=A
##    H[-1,:-1]=v  #assuming t=0
##    find H from C using SVD
##    u,d,v=np.linalg.svd(C,full_matrices=True)# gives u d and v.T  
#    d,u=np.linalg.eig(C)
#    print(d)
#    H=u
#    return(H)

#Homography for Projective Correction using vanishsing Line vectorized
def Hmat_ProjectiveCorrection(imPts):
    """
    Input:
        imPts=[[X,Y],[X,Y],..] min 8 points in the order PQRSTUVW in the image plane
        sets of 4 points give 2 parallel lines in world plane
    Output: Returns the H for Projective Correction using concept of vanishing line
    such that H*im_pt=src_pt or (Inv(H)^T).VL=L_inf
    """
    N=int(np.shape(imPts)[0]/4) #number of vanishing point
    #convert imPts to HC
    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
    im_pts_hc[:,:-1]=imPts
    #find vanishing points
    L1=np.zeros((N,3))#line p1p2
    L2=np.zeros((N,3))#line p3p4
    VP=np.zeros((N,3))#intersection l1 l2
    for i in range(N):
        L1[i,:]=np.cross(im_pts_hc[4*i,:],im_pts_hc[4*i+1,:])
        L2[i,:]=np.cross(im_pts_hc[4*i+2,:],im_pts_hc[4*i+3,:])
        VP[i,:]=np.cross(L1[i,:],L2[i,:])
    #hc vector of vanishing line
    VL=np.cross(VP[0,:],VP[1,:])
    #normalizing VL
    VL=VL/VL[-1]
    print('The Vanishing Line is: '+str(VL))
    H=np.identity(3)
    H[-1,:]=VL
    return(H)


#Homography for Affine Correction using prependicular pairs
def Hmat_AffineCorrection(imPts,H_PC):
    """
    Input:
        imPts=[[X,Y],[X,Y],..] min 8 points in the order PQRSTUVW in the image plane
        Note: imPts should have projective distortion as well
        H_PC = 3x3 Homography that corrects Projective distortion st H_PC.im_pt=affine_pt ie pull-back
        Note: prependicular line pair chosen are (PQ,RS) (TU,VW) and the pairs should have different directions
    Output:purely affine H =[[A,0],[0,1]] : A is 2x2
    such that H*src_pts=im_pt         
    """
    N=int(np.shape(imPts)[0]/4)# number of line pairs
    #convert imPts to HC
    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
    im_pts_hc[:,:-1]=imPts
    #remove Projective distortion
    im_pts_hc=H_PC@im_pts_hc.T
    im_pts_hc=im_pts_hc.T
#    im_pts_hc=np.linalg.inv(np.diag(im_pts_hc[:,-1]))@im_pts_hc
    #find prependicular lines
    L=np.zeros((N,3));M=np.zeros((N,3))
    for i in range(N):
        L[i,:]=np.cross(im_pts_hc[4*i,:],im_pts_hc[4*i+1,:])
        M[i,:]=np.cross(im_pts_hc[4*i+2,:],im_pts_hc[4*i+3,:])
    #normalize lines using third hc coord: imp as info of l1 and l2 is only taken for further step
    L=np.linalg.inv(np.diag(L[:,-1]))@L
    M=np.linalg.inv(np.diag(M[:,-1]))@M
    L=L[:,:-1]
    M=M[:,:-1]
    M_reshape=np.reshape(M,(N*2))
    #fill A and b
    A=np.zeros((N,2)) # coeff matrix
    b=np.zeros((N,1)) # RHS
    temp=np.multiply(L,M)
    A[:,0]=temp[:,0]
    b=-1*temp[:,-1]
    # permutation Matrix
    P=[[0,1],[1,0]] 
    A[:,1]=L@np.kron(np.ones(N),P)@M_reshape
    #solve for S using psuedo inverse
    S=(np.linalg.inv((A.T)@A)@A.T)@b
    S=np.array([[S[0],S[1]],
                [S[1],1]])
    #find A using SVD
    u,d,v= np.linalg.svd(S,full_matrices=True)# gives u d and v.T 
    v=v.T
    A=v@np.diag(np.sqrt(d))@v.T
    # construct H, assuming t=0
    H=np.identity(3)
    H[:2,:2]=A
    return(H)


def mapFitToCanvas(Img,imPts,canvas,H,distype):
    """
    function applies homogrpahy H to source image and fits the resulting matrix to a canvas    
    Input: 
        Img: Image matrices
        canvas: can be [] or any mxn matrix
        imPts =np.array([[X,Y],[X Y]...]) image corners 4x2     
        H is a 3x3 matrix: H*world_pt=image_pt ie push forward
        distype= a string L2, sqL2,BiLinear,RoundDown,RoundUp
    Output: Function will write out the merged image    
    """
    invH=np.linalg.inv(H) 
    
    if canvas==[]:
        Pcor = np.ones((4,3))
        Pcor[:,:-1]=np.array(imPts)
        HP= invH@Pcor.T
        HP = np.linalg.inv(np.diag(HP[-1,:]))@HP.T
        #impts1 = HP[:,:-1]
        x_min=min(HP[:,0]); x_max=max(HP[:,0]);
        y_min=min(HP[:,1]); y_max=max(HP[:,1]);
        canvas=np.zeros(( int(y_max-y_min) , int(x_max-x_min) ,3))
#        canvas=np.zeros(np.shape(img1))

    resultImg=np.zeros(np.shape(canvas))    
     
    #convert imPts to hc
    src_hc=np.ones((3,np.shape(imPts)[0]))
    src_hc[:2,:]=imPts.T
    #find mapping of imPts on canvas
    im_src_hc=invH@src_hc#columns of im_src_hc are images of src_pts
    #divide each column by the value in the last index of that column to get physical coordinates
    im_src_hc=np.linalg.inv(np.diag(im_src_hc[-1,:]))@im_src_hc.T
    #now rows of im_src_hc are physical coordinates with third column all 1s
    #find parameters for coordinate transformation within canvas plane         
    xmin=min(im_src_hc[:,0])
    xmax=max(im_src_hc[:,0])
    ymin=min(im_src_hc[:,1])
    ymax=max(im_src_hc[:,1])
    deltax=xmax-xmin
    deltay=ymax-ymin
    #now find x and y limits for src image coordiantes
    src_xmin=min(imPts[:,0])
    src_xmax=max(imPts[:,0])
    src_ymin=min(imPts[:,1])
    src_ymax=max(imPts[:,1])
    # iterate over points in canvas coordinates
    for i in range(np.shape(canvas)[0]):
        for j in range(np.shape(canvas)[1]):
            #x: col index & y:row index , pt on canvas is [j,i,1]
            #transform [j,i] on canvas to the block of interest
            x_tr=xmin+(deltax/np.shape(canvas)[1])*j
            y_tr=ymin+(deltay/np.shape(canvas)[0])*i
            #find corresponding point in src image using invH
            x = np.dot(H,[x_tr,y_tr,1])
            x=np.array([x[0]/x[-1],x[1]/x[-1]])
            #now we have a non-integer point in src Img
            #check if the point lies inside the window in the source image
            if x[0]>src_xmin and x[0]<src_xmax and x[1]>src_ymin and x[1]<src_ymax:
                #find new pixel value
                cx = InterpolatePixel(x,Img,distype)
                #assign new pixel value to destImg 
                resultImg[i,j,:]=cx
            else:
                resultImg[i,j,:]=canvas[i,j,:]                
    return(resultImg)
#============================================HW3==============================#
#%%
#============================================HW2==============================#

# Homograhpy using points
def HMat_Pts(world,img):
    """
    Returns a homography mat such that H*world=img and H[2,2]=1 ie push forward
    img is on destination Image
    world is on source Image
    img,world=np.array([[x1,y1],[x2,y2],[...],[...]])
    """
    n=np.shape(world)[0]
    m=np.shape(world)[1]
#    A=np.zeros((2*n,8))
#    b=np.reshape(img.T,n*m)
    A=np.zeros((2*n,9))
    
    block1=np.zeros((n,m+1))
    block1[:,-1]=np.ones((1,n))
    block1[:,:2]=world[:,:]
    A[:n,:3]=block1
    A[n:2*n,3:6]=block1

    block2=np.multiply(world.T,-1*img[:,0])
    block3=np.multiply(world.T,-1*img[:,1])
    
    A[:n,6:8]=block2.T    
    A[n:2*n,6:8]=block3.T
    A[:,-1]=-1*np.reshape(img.T,n*m)
#    #solve for h using psuedo inverse
#    h=(np.linalg.inv((A.T)@A)@A.T)@b
#    h=np.concatenate([h,[1]])
    # solve using null space
    u,d,vt=np.linalg.svd(A)
    h=vt[-1,:]
    H=np.reshape(h,(3,3))
    H=H/H[-1,-1]
    return(H)

# function mapFitToCanvas written in HW2 superscedes this one 
def mapImage(srcImg,srcPts,destImg,destPts,H,distype,path,name):
    """    
    Input: 
        srcImg & destImg: Image matrices
        srcpts& destpts =np.array([[X,Y],[X Y]...]) 
        H is a 3x3 matrix
    Output: Function will write out the merged image    
    """
    #find Homography
    resultImg=np.zeros(np.shape(destImg))    
    invH=np.linalg.inv(H)   
    
    xmin=min(srcPts[:,0])
    xmax=max(srcPts[:,0])
    ymin=min(srcPts[:,1])
    ymax=max(srcPts[:,1])
    # iterate over points in destImg
    for i in range(np.shape(destImg)[0]):
        for j in range(np.shape(destImg)[1]):
            # i is row number and corresponds to y coordinate
            #j is col numer and corresponds to x coordinate
            x = np.dot(invH,[j,i,1])
            x=np.array([x[0]/x[-1],x[1]/x[-1]])
            #now we have a non-integer point in src Img
            #check if the point lies insisde the window in the source image
            if x[0]>xmin and x[0]<xmax and x[1]>ymin and x[1]<ymax:
                #find new pixel value
                cx = InterpolatePixel(x,srcImg,distype)# cx is a 3x1 matrix
                #assign new pixel value to destImg 
                resultImg[i,j,:]=cx
            else:
                resultImg[i,j,:]=destImg[i,j,:]
                
    cv2.imwrite(path+name+'.jpg',resultImg)
    return()

# Interpolation of colors from the source image
def InterpolatePixel(x,srcImg,distype):
    """
    Input: 
        x= n.array([X,Y]) coordinates of a point 
        srcImg=source image object a n,m,3 matrix
        distype= a string L2, sqL2,BiLinear,RoundDown,RoundUp
    Output:
        cx= size(1,3) vector of pixel values with only integer values
    The function will interpolate the pixel values of the nearest neighbours and do a rounding
    to get integers
    """
    #finding the nearest interger neighbours
    p1=[int(np.floor(x[0])),int(np.floor(x[1]))]
    p2=[int(np.ceil(x[0])),int(np.floor(x[1]))]
    p3=[int(np.ceil(x[0])),int(np.ceil(x[1]))]
    p4=[int(np.floor(x[0])),int(np.ceil(x[1]))]
    #storing value of pixels at these points 
    #x: col index & y:row index
    c1=srcImg[p1[1],p1[0],:]
    c2=srcImg[p2[1],p2[0],:]
    c3=srcImg[p3[1],p3[0],:]
    c4=srcImg[p4[1],p4[0],:]
    C=np.array([c1,c2,c3,c4])
    #storing weights for interpolation
    w=weights(np.array([p1,p2,p3,p4]),x,distype);#should return a np array 
    #interpolating
    fcx=C.T@w
    fcx=np.floor(fcx)
    cx=fcx.astype(int)
    return(cx)

def weights(pts,x,distype):
    """ 
    function will give a list of weights
    Input:  pts = np.array[[X Y],[X Y]...]
    x=np.array[X,Y]
    distype is a string with options L2, sqL2,BiLinear,RoundDown,RoundUp
    output:
        w=[w1,w2,w3,w4]
    """
    w=[]
    if distype=='L2':
        #use L2 norm for distance
        for i in range(np.shape(pts)[0]):
            w.append(np.linalg.norm(pts[i,:]-x))
        wsum=sum(w)
        w=w/wsum
        return(w)        
    elif distype=='sqL2':
        #use squared L2 norm for distance
        for i in range(np.shape(pts)[0]):
            w.append((np.linalg.norm(pts[i,:]-x))**2)
        wsum=sum(w)
        w=w/wsum
        return(w)
    elif distype=='BiLinear':
        #using bilinear shape functions as weights
        xmin=pts[0,0]
        xmax=pts[1,0]
        ymin=pts[0,1]
        ymax=pts[3,1]
        xi=2*((x[0]-xmin)/(xmax-xmin))-1
        eta=2*((x[1]-ymin)/(ymax-ymin))-1
        w.append((1-xi)*(1-eta)/4.0)
        w.append((1+xi)*(1-eta)/4.0)
        w.append((1+xi)*(1+eta)/4.0)
        w.append((1-xi)*(1+eta)/4.0)
        return(w)
    elif distype=='RoundDown':
        w=[1,0,0,0]
        return(w)
    elif distype=='RoundUp':
        w=[0,0,1,0]
        return(w)    
#============================================HW2==============================#