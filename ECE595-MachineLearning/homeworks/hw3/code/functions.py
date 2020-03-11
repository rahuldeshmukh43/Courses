l"""
ECE 595: ML-1
HW-3: functions
@author- Rahul Deshmukh
"""
# import libraries
import numpy as np

#%%
# function for calculating mean abs error
def mean_abs_error(prediction,truth):
    """
    Input: prediciton: grayscale image normalized to 0-1
           truth: grayscale image normalized to 0-1
           both are of same sizes
    Output: error: scalar value
    """
    error = 0
    M,N = np.shape(prediction)
    error = (1/(M*N))*np.sum(np.abs(prediction-truth))
    return(error)

# function for classifying non overlapping patches of an image
def classify_non_overlapping_patches(img,u,sigma,prior,user_def_label):
    """
    Input: img: np array of image in grayscale(scaled to 0-1)
    u: class means as list of 1d arrays
    sigma: class covariances as list of 2d arrays
    prior: list of priors
    user_def_label: list of label values
    Output: predicted_labels: np array of predicted label
    """
    M,N = np.shape(img)
    predicted_label = np.zeros((M,N))
    for i in range(M//8-1):
        for j in range(N//8-1):
            test_sample  = img[8*i:8*i+8,8*j:8*j+8]
            # convert test sample to a vector
            test_sample = np.reshape(test_sample,(64,1))
            # find class of the sample
            predicted_label[8*i:8*i+8,8*j:8*j+8] = classify_sample(test_sample,u,sigma,prior,user_def_label)
    return(predicted_label)    

#function for classifying the a test sample using MLE decision line
def classify_sample(x,u,sigma,prior,user_def_label):
    """
    Input: x: sample 
    u: class means as list of 1d arrays
    sigma: class covariances as list of 2d arrays
    prior: list of priors
    user_def_label: list of labels
    Output: predicted_label (0,1,2,...K-1)
    """
#    d = np.shape(u)[0]
    # find number of classes
    Num_class = len(u)
    disc = [] #discriminant
    for i in range(Num_class):
        iu = u[i];
        isigma = sigma[i];
        isigma_inv = np.linalg.inv(isigma)
        iprior = prior[i];
        idisc = -(1/2)*(x-iu).T@isigma_inv@(x-iu) \
                -(1/2)*np.log(np.linalg.det(isigma))\
                +np.log(iprior)#-(d/2)*np.log(2*np.pi)
        disc.append(idisc)
    predicted_label = user_def_label[np.argmax(disc)]
    return(predicted_label)
# function for obtainting the image patches 
def classify_overlapping_patches(img,u,sigma,prior,user_def_label):
    """
    Input: img: np array of image in grayscale(scaled to 0-1)
    u: class means as list of 1d arrays
    sigma: class covariances as list of 2d arrays
    prior: list of priors
    user_def_label: list of label values
    Output: predicted_labels: np array of predicted label
    """
    M,N = np.shape(img)
    predicted_label = np.zeros((M,N))
    for i in range(M-8):
        for j in range(N-8):
            test_sample  = img[i:i+8,j:j+8]
            # convert test sample to a vector
            test_sample = np.reshape(test_sample,(64,1))
            # find class of the sample
            predicted_label[i,j] = classify_sample(test_sample,u,sigma,prior,user_def_label)
    return(predicted_label)
# function for finding the covariance using samples
def find_covariance(x,u):
    """
    input: x: samples in the format of a np matrix with col vectors
           u: mean of samples
    output: covar: matrix
    """
    Ndim,Nsample=np.shape(x)
    # substract mean vector from samples
    X = x-np.kron(np.ones((1,Nsample)),u)
    covar = (1/Nsample)*(X@X.T)
    return(covar)
        
