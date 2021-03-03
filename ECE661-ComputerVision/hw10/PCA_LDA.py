"""
ECE661: hw10 Part-1: PCA-LDA
@author: Rahul Deshmukh
email: deshmuk5@purdue.edu
"""
#%% --------------------import libraries -------------------------------------#
import cv2
import numpy as np
import sys
sys.path.append('../../')
import MyCVModule as MyCV
import os
#-----------------------------------------------------------------------------#
#define read path
readpath = '../files/faces/'
savepath = '../data/PCA/'
#-------------------------------Training--------------------------------------#
trainpath = readpath +'train/'
# Read images
img_list=os.listdir(trainpath)
"""
Given info about dataset: 30 classes with 21 images for each class 
naming is such that xx_yy.png where xx is class number and yy the image number
"""
Num_class = 30
Num_img = 21
## Loop over training images
#x= []
#train_labels = []
#for i in range(len(img_list)):
#    # read image 
#    img= cv2.imread(trainpath+img_list[i],0)
#    train_labels.append(int(img_list[i].split('_')[0]))
#    # convert to vectorize format
#    vec_img = np.reshape(img,(img.shape[0]*img.shape[1]))
#    # normalize image
#    vec_img = vec_img/np.linalg.norm(vec_img)
#    #store
#    x.append(list(vec_img))
#
#x= np.array(x).T
## find mean of images
#m = np.mean(x,axis=1)
## make X matrix
#X = x-np.kron(m,np.ones((np.shape(x)[1],1))).T
## save data
#np.save(savepath+'x.npy',x)
#np.save(savepath+'m.npy',m)
#np.save(savepath+'capX.npy',X)
#np.save(savepath+'train_labels.npy',train_labels)
##---------------------------Testing-------------------------------------------#
testpath = readpath+'test/'
test_img_list = os.listdir(testpath)
## read test images
#test_labels=[]
#test_x=[]
#for i in range(len(test_img_list)):
#    # read image 
#    img= cv2.imread(testpath+test_img_list[i],0)
#    test_labels.append(int(test_img_list[i].split('_')[0]))
#    # convert to vectorize format
#    vec_img = np.reshape(img,(img.shape[0]*img.shape[1]))
#    # normalize image
#    vec_img = vec_img/np.linalg.norm(vec_img)
#    #store
#    test_x.append(list(vec_img))
#
#test_x= np.array(test_x).T
## substract mean
#test_X = test_x-np.kron(m,np.ones((np.shape(test_x)[1],1))).T
## save data 
#np.save(savepath+'test_x.npy',test_x)
#np.save(savepath+'captest_X.npy',test_X)
#np.save(savepath+'test_labels.npy',test_labels)
#-----------------------------------------------------------------------------#
#------------------------------- load data -----------------------------------#
x = np.load(savepath+'x.npy')
m = np.load(savepath+'m.npy')
X = np.load(savepath+'capX.npy')
test_x = np.load(savepath+'test_x.npy')
test_X= np.load(savepath+'captest_X.npy')
train_labels = np.load(savepath+'train_labels.npy')
test_labels = np.load(savepath+'test_labels.npy')
#-----------------------------PCA---------------------------------------------#
# find eigen vectors using the trick
w_pca,lam,sorted_lam = MyCV.eig_rankdef(X,0)
# define number of eigen vectors to be taken into consideration
p = np.arange(20,dtype=int)
# make list of eigen vectors w for every p 
W_pca=[]
for i in range(len(p)):
    temp=[]
    for j in range(p[i]+1):
        temp.append(w_pca[:,sorted_lam[j]])
    temp=np.array(temp).T
    W_pca.append(temp)
#-----------------------------------------------------------------------------#

#-----------------------------------LDA---------------------------------------#
# find class means m_i
m_i = []
for i in range(Num_class):
    temp_x = x[:,i*Num_img:(i+1)*Num_img]
    m_i.append(np.mean(temp_x,axis=1))
# Find Sb and Sw
M = np.array(m_i).T
M = M-np.kron(m,np.ones((np.shape(M)[1],1))).T
#Sb = (1/(Num_class))*M@M.T
#Sw = np.zeros((np.shape(x)[0],np.shape(x)[0]))
#for i in range(Num_class):
#    temp_x = x[:,i*Num_img:(i+1)*Num_img] # for one class all images 
#    temp_X = temp_x-np.kron(m_i[i],np.ones((np.shape(temp_x)[1],1))).T
#    iSw = (1/Num_img)*temp_X@temp_X.T
#    Sw+=iSw
#Sw = (1/Num_class)*Sw

## find eigen vectors of Sb using the computational trick
V,mu,sorted_mu = MyCV.eig_rankdef(M,0)
mu = (1/Num_class)*mu
Y = np.zeros_like(V)
Db = np.zeros_like(mu)
for i in range(np.shape(V)[1]):
    Y[:,i]=V[:,sorted_mu[i]]
    Db[i]=mu[sorted_mu[i]]
# find Z
Z = Y@np.linalg.inv(np.diag(np.sqrt(Db)))
# now need to diagonalize Z.T Sw Z using the same trick
Xw = x - np.kron(M,np.ones((1,Num_img)))
ZtXw = Z.T@Xw
U,gamma,sorted_gamma=MyCV.eig_rankdef(ZtXw) # in ascending order
# make list of eigen vectors w for every p 
W_Lda=[]
for i in range(len(p)):
    U_hat=[]
    for j in range(p[i]+1):
        U_hat.append(U[:,sorted_gamma[j]])
    U_hat = np.array(U_hat).T
    Wp = Z@U_hat
    #normalize Wp
    Wp= Wp/np.linalg.norm(Wp,axis=0)
    W_Lda.append(Wp)

#-----------------------------------LDA---------------------------------------#

#------------------Plot------------------------#
# loop over p: chosen sub-space dimnesion
PCA_accuracy = []
LDA_accuracy = []
for i in range(len(p)):
    pca_detected_correctly = 0
    lda_detected_correctly = 0
    iW_pca = W_pca[p[i]]
    iW_lda = W_Lda[p[i]]
    # project training images using iW
    y_pca = iW_pca.T@X
    y_lda = iW_lda.T@X
    # project test images using iW
    test_y_pca = iW_pca.T@test_X
    test_y_lda = iW_lda.T@test_X
    # find Euclidean distance of test_y col vector from all y col vector
    # and use 1-NN classifier
    for j in range(len(test_img_list)):
        dist_pca = (y_pca.T - test_y_pca[:,j]).T
        dist_lda = (y_lda.T - test_y_lda[:,j]).T
        dist_pca = np.linalg.norm(dist_pca,axis=0)
        dist_lda = np.linalg.norm(dist_lda,axis=0)
        j_min_dis_pca = np.argmin(dist_pca)
        j_min_dis_lda = np.argmin(dist_lda)
        if train_labels[j_min_dis_pca]==test_labels[j]:
            pca_detected_correctly+=1
        if train_labels[j_min_dis_lda]==test_labels[j]:
            lda_detected_correctly+=1
    i_accuracy_pca=pca_detected_correctly/len(test_img_list)
    i_accuracy_lda=lda_detected_correctly/len(test_img_list)
    PCA_accuracy.append(i_accuracy_pca)
    LDA_accuracy.append(i_accuracy_lda)

import matplotlib.pyplot as plt
plt.plot(p,PCA_accuracy,c='b',marker='*',label='PCA')
#plt.scatter(p,PCA_accuracy)
plt.plot(p,LDA_accuracy,c='r',marker='^',label='LDA')
plt.xlabel('Dimension of Sub-Space (p)')
plt.ylabel('Accuracy of detection')
plt.legend()
plt.show()