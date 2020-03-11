"""
ECE 595: ML-1
HW-3 Exercise 3
@author- Rahul Deshmukh
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
#%%
#(i) Generate data
NumPts = 1000
# priors
pi_1 = 1/2
pi_2 = 1/2
# define mean and variance
u1 = np.array([0,0])
u2= np.array([10,10])
sigma = np.random.rand(2,2)
sigma = sigma +sigma.T # becomes symmetric
d,v = np.linalg.eig(sigma)
d = np.abs(d)# now semi pos def
# improve the condition number of sigma
if min(d)<1e-6:
    d[np.argmin(d)]=1
if max(d)>1e6:
    d[np.argmax(d)]=1
print('Condition number is: '+str(np.linalg.cond(sigma))+'\n')
print(sigma)

sigma = v@np.diag(d)@v.T
sigma_inv = np.linalg.inv(sigma)
# sample points from gaussian
x1 = np.random.multivariate_normal(u1,sigma,NumPts) #pts for class1
x2 = np.random.multivariate_normal(u2,sigma,NumPts) #pts for class2
x_min = min(min(x1[:,0]),min(x2[:,0]))
x_max = max(max(x1[:,0]),max(x2[:,0]))
#plot data for visual check
plt.scatter(x1[:,0],x1[:,1],c='b')
plt.scatter(x2[:,0],x2[:,1],c='r')
plt.show()
# (ii) plot the decision line 
x_pts = np.linspace(x_min,x_max,100) # for decision line
# decision line using part(a)
beta =sigma_inv@(u1-u2)
beta_0 = -(1/2)*(u1.T@sigma_inv@u1-u2.T@sigma_inv@u2)+np.log(pi_1/pi_2)
# find y coords of line
y_pts = -(beta[0]*x_pts+beta_0)/(beta[1])
# plot line
plt.plot(x_pts,y_pts,c='black')
plt.xlabel('x')
plt.xlabel('y')
plt.show()

#(iii) Decision line using linear least squares
A = np.ones((2*NumPts,3))
A[:NumPts,:-1] = x1
A[NumPts:,:-1] = x2
b = np.vstack((np.ones((NumPts,1)),-1*np.ones((NumPts,1))))
# solve using LLS
theta = np.linalg.inv(A.T@A)@(A.T@b)
y_lls = -(theta[0]*x_pts+theta[2])/(theta[1])
#plot lls line
plt.plot(x_pts,y_lls,c='green',alpha=0.5,linewidth=2,linestyle='--')
plt.legend(['Linear Gaussian','Linear Least Squares'])
plt.show()