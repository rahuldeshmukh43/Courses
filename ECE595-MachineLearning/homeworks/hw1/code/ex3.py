"""
ECE 595: Machine Learning-I
HW-1: Ex 3
@author: rahul
"""
import numpy as np
import matplotlib.pyplot as plt
#%% 3a
def gauss_2d(x,u,s):
    f = (1/np.sqrt((2*np.pi)**2*np.linalg.det(s)))*np.exp((-1/2)*(x-u).T@np.linalg.inv(s)@(x-u))
    return(f)

u=np.array([2,6])
sigma = np.array([[2,1],[1,2]])
N=100
x = np.linspace(-1,5,N)
y = np.linspace(0,10,N)
X,Y=np.meshgrid(x,y)

F = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        x = np.array([X[i,j],Y[i,j]])
        F[i,j]= gauss_2d(x,u,sigma)

#plot
plt.figure(1)
plt.contour(X,Y,F)
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-1,5,0,10])
plt.title('contour plot of 2d Gaussian')
plt.show()
plt.savefig('3a2')
#%% 3c
# i
N_sample = 5000
samples = np.random.multivariate_normal(np.zeros(2),np.eye(2),N_sample)
#plot 
plt.figure(2)
plt.subplot(3,1,1)
plt.scatter(samples[:,0],samples[:,1],c='b',marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of random samples')
# ii
#sqrt_2 = np.sqrt(2)
sqrt_3 = np.sqrt(3)
#A =(1/sqrt_2)*np.array([[sqrt_3,-1],[sqrt_3,1]])
A =(1/2)*np.array([[sqrt_3+1,sqrt_3-1],[sqrt_3-1,sqrt_3+1]])
b = np.array([2,6])
Y = ((A@samples.T).T+b)
plt.subplot(3,1,2)
plt.scatter(Y[:,0],Y[:,1],c='r',marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of transformed samples (hand calc)')
# iii
D,U = np.linalg.eig(sigma)
A_py = U@np.diag(np.sqrt(D))@U.T
Y_py = ((A_py@samples.T).T+b)
plt.subplot(3,1,3)
plt.scatter(Y_py[:,0],Y_py[:,1],c='r',marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of random samples (using python)')
# check 
print('Sample observed mean:'+str(np.mean(Y,axis=0)))
print('Sample observed covariance:'+str(np.cov(Y.T)))
