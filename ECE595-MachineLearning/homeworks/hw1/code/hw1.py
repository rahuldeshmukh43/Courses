"""
ECE 595: Machine Learning-I
HW-1: Ex 2
@author: rahul
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#%% Exercise 2 
#%% 2b
u = 0
sigma= 1
numpts = 100
lb=-3;ub=3;
x = np.linspace(lb,ub,numpts)
fx = (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-u)**2/(2*sigma**2))
plt.plot(x,fx)
plt.xlabel('x')
plt.ylabel('fx')
plt.title('2(b) Normal Distribution u=0 sigma=1 in [-3,3]')
plt.savefig('2b_fig')
#%% 2c
# i
n=1000
s = np.random.normal(0,1,n)
# ii
bins = 4
plt.figure()
plt.hist(s,bins)
plt.xlabel('x')
plt.title('Histogram with '+str(bins)+' bins')
plt.savefig('2c1_fig')
bins=1000
plt.figure()
plt.hist(s,bins)
plt.xlabel('x')
plt.title('Histogram with '+str(bins)+' bins')
plt.savefig('2c2_fig')
# iii
u_est,sig_est = norm.fit(s)
print('estimated u:'+str(u_est)+' estimated sigma:'+str(sig_est))
# iv
bins = 4
plt.figure()
plt.hist(s,bins,normed=True,alpha=0.2)
plt.plot(x,norm.pdf(x,u_est,sig_est))
plt.xlabel('x')
plt.title('Histogram with '+str(bins)+' bins')
plt.savefig('2c3_fig')
bins = 1000
plt.figure()
plt.hist(s,bins,normed=True,alpha=0.2)
plt.plot(x,norm.pdf(x,u_est,sig_est))
plt.xlabel('x')
plt.title('Histogram with '+str(bins)+' bins')
plt.savefig('2c4_fig')
#%% 2d
Jh = []
range_s = max(s)-min(s)
for m in range(1,200+1):
    h = range_s/m
    temp = 2/(h*(n-1))-((n+1)/(h*(n-1)))*\
    np.sum(np.square(np.histogram(s,bins=m)[0]/n))
    Jh.append([temp])

m_star = np.argmin(Jh)+1
print(m_star)
plt.figure()
plt.plot(np.arange(1,200+1),Jh)
plt.scatter(m_star,Jh[m_star-1],c='r',marker='*')
plt.xlabel('m')
plt.ylabel('J_hat(h)')
plt.title('Plot of CVER wrt number of bins(m)')
plt.savefig('2c5_fig')
plt.figure()
plt.hist(s,m_star,normed=True,alpha=0.2)
plt.plot(x,norm.pdf(x,u_est,sig_est))
plt.xlabel('m')
plt.title('Histogram with optimal bin width using CVER')
plt.savefig('2c6_fig')

