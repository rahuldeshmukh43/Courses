#!/bin/python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0,0],[0,1],[1,0],[1,1]])
f = np.bitwise_xor(x[:,0],x[:,1])
#print(f)
plt.figure(1)
for i,ix in enumerate(x):
    if f[i]==1:
        plt.scatter(x[i,0],x[i,1],color='r',marker='*',s=200)
    else:
        plt.scatter(x[i,0],x[i,1],color='b',marker='^',s=200)
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
#plt.title('XOR function')
#plt.hold('on')
plt.grid()
#plt.show()
#plt.savefig('./XOR.png')

w = np.array([1,-2])
c = [0,-1]
W = np.ones((2,2))
b = 0

def ReLU(z):
    temp = (z>0).astype(int)
    return np.multiply(temp,z)
z = x@W + c
h =np.array([ReLU(z[i,:]) for i in range(z.shape[0])])
y = np.dot(h,w) + b
print(y)

plt.figure(2)
for i,iz in enumerate(z):
    if f[i]==1:
        plt.scatter(z[i,0],z[i,1],color='r',marker='*',s=200)
    else:
        plt.scatter(z[i,0],z[i,1],color='b',marker='^',s=200)
plt.xlabel('$z_1$'); plt.ylabel('$z_2$')
plt.grid()
#plt.show()
#plt.savefig('./1_Z.png')
#
plt.figure(3)
for i,ih in enumerate(h):
    if f[i]==1:
        plt.scatter(h[i,0],h[i,1],color='r',marker='*',s=200)
    else:
        plt.scatter(h[i,0],h[i,1],color='b',marker='^',s=200)
plt.xlabel('$h_1$'); plt.ylabel('$h_2$')
plt.grid()
#plt.show()
#plt.savefig('./1_H.png')

def sigmoid(x): return 1/(1+np.exp(-x))

xx = np.linspace(-100,100,1000)
s = sigmoid(xx)
plt.figure(4)
plt.plot(xx,s,'b')
plt.xlabel('x')
plt.ylabel('$\sigma(x)$')
plt.savefig('./sigmoid.png')

plt.figure(5)
plt.plot(xx,ReLU(xx))
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.savefig('./relu.png')

def abs_relu(z):
    temp= (z>0).astype(int)
    return np.multiply(temp,z) - np.multiply(1-temp,z)

def leaky_relu(z,a):
    temp= (z>0).astype(int)
    return np.multiply(temp,z) + a*np.multiply(1-temp,z)

plt.figure(6)
plt.plot(xx,abs_relu(xx))
plt.xlabel('x')
plt.ylabel('abs ReLU(x)')
plt.savefig('./abs_relu.png')

a =0.1
plt.figure(7)
plt.plot(xx,leaky_relu(xx,a))
plt.xlabel('x')
plt.ylabel('leaky ReLU(x)')
plt.savefig('./leaky_relu.png')
