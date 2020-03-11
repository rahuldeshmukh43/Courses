"""
ECE 595: ML-1
HW-2
@author- Rahul Deshmukh
"""

#import libraries
import numpy as np
import csv
import cvxpy as cp
import matplotlib.pyplot as plt
#%% 1 Read data from files
read_path = '../data/'

def read_csv(filename):
    ans=[]
    with open(filename,'r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        csv_reader = list(csv_reader)
        nrow=len(csv_reader)
        for i in range(1,nrow):
            temp=np.array(list(map(float,csv_reader[i][1:])))
            temp=temp/np.array([1,100]) # for scaling
            ans.append(list(temp))
    csv_file.close()
    return(ans)

male_train_data = read_csv(read_path+'male_train_data.csv')
female_train_data =  read_csv(read_path+'female_train_data.csv')

print(male_train_data[0])
print(female_train_data[0])
#%% 2
#(c)
A = np.vstack((np.array(male_train_data),np.array(female_train_data)))
A = np.hstack((A,np.ones((np.shape(A)[0],1))))
b = np.vstack((np.ones((len(male_train_data),1)),-1*np.ones((len(female_train_data),1))))

theta_star = (np.linalg.inv((A.T@A))@A.T)@b
#(d) Using cvxpy
x = cp.Variable(A.shape[1])
obj = cp.Minimize(cp.sum_squares(A*x-b))
prob = cp.Problem(obj)
result = prob.solve(solver = cp.SCS)
print('Norm of difference of solutions:'+str(np.linalg.norm(x.value-theta_star)))
#Norm of difference of solutions:0.0019636385053896374

#%% 3
#(a)
plt.figure(1)
plt.scatter(A[:len(male_train_data),0],A[:len(male_train_data),1],c='r',marker='o',label='male')
plt.scatter(A[len(male_train_data):,0],A[len(male_train_data):,1],c='b',marker='^',label='female')
N=1000
x_min = min(A[:,0])
x_max = max(A[:,0])
x_line = np.linspace(x_min,x_max,N)
y_line = (theta_star[0]*x_line+theta_star[-1])/(-1*theta_star[1])
plt.plot(x_line,y_line,c='k')
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Stature')
#(b)
male_test_data = read_csv(read_path+'male_test_data.csv')
male_test_label= np.ones((len(male_test_data),1))

female_test_data = read_csv(read_path+'female_test_data.csv')
female_test_label= -1*np.ones((len(female_test_data),1))

def classify(x,w):
    """
    x: query data 2d
    w: decision line coefficients
    """
    g = w[0]*x[0]+w[1]*x[1]+w[-1]
    if g>=0:
        label=+1
    else:
        label=-1
    return(label)

male_pred_label = np.zeros_like(male_test_label)
for i in range(len(male_test_data)): 
    male_pred_label[i]=classify(male_test_data[i],theta_star)

female_pred_label = np.zeros_like(female_test_label)
for i in range(len(female_test_data)): 
    female_pred_label[i]=classify(female_test_data[i],theta_star)
    
sm = np.where(male_test_label==male_pred_label,1,0)
sm = np.sum(sm)

sf = np.where(female_test_label==female_pred_label,1,0)
sf = np.sum(sf)

success_rate = (sm+sf)/(len(male_test_data)+len(female_test_data))
print('The success rate of our Linear classifier is:'+str(success_rate*100)+'%')
#The success rate of our Linear classifier is:83.93213572854292%

#%% 4 (a)
lambda_list = np.arange(0.1, 10, 0.1)
theta_lam=[]
residue_lam = []
norm_th_lam = []

for lam in lambda_list:
# Solve the regularized least-squares problem depending on lambda
    x = cp.Variable(A.shape[1])
    obj = cp.Minimize(cp.sum_squares(A*x-b)+lam*cp.sum_squares(x))
    prob = cp.Problem(obj)
    result = prob.solve(solver = cp.SCS)
    theta_lam.append(x.value)   
    residue_lam.append((np.linalg.norm(A@theta_lam[-1]-b))**2)
    norm_th_lam.append((np.linalg.norm(theta_lam[-1]))**2)    

plt.figure(2)
plt.plot(norm_th_lam,residue_lam)
plt.xlabel('$||\Theta _{\lambda}||^2$')
plt.ylabel('$||A\Theta _{\lambda}-b||^2$')

plt.figure(3)
plt.plot(lambda_list,residue_lam,c='b')
plt.xlabel('$\lambda$')
plt.ylabel('$||A\Theta _{\lambda}-b||^2$')

plt.figure(4)
plt.plot(lambda_list,norm_th_lam,c='b')
plt.xlabel('$\lambda$')
plt.ylabel('$||\Theta _{\lambda}||^2$')
    
x = np.linspace(x_min, x_max, 200)
legend_str = []
plt.figure(figsize=(15,7.5))
for i in range(len(lambda_list))[0::10]:
    y = (theta_lam[i][-1] + theta_lam[i][0]*x)/(-theta_lam[i][1])
    plt.plot(x, y.T)
    legend_str.append('$\lambda = $' + str(lambda_list[i]))
    
plt.scatter(A[:len(male_train_data),0],A[:len(male_train_data),1],c='r',marker='o')
plt.scatter(A[len(male_train_data):,0],A[len(male_train_data):,1],c='b',marker='^')
plt.legend(legend_str)
plt.xlabel('BMI')
plt.ylabel('Stature')

#%% 4(c)
# (i)
alpha_star = norm_th_lam[0]
alpha_list = alpha_star+2*np.arange(-50, 50+1, 1)
theta_alpha=[]
residue_alpha = []
norm_th_alpha = []

for alpha in alpha_list:
# Solve the regularized least-squares problem depending on lambda
    x = cp.Variable(A.shape[1])
    obj = cp.Minimize(0.1*cp.sum_squares(A*x-b))
    cons = [(cp.sum_squares(x)-alpha)<=0]
    prob = cp.Problem(obj,cons)    
    result = prob.solve(solver = cp.ECOS_BB)
    theta_alpha.append(x.value)   
    residue_alpha.append((np.linalg.norm(A@theta_alpha[-1]-b))**2)
    norm_th_alpha.append((np.linalg.norm(theta_alpha[-1]))**2)    

plt.figure(6)
plt.plot(norm_th_alpha,residue_alpha)
plt.xlabel('$||\Theta||^2$')
plt.ylabel('$||A\Theta-b||^2$')
plt.savefig('fig_4c_i1.png')

plt.figure(7)
plt.plot(alpha_list,residue_alpha,c='b')
plt.xlabel(chr(945))
plt.ylabel('$||A\Theta-b||^2$')
plt.savefig('fig_4c_i2.png')

plt.figure(8)
plt.plot(alpha_list,norm_th_alpha,c='b')
plt.xlabel(chr(945))
plt.ylabel('$||\Theta||^2$')
plt.savefig('fig_4c_i3.png')

# check solution
print('-----------------------')
print('The difference in solution btw problem 1 and problem 2 is:  '+str(np.linalg.norm(theta_lam[0]-theta_alpha[51-1])))
print('-----------------------')

# (ii)
eps_star = residue_lam[0]
eps_list = eps_star+2*np.arange(0, 100+1, 1)
theta_eps=[]
residue_eps= []
norm_th_eps= []

for eps in eps_list:
# Solve the regularized least-squares problem depending on lambda
    x = cp.Variable(A.shape[1])
    obj = cp.Minimize(0.1*cp.sum_squares(x))
    cons = [(cp.sum_squares(A*x-b)-eps)<=0]
    prob = cp.Problem(obj,cons)    
    result = prob.solve(solver = cp.ECOS_BB)
    theta_eps.append(x.value)   
    residue_eps.append((np.linalg.norm(A@theta_eps[-1]-b))**2)
    norm_th_eps.append((np.linalg.norm(theta_eps[-1]))**2)    

plt.figure(9)
plt.plot(norm_th_eps,residue_eps)
plt.xlabel('$||\Theta||^2$')
plt.ylabel('$||A\Theta-b||^2$')
plt.savefig('fig_4c_ii1.png')

plt.figure(10)
plt.plot(eps_list,residue_eps,c='b')
plt.xlabel('$\u03B5$')
plt.ylabel('$||A\Theta-b||^2$')
plt.savefig('fig_4c_ii2.png')

plt.figure(11)
plt.plot(eps_list,norm_th_eps,c='b')
plt.xlabel('$\u03B5$')
plt.ylabel('$||\Theta||^2$')
plt.savefig('fig_4c_ii3.png')

# check solution
print('-----------------------')
print('The difference in solution btw problem 1 and problem 3 is:  '+str(np.linalg.norm(theta_lam[0]-theta_eps[0])))
print('-----------------------')

