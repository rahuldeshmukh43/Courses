"""
ECE661: hw8 main file
@author: Rahul Deshmukh
email: deshmuk5@purdue.edu
"""
#%%
# import libraries
import cv2
import os
import numpy as np
import sys
sys.path.append('../../')
import MyCVModule as MyCV

#%%
#-------------------Corner detection----------------#
# define readpath
readpath='../Files/Dataset2/raw'
savepath='../Files/Dataset2/'
img_list=os.listdir(readpath)
# calibration pattern grid def
pattern_dim=(5,4)
# world coordinates of the grid
# measured sizes of grid
sq=25.0 # length of side of square in the pattern
world_coord=[]
for i in range(2*pattern_dim[0]):
    for j in range(2*pattern_dim[1]):
        world_coord.append([j*sq,i*sq])

# define thresholds for Canny
minVal=500 #200 for provided 500 for my
maxVal=600  #500 for provided  600 for my
# Hough transform related constants
HoughThresh=50 # 52 for provided dataset 50 for my
font = cv2.FONT_HERSHEY_SIMPLEX

# loop over dataset images
# initialize
H=[] # list to store iH for all images
V=[] # list to store zhangs V for all images
AllCorners=[] # list to store all corners in the images
for i in range(len(img_list)):
    # read image in color
    color_img=cv2.imread(readpath+'/Pic_'+str(i+1)+'.jpg',1)
    gray_img=cv2.imread(readpath+'/Pic_'+str(i+1)+'.jpg',0)
    # find image size
    m,n=gray_img.shape
    # find lines using canny and hough
    edges,lines=MyCV.findLines(gray_img,minVal,maxVal,
                                HoughThresh)
#    cv2.imwrite(savepath+'edges/Pic_'+str(i+1)+'.jpg',edges)
    # draw lines onto the color image
#    temp_img=MyCV.drawLine_old(color_img,lines)
#    cv2.imwrite(savepath+'labelled/Pic_'+str(i+1)+'temp.jpg',temp_img)
    # find corners
    corners,hor_lines,ver_lines=MyCV.findCorners(lines,(m,n),pattern_dim)
    #draw hor lines and ver lines
#    draw_img=MyCV.drawLine(color_img,hor_lines)
#    draw_img=MyCV.drawLine(draw_img,ver_lines)
    corners=MyCV.refine_corners(gray_img,corners,10)
    AllCorners.append(corners)
#    for j in range(len(corners)):
#        draw_img=cv2.circle(draw_img,(int(corners[j][0]),int(corners[j][1])),4,(0,0,255),-1)
#        draw_img=cv2.putText(draw_img,str(j+1),(int(corners[j][0]),int(corners[j][1])), font,0.5,(0,0,255),1,cv2.LINE_AA)
#    cv2.imwrite(savepath+'labelled/Pic_'+str(i+1)+'.jpg',draw_img)
    # find homography using world coordinates
    iH=MyCV.HMat_Pts(np.array(world_coord),np.array(corners))
    H.append(iH)
    iV=MyCV.Zhang_V(iH)
    V.append(iV)
    
np.save('./saved_data/V.npy',V)
np.save('./saved_data/H.npy',H)
np.save('./saved_data/AllCorners.npy',AllCorners)
#-------------------------------------------------------#
#%%
#-------------------Zhang's Algo II--------------------#    
# find w from the equation Vb=0 using svd and null vector
# convert V list to a matrix
Vmat=np.zeros((2*len(img_list),6),dtype=np.float32)
for i in range(len(img_list)): Vmat[2*i:2*i+2,:]=V[i]
# do SVD of Vmat
u,d,vt=np.linalg.svd(Vmat)
b=vt[-1,:]# last row of vt
w=np.array([[b[0],b[1],b[3]],
            [b[1],b[2],b[4]],
            [b[3],b[4],b[5]]])    
# find intrinsic parameters of K from w
y0=(w[0,1]*w[0,2]-w[0,0]*w[1,2])/(w[0,0]*w[1,1]-w[0,1]**2)
lam=w[2,2]-(w[0,2]**2+y0*(w[0,1]*w[0,2]-w[0,0]*w[1,2]))/(w[0,0])
a_x=np.sqrt(lam/w[0,0])
a_y=np.sqrt((lam*w[0,0])/(w[0,0]*w[1,1]-w[0,1]**2))
s=-1*(w[0,1]*(a_x**2)*(a_y))/(lam)
x0= s*y0/a_y - w[0,2]*a_x**2/lam
K=np.array([[a_x,s,x0],
            [0,a_y,y0],
            [0,0,1]])
np.save('./saved_data/K.npy',K)
#-------------------------------------------------------#  
#%%
#--------Extrinsic Parameters: R and t------------------#
R = [] # list to store R
t = [] # list to store t
w = [] # list to store w: rodriguez rep of Rot
K_inv=np.linalg.inv(K)
for i in range(len(img_list)):
    y=1/np.linalg.norm(K_inv@H[i][:,0]) # constant for scaling
    r1= y*(K_inv@(H[i][:,0]))
    r2= y*(K_inv@(H[i][:,1]))
    r3= np.cross(r1,r2)
    it= y*(K_inv@(H[i][:,-1]))
    iR=np.array([r1.T,r2.T,r3.T])
    iR=iR.T
    # conditioning of R
    u,d,vt=np.linalg.svd(iR)
    iR=u@vt
    R.append(iR)
    t.append(it)
    # find w using rodriguez 
    iw=MyCV.Rot_mat2vec(iR)
    w.append(iw)

np.save('./saved_data/R.npy',R)
np.save('./saved_data/t.npy',t)
np.save('./saved_data/w.npy',w)
#-------------------------------------------------------# 
#%%
#------------Reprojection with linear least squares estimate--------#
linlsq_savepath=savepath+'reproject_linlsq/'
mean_linlsq=[]
var_linlsq=[]
for i in range(len(img_list)):
    img=cv2.imread(readpath+'/Pic_'+str(i+1)+'.jpg',1)
    rep_img,imean,ivar=MyCV.ReprojectPoints(img,world_coord,
                                            AllCorners[i],K,R[i],t[i])
    mean_linlsq.append(imean)
    var_linlsq.append(ivar)
    cv2.imwrite(linlsq_savepath+'Pic_'+str(i+1)+'.jpg',rep_img)
    
np.save('./saved_data/mean_linlsq.npy',mean_linlsq)
np.save('./saved_data/var_linlsq.npy',var_linlsq)
#-------------------------------------------------------------------# 
#%%
#-----------Refinement of Calibration Parameters----------#    
rad_dist = 1 # user parameter defining to include radial distortion
k1,k2= np.zeros(2) # initialization fo radial distortion parameters
#k1=1;k2=1;
# prepare initial solution p for LM
if rad_dist==0:
    # p= [K,w1,t1,w2,t2,...wn,tn]
    p0=np.zeros(5+6*len(img_list))
    p0[:5]=np.array([a_x,a_y,s,x0,y0])
    for i in range(len(img_list)):
        p0[6*i+5:6*i+8]=w[i]
        p0[6*i+8:6*i+11]=t[i]
#    #bounds
#    ub=np.inf
#    lb= [0,0,0,0,0]+list(np.kron(-1*np.inf,np.ones(6*len(img_list))))
#    bounds=(lb,ub)
else:
    # p = [K,w1,t1,w2,t2,...wn,tn,k1,k2]
    p0=np.zeros(7+6*len(img_list))
    p0[:5]=np.array([a_x,a_y,s,x0,y0])
    for i in range(len(img_list)):
        p0[6*i+5:6*i+8]=w[i]
        p0[6*i+8:6*i+11]=t[i] 
    p0[-2]=k1;  p0[-1]=k2
#    #bounds
#    ub=np.inf
#    lb= [0,0,-1*np.inf,0,0]+list(np.kron(-1*np.inf,np.ones(6*len(img_list)+2)))
#    bounds=(lb,ub)
# call to LM for refining the parameters
from scipy.optimize import least_squares
import time

start_time=time.time()

if rad_dist==0:
    optim=least_squares(MyCV.CostFun_cam_cal_linear,p0,
                        method='lm',args=(AllCorners,world_coord))
    p_star=optim['x']
    np.save('./saved_data/p_star_linear.npy',p_star)
else:
    optim=least_squares(MyCV.CostFun_cam_cal_radial,p0,
                        method='lm',args=(AllCorners,world_coord))
    p_star=optim['x']
    np.save('./saved_data/p_star_radial.npy',p_star)
np.save('./saved_data/optim.npy',optim)
#-------------------------------------------------------# 
end_time=time.time()
total_time=end_time-start_time
np.save('./saved_data/total_time.npy',total_time)
#%%
# make final refined K_ref and R_ref from p_star
a_x=p_star[0]; a_y=p_star[1]; s=p_star[2]
x0=p_star[3]; y0=p_star[4];
if rad_dist==1:
    k1=p_star[-2]; k2=p_star[-1]
    print('Radial Distortion parameters: k1='+str(k1)+' k2='+str(k2))
K_ref=np.array([[a_x,s,x0],
            [0,a_y,y0],
            [0,0,1]])
R_ref=[]
t_ref=[]
for i in range(len(img_list)):
    iw=p_star[6*i+5:6*i+8]
    it=p_star[6*i+8:6*i+11]
    iR=MyCV.Rot_vec2mat(iw)
    R_ref.append(iR)
    t_ref.append(it)

np.save('./saved_data/K_ref.npy',K_ref)
np.save('./saved_data/R_ref.npy',R_ref)
np.save('./saved_data/t_ref.npy',t_ref)
#%%
#-----------------------Reprojection------------------------------------#
linlsq_savepath=savepath+'reproject_LM/'
mean_LM=[]
var_LM=[]
for i in range(len(img_list)):
    img=cv2.imread(readpath+'/Pic_'+str(i+1)+'.jpg',1)
    rep_img,imean,ivar=MyCV.ReprojectPoints(img,world_coord,AllCorners[i],
                                            K_ref,R_ref[i],t_ref[i])
    mean_LM.append(imean)
    var_LM.append(ivar)
    cv2.imwrite(linlsq_savepath+'Pic_'+str(i+1)+'.jpg',rep_img)
np.save('./saved_data/mean_LM.npy',mean_LM)
np.save('./saved_data/var_LM.npy',var_LM)
#-----------------------------------------------------------------------#
# make plots of mean and var for both cases linlsq and LM
import matplotlib.pyplot as plt
plt.scatter(np.arange(len(mean_linlsq)),mean_linlsq,c='b',label='lin_lsq')
plt.scatter(np.arange(len(mean_LM)),mean_LM,c='r',label='LM')
plt.xlabel('Images')
plt.ylabel('Mean')
plt.title('Plot of mean of error')
plt.legend()
plt.show()
plt.scatter(np.arange(len(var_linlsq)),var_linlsq,c='b',label='lin_lsq')
plt.scatter(np.arange(len(var_LM)),var_LM,c='r',label='LM')
plt.xlabel('Images')
plt.ylabel('Variance')
plt.title('Plot of Variance of error')
plt.legend()
plt.show()

