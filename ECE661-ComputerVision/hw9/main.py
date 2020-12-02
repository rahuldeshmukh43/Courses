"""
ECE661: hw9 main file
@author: Rahul Deshmukh
email: deshmuk5@purdue.edu
PUID: 0030004932
"""
#%% --------------------import libraries -------------------------------------#
import cv2
import numpy as np
import sys
sys.path.append('../../')
import MyCVModule as MyCV
#%% ---------------------define readpath--------------------------------------#
readpath='../files/original/'
savepath='../files/'
#read images
img1= cv2.imread(readpath+'1 (1).jpeg')
img2= cv2.imread(readpath+'1 (2).jpeg')
img1_gray= cv2.imread(readpath+'1 (1).jpeg',0)
img2_gray= cv2.imread(readpath+'1 (2).jpeg',0)

#%%------------------------ define coordinates -------------------------------#
pts1=[
[731,	486], #1
[725,590], #2
[308,301], #3
[333,411], #4
[971,143], #5
[652,58], #6
[838,220], 
[819,396],
[603,147],
[340,276],
[888,296],
[619,274],
[957,223]]#7

pts2=[
[461,	558],
[478,	648],
[193,	220],
[228,	327],
[1017,207],
[721,16],
[790,269],
[657,470],       
[593,113],
[239,201],
[814,370],
[510,272],
[987,299]]
#------------------------------RANSAC-----------------------------------------#
## find pts using SIFT on both the images
## find sift points of images
#pts1,des1=MyCV.SIFT_detector(img1_gray)
#pts2,des2=MyCV.SIFT_detector(img2_gray)
## find euclidean matches between images
#match_pts1,match_pts2=MyCV.Euclidean_sift_match(pts1,des1,pts2,des2,3) # returns list of lists
#in_indices= MyCV.F_Ransac(match_pts1,match_pts2)
#pts1 = np.array(pts1)
#pts2 = np.array(pts2)
#pts1 = pts1[in_indices,:]
#pts2 = pts2[in_indices,:]
#%%-----plot images with points and lines connecting correspondences----------#
pt_pairs=[]
for i in range(len(pts1)): pt_pairs.append( [tuple(pts1[i]),tuple(pts2[i])] )
img= MyCV.plot_matching_points(img1,pts1,img2,pts2,pt_pairs)
cv2.imwrite(savepath+'plots/'+'correspondences'+'.jpg',img)

#%%----------------- find F using Linear Least Squares------------------------#
# convert points to HC
pts1_hc = MyCV.convert2HC(pts1)
pts2_hc = MyCV.convert2HC(pts2)
F0 = MyCV.Funda_using_pts(pts1_hc,pts2_hc)
P1,P2 =MyCV.get_Proj_mat(F0)
# refine F using LM
F_ref,P1_ref,P2_ref,M_ref=MyCV.Refine_F(F0,pts1,pts2)
e2 = MyCV.get_left_nullvec(F_ref)
#%%---------------------------Image Rectification-----------------------------#
H1,H2 = MyCV.get_rectify_homo(e2,P1_ref,P2_ref,M_ref,img1,img2,pts1,pts2)
#distype = 'BiLinear'
#img1P=[0,0]
#img1Q=[np.shape(img1)[1]-1,0]
#img1R=[0,np.shape(img1)[0]-1]
#img1S=[np.shape(img1)[1]-1,np.shape(img1)[0]-1]
#impts1=np.array([img1P,img1Q,img1S,img1R])
##ar=np.shape(img1)[1]/np.shape(img1)[0]
##canvas=np.zeros(np.shape(img1))
#rec_img1 = MyCV.mapFitToCanvas(img1,impts1,[],np.linalg.inv(H1),distype)
#cv2.imwrite(savepath+'plots/'+'1_Rec.jpeg',rec_img1)
#
#
#img1P=[0,0]
#img1Q=[np.shape(img2)[1]-1,0]
#img1R=[0,np.shape(img2)[0]-1]
#img1S=[np.shape(img2)[1]-1,np.shape(img2)[0]-1]
#impts1=np.array([img1P,img1Q,img1S,img1R])
###ar=np.shape(img2)[1]/np.shape(img2)[0]
###canvas=np.zeros((200,int(ar*200),3))
##Pcor = np.ones((4,3))
##Pcor[:,:-1]=np.array([img1P,img1Q,img1S,img1R])
##HP= H2@Pcor.T
##HP = np.linalg.inv(np.diag(HP[-1,:]))@HP.T
###impts1=HP[:,:-1]
##x_min=min(HP[:,0]); x_max=max(HP[:,0]);
##y_min=min(HP[:,1]); y_max=max(HP[:,1]);
##canvas=np.zeros(( int(y_max-y_min) , int(x_max-x_min) ,3))
##canvas=np.zeros(np.shape(img2))
#rec_img2 = MyCV.mapFitToCanvas(img2,impts1,[],np.linalg.inv(H2),distype)
#cv2.imwrite(savepath+'plots/'+'2_Rec.jpeg',rec_img2)

#plot of correspondences in rectified images
pts1_rec_hc = H1@pts1_hc.T
pts2_rec_hc = H2@pts2_hc.T
pts1_rec = MyCV.convert2phy(pts1_rec_hc)
pts2_rec = MyCV.convert2phy(pts2_rec_hc)

#pt_pairs_rec=[]
#for i in range(len(pts1)): pt_pairs_rec.append( [tuple(pts1_rec[i,:]),tuple(pts2_rec[i,:])] )
#img= MyCV.plot_matching_points(rec_img1,pts1_rec,rec_img2,pts2_rec,pt_pairs_rec)
#cv2.imwrite(savepath+'plots/'+'correspondences_Rec'+'.jpg',img)
#
img=MyCV.plot_rectify_img(img1,img2,H1,H2)
cv2.imwrite(savepath+'plots/'+'correspondences_Rec'+'.jpg',img)
#%%---------------------Interest Point Detection------------------------------#
pts1_sift,des1=MyCV.SIFT_detector(img1_gray)
pts2_sift,des2=MyCV.SIFT_detector(img2_gray)

# now we need to do row scans on the rectified images to make pairs with 
# descriptor vectors as my metric 

# find pt pairs using row-scans in the rectified images
scan_tol = 2 # pm row scan window
pt_pairs = MyCV.find_pairs_rectifed_img(img1,pts1_sift,des1,img2,pts2_sift,des2,H1,H2,scan_tol)
#img= MyCV.plot_matching_points(img1,pts1_sift,img2,pts2_sift,pt_pairs)
#cv2.imwrite(savepath+'plots/'+'sift_matches'+'.jpg',img)
#%%--------------------------Projective Reconstruction------------------------#
# separate pt_pairs for each image
pt_pair1 = [] # for pair poin on image1 
pt_pair2 = [] # for pair poin on image12
for i in range(len(pt_pairs)):
    pt_pair1.append(pt_pairs[i][0])
    pt_pair2.append(pt_pairs[i][1])

# convert pt_pairs to HC for triangulation
pt_pair1_hc = MyCV.convert2HC(pt_pair1)
pt_pair2_hc = MyCV.convert2HC(pt_pair2)
# Triangulate the pt_pairs to world pts for projective reconstruction
world_pt_hc =[]
for i in range(len(pt_pairs)):
    world_pt_hc.append(MyCV.Triangulate(pt_pair1_hc[i,:],pt_pair2_hc[i,:],P1_ref,P2_ref))

world_pt_hc = np.array(world_pt_hc)
world_pt = MyCV.convert2phy(world_pt_hc.T)
#%%-------------------------- 3D Visual Inspection----------------------------#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
#plots 3D points from sift in blue
ax.scatter(world_pt[:,0], world_pt[:,1], world_pt[:,2], c='b', marker='o')
# plot Triangulation of original inliers in red 
inlier_3D_hc = []
for i in range(len(pts1)):
    inlier_3D_hc.append(MyCV.Triangulate(pts1_hc[i,:],pts2_hc[i,:],P1_ref,P2_ref))
inlier_3D_hc = np.array(inlier_3D_hc)
inlier_3D = MyCV.convert2phy(inlier_3D_hc.T)
ax.scatter(inlier_3D[:,0], inlier_3D[:,1], inlier_3D[:,2], c='r', marker='^')
# draw line joininG the vertices which were picked manually
# LINES: 12,24,43,31,15,56,63,57,72
lookup = np.array([[1,2,3,4,5,6, 7],
                   [0,1,2,3,4,5,-1]])
#edges = np.array([[1,2],[2,4],[4,3],[3,1],[1,5],[5,6],[6,3],[5,7],[7,2]])
face1= np.array([[1,2],[2,4],[4,3],[3,1]])
for i in range(np.shape(face1)[0]):
    ip1x,ip1y,ip1z = inlier_3D[lookup[1,(face1[i,0])-1],:]
    ip2x,ip2y,ip2z = inlier_3D[lookup[1,(face1[i,1])-1],:]
    ax.plot([ip1x,ip2x],[ip1y,ip2y],[ip1z,ip2z],c='r')

face2=np.array([[1,5],[5,6],[6,3],[5,7],[7,2]])
for i in range(np.shape(face2)[0]):
    ip1x,ip1y,ip1z = inlier_3D[lookup[1,(face2[i,0])-1],:]
    ip2x,ip2y,ip2z = inlier_3D[lookup[1,(face2[i,1])-1],:]
    ax.plot([ip1x,ip2x],[ip1y,ip2y],[ip1z,ip2z],c='black')
    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.xlim(min(inlier_3D[:,0]),max(inlier_3D[:,0]))
plt.ylim(min(inlier_3D[:,1]),max(inlier_3D[:,1]))
ax.set_zlim(min(inlier_3D[:,2]),max(inlier_3D[:,2]))
plt.show()

# write a ply file for visualization in openmesh
all_3D = np.vstack((world_pt,inlier_3D))
MyCV.write_ply_file(all_3D,savepath+'Point_cloud')

