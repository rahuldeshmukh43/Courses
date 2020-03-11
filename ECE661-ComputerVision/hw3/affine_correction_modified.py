import numpy as np
import cv2


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


#Homography for Projective Correction using vanishsing Line
def Hmat_ProjectiveCorrection(imPts):
    """
    Input:
        imPts=[[X,Y],[X,Y],..] 4 points in the order PQRS in the image plane
    Output: Returns the H for Projective Correction using concept of vanishing line
    such that H*im_pt=src_pt or (Inv(H)^T).VL=L_inf
    """
    #convert imPts to HC
    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
    im_pts_hc[:,:-1]=imPts
    #VP1 is intersection of PQ and RS
    L_PQ=np.cross(im_pts_hc[0,:],im_pts_hc[1,:])
    L_RS=np.cross(im_pts_hc[-2,:],im_pts_hc[-1,:])
    VP1=np.cross(L_PQ,L_RS)
    #VP2 is intersection of PR and QS
    L_PR=np.cross(im_pts_hc[0,:],im_pts_hc[-2,:])
    L_QS=np.cross(im_pts_hc[1,:],im_pts_hc[-1,:])
    VP2=np.cross(L_PR,L_QS)
    #hc vector of vanishing line
    VL=np.cross(VP1,VP2)
    #normalizing VL
    VL=VL/VL[-1]
    print('The Vanishing Line is: '+str(VL))
    H=np.identity(3)
    H[-1,:]=VL
    return(H)