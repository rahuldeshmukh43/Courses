"""
MyCVModule: all the functions created in ECE661
Created on Sun Sep  9 21:05:39 2018

Author: Rahul Deshmukh
email: deshmuk5@purdue.edu
"""
# all the functions created in homeworks will be called from this central library 
# functions added in chronological order with latest hw first
#----make sure to add documentation in each function and also link to the home works---#
import numpy as np
import cv2

#============================================HW3=============================================#

#function for Euclidean match for SIFT points 
def Euclidean_sift_match(pts1,des1,pts2,des2):
    """
    Input: pts1,pts2= [(x,y),(),..] list of tuples of interest points
           des1,des2 = vector of 128 size, sift descriptor
    Output: pt_pairs= [[(x1,y1),(x2,y2)],....] list of list of point pair tuples such that
            Euclidean distance is less than a dynamic threshold
    """
    pt_pairs=[]
    sq_eu=np.zeros((len(pts1),len(pts2)))    
    for i in range(len(pts1)):
        for j in range(len(pts2)):
            sq_eu[i,j]=np.linalg.norm(des1[i,:]-des2[j,:])
    
    sq_eu=sq_eu/np.min(sq_eu)
    dy_threshold=2
    print(dy_threshold)
    for i in range(len(pts1)):            
        sq_eu_min=np.min(sq_eu[i,:])
        if sq_eu_min<dy_threshold:
            j_min=np.argmin(sq_eu[i,:])
            pt_pairs.append([pts1[i],pts2[j_min]])
    return(pt_pairs)            

#function for converting sift key points into my format 
def convert_SIFT_myFormat(kp):
    """
    Input: kp: keypoint structure given by SIFT, point coordinates in kp.pt
    Output: [(x,y),...] : list of tuples of coorinates
    """
    mykp=[]
    for ikp in kp: 
        #convert points to floored integer
        mykp.append((int(ikp.pt[0]),int(ikp.pt[1])))
    return(mykp)

#function for Squared Sum Differences for Point correspondences
def SSD(img1,pts1,img2,pts2):
    """
    Input: img1, img2 = grayscale images of a scene from different viewpoints, sizes are identical
            pts1,pt2 =[(x,y),...] list of tuples interest points detected for img1 and img2 respectively
    Output: pt_pairs= [[(x1,y1),(x2,y2)],....] list of list of point pair tuples which pass a certain
            confidence threshold
    """
    M=51 # window is of size M
    pad = int((M-1)/2)
    #pad img1 and img2
    img1_pad=cv2.copyMakeBorder(img1,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    img2_pad=cv2.copyMakeBorder(img2,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    # loop over length of pts1 and pts2: exhaustive search
    pt_pairs=[]
    SSD=np.zeros((len(pts1),len(pts2)))
    i=0
    for ipts1 in pts1:
        j=0
        for ipts2 in pts2:
            #make window of size MxM at both the point locations 
            win1=img1_pad[ipts1[1]+pad-pad:ipts1[1]+pad+pad+1,ipts1[0]+pad-pad:ipts1[0]+pad+pad+1]
            win2=img2_pad[ipts2[1]+pad-pad:ipts2[1]+pad+pad+1,ipts2[0]+pad-pad:ipts2[0]+pad+pad+1]
            #find SSD
            SSD[i,j]=np.sum(np.power((win1-win2),2))
            j+=1
        i+=1
    #using a dynamic threshold as criteria for matching points
    #converting SSD to 0-1 range and then using a dynamic threshold
    SSD=(SSD-np.min(SSD)*np.ones(np.shape(SSD)))/(np.max(SSD)-np.min(SSD))
    dy_threshold=0.05
    i=0
    for ipts1 in pts1:        
        #check if SSD_max is less than dy_threshold
        SSD_min=np.min(SSD[i,:])
        if SSD_min<dy_threshold:
            j_min=np.argmin(SSD[i,:])
            pt_pairs.append([ipts1,pts2[j_min]])
        i+=1
    return(pt_pairs)


#function for plotting image with interest points of two scenes and lines joining matching points
def plot_matching_points(img1,pts1,img2,pts2,pt_pairs):
    """
    Input: img1,img2: colored image matrices of same size of the same scene from diff viewpoints
            pts1,pts2: [(x,y),..] list of tuples of intresting points in img1 and img2 respectively
            pt_paris:[ [(x1,y1), (x2,y2)],... ] list of list of tuples of point pairs with matching metric
    OutPut: img3: image matrix with both scenes placed side by side horizontally,
            with marked points of interests in both the images and lines joining elements of pt_pairs
    """
    m=np.shape(img1)[0];n1=np.shape(img1)[1];n2=np.shape(img2)[1]
    img3=np.zeros((m,n1+n2,3))
    #assign img1 and img2 pixels to img3
    img3[:,:n1,:]=img1
    img3[:,n1:,:]=img2
    #draw interest points on both images
    for ipts1 in pts1: cv2.circle(img3,ipts1,2,(0,0,255),-1)    
    for ipts2 in pts2:
        x=n1+ipts2[0]
        cv2.circle(img3,(x,ipts2[1]),2,(0,0,255),-1)    
    #draw line joining matching points
    for i in range(len(pt_pairs)):
        cv2.line(img3,pt_pairs[i][0],(pt_pairs[i][1][0]+n1,pt_pairs[i][1][1]),(0,255,0),1)    
    return(img3)

#function for Normalized Cross Correlation for point correspondences
def NCC(img1,pts1,img2,pts2):
    """
    Input: img1, img2 = grayscale images of a scene from different viewpoints, sizes are identical
            pts1,pt2 =[(x,y),...] list of tuples interest points detected for img1 and img2 respectively
    Output: pt_pairs= [[(x1,y1),(x2,y2)],....] list of list of point pair tuples which pass a certain
            confidence threshold
    """
    M=31 # window is of size M
    pad = int((M-1)/2)
    #pad img1 and img2
    img1_pad=cv2.copyMakeBorder(img1,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    img2_pad=cv2.copyMakeBorder(img2,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    # loop over length of pts1 and pts2: exhaustive search
    threshold=0.95
    pt_pairs=[]
    
    for ipts1 in pts1:
        NCC=np.zeros((len(pts2),1))
        k=0
        for ipts2 in pts2:
            #make window of size MxM at both the point locations 
            win1=img1_pad[ipts1[1]+pad-pad:ipts1[1]+pad+pad+1,ipts1[0]+pad-pad:ipts1[0]+pad+pad+1]
            win2=img2_pad[ipts2[1]+pad-pad:ipts2[1]+pad+pad+1,ipts2[0]+pad-pad:ipts2[0]+pad+pad+1]
            #find means for both the window
            mean1=np.mean(win1)
            mean2=np.mean(win2)
            #find NCC 
            num=np.sum(np.multiply( (win1-mean1*np.ones((M,M))),(win2-mean2*np.ones((M,M)))) )
            deno=np.sum(np.power((win1-mean1*np.ones((M,M))),2))
            deno=deno*(np.sum(np.power((win2-mean2*np.ones((M,M))),2)))
            deno=np.sqrt(deno)
            NCC[k]=num/deno
            k+=1
        #check if NCC_max is greater than threshold
#        NCC=(NCC-np.min(NCC)*np.ones(np.shape(NCC)))/(np.max(NCC)-np.min(NCC))
        NCC_max=np.max(NCC)
        if NCC_max>threshold:
            i_max=np.argmax(NCC)
            pt_pairs.append([ipts1,pts2[i_max]])
            #drop pts2[i_max] from pts2 list
#            del pts2[i_max]            
    return(pt_pairs)

#harris corner detection
def Harris_Corners(img,scale):
    """
    Input: img= image object for which corners are to be detected
        scale= smoothing scale for image 
    Output: pts=[[x,y],[],[]..] List of corner coordinate
        Ix: xderivative Image
        Iy: y derivative image
    """
    #find derivatives of image dx and dy
    dx,dy=Haar_filter(scale)
    #convolve dx dy to get Ix Iy
    Ix=cv2.filter2D(img,-1,dx) #by default pixelExtrapolation at borders
    Iy=cv2.filter2D(img,-1,dy)
    #pad Ix and Iy for next step
    #find pad
    window_size=int(np.ceil(5*scale))
    if window_size%2==0: window_size+=1
    pad=int((window_size-1)/2.0)
    Ix_pad=cv2.copyMakeBorder(Ix,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    Iy_pad=cv2.copyMakeBorder(Iy,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    R=np.zeros(np.shape(img))#stores info of eigenvalue ratios
    #loop over pixels and update R if a corner
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            #take slice of Ix Iy with window of 2*pad+1 size
            Ix_window=Ix_pad[(i+pad)-pad:(i+pad)+pad+1,(j+pad)-pad:(j+pad)+pad+1]
            Iy_window=Iy_pad[(i+pad)-pad:(i+pad)+pad+1,(j+pad)-pad:(j+pad)+pad+1]
            #find Ix**2 Iy**2 IxIy
            Ix2=np.sum(np.multiply(Ix_window,Ix_window))
            Iy2=np.sum(np.multiply(Iy_window,Iy_window))
            Ixy=np.sum(np.multiply(Ix_window,Iy_window))
            #now C=[[Ix2,Ixy],[Ixy,Iy2]] check if a corner
            #using determinant of C as check for rank2
            detC=Ix2*Iy2-(Ixy**2)
            if detC>0:
                R[i,j]=detC/((Ix2+Iy2)**2)                
    #remove points on the edges of image upto a pixel wdith of p pixels
    p=10
    R[:p,:]=0*R[:p,:]#top border
    R[-p:,:]=0*R[-p:,:]#bottom border
    R[:,:p]=0*R[:,:p]#left border
    R[:,-p:]=0*R[:,-p:]#right border
    #Suppression of non-maxium points in vicinity of a maxima
    #so that we dont get a cluster of points at a single location
    non_max_window=31 #pixels, should be odd
    threshold=np.mean(R)  
    pad = int((non_max_window-1)/2)
    R_pad=cv2.copyMakeBorder(R,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    pts=[] 
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if R[i,j]>0:
                R_window=R_pad[(i+pad)-pad:(i+pad)+pad+1,(j+pad)-pad:(j+pad)+pad+1]
                if R[i,j]==np.max(R_window) and R[i,j]>threshold:
                     pts.append((j,i))# x is j col y is i row
    return(pts,Ix,Iy)

#function for haar filter
def Haar_filter(scale):
    """
    Input: scale= defines the sigma for the filter and also the size of filter
    Output: dx, dy haar filter kernels
    """
    N=int(np.ceil(4*scale)) # size of kernel is smallest even integer greater than 4*sigma
    if N%2!=0:
        N=N+1 #if N was odd: add 1 to it
    dx=np.ones((N,N))
    dy=np.ones((N,N))
    dx[:,:int(N/2)]=-1*dx[:,:int(N/2)]
    dy[:int(N/2),:]=-1*dy[:int(N/2),:]
    return(dx,dy)  

#============================================================================================#


#============================================HW2=============================================#
# H using one step using SVD
def HMat_OneStep(imPts):
    """
    Input:
        imPts: 20x2 [P1,P2,P3,P4,...] array with information of 20 points in image plane
        can be more than 20 also but always in multiples of 4
        pts ordered such that line P1P2 and line P3P4 is prependicular
        therefore we have atleast 5 sets of prependicular lines
    Output:
        H: 3x3 homography matrix which will correct both 
        affine and projective distortion in one step
        H*world_pt = image_pt : push fwd
    """
    N=int(np.shape(imPts)[0]/4)# number of line pairs
    #convert imPts to HC
    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
    im_pts_hc[:,:-1]=imPts
    # find lines
    L=np.zeros((N,3));M=np.zeros((N,3))
    for i in range(N):
        L[i,:]=np.cross(im_pts_hc[4*i,:],im_pts_hc[4*i+1,:])
        M[i,:]=np.cross(im_pts_hc[4*i+2,:],im_pts_hc[4*i+3,:])
    #normalize lines: not really needed  and experimentation confirmed it: also it increases the condition number of A
#    L=np.linalg.inv(np.diag(L[:,-1]))@L
#    M=np.linalg.inv(np.diag(M[:,-1]))@M
    M_reshape=np.reshape(M,(N*3))
    # fill out A 
    A=np.zeros((N,6))
    temp=np.multiply(L,M)
    A[:,:3]=temp
    #permutation matrices
    P1=[[0,1,0],[1,0,0],[0,0,0]]
    P2=[[0,0,1],[0,0,0],[1,0,0]]
    P3=[[0,0,0],[0,0,1],[0,1,0]]
    A[:,3]= L@np.kron(np.ones(N),P1)@M_reshape  #l1m2+l2m1
    A[:,4]= L@np.kron(np.ones(N),P2)@M_reshape  #l1m3+l3m1
    A[:,5]= L@np.kron(np.ones(N),P3)@M_reshape  #l2m3+l3m2
    #solve for C using SVD 
    uc,dc,vct=np.linalg.svd(A,full_matrices=True)
    C=vct[-1,:]  #C=[a,c,f,b/2,d/2,e/2]
#    C=C/C[2] #make f as 1
    C=np.array([[C[0],C[3],C[4]],
       [C[3],C[1],C[5]],
       [C[4],C[5],C[2]]])
#   #find H from C using SVD
    u,d,v=np.linalg.svd(C,full_matrices=True)
    print(d)
    H=u
    return(H)



## H using one step: w/o SVD
#def HMat_OneStep(imPts):
#    """
#    Input:
#        imPts: 20x2 [P1,P2,P3,P4,...] array with information of 20 points in image plane
#        can be more than 20 also but always in multiples of 4
#        pts ordered such that line P1P2 and line P3P4 is prependicular
#        therefore we have atleast 5 sets of prependicular lines
#    Output:
#        H: 3x3 homography matrix which will correct both 
#        affine and projective distortion in one step
#        H*world_pt = image_pt : push fwd
#    """
#    N=int(np.shape(imPts)[0]/4)# number of line pairs
#    #convert imPts to HC
#    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
#    im_pts_hc[:,:-1]=imPts
#    # find lines
#    L=np.zeros((N,3));M=np.zeros((N,3))
#    for i in range(N):
#        L[i,:]=np.cross(im_pts_hc[4*i,:],im_pts_hc[4*i+1,:])
#        M[i,:]=np.cross(im_pts_hc[4*i+2,:],im_pts_hc[4*i+3,:])
#    #normalize lines: not really needed  and experimentation confirmed it: also it increases the condition number of A
##    L=np.linalg.inv(np.diag(L[:,-1]))@L
##    M=np.linalg.inv(np.diag(M[:,-1]))@M
#    print('L is \n')
#    print(L)
#    print('M is\n')
#    print(M)
#    M_reshape=np.reshape(M,(N*3))
#    # fill out A and b
#    A=np.zeros((N,5))
#    b=np.zeros((N,1))
#    temp=np.multiply(L,M)
#    A[:,:2]=temp[:,:-1]  #first two columns with l1m1, l2m2
#    b=-1*temp[:,-1] #RHS = -l3m3
#    #permutation matrices
#    P1=[[0,1,0],[1,0,0],[0,0,0]]
#    P2=[[0,0,1],[0,0,0],[1,0,0]]
#    P3=[[0,0,0],[0,0,1],[0,1,0]]
#    A[:,2]= L@np.kron(np.ones(N),P1)@M_reshape  #l1m2+l2m1
#    A[:,3]= L@np.kron(np.ones(N),P2)@M_reshape  #l1m3+l3m1
#    A[:,4]= L@np.kron(np.ones(N),P3)@M_reshape  #l2m3+l3m2
#    #solve for coeffs using psuedo inverse
#    C=(np.linalg.inv((A.T)@A)@A.T)@b  #C=[a,c,b/2,d/2,e/2]
##    C=C/max(np.abs(C)) #not needed
#    C=np.array([[C[0],C[2],C[3]],
#       [C[2],C[1],C[4]],
#       [C[3],C[4],1]])
##    #find A and v
##    S=C[:2,:2]
##    U,D,V=np.linalg.svd(S,full_matrices=True)
##    A=V@np.diag(np.sqrt(D))@V.T
##    v=np.linalg.inv(A)@C[:2,-1]
##    # construct H
##    H=np.identity(3)
##    H[:2,:2]=A
##    H[-1,:-1]=v  #assuming t=0
##    find H from C using SVD
##    u,d,v=np.linalg.svd(C,full_matrices=True)# gives u d and v.T  
#    d,u=np.linalg.eig(C)
#    print(d)
#    H=u
#    return(H)

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


#Homography for Affine Correction using prependicular pairs
def Hmat_AffineCorrection(imPts,H_PC):
    """
    Input:
        imPts=[[X,Y],[X,Y],..] min 8 points in the order PQRSTUVW in the image plane
        Note: imPts should have projective distortion as well
        H_PC = 3x3 Homography that corrects Projective distortion st H_PC.im_pt=affine_pt ie pull-back
        Note: prependicular line pair chosen are (PQ,RS) (TU,VW) and the pairs should have different directions
    Output:purely affine H =[[A,0],[0,1]] : A is 2x2
    such that H*src_pts=im_pt         
    """
    N=int(np.shape(imPts)[0]/4)# number of line pairs
    #convert imPts to HC
    im_pts_hc=np.ones((np.shape(imPts)[0],np.shape(imPts)[1]+1))
    im_pts_hc[:,:-1]=imPts
    #remove Projective distortion
    im_pts_hc=H_PC@im_pts_hc.T
    im_pts_hc=im_pts_hc.T
#    im_pts_hc=np.linalg.inv(np.diag(im_pts_hc[:,-1]))@im_pts_hc
    #find prependicular lines
    L=np.zeros((N,3));M=np.zeros((N,3))
    for i in range(N):
        L[i,:]=np.cross(im_pts_hc[4*i,:],im_pts_hc[4*i+1,:])
        M[i,:]=np.cross(im_pts_hc[4*i+2,:],im_pts_hc[4*i+3,:])
    #normalize lines using third hc coord: imp as info of l1 and l2 is only taken for further step
    L=np.linalg.inv(np.diag(L[:,-1]))@L
    M=np.linalg.inv(np.diag(M[:,-1]))@M
    L=L[:,:-1]
    M=M[:,:-1]
    M_reshape=np.reshape(M,(N*2))
    #fill A and b
    A=np.zeros((N,2)) # coeff matrix
    b=np.zeros((N,1)) # RHS
    temp=np.multiply(L,M)
    A[:,0]=temp[:,0]
    b=-1*temp[:,-1]
    # permutation Matrix
    P=[[0,1],[1,0]] 
    A[:,1]=L@np.kron(np.ones(N),P)@M_reshape
    #solve for S using psuedo inverse
    S=(np.linalg.inv((A.T)@A)@A.T)@b
    S=np.array([[S[0],S[1]],
                [S[1],1]])
    #find A using SVD
    u,d,v= np.linalg.svd(S,full_matrices=True)# gives u d and v.T 
    v=v.T
    A=v@np.diag(np.sqrt(d))@v.T
    # construct H, assuming t=0
    H=np.identity(3)
    H[:2,:2]=A
    return(H)


def mapFitToCanvas(Img,imPts,canvas,H,distype):
    """
    function applies homogrpahy H to source image and fits the resulting matrix to a canvas    
    Input: 
        Img & canvas: Image matrices
        imPts =np.array([[X,Y],[X Y]...])        
        H is a 3x3 matrix: H*world_pt=image_pt ie push forward
        distype= a string L2, sqL2,BiLinear,RoundDown,RoundUp
    Output: Function will write out the merged image    
    """
    resultImg=np.zeros(np.shape(canvas))    
    invH=np.linalg.inv(H)   
    #convert imPts to hc
    src_hc=np.ones((3,np.shape(imPts)[0]))
    src_hc[:2,:]=imPts.T
    #find mapping of imPts on canvas
    im_src_hc=invH@src_hc#columns of im_src_hc are images of src_pts
    #divide each column by the value in the last index of that column to get physical coordinates
    im_src_hc=np.linalg.inv(np.diag(im_src_hc[-1,:]))@im_src_hc.T
    #now rows of im_src_hc are physical coordinates with third column all 1s
    #find parameters for coordinate transformation within canvas plane         
    xmin=min(im_src_hc[:,0])
    xmax=max(im_src_hc[:,0])
    ymin=min(im_src_hc[:,1])
    ymax=max(im_src_hc[:,1])
    deltax=xmax-xmin
    deltay=ymax-ymin
    #now find x and y limits for src image coordiantes
    src_xmin=min(imPts[:,0])
    src_xmax=max(imPts[:,0])
    src_ymin=min(imPts[:,1])
    src_ymax=max(imPts[:,1])
    # iterate over points in canvas coordinates
    for i in range(np.shape(canvas)[0]):
        for j in range(np.shape(canvas)[1]):
            #x: col index & y:row index , pt on canvas is [j,i,1]
            #transform [j,i] on canvas to the block of interest
            x_tr=xmin+(deltax/np.shape(canvas)[1])*j
            y_tr=ymin+(deltay/np.shape(canvas)[0])*i
            #find corresponding point in src image using invH
            x = np.dot(H,[x_tr,y_tr,1])
            x=np.array([x[0]/x[-1],x[1]/x[-1]])
            #now we have a non-integer point in src Img
            #check if the point lies inside the window in the source image
            if x[0]>src_xmin and x[0]<src_xmax and x[1]>src_ymin and x[1]<src_ymax:
                #find new pixel value
                cx = InterpolatePixel(x,Img,distype)
                #assign new pixel value to destImg 
                resultImg[i,j,:]=cx
            else:
                resultImg[i,j,:]=canvas[i,j,:]                
    return(resultImg)
#============================================HW2=============================================#

#============================================HW1=============================================#

# Homograhpy using points
def HMat_Pts(world,img):
    """
    Returns a homography mat such that H*world=img and H[2,2]=1 ie push forward
    img is on destination Image
    world is on source Image
    img,world=np.array([[x1,y1],[x2,y2],[...],[...]])
    """
    n=np.shape(world)[0]
    m=np.shape(world)[1]
    A=np.zeros((2*n,8))
    b=np.reshape(img.T,n*m)

    block1=np.zeros((n,m+1))
    block1[:,-1]=np.ones((1,n))
    block1[:,:2]=world[:,:]
    A[:n,:3]=block1
    A[n:2*n,3:6]=block1

    block2=np.multiply(world.T,-1*img[:,0])
    block3=np.multiply(world.T,-1*img[:,1])
    
    A[:n,6:8]=block2.T    
    A[n:2*n,6:8]=block3.T
    #solve for h using psuedo inverse
    h=(np.linalg.inv((A.T)@A)@A.T)@b
    h=np.concatenate([h,[1]])
    H=np.reshape(h,(3,3))
    return(H)

# function mapFitToCanvas written in HW2 superscedes this one 
def mapImage(srcImg,srcPts,destImg,destPts,H,distype,path,name):
    """    
    Input: 
        srcImg & destImg: Image matrices
        srcpts& destpts =np.array([[X,Y],[X Y]...]) 
        H is a 3x3 matrix
    Output: Function will write out the merged image    
    """
    #find Homography
    resultImg=np.zeros(np.shape(destImg))    
    invH=np.linalg.inv(H)   
    
    xmin=min(srcPts[:,0])
    xmax=max(srcPts[:,0])
    ymin=min(srcPts[:,1])
    ymax=max(srcPts[:,1])
    # iterate over points in destImg
    for i in range(np.shape(destImg)[0]):
        for j in range(np.shape(destImg)[1]):
            # i is row number and corresponds to y coordinate
            #j is col numer and corresponds to x coordinate
            x = np.dot(invH,[j,i,1])
            x=np.array([x[0]/x[-1],x[1]/x[-1]])
            #now we have a non-integer point in src Img
            #check if the point lies insisde the window in the source image
            if x[0]>xmin and x[0]<xmax and x[1]>ymin and x[1]<ymax:
                #find new pixel value
                cx = InterpolatePixel(x,srcImg,distype)# cx is a 3x1 matrix
                #assign new pixel value to destImg 
                resultImg[i,j,:]=cx
            else:
                resultImg[i,j,:]=destImg[i,j,:]
                
    cv2.imwrite(path+name+'.jpg',resultImg)
    return()

# Interpolation of colors from the source image
def InterpolatePixel(x,srcImg,distype):
    """
    Input: 
        x= n.array([X,Y]) coordinates of a point 
        srcImg=source image object a n,m,3 matrix
        distype= a string L2, sqL2,BiLinear,RoundDown,RoundUp
    Output:
        cx= size(1,3) vector of pixel values with only integer values
    The function will interpolate the pixel values of the nearest neighbours and do a rounding
    to get integers
    """
    #finding the nearest interger neighbours
    p1=[int(np.floor(x[0])),int(np.floor(x[1]))]
    p2=[int(np.ceil(x[0])),int(np.floor(x[1]))]
    p3=[int(np.ceil(x[0])),int(np.ceil(x[1]))]
    p4=[int(np.floor(x[0])),int(np.ceil(x[1]))]
    #storing value of pixels at these points 
    #x: col index & y:row index
    c1=srcImg[p1[1],p1[0],:]
    c2=srcImg[p1[1],p1[0],:]
    c3=srcImg[p1[1],p1[0],:]
    c4=srcImg[p1[1],p1[0],:]
    C=np.array([c1,c2,c3,c4])
    #storing weights for interpolation
    w=weights(np.array([p1,p2,p3,p4]),x,distype);#should return a np array 
    #interpolating
    fcx=C.T@w
    fcx=np.floor(fcx)
    cx=fcx.astype(int)
    return(cx)

def weights(pts,x,distype):
    """ 
    function will give a list of weights
    Input:  pts = np.array[[X Y],[X Y]...]
    x=np.array[X,Y]
    distype is a string with options L2, sqL2,BiLinear,RoundDown,RoundUp
    output:
        w=[w1,w2,w3,w4]
    """
    w=[]
    if distype=='L2':
        #use L2 norm for distance
        for i in range(np.shape(pts)[0]):
            w.append(np.linalg.norm(pts[i,:]-x))
        wsum=sum(w)
        w=w/wsum
        return(w)        
    elif distype=='sqL2':
        #use squared L2 norm for distance
        for i in range(np.shape(pts)[0]):
            w.append((np.linalg.norm(pts[i,:]-x))**2)
        wsum=sum(w)
        w=w/wsum
        return(w)
    elif distype=='BiLinear':
        #using bilinear shape functions as weights
        xmin=pts[0,0]
        xmax=pts[1,0]
        ymin=pts[0,1]
        ymax=pts[3,1]
        xi=2*((x[0]-xmin)/(xmax-xmin))-1
        eta=2*((x[1]-ymin)/(ymax-ymin))-1
        w.append((1-xi)*(1-eta)/4.0)
        w.append((1+xi)*(1-eta)/4.0)
        w.append((1+xi)*(1+eta)/4.0)
        w.append((1-xi)*(1+eta)/4.0)
        return(w)
    elif distype=='RoundDown':
        w=[1,0,0,0]
        return(w)
    elif distype=='RoundUp':
        w=[0,0,1,0]
        return(w)
    
#============================================HW1=============================================#