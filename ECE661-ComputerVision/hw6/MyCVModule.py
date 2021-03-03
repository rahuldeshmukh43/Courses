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
from scipy.optimize import least_squares
import os
import matplotlib.pyplot as plt

#%%
#============================================HW6=============================================#
#%%
# function for contour extraction from binary mask
def contour_img(img,option,sigma,N):
    """
    Using 8-neighbours for defining foreground
    Input: img= binary image mxn
            option= 0 or 1 for gaussian smoothing of binary image
            prior to extraction
            sigma= std dev for gaussian filter
            N: size of gaussain filter
    Output: cont_img= contour image (binary image)
    """
    m=img.shape[0];n=img.shape[1]
    # do gaussian smoothing based on option value
    if option==1:
        img = cv2.GaussianBlur(img,(5,5),sigma)
    # create empty image
    cont_img=np.zeros((m,n))
    N=3# size of neighbours window 
    pad_img=np.zeros((m+2*N//2,n+2*N//2))
    pad_img[N//2:-N//2,N//2:-N//2]=img
    # loop over img xy coordinates
    for i in range(m):
        for j in range(n):
            # make NxN window 
            window=pad_img[i:i+2*N//2,j:j+2*N//2]
            # window center should be 1 for contour point
            if window[1,1]==1:
                # if all 1s in window then not a contour point
                if np.sum(window)!=N**2:
                    cont_img[i,j]=1                
    return(cont_img)
#%%
# function for texture based segmentation
def textured_img(img,N):
    """
    Input: img: grayscale image matrix mxn
            N: list of 3x1 specifying sizes of window 
            all N_i are assumed to be odd numbers
    Output: text_img: mxnx3 image matrix with window variances as pixel 
                        values.
    """
    m=img.shape[0];n=img.shape[1]
    #define empty texture image
    text_img=np.zeros((m,n,3))
    #loop over pixel pages in text_img
    for i in range(3):
        #pad the original image using Ni
        pad = N[i]//2
        pad_img=cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
#        pad_img=np.zeros((m+2*(N[i]//2),n+2*(N[i]//2)))
#        pad_img[(N[i]//2):-(N[i]//2),(N[i]//2):-(N[i]//2)]=img
        #loop over xy coordinates of text_img
        for j in range(m):
            for k in range(n):
                window=pad_img[j:j+2*(N[i]//2),k:k+2*(N[i]//2)]
                text_img[j,k,i]=np.var(window)
        #convert text_img pixels to the range of 0-255
        min_x=np.min(text_img[:,:,i])
        max_x=np.max(text_img[:,:,i])
        text_img[:,:,i]=(255/(max_x-min_x))*(text_img[:,:,i]-min_x*np.ones((m,n)))
    #convert text_img pixel values to integers
    text_img=text_img.astype(int)
    return(text_img)
#%%
# function for inverting a biinary image
def or_img(img1,img2):
    """
    Input: img1,2: binary images
    output: logical OR of image
    """
    op=(img1+img2-img1*img2)
    return(op)
#%%
# function for inverting a biinary image
def invert_img(img):
    """
    Input: img: single color channel image
    output: inverted color matrix
    """
    op=255*np.ones((img.shape[0],img.shape[1]))-img
    return(op)
#%%
#function for finding not a binary image
def not_img(mat):
    """
    Input: mat:2d matrix with only ones and zeros
    Output: logical not of mat
    """
    op=np.ones((mat.shape[0],mat.shape[1]))
    op=op-mat
    return(op)
#%%
# function for generating binary mask
def img_mask(img,L,k):
    """
    Input:img: image matrix mxn
            L: number of levels in histogram
            k: separation threshold using Otsu
    Output:mask: binary image size mxn
    """
    m=np.shape(img)[0];n=np.shape(img)[1]
    mask=np.zeros((m,n))
    nbins=256/L
    for i in range(m):
        for j in range(n):
            if img[i,j]//nbins>=k:
                mask[i,j]=1
    return(mask)
#%%
def Otsu(img,L,N):
    """
    Input: img: image matrix mxn
            L: number of levels in histogram
            N: number of iterations to be performed
    Outut: k_star: threshold that separates foreground and background
    """
    # get initial PDF of image pixels
    h=img_histogram(img,L)
    p=h/np.sum(h)
    p_ini=p
    # find u_t total mean
    ip=np.arange(1,L+1)
    u_t=ip@p
    # loop over number of iterations
    k_star=[0]
    ipnew=ip
    for iN in range(N):
        # initialize omega,u sigma as zeros        
        omega=np.zeros((L-k_star[iN],1))
        u=np.zeros((L-k_star[iN],1))
        sigma2_b=np.zeros((L-k_star[iN],1))
        for k in range(L-k_star[iN]):
            omega[k]=np.sum(p[:k])
            u[k]=ipnew[:k]@p[:k]
            if omega[k]>0 and omega[k]<1:
                sigma2_b[k]= ((u_t*omega[k]-u[k])**2)/(omega[k]*(1-omega[k])) 
        # find k_star at max sigma2_b value
        k_star.append(k_star[-1]+np.argmax(sigma2_b))
        # find image with pixel values greater than k_star only
        # find new probability in the region k_star-L
        hnew=h[k_star[-1]:L]
        ipnew=np.arange(k_star[-1]+1,L+1)
        p=hnew/np.sum(hnew)        
        u_t=ipnew@p
    # make plots
    plt.plot(ip,p_ini)
    plt.plot([k_star,k_star],[0,np.max(p_ini)])
    plt.title('PDF for img with '+str(L)+' levels')
    plt.xlabel(str(L)+' levels')
    plt.ylabel('probability')
    plt.show()
    return(k_star[-1])
#%%
# function for generating histogram of image pixels
def img_histogram(img,L):
    """
    Input: img: image matrix mxn
            L: number of levels in histogram
    Outut: h:Lx1 vector with histogram values ni/N
    """
    nbins=256/L
    h=np.zeros((L,1))
    # raster scan of image to read pixel values
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            #update ith bin in p by 1, where i depends on img[i,j]
                h[int(img[i,j]//nbins)]+=1
    return(h)
#============================================HW6=============================================#

#============================================HW5=============================================#
#%%
#function for panoramic image generation
def Panorama(readpath,savepath,savename,plot):
    """
    Input:  readpath: string type path of a folder containing all images ...
            named sequentially as 1.jpg,2,3.... folder should have only images
            assuming all images are of the same sizes m,n
            savepath: saving location of the panoramic image, string
            savename: name of the saved image, string
            plot: 1 or 0: 1= plot pairwise images with inliers an outliers
    Output: saved image at savepath location , empty return
    """
    print('panorama begins')
    img_names=os.listdir(readpath)
    print(img_names)
    N=len(img_names)#total number of images
    # read all images in the sequence 1,2,3,4....
    img=[];color_img=[]
    for i in range(N):
        temp=cv2.imread(readpath+img_names[i],0)
        temp2=cv2.imread(readpath+img_names[i],-1)
        img.append(temp)
        color_img.append(temp2)
    m=np.shape(img[0])[0];n=np.shape(img[0])[1]
    print('Images read')
    # find homographies sequentially ie H12, H23, H34, ...
    H=[]
    for i in range(N-1):
        temp=AutoHomoCalc(img[i],img[i+1],(plot,savepath+savename+str(i)+str(i+1)),color_img[i],color_img[i+1])
        H.append(temp)
    print('Individual Homographies calculated')
    # find middle image: 
    if N%2==0:
        mid=int(N/2)
    else:
        mid=int((N+1)/2)
    #find homographies to middle image ie projecting to mid-1 image: numbering from 0-(N-1)
    H_to_mid=[]
    for i in range(N-1):
        temp=np.identity(3)
        if i<mid-1:
            for j in range((mid-1)-i):
                temp=np.array(H[i+j])@temp
            H_to_mid.append(temp)
        else:
            for j in range(i-(mid-1)+1):
                temp=np.array(H[(mid-1)+j])@temp
            H_to_mid.append(np.linalg.inv(temp))
    # add H=I for center image
    if N%2==0:
        H_to_mid.insert(mid,np.identity(3))
    else:
        H_to_mid.insert(mid-1,np.identity(3))
    # find transformed corners of all images in the plane of mid
    corners=np.array([[0,0,1],
                      [n,0,1],
                      [n,m,1],
                      [0,m,1]])
    all_corners=np.kron(np.ones((N,1)),corners)# col vector 4Nx3
    all_corners=all_corners.T # 3x4N vector last row all 1
    all_H=np.zeros((3,3*N)) #row vector 3x3N: need as row vector only
    for i in range(N): all_H[:,3*i:3*i+3]=H_to_mid[i] 
    all_H_inv=np.zeros((3,3*N))#row vector 3x3N: need as row vector only
    for i in range(N): all_H_inv[:,3*i:3*i+3]=np.linalg.inv(H_to_mid[i]) 
    
    temp=np.zeros((3*N,4*N))
    for i in range(N): temp[3*i:3*i+3,4*i:4*i+4]=all_corners[:,4*i:4*i+4]
    tr_corners=all_H@temp# tarnsformed corners 3x4N last row all 1
    # scale tr_corners to real coordinates
    tr_corners=tr_corners.T # 4Nx3 
    tr_corners=np.linalg.inv(np.diag(tr_corners[:,-1]))@tr_corners    
    #find canvas size
    xmin=int(np.floor(np.min(tr_corners[:,0])))
    xmax=int(np.ceil(np.max(tr_corners[:,0])))
    deltax=xmax-xmin
    ymin=int(np.floor(np.min(tr_corners[:,1])))
    ymax=int(np.ceil(np.max(tr_corners[:,1])))
    deltay=ymax-ymin
    #create empty canvas 
    canvas=np.zeros(((ymax-ymin),(xmax-xmin),3)) #filan image of this size, no scaling
    print('canvas size:'+str(ymax-ymin)+'x'+str(xmax-xmin))
    # loop over canvas indices to assign pixel values
    distype='RoundDown'
    for i in range(deltay):
        for j in range(deltax):
            #transform [j,i] on canvas to the block of interest
            x_tr=xmin+j
            y_tr=ymin+i
            hc_mp=[x_tr,y_tr,1] # hc coordinate in mid plane
            #find transformed back hc coordinate
            hc_op=all_H_inv@(np.kron(np.identity(N),hc_mp)).T # hc own plane 3xN
            #scale to convert to real coordinates
            hc_op=hc_op.T # Nx3
            hc_op=np.linalg.inv(np.diag(hc_op[:,-1]))@hc_op
            x=hc_op[:,:-1] # only x and y coordinates in all planes            
            #find if coordinates lie in the image region of the specific plane
            count=0
            p=np.zeros((1,3))#pixel values
            for k in range(N):
                temp_x=x[k,:]
                if temp_x[0]>0 and temp_x[0]<n and temp_x[1]>0 and temp_x[1]<m:
                    p=p+InterpolatePixel(temp_x,color_img[k],distype)
                    count+=1
                    break
#             take average of summed pixel values and make integer
#            if count>0:
#                p=p/count
#            p=np.floor(p)
#            p=p.astype(int)
            #assign pixel value
            canvas[i,j,:]=p
    # write image
    cv2.imwrite(savepath+savename+'.jpg',canvas)
    return()
#%%
# function for automatic calculation of homography between two images
# using SIFT interest points, RANSAC, and Non-Linear Least squares Levenberg marqaudt
def AutoHomoCalc(img1,img2,options,img1c,img2c):
    """
    Input: img1,img2: grayscale image matrices
    img1c,img2c: colored images only for plotting purposes    
    options=tuple(plot,name)        
    plot: 0 or 1, 1= make a plot of inliers
    name: name of saved image, full name with path            
    Output: H: homograhy
    """
    # find sift points of images
    pts1,des1=SIFT_detector(img1)
    pts2,des2=SIFT_detector(img2)
    # find euclidean matches between images
    match_pts1,match_pts2=Euclidean_sift_match(pts1,des1,pts2,des2) # returns list of lists
    # find inlier points using RANSAC    
    in_indices=RANSAC(match_pts1,match_pts2)  # returns array of indices
    #store inliers and outlier separately
    in_pts1=np.array(match_pts1)
    in_pts1=in_pts1[in_indices,:]
    out_pts1=match_pts1
    for i in sorted(in_indices,reverse=True): del out_pts1[i]
    out_pts1=np.array(out_pts1)
    
    in_pts2=np.array(match_pts2)
    in_pts2=in_pts2[in_indices,:]
    out_pts2=match_pts2
    for i in sorted(in_indices,reverse=True): del out_pts2[i]
    out_pts2=np.array(out_pts2)
    ## plot inliers(green) and outliers(red) on one image and save image
    if options[0]==1:
        plot_inliers(img1c,img2c,in_pts1,out_pts1,in_pts2,out_pts2,options[1])
        
    # find Initial homography using Linear Least sqaures with all inliers
    H=HMat_Pts(in_pts1,in_pts2)
    # Non-Linear Least Squares fit to find better Homography
    # find intial solution for LM
    p0=np.reshape(H,np.shape(H)[0]*np.shape(H)[1]) # p0=[h11,h12,h13,h21,h22,h23,h31,h32,h33]
    # call Least squares with cost function as argument
    optim_results=least_squares(cost_fun,p0,method='lm',args=(in_pts1,in_pts2))
#    print('---------------------LM results-------------------------')
#    print('solution is :'+str(optim_results['x']))
#    print('min function value is:'+str(optim_results['cost']))
#    print('gradient is:'+str(optim_results['grad']))
#    print('number of function evalutaions were:'+str(optim_results['nfev']))
#    print('termination status:'+str(optim_results['status']))
#    print('termination message:'+str(optim_results['message']))
    
    p=optim_results['x']
    H=np.reshape(p,(3,3))
    H=H/H[-1,-1]
    return(H)
#%%
# cost function for Non-Linear Least Squares Fit
def cost_fun(p,in_pts1,in_pts2):
    """
    We are estimating homography such that H*in_pts1=in_pts2
    Input: p is an array of size 9 in the format np.array([h11,h12,h13,h21,h22,h23,h31,h32,h33])
           in_pts1,in_pts2: inlier points as np.array([[x1,y1],[x2,y2],...])
    Output: scalar cost function value: sum of sqaures of error 
    """
    #reshape p to form H
    H=np.reshape(p,(3,3))    
    #convert in_pts1 to hc representation
    in_pts1_hc=np.ones((np.shape(in_pts1)[0],3))
    in_pts1_hc[:,:-1]=in_pts1
    # find estimated point2 using homography
    est_hc= H@in_pts1_hc.T#estimated point as stacked col vectors
    est_hc=est_hc.T #points as row vectors
    #scale with 3rd ordinate to get real point
    est=np.linalg.inv(np.diag(est_hc[:,-1]))@est_hc
    est=est[:,:-1] # removing third column
    # define error and cost function
    X=np.reshape(in_pts2,np.shape(in_pts2)[0]*np.shape(in_pts2)[1]) #col vector
    f=np.reshape(est,np.shape(est)[0]*np.shape(est)[1]) #col vector
    error=X-f #col vector
    cost=error# Least sqaures function with lm option requires a M dimensional vector as objective function
    #cost= error.T@error
    return(cost)

#%%
def plot_inliers(img1,img2,in_pts1,out_pts1,in_pts2,out_pts2,name):
    """
    Input: img1,img2: colored images of the same sizes mxn
           in_pts1,in_pts2: np array [[x,y],..] inliers to be plotted in green circle
           out_pts1,out_pts2: np array [[x,y],..] outliers to be plotted in red circles
           name: filename of image with full path
    Output:  save the image, empty return
    """
    m=np.shape(img1)[0];n=np.shape(img1)[1]
    #convert in_pts and out_pts to integers
    in_pts1=in_pts1.astype(int)
    in_pts2=in_pts2.astype(int)
    out_pts1=out_pts1.astype(int)
    out_pts2=out_pts2.astype(int)
    # make empty canvas of twice width
    img=np.zeros((m,2*n,3))
    #copy original images onto the two halves of canvas
    img[:,:n,:]=img1
    img[:,n:,:]=img2
    #plot inliers in green
    r=4
    for i in range(len(in_pts1)):
        cv2.circle(img,tuple(in_pts1[i,:]),r,(0,255,0),-1)# left image
        cv2.circle(img,(n+in_pts2[i,0],in_pts2[i,1]),r,(0,255,0),-1)# right image
        cv2.line(img,tuple(in_pts1[i,:]),(n+in_pts2[i,0],in_pts2[i,1]),(0,255,0),1)#line joining inliers   
    #plot outliers in red
    for i in range(len(out_pts1)):
        cv2.circle(img,tuple(out_pts1[i]),r,(0,0,255),-1)# left image
        cv2.circle(img,(n+out_pts2[i,0],out_pts2[i,1]),r,(0,0,255),-1)# right image
    #save image
    cv2.imwrite(name+'.jpg',img)
    return()
#%%
# function for Random sampling and consensus
def RANSAC(pts1,pts2):
    """
    Input: pts1,pts2: matching points with inliers and outliers 
                      format:[[x1,y1],[x2,y2],....] list of lists of points 
                      
    Output: in_indices_store: indices of inlier points of image  1 and image 2 respectively
                            format:[index1,index2,....] list of index numbers
    """
    #----define RANSAC parameters:-----#
    p=0.99  # probability that at least one of the N trials will be free of outliers
    n=6     # minimal set of correspondences chosen randomly for constructing homography
    delta=2 # decision threshold to construct inlier set    
    e=0.1   # probability that a correspondence is an outlier
    N=int(np.ceil(np.log(1-p)/np.log(1-(1-e)**n))) #number of iterations needed
    #----------------begin-------------------#
    n_total=len(pts1) # total number of correspondences
    # convert pts lists to array and into homogeneous form
    pts1_hc=np.ones((n_total,3))
    pts1_hc[:,:-1]=pts1
    pts2_hc=np.ones((n_total,3))
    pts2_hc[:,:-1]=pts2
    

    badinliers=True
    while badinliers:
        count=0
        size_old=0        
        while count<N:
            #randomly sample n points from pts1 and pts2
            rand_indices=np.random.randint(n_total,size=n)
            ipts1=pts1_hc[rand_indices,:]
            ipts2=pts2_hc[rand_indices,:]
            # find homography between these points: linear least sqaures
            iH=HMat_Pts(ipts1[:,:-1],ipts2[:,:-1])#gives H12*ipts1=ipts2: push fwd
            # find estimated pts2 and error distance
            est_pts2=iH@(pts1_hc.T) # est pts as col vectors
            est_pts2=est_pts2.T # est pts as row vectors
            # scale est_pts2 to get third col as 1
            temp=est_pts2[:,-1]
            est_pts2=np.linalg.inv(np.diag(temp))@est_pts2
            error=pts2_hc[:,:-1]-est_pts2[:,:-1]
            distance=np.power(error,2)
            distance=distance@np.ones((2,1))
            distance=np.sqrt(distance)
            # consensus
            in_indices=np.where(distance<=delta)[0] # inlier indices
            size_new=len(in_indices) #size of ith inlier set           
            if size_new>size_old:
                in_indices_store=in_indices #store the set of indices with largest size
                size_old=size_new       
            count+=1
        if len(in_indices_store)>(1-e)*n_total:
            badinliers=False
        else:
            e=e*2 #bad dataset than what was assumed
            #update N ie more iterations of RANSAC
            N=int(np.ceil(np.log(1-p)/np.log(1-(1-e)**n))) 
    print('RANSAC max inlier set was'+str(len(in_indices_store)))
    return(in_indices_store)
#%%
#function for Euclidean match for SIFT points 
#V1: for HW5 script: make sure V0 is commented out!!
def Euclidean_sift_match(pts1,des1,pts2,des2):
    """
    Input: pts1,pts2= [[x,y],[],..] list of lists of interest points
           des1,des2 = vector of 128 size, sift descriptor stacked as row vectors
    Output: match_pts1,match_pts2= [[x1,y1],[x2,y2],....] list of lists of points 
                                    both have same sizes
            Euclidean distance is less than a dynamic threshold
    """
    match_pts1=[]
    match_pts2=[]
    eu=np.zeros((len(pts1),len(pts2)))    
    for i in range(len(pts1)):
        for j in range(len(pts2)):
            eu[i,j]=np.linalg.norm(des1[i,:]-des2[j,:])
    
    eu=eu/np.min(eu)
    dy_threshold=2
    for i in range(len(pts1)):            
        eu_min=np.min(eu[i,:])
        if eu_min<dy_threshold:
            j_min=np.argmin(eu[i,:])
            match_pts1.append(pts1[i])
            match_pts2.append(pts2[j_min])
    print('size of Euclidean match was:'+str(len(match_pts1)))
    return(match_pts1,match_pts2)   
#%%
#function for SIFT interest points detection
def SIFT_detector(img):
    """
    detects SIFT interest points using openCV function and returns pts and des
    in my particular format.
    Input: img: image matrix
    Output: pts: [[x,y],...] : list of lists of pt coordinates as row vectors
            des: 128 bit  SIFT descriptor matrix arranged as stacked row vector 
            =[d1,d2,d3,...] di is row vector of size 128
    """
    #sift key points
    sift=cv2.xfeatures2d.SIFT_create() #defining structure
    sift_pts,des=sift.detectAndCompute(img,None)
    pts=convert_SIFT_myFormat(sift_pts)
    return(pts,des)
#%%
#function for converting sift key points into my format 
#V1: to be used for HW5: make sure V0 is commented out!!
def convert_SIFT_myFormat(kp):
    """
    Input: kp: keypoint structure given by SIFT, point coordinates in kp.pt
    Output: [[x,y],...] : list of lists of pt coordinates as row vectors
    """
    mykp=[]
    for ikp in kp: 
        mykp.append([ikp.pt[0],ikp.pt[1]])
    return(mykp)
#============================================================================================#
#%%
#============================================HW4=============================================#

##function for Euclidean match for SIFT points 
##V0: for HW4 script
#def Euclidean_sift_match(pts1,des1,pts2,des2):
#    """
#    Input: pts1,pts2= [(x,y),(),..] list of tuples of interest points
#           des1,des2 = vector of 128 size, sift descriptor
#    Output: pt_pairs= [[(x1,y1),(x2,y2)],....] list of list of point pair tuples such that
#            Euclidean distance is less than a dynamic threshold
#    """
#    pt_pairs=[]
#    sq_eu=np.zeros((len(pts1),len(pts2)))    
#    for i in range(len(pts1)):
#        for j in range(len(pts2)):
#            sq_eu[i,j]=np.linalg.norm(des1[i,:]-des2[j,:])
#    
#    sq_eu=sq_eu/np.min(sq_eu)
#    dy_threshold=2
#    print(dy_threshold)
#    for i in range(len(pts1)):            
#        sq_eu_min=np.min(sq_eu[i,:])
#        if sq_eu_min<dy_threshold:
#            j_min=np.argmin(sq_eu[i,:])
#            pt_pairs.append([pts1[i],pts2[j_min]])
#    return(pt_pairs)            

##function for converting sift key points into my format 
##V0: was used for HW3 script
#def convert_SIFT_myFormat(kp):
#    """
#    Input: kp: keypoint structure given by SIFT, point coordinates in kp.pt
#    Output: [(x,y),...] : list of tuples of coorinates
#    """
#    mykp=[]
#    for ikp in kp: 
#        #convert points to floored integer
#        mykp.append((int(ikp.pt[0]),int(ikp.pt[1])))
#    return(mykp)

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

#%%
#============================================HW3=============================================#
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
#============================================HW3=============================================#
#%%
#============================================HW2=============================================#

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
    
#============================================HW2=============================================#
