"""
ECE661: hw5 main file
@author: rahul deshmukh
email: deshmuk5@purdue.edu
"""

#import libraries
import sys

sys.path.append('../../')
import MyCVModule as MyCV
#define path
readpath='../images/myfountain/' # path of images to be read
savepath='../results/' #path for saving results of images
savename='myfountain'

plot=1 # print pair wise images of inliers and outliers
MyCV.Panorama(readpath,savepath,savename,plot)
