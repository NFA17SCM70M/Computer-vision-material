# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:52:49 2017

@author: prate
"""

import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join
import numpy as np
b=[]


def PCA(data, dims_rescaled_data=895):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m,n = data.shape
    # mean center the data
    np.subtract(data,data.mean(axis=0))
#    np.add(a, b, out=a, casting="unsafe")
#    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs



























imgarr=[]

path = r'D:\homework and assignments\computer vision\face and gender recognition\images' # Source Folder
dstpath = r'D:\homework and assignments\computer vision\face and gender recognition\test' # Destination Folder

userinput=int(input('Enter 1-->Img capture 2-->predefined image :'))
if(userinput==1):
    try: 
        os.remove("frame1.jpg")
    except:
        pass
    vidcap = cv2.VideoCapture(0)
    success,image = vidcap.read()
    count = 0
    if(count==0):
        success = True
        count+=1   
    while success:
        success,image = vidcap.read()
        cv2.imwrite("frame%d.jpg" % count, image)
#        imgloc = r"D:\homework and assignments\computer vision\opencv python\frame1.jpg"
        vidcap.release()
        if(count>0):
            success=False 
    img=cv2.imread('frame1.jpg')
#    img2=cv2.imread('frame1.jpg')
    imggray=cv2.imread('frame1.jpg',cv2.IMREAD_GRAYSCALE)
    imgnew=cv2.equalizeHist(imggray)
    cv2.imwrite('Frame2.jpg',imgnew)
            
if(userinput==2):
    
    try:
        makedirs(dstpath)
    except:
        print ("Directory already exist, images will be written in asme folder")


# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))] 

for image in files:
    try:
        img = cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
        imgnew=cv2.equalizeHist(img)
        dstPath = join(dstpath,image)
        imgarr.append(imgnew)
        cv2.imwrite(dstPath[:-4]+ '.pgm',imgnew)
        
    except:
        print ("{} is not converted".format(image))

a=np.arrray(imgarr)
    
else:
    pass
for i in imgarr:
    b=PCA(i,dims_rescaled_data=895)
    
    


