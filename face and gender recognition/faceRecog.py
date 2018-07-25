# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:44:42 2017

@author: Quantum
"""
import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join


path = r'D:\homework and assignments\computer vision\face and gender recognition\images' # Source Folder
dstpath = r'D:\homework and assignments\computer vision\face and gender recognition\test' # Destination Folder
dstpath2= r'D:\homework and assignments\computer vision\face and gender recognition\Histeq'

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
        cv2.imwrite(dstPath,imgnew)
    except:
        print ("{} is not converted".format(image))
        
        

#        
