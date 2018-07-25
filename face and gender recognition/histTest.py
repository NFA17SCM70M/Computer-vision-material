# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:12:17 2017

@author: prate
"""
import cv2
im = cv2.imread('1.jpg',0)         #当前目录下的图片：vi.png
cv2.imshow('image1', im)
cv2.waitKey(0)

eq = cv2.equalizeHist(im)         #灰度图片直方图均衡化
cv2.imshow('image2',eq)
cv2.waitKey(0)

cv2.imwrite('vi2.jpg',eq)

