import cv2
import numpy as np
from matplotlib import pyplot as plt
def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),10,color,-1)
        cv2.circle(img2,tuple(pt2),10,color,-1)

    return img1,img2
    
img1 = cv2.imread('./data/left.jpg')  
img2 = cv2.imread('./data/right.jpg') 
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
#cv2.imshow('image1',img1)
#cv2.imshow('image2',img2)
#global pts1
#global pts2
def my_mouse_callbackL(event,x,y,flags,param): 
 cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
 cv2.namedWindow('image2', cv2.WINDOW_NORMAL) 
 if event == cv2.EVENT_LBUTTONDOWN:
  mouseX,mouseY = x,y
  img1 = cv2.imread('./data/left.jpg',0)  
  img2 = cv2.imread('./data/right.jpg',0) 
  cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
  cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
  orb = cv2.ORB_create()
  # find the keypoints and descriptors with ORB
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)
  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  
  
  # Match descriptors.
  matches = bf.match(des1,des2)
  
  good = []
  pts1 = []
  pts2 = []
  
  D_MATCH_THRES = 65.0
  for m in matches:
      if m.distance < D_MATCH_THRES:
          good.append(m)
          pts2.append(kp2[m.trainIdx].pt)
          pts1.append(kp1[m.queryIdx].pt)
  
  pts1 = np.float32(pts1)
  pts2 = np.float32(pts2)
  
  # compute F
  F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC) #cv2.FM_LMEDS
  pts1 = np.array([[x, y]])
  
  lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
  lines2 = lines2.reshape(-1,3)
  img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
  plt.subplot(122),plt.imshow(img3)
  plt.show()
def my_mouse_callbackR(event,x,y,flags,param):
 cv2.namedWindow('image5', cv2.WINDOW_NORMAL)
 if event == cv2.EVENT_LBUTTONDOWN:
  mouseX,mouseY = x,y
  img1 = cv2.imread('./data/left.jpg',0)  
  img2 = cv2.imread('./data/right.jpg',0) 
  cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
  cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
  orb = cv2.ORB_create()
  # find the keypoints and descriptors with ORB
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)
  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  
  
  # Match descriptors.
  matches = bf.match(des1,des2)
  
  good = []
  pts1 = []
  pts2 = []
  
  D_MATCH_THRES = 65.0
  for m in matches:
      if m.distance < D_MATCH_THRES:
          good.append(m)
          pts2.append(kp2[m.trainIdx].pt)
          pts1.append(kp1[m.queryIdx].pt)
  
  pts1 = np.float32(pts1)
  pts2 = np.float32(pts2)
  
  # compute F
  F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC) #cv2.FM_LMEDS
  pts1 = np.array([[x, y]])
  
  lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
  lines1 = lines1.reshape(-1,3)
  img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
  cv2.imshow("image5",img5)
  

cv2.setMouseCallback('image1',my_mouse_callbackL)
cv2.setMouseCallback('image2',my_mouse_callbackR)	
    
while True:
 cv2.imshow("image1",img1)
 cv2.imshow("image2",img2)
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(pts1)
cv2.destroyAllWindows()