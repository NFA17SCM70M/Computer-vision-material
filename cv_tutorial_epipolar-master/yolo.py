import cv2
import numpy as np
from matplotlib import pyplot as plt
global pts1
global pts2
global pts3
global pts4
pts3=[]
pts4=[]
pts5=[]
pts6=[]
C=[]


def fundamental_matrix(point1,point2):
    try:
#        x1,x2,npts = process_input_pointpairs(args)
        F = eight_point_algorithm(x1,x2)
        return F
    except( ShapeError, e):
        print ('ShapeError, exception message:', e)
        return None
    
    
    
def eight_point_algorithm(x1,x2):
    
    # perform the normalization (translation and scaling)
    x1, T1 = normalize2dpts(x1);
    x2, T2 = normalize2dpts(x2);
    
    # assemble the constraint matrix
    A = constraint_matrix(x1,x2)
    
    # A*vec(F) = 0 implies that the fundamental matrix F can be extracted from 
    # singular vector of V corresponding to smallest singular value
    (U, S, V) = np.linalg.svd(A)
    V = V.conj().T;
    F = V[:,8].reshape(3,3).copy()
    
    # recall that F should be of rank 2, do the lower-rank approximation by svd
    (U,D,V) = np.linalg.svd(F);
    F = np.dot(np.dot(U,np.diag([D[0], D[1], 0])),V);

    # denormalize
    F = np.dot(np.dot(T2.T,F),T1);
    return F
    
    
def constraint_matrix(x1,x2):
    npts = x1.shape[1]
    # stack column by column
    A = C[x2[0]*x1[0], x2[0]*x1[1], x2[0], x2[1]*x1[0], x2[1]*x1[1], x2[1], x1[0], x1[1], np.ones((npts,1))]
    return A

def normalize2dpts(pts):
    ''' This function translates and scales the input (homogeneous) points 
    such that the output points are centered at origin and the mean distance
    from the origin is sqrt(2). As shown in Hartley (1997), this normalization
    process typically improves the condition number of the linear systems used
    for solving homographies, fundamental matrices, etc.
    
    References:
        Richard Hartley, PAMI 1997
        Peter Kovesi, MATLAB functions for computer vision and image processing,
        http://www.csse.uwa.edu.au/~pk
     '''
    if pts.shape[0]!=3:
        raise ShapeError('pts must be 3xN')
    
    finiteind = abs(pts[2]) > finfo(float).eps
    pts[0,finiteind] = pts[0,finiteind]/pts[2,finiteind]
    pts[1,finiteind] = pts[1,finiteind]/pts[2,finiteind]
    pts[2,finiteind] = 1
    
    # Centroid of finite points
    c = [np.mean(pts[0,finiteind]), np.mean(pts[1,finiteind])] 
    
    
    # Shift origin to centroid.
    newp0 = pts[0,finiteind]-c[0] 
    newp1 = pts[1,finiteind]-c[1] 

    meandist = np.mean(np.sqrt(newp0**2 + newp1**2));
    
    scale = np.sqrt(2)/meandist;
    '''
    T = [scale   0   -scale*c(1)
         0     scale -scale*c(2)
         0       0      1      ];
    '''
    T = np.eye(3)
    T[0][0] = scale
    T[1][1] = scale
    T[0][2] = -scale*c[0]
    T[1][2] = -scale*c[1]
    newpts = np.dot(T, pts)    
    
    return newpts, T








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

def my_mouse_callbackL(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img1,(x, y), 5, (0, 0, 255), -1)
     pts3.append([x,y])
     pts5.append([x,y])
     
     print(x, y)

  
  
  
def my_mouse_callbackR(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img2,(x, y), 5, (0, 0, 255), -1)
     pts4.append([x,y])
     pts6.append([x,y])
     print(x, y)
     
     
     
     
def my_mouse_callbackL1(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img1,(x, y), 5, (0, 0, 255), -1)
     index1=pts5.index([x,y])
     img7,img8=draw_single_Line(imgx,imgy,lines1,pts5[index1],pts5[index1])
     plt.subplot(121),plt.imshow(img7)
     plt.subplot(122),plt.imshow(img8)
     plt.show()
     
  
  
  
def my_mouse_callbackR1(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img2,(x, y), 5, (0, 0, 255), -1)
     index2=pts5.index([x,y])
     img9,img10=draw_single_Line(imgy,imgx,lines1,pts5[index2],pts5[index2])
     plt.subplot(121),plt.imshow(img9)
     plt.subplot(122),plt.imshow(img10)
     plt.show()
     
def draw_single_Line(img1,img2,lines,pts1,pts2):
    
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r in lines:
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pts1),10,color,-1)
        cv2.circle(img2,tuple(pts2),10,color,-1)

    return img1,img2








img1 = cv2.imread('./data/001.jpg',0)  
img2 = cv2.imread('./data/002.jpg',0) 
imgx = cv2.imread('./data/001.jpg',0)  
imgy = cv2.imread('./data/002.jpg',0) 
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)


orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print(des1, des2)
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
print(F)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
pts1 = np.array([[0,0]])
pts2 = np.array([[0,0]])


cv2.setMouseCallback('image1',my_mouse_callbackL)
cv2.setMouseCallback('image2',my_mouse_callbackR)
while True:
 cv2.imshow("image1",img1)
 cv2.imshow("image2",img2)
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pts3=np.float32(np.array(pts3))
pts4=np.float32(np.array(pts4))
X, mask = cv2.findFundamentalMat(pts3,pts4)


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts4.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts3,pts4)



# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts3.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts4,pts3)


plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

print("The coordinates of image 1 are",pts5)
print("The coordinates of image 2 are",pts6)

cv2.namedWindow('imagex',cv2.WINDOW_NORMAL)
cv2.namedWindow('imagey',cv2.WINDOW_NORMAL)

cv2.setMouseCallback('imagex',my_mouse_callbackL1)
cv2.setMouseCallback('imagey',my_mouse_callbackR1)
while True:
 cv2.imshow("imagex",imgx)
 cv2.imshow("imagey",imgy)
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break




