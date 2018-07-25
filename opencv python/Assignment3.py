import cv2
import operator
from matplotlib import pyplot as plt

def findCorners1(img1, window_size, k, thresh):
   
    #Find x and y derivatives
    dx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img1.shape[0]
    width = img1.shape[1]

    cornerList1 = []
    newImg1 = img1.copy()
    color_img1 = cv2.cvtColor(newImg1, cv2.COLOR_GRAY2RGB)
    offset = int(window_size/4)

    #Loop through image and find our corners
    for y in range(offset, height):
        for x in range(offset , width):
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
#                print(x, y, r)
                cornerList1.append([x, y, r])
                color_img1.itemset((y, x, 0), 0)
                color_img1.itemset((y, x, 1), 0)
                color_img1.itemset((y, x, 2), 255)
                
    return color_img1, cornerList1





def findCorners(img, window_size, k, thresh):
   
    #Find x and y derivatives
    dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    Ixx = cv2.GaussianBlur(dx*dx,(5,5),0)
    Iyy = cv2.GaussianBlur(dy*dy,(5,5),0)
    Ixy = cv2.GaussianBlur(dx*dy,(5,5),0)
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = int(window_size/2)

    #Loop through image and find our corners
    print("Finding Corners...")
    for y in range(offset, height):
        for x in range(offset , width):
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
#                print(x, y, r)
                cornerList.append([x, y, r])
                color_img.itemset((y, x, 0), 0)
                color_img.itemset((y, x, 1), 0)
                color_img.itemset((y, x, 2), 255)
                
    return color_img, cornerList

def main():
#    sigma=int(input('Enter variance in gaussian ---->'))
    window_size =int(input('Enter window size--->'))
    k = float(input('Enter corner response--->'))
    thresh =int(input('Enter threshold size--->')) 
    print("Window Size: " + str(window_size))
    print("K Corner Response: " + str(k))
    print("Corner Response Threshold:" + str(thresh))

    img = cv2.imread('test.png',0)
#    gray = np.float32(img)
    img1=cv2.imread('test.png',0)
#    gray1 = np.float32(img1)
    if img is not None:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(img.shape) == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        print("Shape: " + str(img.shape))
        print("Size: " + str(img.size))
        print ("Type: " + str(img.dtype))
        print("Printing Original Image...")
        print(img)
        finalImg, cornerList = findCorners(img, int(window_size), float(k), int(thresh))
        finalImg1, cornerList1 = findCorners(img1, int(window_size), float(k), int(thresh))
#        dst = cv2.dilate(gray,None)
#        dst1 = cv2.dilate(gray1,None)
#        ret, dst = cv2.threshold(img,0.01*dst.max(),255,0)
#        ret1, dst1 = cv2.threshold(img1,0.01*dst.max(),255,0)
#        dst = np.uint8(dst)
#        dst1 = np.uint8(dst1)

        if finalImg is not None:
            cv2.imwrite("finalimage.png", finalImg)
        if finalImg1 is not None:
            cv2.imwrite("finalimage1.png", finalImg1)    
            
        cv2.imshow("image 1", finalImg)
        cv2.imshow("image 2", finalImg1)
        cv2.waitKey(0)

        # Write top 100 corners to file
        cornerList.sort(key=operator.itemgetter(2))
        outfile = open('corners.txt', 'w')
        for i in range(100):
            outfile.write(str(cornerList[i][0]) + ' ' + str(cornerList[i][1]) + ' ' + str(cornerList[i][2]) + '\n')
        outfile.close()
        
        
         # Write top 100 corners to file for image 2
        cornerList1.sort(key=operator.itemgetter(2))
        outfile = open('corners1.txt', 'w')
        for i in range(100):
            outfile.write(str(cornerList1[i][0]) + ' ' + str(cornerList1[i][1]) + ' ' + str(cornerList1[i][2]) + '\n')
        outfile.close()
        
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(finalImg,None)
        kp2, des2 = orb.detectAndCompute(finalImg1,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(finalImg,kp1,finalImg1,kp2,matches[:10],None, flags=2)
        plt.imshow(img3)
        plt.show()


if __name__ == "__main__":
    main()