import cv2
import numpy as np
import os

pathsrc = '../images/HaarCascade_Train/Negative/AC.jpg' #change your source picture path here

pathdst = "../images/HaarCascade_Train/Negative/" #change your destination path here
    
if __name__ == '__main__' :

    # Read image
    im = cv2.imread(os.path.join(pathsrc)) #let's choose the image inside the path
    i = 863
   #change here
    while(True):
        # Select ROI
        showCrosshair = False
        fromCenter = False

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)

        r = cv2.selectROI("image", im, fromCenter, showCrosshair)

        # Crop image
        imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        # Display cropped image
        cv2.imwrite(os.path.join(pathdst, str(i)+".png"), imCrop)
        i=i+1
    cv2.waitKey(0)

    # im = cv2.imread(os.path.join(path, 'bce128noise20e15_1.png'))
    # showCrosshair = False
    # fromCenter = False
    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # r = cv2.selectROI("image", im, fromCenter, showCrosshair)
    # imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    # cv2.imwrite(os.path.join(path1, "generated.png"), imCrop)
    
