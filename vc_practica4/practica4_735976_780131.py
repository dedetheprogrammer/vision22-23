from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import tkinter as tk
def harris():
    image = cv.imread("building1.JPG")
    gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #images.append(cv.cvtColor(cv.imread("building2.JPG"), cv.COLOR_BGR2GRAY))
    #images.append(cv.cvtColor(cv.imread("building3.JPG"), cv.COLOR_BGR2GRAY))
    #images.append(cv.cvtColor(cv.imread("building4.JPG"), cv.COLOR_BGR2GRAY))
    #images.append(cv.cvtColor(cv.imread("building5.JPG"), cv.COLOR_BGR2GRAY))
    cv.imshow('fst', image)
    cv.imshow('fst', gray)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.01*dst.max()]=[0,0,255]

    cv.imshow('dst',image)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def orb():
    img1 = cv.imread("building1.JPG", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("building2.JPG", cv.IMREAD_GRAYSCALE)
    # Initiate ORB detector
    orb = cv.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("dst", img3)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

orb()