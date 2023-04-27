import cv2 as cv
import numpy as np
# https://www.geeksforgeeks.org/feature-detection-and-matching-with-opencv-python/
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def HARRIS(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayscale = np.float32(grayscale)
    harris    = cv.cornerHarris(grayscale, blockSize=2, ksize=3, k=0.04)
    # Result is dilated for marking the corners, not important
    harris    = cv.dilate(harris, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[harris > 0.01 * harris.max()] = [0,255,0]
    return img

# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
def SIFT(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift      = cv.SIFT_create()
    kp        = sift.detect(grayscale, None)
    return cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/
def ORB(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb       = cv.ORB_create()
    kp        = orb.detect(grayscale, None)
    return cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html
def AKAZE(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    akaze     = cv.AKAZE_create()
    kp        = akaze.detect(grayscale, None)
    return cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img = cv.imread('./BuildingScene/building1.JPG')
cv.namedWindow('Practica 4', cv.WINDOW_AUTOSIZE)
cv.imshow('Practica 4', AKAZE(img))
cv.waitKey(0)

# while 1:
#     cv.imshow('Practica 4', Harris(img))
#     if cv.waitKey(0) == 27:
#         break
cv.destroyWindow('Practica 4')