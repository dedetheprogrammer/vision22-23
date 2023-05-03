import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

#------------------------------------------------------------------------------
# Features detection operators
# https://www.geeksforgeeks.org/feature-detection-and-matching-with-opencv-python/
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def HARRIS(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayscale = np.float32(grayscale)
    harris    = cv.cornerHarris(grayscale, blockSize=2, ksize=3, k=0.04)
    # Result is dilated for marking the corners, not important
    return  cv.dilate(harris, None)

# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
def SIFT(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift      = cv.SIFT_create()
    kp, des   = sift.detectAndCompute(grayscale, None)
    return kp, des, cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/
def ORB(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb       = cv.ORB_create()
    kp, des   = orb.detectAndCompute(grayscale, None)
    return kp, des, cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html
def AKAZE(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    akaze     = cv.AKAZE_create()
    kp, des   = akaze.detectAndCompute(grayscale, None)
    return kp, des, cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#------------------------------------------------------------------------------
# Features matching matchers:
def BRUTE_FORCE_MATCH(src_des, dst_des, k):
    bf      = cv.BFMatcher()
    matches = bf.match(src_des, dst_des)
    matches = sorted(matches, key= lambda x:x.distance)
    return matches[:k]
    # return matches, cv.drawMatches(fst, fst_kp, snd, snd_kp, matches[:200], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 

def KNN_MATCH(src_des, dst_des, k=2, threshold=0.75):
    bf      = cv.BFMatcher()
    matches = bf.knnMatch(src_des, dst_des, k=2)
    good    = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append(m)
    return np.asarray(good)

def FLANN_MATCH(src_des, dst_des, FLANN_INDEX_KDTREE=1, trees=5, checks=50, k=2, threshold=0.75):
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
    search_params = dict(checks=checks)
    flann         = cv.FlannBasedMatcher(index_params, search_params)
    matches       = flann.knnMatch(src_des, dst_des, k=k)
    good          = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append(m)
    return np.asarray(good)
    ## Need to draw only good matches, then have to create a mask
    #matchesMask        = [[0,0] for i in range(len(matches))]
    ## Ratio test as per Lowe's paper
    #for i,(m,n) in enumerate(matches):
    #    if m.distance < 0.7*n.distance:
    #        matchesMask[i]=[1,0]
    #return matches, cv.drawMatchesKnn(fst, fst_kp, snd, snd_kp, matches, None, matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)
    #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #        singlePointColor = None,
    #        matchesMask = matchesMask, # draw only inliers
    #        flags = 2)
    #return matches, cv.drawMatches(fst, fst_kp, snd, snd_kp, good, None, **draw_params)

def ImageSnitching(src_img, dst_img, operator, matcher, threshold):

    if (operator.__name__ == 'HARRIS'):

        # Features detection
        ti0 = time.time()
        src_kp = HARRIS(src_img)
        dst_kp = HARRIS(dst_img)
        tf0 = time.time()
        # Threshold for an optimal value, it may vary depending on the image.
        src_img[src_kp > 0.01 * src_kp.max()] = [0,255,0]
        dst_img[dst_kp > 0.01 * dst_kp.max()] = [0,0,255]

        print('========================================================')
        print(f' Deteccion mediante {operator.__name__}')
        print('========================================================')
        print(f'> Caracteristicas detectadas en la imagen de origen: {len(src_kp)}')
        print(f'> Caracteristicas detectadas en la imagen de origen: {len(dst_kp)}')
        print(f'> Tiempo de deteccion: {(tf0 - ti0):.4f} s')
        print('========================================================')
        return cv.hconcat([src_img, dst_img])

    else:
        # Features detection
        ti0 = time.time()
        src_kp, src_des, _ = operator(src_img)
        dst_kp, dst_des, _ = operator(dst_img)
        tf0 = time.time()

        # Features matching
        ti1 = time.time()
        matches = matcher(src_des, dst_des, threshold)
        tf1 = time.time()

        print('========================================================')
        print(f' Deteccion mediante {operator.__name__} y matching mediante {matcher.__name__}')
        print('========================================================')
        print(f'> Caracteristicas y emparejamientos detectados: {len(src_kp)}')
        print(f'> Numero de emparejamientos buenos: {len(matches)}')
        print(f'> Tiempo de deteccion: {(tf0 - ti0):.4f} s')
        print(f'> Tiempo de emparejamiento: {(tf1 - ti1):.4f} s')
        print('========================================================')

        # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
        if (len(matches) >= 4):
            src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            #snd = cv.polylines(snd, [np.int32(dst)],True,255,3, cv.LINE_AA)
            
            res = cv.warpPerspective(src_img, H, ((dst_img.shape[1] + src_img.shape[1]), dst_img.shape[0])) #wraped image
            res[0:dst_img.shape[0], 0:dst_img.shape[1]] = dst_img
            return res
        
            # h, w = src_img.shape[:2]
            # pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
            # res = cv.perspectiveTransform(pts, H)
            # return cv.polylines(dst_des,[np.int32(res)],True,255,3, cv.LINE_AA)
            # return matches, cv.drawMatches(fst, fst_kp, snd, snd_kp, good, None, matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        else:
            raise AssertionError("Can't find enough keypoints.")
            return []
            # return matches, cv.drawMatchesKnn(fst, fst_kp, snd, snd_kp, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
