import cv2 as cv
import numpy as np
import time

nfeatures = 10
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

def matching(fst, fst_kp, fst_des, snd, snd_kp, snd_des, mode = 'Brute force'):
    bf = cv.BFMatcher()
    if (mode == 'Brute force'):
        matches = bf.match(fst_des, snd_des)
        matches = sorted(matches, key = lambda x:x.distance)
        return matches, cv.drawMatches(fst, fst_kp, snd, snd_kp, matches[:200], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif (mode == 'KNN'):
        matches = bf.knnMatch(fst_des, snd_des, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return matches, cv.drawMatchesKnn(fst, fst_kp, snd, snd_kp, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif (mode == 'FLANN'):
        # FLANN parameters:
        FLANN_INDEX_KDTREE = 1
        index_params       = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params      = dict(checks=50)
        flann              = cv.FlannBasedMatcher(index_params, search_params)
        matches            = flann.knnMatch(fst_des, snd_des, k=2)
        # Need to draw only good matches, then have to create a mask
        matchesMask        = [[0,0] for i in range(len(matches))]
        # Ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        return matches, cv.drawMatchesKnn(fst, fst_kp, snd, snd_kp, matches, None, matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)

# Cambiar lo que devuelven los operadores:
def FeaturesMatching(fst, snd, operator, mode = 'Brute force'):
    if (operator == 'Harris'):
        # Detection
        inicio_deteccion = time.time()
        fst_output = HARRIS(fst)
        snd_output = HARRIS(snd)
        t_deteccion = time.time() - inicio_deteccion

        # Log output
        print('========================================================')
        print(f'Caracteristicas detectadas en imagen 1: -')
        print(f'Caracteristicas detectadas en imagen 2: -')
        print(f'Número de emparejamientos: -')
        print(f'Tiempo de detección con Harris: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento: - ')
        print('========================================================')
        return cv.hconcat([fst_output, snd_output])
    
    elif (operator == 'Sift'):
        # Detection
        inicio_deteccion = time.time()
        fst_kp, fst_des, _ = SIFT(fst)
        snd_kp, snd_des, _ = SIFT(snd)
        t_deteccion = time.time() - inicio_deteccion

        # Matching
        inicio_emparejamiento = time.time()
        matches, output = matching(fst, fst_kp, fst_des, snd, snd_kp, snd_des, mode)
        t_emparejamiento = time.time() - inicio_emparejamiento

        # Log output
        print('========================================================')
        print(f'Caracteristicas detectadas en imagen 1: {len(fst_kp)}')
        print(f'Caracteristicas detectadas en imagen 2: {len(snd_kp)}')
        print(f'Número de emparejamientos: {len(matches)}')
        print(f'Tiempo de detección con Sift: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento con {mode}: {t_emparejamiento:.4f} segundos')
        print('========================================================')
        return output
    
    elif (operator == 'Orb'):
        # Detection
        inicio_deteccion = time.time()
        fst_kp, fst_des, _ = ORB(fst)
        snd_kp, snd_des, _ = ORB(snd)
        t_deteccion = time.time() - inicio_deteccion

        # Matching
        inicio_emparejamiento = time.time()
        matches, output = matching(fst, fst_kp, fst_des, snd, snd_kp, snd_des, mode)
        t_emparejamiento = time.time() - inicio_emparejamiento

        # Log output
        print('========================================================')
        print(f'Caracteristicas detectadas en imagen 1: {len(fst_kp)}')
        print(f'Caracteristicas detectadas en imagen 2: {len(snd_kp)}')
        print(f'Número de emparejamientos: {len(matches)}')
        print(f'Tiempo de detección con Orb: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento con {mode}: {t_emparejamiento:.4f} segundos')
        print('========================================================')
        return output
    
    elif (operator == 'Akaze'):
        # Detection
        inicio_deteccion = time.time()
        fst_kp, fst_des, _ = AKAZE(fst)
        snd_kp, snd_des, _ = AKAZE(snd)
        t_deteccion = time.time() - inicio_deteccion

        # Matching
        inicio_emparejamiento = time.time()
        matches, output = matching(fst, fst_kp, fst_des, snd, snd_kp, snd_des, mode)
        t_emparejamiento = time.time() - inicio_emparejamiento

        # Log output
        print('========================================================')
        print(f'Caracteristicas detectadas en imagen 1: {len(fst_kp)}')
        print(f'Caracteristicas detectadas en imagen 2: {len(snd_kp)}')
        print(f'Número de emparejamientos: {len(matches)}')
        print(f'Tiempo de detección con Akaze: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento con {mode}: {t_emparejamiento:.4f} segundos')
        print('========================================================')
        return output
    
fst = cv.imread('./BuildingScene/building1.JPG')
snd = cv.imread('./BuildingScene/building2.JPG')
#cv.namedWindow('Practica 4', cv.WINDOW_AUTOSIZE)
cv.imshow('0', FeaturesMatching(fst, snd, 'Harris'))
cv.imshow('1', FeaturesMatching(fst, snd, 'Sift'))
cv.imshow('2', FeaturesMatching(fst, snd, 'Sift', 'KNN'))
cv.imshow('3', FeaturesMatching(fst, snd, 'Sift', 'FLANN'))
cv.imshow('4', FeaturesMatching(fst, snd, 'Orb'))
cv.imshow('5', FeaturesMatching(fst, snd, 'Orb', 'KNN'))
cv.imshow('7', FeaturesMatching(fst, snd, 'Akaze'))
cv.imshow('8', FeaturesMatching(fst, snd, 'Akaze', 'KNN'))

cv.waitKey(0)

# while 1:
#     cv.imshow('Practica 4', Harris(img))
#     if cv.waitKey(0) == 27:
#         break
cv.destroyWindow('Practica 4')