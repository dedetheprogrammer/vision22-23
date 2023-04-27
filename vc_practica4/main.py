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

# Cambiar lo que devuelven los operadores:
def MATCH(fst, snd, operator, mode = 'Brute force'):
    if (operator == 'Harris'):
        HARRIS(fst)
        HARRIS(snd)
        return []
    elif (operator == 'Sift'):
        inicio_deteccion = time.time()
        fst_kp, fst_des, _ = SIFT(fst)
        snd_kp, snd_des, _ = SIFT(snd)
        t_deteccion = time.time() - inicio_deteccion

        inicio_emparejamiento = time.time()
        bf      = cv.BFMatcher()
        matches = bf.match(fst_des, snd_des)
        matches = sorted(matches, key = lambda x:x.distance)
        t_emparejamiento = time.time() - inicio_emparejamiento
        print(f'Característica detectadas en imagen 1: {len(fst_kp)}')
        print(f'Característica detectadas en imagen 2: {len(snd_kp)}')
        print(f'Número de emparejamientos: {len(matches)}')
        print(f'Tiempo de detección con ORB: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento por fuerza bruta: {t_emparejamiento:.4f} segundos')
        return cv.drawMatches(fst, fst_kp, snd, snd_kp, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif (operator == 'Orb'):
        inicio_deteccion = time.time()
        fst_kp, fst_des, _ = ORB(fst)
        snd_kp, snd_des, _ = ORB(snd)
        t_deteccion = time.time() - inicio_deteccion

        inicio_emparejamiento = time.time()
        bf      = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = bf.match(fst_des, snd_des)
        matches = sorted(matches, key = lambda x:x.distance)
        t_emparejamiento = time.time() - inicio_emparejamiento
        print(f'Característica detectadas en imagen 1: {len(fst_kp)}')
        print(f'Característica detectadas en imagen 2: {len(snd_kp)}')
        print(f'Número de emparejamientos: {len(matches)}')
        print(f'Tiempo de detección con ORB: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento por fuerza bruta: {t_emparejamiento:.4f} segundos')
        return cv.drawMatches(fst, fst_kp, snd, snd_kp, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif (operator == 'Akaze'):
        inicio_deteccion = time.time()
        fst_kp, fst_des, _ = AKAZE(fst)
        snd_kp, snd_des, _ = AKAZE(snd)
        t_deteccion = time.time() - inicio_deteccion

        inicio_emparejamiento = time.time()
        bf      = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        if (mode == 'Brute force'):
            matches = bf.match(fst_des, snd_des)
            matches = sorted(matches, key = lambda x:x.distance)
        elif (mode == 'KNN'):
            matches = bf.knnMatch(fst_des, snd_des, k=2)
        t_emparejamiento = time.time() - inicio_emparejamiento
        print(f'Característica detectadas en imagen 1: {len(fst_kp)}')
        print(f'Característica detectadas en imagen 2: {len(snd_kp)}')
        print(f'Número de emparejamientos: {len(matches)}')
        print(f'Tiempo de detección con ORB: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento por fuerza bruta: {t_emparejamiento:.4f} segundos')
        return cv.drawMatches(fst, fst_kp, snd, snd_kp, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fst = cv.imread('./BuildingScene/building1.JPG')
snd = cv.imread('./BuildingScene/building2.JPG')
cv.namedWindow('Practica 4', cv.WINDOW_AUTOSIZE)
cv.imshow('Practica 4', MATCH(fst, snd, 'Akaze', 'KNN'))
cv.waitKey(0)

# while 1:
#     cv.imshow('Practica 4', Harris(img))
#     if cv.waitKey(0) == 27:
#         break
cv.destroyWindow('Practica 4')