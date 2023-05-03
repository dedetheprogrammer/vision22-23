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


def Blend(src, dst):
    G = src.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(gpA[i])
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = dst.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(gpB[i])
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0]) 
        GE   = cv.pyrUp(gpA[i], dstsize=size)
        L    = cv.subtract(gpA[i-1],GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        GE = cv.pyrUp(gpB[i], dstsize=size)
        L = cv.subtract(gpB[i-1],GE)
        lpB.append(L)

    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv.pyrUp(ls_, dstsize = size)
        ls_ = cv.add(ls_, LS[i])
    # image with direct connecting each half
    real = np.hstack((src[:,:cols//2], dst[:,cols//2:]))
    return ls_, real


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
        cv.imshow("puntos", output)

        src_pts = np.float32([fst_kp[match.queryIdx].pt for match in matches[:200]]).reshape(-1, 1, 2)
        dst_pts = np.float32([snd_kp[match.trainIdx].pt for match in matches[:200]]).reshape(-1, 1, 2)
        M, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
        
        h1, w1 = fst.shape[:2]
        h2, w2 = snd.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv.perspectiveTransform(pts2, M)
        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        # Warp the second image to align with the first
        aligned_img = cv.warpPerspective(snd, Ht.dot(M), (xmax - xmin, ymax - ymin))
        aligned_img[t[1]:h1 + t[1], t[0]:w1 + t[0]] = fst
        # Log output
        print('========================================================')
        print(f'Caracteristicas detectadas en imagen 1: {len(fst_kp)}')
        print(f'Caracteristicas detectadas en imagen 2: {len(snd_kp)}')
        print(f'Número de emparejamientos: {len(matches)}')
        print(f'Tiempo de detección con Akaze: {t_deteccion:.4f} segundos')
        print(f'Tiempo de emparejamiento con {mode}: {t_emparejamiento:.4f} segundos')
        print('========================================================')
        return aligned_img

images = []
images.append(cv.imread('./BuildingScene/building1.JPG'))
images.append(cv.imread('./BuildingScene/building2.JPG'))
images.append(cv.imread('./BuildingScene/building3.JPG'))
images.append(cv.imread('./BuildingScene/building4.JPG'))
images.append(cv.imread('./BuildingScene/building5.JPG'))
#cv.namedWindow('Practica 4', cv.WINDOW_AUTOSIZE)
#cv.imshow('0', FeaturesMatching(fst, snd, 'Harris'))
#cv.imshow('1', FeaturesMatching(fst, snd, 'Sift'))
#cv.imshow('2', FeaturesMatching(fst, snd, 'Sift', 'KNN'))
#cv.imshow('3', FeaturesMatching(fst, snd, 'Sift', 'FLANN'))
#cv.imshow('4', FeaturesMatching(fst, snd, 'Orb'))
#cv.imshow('5', FeaturesMatching(fst, snd, 'Orb', 'KNN'))
#cv.imshow('7', FeaturesMatching(img1, img2, 'Akaze'))
#cv.imshow('8', FeaturesMatching(fst, snd, 'Akaze', 'KNN'))

res1 = FeaturesMatching(images[0], images[1], 'Akaze')
res2 = FeaturesMatching(images[3], images[4], 'Akaze')
res3 = FeaturesMatching(images[2], res2, 'Akaze')
res4 = FeaturesMatching(res3, res1, 'Akaze')
cv.imshow('Panorama', res4)
cv.waitKey(0)

# while 1:
#     cv.imshow('Practica 4', Harris(img))
#     if cv.waitKey(0) == 27:
#         break
cv.destroyWindow('Practica 4')