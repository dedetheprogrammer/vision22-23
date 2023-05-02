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

#------------------------------------------------------------------------------
# Features matching matchers:
def BRUTE_FORCE_MATCH(src_des, dst_des, k):
    bf      = cv.BFMatcher()
    matches = bf.match(src_des, dst_des)
    matches = sorted(matches, key= lambda x:x.distance)
    return matches[:k]

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

# -----------------------------------------------------------------------------
# Thresholding methods

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
                good.append(m)

        if len(good)> 10:
            src_pts = np.float32([ fst_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ snd_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w,_ = fst.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            snd = cv.polylines(snd ,[np.int32(dst)],True,255,3, cv.LINE_AA)
            return matches, cv.drawMatches(fst, fst_kp, snd, snd_kp, good, None, matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), 10) )
            matchesMask = None 
            return matches, cv.drawMatchesKnn(fst, fst_kp, snd, snd_kp, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif (mode == 'FLANN'):
        # FLANN parameters:

        ## Need to draw only good matches, then have to create a mask
        #matchesMask        = [[0,0] for i in range(len(matches))]
        ## Ratio test as per Lowe's paper
        #for i,(m,n) in enumerate(matches):
        #    if m.distance < 0.7*n.distance:
        #        matchesMask[i]=[1,0]
        #return matches, cv.drawMatchesKnn(fst, fst_kp, snd, snd_kp, matches, None, matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)
        
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
        res = []
        if len(good) > 10:
            src_pts     = np.float32([ fst_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts     = np.float32([ snd_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            H, mask     = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h,w,_ = fst.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,H)
            #snd = cv.polylines(snd, [np.int32(dst)],True,255,3, cv.LINE_AA)

            res = cv.warpPerspective(fst, H, ((fst.shape[0]+snd.shape[0]), snd.shape[1])) #wraped image
            res[0:snd.shape[0], 0:snd.shape[1]] = snd
            res[0:fst.shape[0], 0:fst.shape[1]] = fst
        else:
            res = cv.hconcat(fst, snd)
            print(f"Not enough matches are found - {len(good)}/{10}")
            matchesMask = None
        
        cv.imshow('Hola', res)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
        return matches, cv.drawMatches(fst, fst_kp, snd, snd_kp, good, None, **draw_params)

def FeaturesMatching(src_img, dst_img, operator, matcher, threshold):

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
    print(f'> Numero de emparejamientos buenos: {len(matches)}.')
    print(f'> Tiempo de deteccion: {(tf0 - ti0):.4f} s')
    print(f'> Tiempo de emparejamiento: {(tf1 - ti1):.4f} s')
    print('========================================================')

    return [src_kp, src_des], [dst_kp, dst_des]

# Cambiar lo que devuelven los operadores:
def FeaturesMatching2(fst, snd, operator, mode = 'Brute force'):
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

src_img = cv.imread('./BuildingScene/building1.JPG')
dst_img = cv.imread('./BuildingScene/building1.JPG')
FeaturesMatching(src_img, dst_img, AKAZE, KNN_MATCH, {"type":"DISTANCE_THRESHOLD", "k":0.25 })

# 1. Obtenemos las imagenes originales:
#train_img  = cv.imread('./BuildingScene/building1.JPG')
#query_img = cv.imread('./BuildingScene/building2.JPG')
##cv.imshow("Test", cv.hconcat([train_img, query_img]))
#
## 2. Las pasamos a escala de grises:
#train_gsimg = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)
#query_gsimg = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
##cv.imshow("GTest", cv.hconcat([train_gsimg, query_gsimg]))
#
#
## 3. Obtenemos los key points y los descriptores:
#sift = cv.SIFT_create()
#train_kp, train_des = sift.detectAndCompute(train_gsimg, None)
#query_kp, query_des = sift.detectAndCompute(query_gsimg, None)
#
## 4. Inicializamos el matching de vecinos por Flann:
#FLANN_INDEX_KDTREE = 1
#index_params       = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#search_params      = dict(checks=50)
#flann              = cv.FlannBasedMatcher(index_params, search_params)
#matches            = flann.knnMatch(train_des, query_des, k=2)
#good = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:
#        good.append(m)
#matches = np.asarray(good)
#
## 5. Alinear las imagenes con su homografía:
## https://stackoverflow.com/a/58131728/20853085
#if (len(matches) >= 4):
#    src = np.float32([train_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
#    dst = np.float32([query_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
#    H, masked = cv.findHomography(dst, src, cv.RANSAC, 5.0)
#    dst = cv.warpPerspective(query_img,H,((train_img.shape[1] + query_img.shape[1]), train_img.shape[0])) #wraped image
#    dst[0:train_img.shape[0], 0:train_img.shape[1]] = train_img
#    cv.imshow("Hola?", dst)
#else:
#    raise AssertionError("Can't find enough keypoints.")

cv.waitKey(0)


#FeaturesMatching(fst, snd, 'Sift', 'FLANN')
#cv.imshow('Practica 4', FeaturesMatching(fst, snd, 'Sift', 'FLANN'))
#cv.waitKey(0)

# while 1:
#     cv.imshow('Practica 4', Harris(img))
#     if cv.waitKey(0) == 27:
#         break
#cv.destroyWindow('Practica 4')