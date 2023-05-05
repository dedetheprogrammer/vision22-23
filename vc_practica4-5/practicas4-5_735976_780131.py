import cv2 as cv
import math
import numpy as np
import time

nfeatures = 10
# https://www.geeksforgeeks.org/feature-detection-and-matching-with-opencv-python/
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def HARRIS(img, get_output):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayscale = np.float32(grayscale)
    harris    = cv.cornerHarris(grayscale, blockSize=2, ksize=3, k=0.04)
    # Result is dilated for marking the corners, not important
    harris    = cv.dilate(harris, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[harris > 0.01 * harris.max()] = [0,255,0]
    if get_output:
        return harris, img
    else:
        return harris, None

# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
def SIFT(img, get_output):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift      = cv.SIFT_create()
    kp, des   = sift.detectAndCompute(grayscale, None)
    if get_output:
        return kp, des, cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        return kp, des, None

# https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/
def ORB(img, get_output):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb       = cv.ORB_create()
    kp, des   = orb.detectAndCompute(grayscale, None)
    if get_output:
        return kp, des, cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        return kp, des, None

# https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html
def AKAZE(img, get_output):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    akaze     = cv.AKAZE_create()
    kp, des   = akaze.detectAndCompute(grayscale, None)
    if get_output:
        return kp, des, cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        return kp, des, None

#------------------------------------------------------------------------------
# Features matching matchers:
def BRUTE_FORCE_MATCH(fst, fst_kp, fst_des, snd, snd_kp, snd_des, get_output, ratio=200):
    bf      = cv.BFMatcher()
    matches = bf.match(fst_des, snd_des)
    matches = sorted(matches, key= lambda x:x.distance)
    if get_output:
        return matches[:ratio], cv.drawMatches(fst, fst_kp, snd, snd_kp, matches[:200], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        return matches[:ratio], None

def KNN_MATCH(fst, fst_kp, fst_des, snd, snd_kp, snd_des, get_output, k=2, ratio=0.75):
    bf      = cv.BFMatcher()
    matches = bf.knnMatch(fst_des, snd_des, k=2)
    good    = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    if get_output:
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = None,
                   flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return np.asarray(good), cv.drawMatches(fst, fst_kp, snd, snd_kp, good, None, **draw_params)
    else:
        return np.asarray(good), None

def FLANN_MATCH(fst, fst_kp, fst_des, snd, snd_kp, snd_des, get_output, FLANN_INDEX_KDTREE=1, trees=5, checks=50, k=2, ratio=0.75):
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
    search_params = dict(checks=checks)
    flann         = cv.FlannBasedMatcher(index_params, search_params)
    matches       = flann.knnMatch(fst_des, snd_des, k=k)
    good          = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    if get_output:
        matchesMask = [[0,0] for i in range(len(matches))]
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = None,
                   flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return np.asarray(good), cv.drawMatches(fst, fst_kp, snd, snd_kp, good, None, **draw_params)
    else:
        return np.asarray(good), None

# A try to aproximate panoramic images blend without any results :(
# https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html
# https://stackoverflow.com/a/32970947/20853085
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

# To estabilize images perspectives:
#https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b
def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA, borderMode=cv.BORDER_TRANSPARENT)
 
# Trying to normalize the panoramic color
# https://stackoverflow.com/a/62441609/20853085
def normalize_light(image):
    hh, ww = image.shape[:2]
    print(hh, ww)
    m = max(hh, ww)

    # illumination normalize
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    # separate channels
    y, cr, cb = cv.split(ycrcb)

    # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
    # account for size of input vs 300
    sigma = int(5 * m / 300)
    print('sigma: ',sigma)
    gaussian = cv.GaussianBlur(y, (0, 0), sigma, sigma)

    # subtract background from Y channel
    y = (y - gaussian + 100)

    # merge channels back
    ycrcb = cv.merge([y, cr, cb])

    #convert to BGR
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR).astype(np.uint8)

def histogram_equalization(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(img)
    # https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
    # Images: [img].
    # Channels: [0] if grayscale, [1] if blue, [2] if green, [3] if red.
    # Mask: mask image.
    # HistSize: BIN count.
    # Ranges: histogram range.
    hist = cv.calcHist([y], [0], None, [256], [0, 256])
    cdf  = hist.cumsum()
    # Para normalizar:
    # - Multiplicamos el cumsum por el valor de escala de grises más alto.
    # - Dividimos el cumsum por el número total de pixeles (que equivale al máximo del cumsum).
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cv.cvtColor(cv.merge((cdf[y], cr, cb)), cv.COLOR_YCR_CB2BGR)

def crop(panorama):
    gpanorama = cv.cvtColor(panorama, cv.COLOR_BGR2GRAY) 
    pts = cv.findNonZero(gpanorama)
    x,y,w,h = cv.boundingRect(pts)
    return panorama[y:y+h, x:x+w]

# Cambiar lo que devuelven los operadores:
def ImageStitching(fst, snd, md, feature_matcher, ratio, get_output = False):

    if md.__name__ == 'HARRIS':
        # 1. Detection
        td0 = time.time()
        fst_kp, fst_kp_img = md(fst, get_output)
        snd_kp, snd_kp_img = md(snd, get_output)
        td1 = time.time() - td0

        # 2. Log output
        print('========================================================')
        print(f'Deteccion de esquinas con {md.__name__}')
        print('========================================================')
        print(f'Esquinas detectadas en imagen 1: {len(fst_kp)}')
        print(f'Esquinas detectadas en imagen 2: {len(snd_kp)}')
        print(f'Tiempo de detección: {td1:.4f} segundos')
        print('========================================================')

        return cv.hconcat([fst_kp_img, snd_kp_img]), None, None, None
    
    else:
        # 1. Detection
        td0 = time.time()
        fst_kp, fst_des, fst_kp_img = md(fst, get_output=get_output)
        snd_kp, snd_des, snd_kp_img = md(snd, get_output=get_output)
        td1 = time.time() - td0
        
        # 2. Matching
        tm0     = time.time()
        matches, matching_img = feature_matcher(fst, fst_kp, fst_des, snd, snd_kp, snd_des, ratio=ratio, get_output=get_output)
        tm1     = time.time() - tm0

        # 3. Wrapping
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

        # 4. Blending
        aligned_img = cv.warpPerspective(snd, Ht.dot(M), (xmax - xmin, ymax - ymin))
        mask = fst != 0
        aligned_img[t[1]:h1 + t[1], t[0]:w1 + t[0]][mask] = fst[mask]

        # 5. Log output
        print('========================================================')
        print(f'Deteccion mediante {md.__name__} y matching mediante {feature_matcher.__name__}')
        print('========================================================')
        print(f'Caracteristicas detectadas en imagen 1: {len(fst_kp)}')
        print(f'Número de buenos emparejamientos: {len(matches)}')
        print(f'Tiempo de detección: {td1:.4f} segundos')
        print(f'Tiempo de matching: {tm1:.4f} segundos')
        print('========================================================')
        # 6. Final crop
        return crop(aligned_img), fst_kp_img, snd_kp_img, matching_img

# No funciona :( (tampoco esta depurado)
# Su objetivo era obtener el par con el mejor porcentaje de matching, pero 
# no funciona y las imagenes se deformaban demasiado o no aparecian 
# directamente.
def MultipleImageStitching(images, operator, matcher, threshold):

    best_nmatches = 0
    # Source
    best_fst_img  = 0
    fst_kp        = []
    # Destination
    best_snd_img  = 0
    snd_kp        = []

    t0 = time.time()
    # Find the best pair to start with:
    for i, local_fst in enumerate(images):
        for j, local_snd in enumerate(images[i+1:], start=i+1):
            # Features detection
            local_fst_kp, src_des, _ = operator(local_fst)
            local_snd_kp, dst_des, _ = operator(local_snd)
            # Features matching
            matches = matcher(src_des, dst_des, threshold)
            if (len(matches) > best_nmatches):
                best_nmatches = len(matches)
                # Source
                best_fst_img  = i
                fst_kp        = local_fst_kp
                # Destination
                best_snd_img  = j
                snd_kp        = local_snd_kp

    res = images[0]
    if (best_nmatches >= 4):

        fst = images[best_fst_img]
        snd = images[best_snd_img]

        # 3. Wrapping
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

        # 4. Blending
        aligned_img = cv.warpPerspective(snd, Ht.dot(M), (xmax - xmin, ymax - ymin))
        mask = fst != 0
        aligned_img[t[1]:h1 + t[1], t[0]:w1 + t[0]][mask] = fst[mask]

        del images[best_fst_img]
        del images[best_snd_img]

        fst = aligned_img = crop(aligned_img)

    # Continue with the best pair until there is no more image to merge with:
    while len(images) > 0:

        best_nmatches   = 0
        fst_kp, fst_des = operator(fst) 

        for i,local_img in enumerate(images):
            # Features detection
            local_snd_kp, snd_des, _ = operator(local_img)
            # Features matching
            matches = matcher(fst_des, snd_des, threshold)

            if (len(matches) > best_nmatches):
                best_nmatches = len(matches)
                # Destination
                best_fst_img  = i
                snd_kp        = local_snd_kp

        if (len(matches) >= 4):
            snd = images[best_snd_img]

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

            # 4. Blending
            aligned_img = cv.warpPerspective(snd, Ht.dot(M), (xmax - xmin, ymax - ymin))
            mask = fst != 0
            aligned_img[t[1]:h1 + t[1], t[0]:w1 + t[0]][mask] = fst[mask]

            del images[best_fst_img]

    t1 = time.time()

    print('========================================================')
    print(f' Deteccion mediante {operator.__name__} y matching mediante {matcher.__name__}: {(t1-t0):.4f} s')
    print('========================================================')
    return res

option = 1
images = []
if option == 0:
    #for i in range(1,6):
    #    images.append(cv.imread(f'./BuildingScene/building{i}.JPG'))  
    #res,_,_,_ = ImageStitching(images[0], images[3], HARRIS, None, 0, True)
    #cv.imshow("Building", res)
    #cv.waitKey(0)

    #res,fst,snd,match = ImageStitching(images[0], images[1], SIFT,  BRUTE_FORCE_MATCH, 200, True)
    #cv.imshow("Building", cv.hconcat([fst, snd]))
    #cv.waitKey(0)
    #cv.imshow("Building", match)
    #cv.waitKey(0)
#
    #res,fst,snd,match = ImageStitching(images[0], images[1], ORB,   BRUTE_FORCE_MATCH, 200, True)
    #cv.imshow("Building", cv.hconcat([fst, snd]))
    #cv.waitKey(0)
    #cv.imshow("Building", match)
    #cv.waitKey(0)
#
    #res,fst,snd,match = ImageStitching(images[0], images[1], AKAZE, BRUTE_FORCE_MATCH, 200, True)
    #cv.imshow("Building", cv.hconcat([fst, snd]))
    #cv.waitKey(0)
    #cv.imshow("Building", match)
    #cv.waitKey(0)
#
    #res,fst,snd,match = ImageStitching(images[0], images[1], SIFT,  KNN_MATCH, 0.75, True)
    #cv.imshow("Building", cv.hconcat([fst, snd]))
    #cv.waitKey(0)
    #cv.imshow("Building", match)
    #cv.waitKey(0)
#
    #res,fst,snd,match = ImageStitching(images[0], images[1], ORB,   KNN_MATCH, 0.75, True)
    #cv.imshow("Building", cv.hconcat([fst, snd]))
    #cv.waitKey(0)
    #cv.imshow("Building", match)
    #cv.waitKey(0)
#
    #res,fst,snd,match = ImageStitching(images[0], images[1], AKAZE, KNN_MATCH, 0.75, True)
    #cv.imshow("Building", cv.hconcat([fst, snd]))
    #cv.waitKey(0)
    #cv.imshow("Building", match)
    #cv.waitKey(0)
#
    #res,fst,snd,match = ImageStitching(images[0], images[1], SIFT,  FLANN_MATCH, 0.75, True)
    #cv.imshow("Building", cv.hconcat([fst, snd]))
    #cv.waitKey(0)
    #cv.imshow("Building", match)
    #cv.waitKey(0)

    images = []
    for i in range(1,8):
        img = cv.imread(f'./BuildingScene/EINA{i}.jpg')
        img = cv.resize(img, (int(math.ceil(img.shape[1] * 0.16)), int(math.ceil(img.shape[0] * 0.16))), interpolation=cv.INTER_AREA)
        images.append(img)
    for i in range(1,7):
        images[i-1] = cv.resize(images[i-1], images[2].shape[:2], interpolation=cv.INTER_AREA)
    # res,_,_,_ = ImageStitching(images[2], images[4], HARRIS, None, 0, True)
    # res = cv.resize(res, (int(res.shape[1] * 0.2), int(res.shape[0] * 0.2)), interpolation=cv.INTER_AREA)
    # cv.imshow("EINA", res)

    res,fst,snd,match = ImageStitching(images[2], images[3], SIFT,  BRUTE_FORCE_MATCH, 200, True)
    cv.imshow("Building", cv.hconcat([fst, snd]))
    cv.waitKey(0)
    cv.imshow("Building", match)
    cv.waitKey(0)

    res,fst,snd,match = ImageStitching(images[2], images[3], ORB, BRUTE_FORCE_MATCH, 200, True)
    cv.imshow("Building", cv.hconcat([fst, snd]))
    cv.waitKey(0)
    cv.imshow("Building", match)
    cv.waitKey(0)

    res,fst,snd,match = ImageStitching(images[2], images[3], AKAZE, BRUTE_FORCE_MATCH, 200, True)
    cv.imshow("Building", cv.hconcat([fst, snd]))
    cv.waitKey(0)
    cv.imshow("Building", match)
    cv.waitKey(0)

    res,fst,snd,match = ImageStitching(images[2], images[3], SIFT, KNN_MATCH, 0.75, True)
    cv.imshow("Building", cv.hconcat([fst, snd]))
    cv.waitKey(0)
    cv.imshow("Building", match)
    cv.waitKey(0)

    res,fst,snd,match = ImageStitching(images[2], images[3], ORB, KNN_MATCH, 0.75, True)
    cv.imshow("Building", cv.hconcat([fst, snd]))
    cv.waitKey(0)
    cv.imshow("Building", match)
    cv.waitKey(0)

    res,fst,snd,match = ImageStitching(images[2], images[3], AKAZE, KNN_MATCH, 0.75, True)
    cv.imshow("Building", cv.hconcat([fst, snd]))
    cv.waitKey(0)
    cv.imshow("Building", match)
    cv.waitKey(0)

    res,fst,snd,match = ImageStitching(images[2], images[3], SIFT,  FLANN_MATCH, 0.75, True)
    cv.imshow("Building", cv.hconcat([fst, snd]))
    cv.waitKey(0)
    cv.imshow("Building", match)
    cv.waitKey(0)

elif option == 1:
    for i in range(1,6):
        img  = cv.imread(f'./BuildingScene/building{i}.JPG')
        h, w = img.shape[:2]
        images.append(cylindricalWarp(img, np.array([[800,0,w/2],[0,800,h/2],[0,0,1]])))
    print('========================================================')
    t0  = time.time()
    res1,_,_,_ = ImageStitching(images[0], images[1], AKAZE, KNN_MATCH, 0.75)
    res2,_,_,_ = ImageStitching(images[3], images[4], AKAZE, KNN_MATCH, 0.75)
    res3,_,_,_ = ImageStitching(images[2], res2, AKAZE, KNN_MATCH, 0.75)
    res,_,_,_  = ImageStitching(res3, res1, AKAZE, KNN_MATCH, 0.75)
    cv.imshow('Practica 5', histogram_equalization(res))
    t1  = time.time() - t0
    print('========================================================')
    print(f'Tiempo total: {t1} s')
    print('========================================================')
else:
    for i in range(1,8):
        img  = cv.imread(f'./BuildingScene/EINA{i}.jpg')
        img  = cv.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)), interpolation=cv.INTER_AREA)
        h, w = img.shape[:2]
        images.append(cylindricalWarp(img, np.array([[800,0,w/2],[0,800,h/2],[0,0,1]])))
    print('========================================================')
    t0  = time.time()
    res1,_,_,_ = ImageStitching(images[1], images[2], AKAZE, KNN_MATCH, 0.75)
    res2,_,_,_ = ImageStitching(res1, images[3], AKAZE, KNN_MATCH, 0.75)
    res3,_,_,_ = ImageStitching(res2, images[0], AKAZE, KNN_MATCH, 0.75)
    res,_,_,_  = ImageStitching(res3, images[4], AKAZE, KNN_MATCH, 0.75)
    res = cv.resize(res, (int(res.shape[1] * 0.4), int(res.shape[0] * 0.4)), interpolation=cv.INTER_AREA)
    cv.imshow('Practica 5', histogram_equalization(res))

    t1  = time.time() - t0
    print('========================================================')
    print(f'Tiempo total: {t1} s')
    print('========================================================')


cv.waitKey(0)
