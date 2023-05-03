import cv2 as cv
import numpy as np
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

def ImageStitching(images, operator, matcher, threshold):

    best_nmatches = 0
    # Source
    best_src_img  = 0
    src_kp        = []
    # Destination
    best_dst_img  = 0
    dst_kp        = []

    t0 = time.time()
    # Find the best pair to start with:
    for i, src_img in enumerate(images):
        for j, dst_img in enumerate(images[i+1:], start=i+1):
            # Features detection
            local_src_kp, src_des, _ = operator(src_img)
            local_dst_kp, dst_des, _ = operator(dst_img)
            # Features matching
            matches = matcher(src_des, dst_des, threshold)
            if (len(matches) > best_nmatches):
                best_nmatches = len(matches)
                # Source
                best_src_img  = i
                src_kp        = local_src_kp
                # Destination
                best_dst_img  = j
                dst_kp        = local_dst_kp

    res = images[0]
    if (best_nmatches >= 4):

        src_img = images[best_src_img]
        dst_img = images[best_dst_img]
        cv.waitKey(0)

        src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = src_img.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,H)
        dst_img = cv.polylines(dst_img,[np.int32(dst)],True,255,3, cv.LINE_AA)

        #src_h, src_w = src_img.shape[:2]
        #dst_h, dst_w = src_img.shape[:2]
        #src_pts = np.float32([[0, 0], [0, src_h], [src_w, src_h], [src_w, 0]]).reshape(-1, 1, 2)
        #dst_pts = np.float32([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]]).reshape(-1, 1, 2)
        #
        #pts = np.concatenate((src_pts, cv.perspectiveTransform(dst_pts, H)), axis=0)
        #[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        #[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        #t = [-xmin, -ymin]
        #Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        ## Warp the second image to align with the first
        #aligned_img = cv.warpPerspective(dst_img, Ht.dot(H), (xmax - xmin, ymax - ymin))
        #aligned_img[t[1]:src_h + t[1], t[0]:src_w + t[0]] = src_img
        #return aligned_img

        res = cv.warpPerspective(src_img, H, ((dst_img.shape[1] + src_img.shape[1]), dst_img.shape[0])) #wraped image
        res[0:dst_img.shape[0], 0:dst_img.shape[1]] = dst_img
        #res = cv.warpPerspective(dst_img, H, ((src_img.shape[1] + dst_img.shape[1]), src_img.shape[0])) #wraped image
        #res[0:src_img.shape[0], 0:src_img.shape[1]] = src_img
        

        #del images[best_src_img]
        #del images[best_dst_img]
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = None, # draw only inliers
        flags = 2)
    cv.imshow("Test", cv.drawMatches(src_img, src_kp, dst_img, dst_kp, matches, None, **draw_params))
    return res
 
    # Continue with the best pair until there is no more image to merge with:
    ##while len(images) > 0:
    ##    best_nmatches   = 0
    ##    src_kp, src_des = operator(src_img) 
##
    ##    for image in images:
    ##        # Features detection
    ##        local_dst_kp, dst_des, _ = operator(image)
    ##        # Features matching
    ##        matches = matcher(src_des, dst_des, threshold)
##
    ##        if (len(matches) > best_nmatches):
    ##            best_nmatches = len(matches)
    ##            # Destination
    ##            dst_img  = image
    ##            dst_kp   = local_dst_kp
##
    ##    if (len(matches) >= 4):
    ##        src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    ##        dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    ##        H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    ##        res = cv.warpPerspective(src_img, H, ((dst_img.shape[1] + src_img.shape[1]), dst_img.shape[0])) #wraped image
    ##        res[0:dst_img.shape[0], 0:dst_img.shape[1]] = dst_img
    ##        src_img = res
##
    ##        images.remove(dst_img)
    ##        
    ##t1 = time.time()
##
    ##print('========================================================')
    ##print(f' Deteccion mediante {operator.__name__} y matching mediante {matcher.__name__}: {(t1-t0):.4f} s')
    ##print('========================================================')

    return res

def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask    = np.zeros((height, width))
    offset = int(smoothing_window / 2)
    try:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset + 1).T, (height, 1))
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset + 1).T, (height, 1))
            mask[:, barrier + offset :] = 1
    except BaseException:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset).T, (height, 1))
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset).T, (height, 1))
            mask[:, barrier + offset :] = 1

    return cv.merge([mask, mask, mask])

# https://www.kaggle.com/code/deepzsenu/multiple-image-stitching
# https://github.com/ndvinh98/Panorama
def ImageStitching(src_img, dst_img):
    # 1. Finding features:
    sift = cv.SIFT_create()
    src_kp, src_features = sift.detectAndCompute(cv.cvtColor(src_img, cv.COLOR_BGR2GRAY), None)
    src_features = np.float32(src_features)
    cv.drawKeypoints(src_img, src_kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    dst_kp, dst_features = sift.detectAndCompute(cv.cvtColor(dst_img, cv.COLOR_BGR2GRAY), None)
    dst_features = np.float32(dst_features)
    cv.drawKeypoints(dst_img, dst_kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 2. Matching features:
    FLANN_INDEX_KDTREE = 0
    index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann         = cv.FlannBasedMatcher(index_params, search_params)
    matches       = flann.knnMatch(src_features, dst_features, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # 3. Reshape:
    src_pts = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]

    pts1 = np.float32([[0,0],[0,height_src],[width_src,height_src],[width_src,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,height_dst],[width_dst,height_dst],[width_dst,0]]).reshape(-1,1,2)

    pts1_ = cv.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts1_, pts2), axis=0)
    [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]

    # 4. Translation
    Ht  = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    src_img_warped = cv.warpPerspective(src_img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    dst_img_rz     = np.zeros(src_img_warped.shape, dtype=np.uint8)
    dst_img_rz[t[1] : height_dst + t[1], t[0]: width_dst + t[0]] = dst_img
    
    G = src_img_warped.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(gpA[i])
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = dst_img_rz.copy()
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
    real = np.hstack((src_img_warped[:,:cols//2],dst_img_rz[:,cols//2:]))
    return ls_, real



images = []
for i in range(1,6):
    images.append(cv.imread(f'./BuildingScene/building{i}.JPG'))


_, res1 = ImageStitching(images[0], images[1])
_, res2 = ImageStitching(images[3], images[4])
_, res3 = ImageStitching(images[2], res2)
_, res4 = ImageStitching(res3, res1)

print(res4)
#res = ImageStitching(res, images[3])
#res = ImageStitching(res, images[4])
cv.imshow("Practica 5", res3.npuin)

cv.waitKey(0)