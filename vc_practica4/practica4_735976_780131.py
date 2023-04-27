from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import time

def harris():
    nfeatures = 50
    img1 = cv.imread("BuildingScene/building1.JPG")
    img2 = cv.imread("BuildingScene/building2.JPG")
    gray1  = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2  = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    
    # Inicio de detección
    inicio_deteccion = time.time()
    gray1 = np.float32(gray1)
    dst = cv.cornerHarris(gray1,2,3,0.04)
    dst_norm = np.empty_like(dst)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    img1_aux = img1.copy()
    img1_aux[dst>0.01*dst.max()]=[0,0,255]
    kp1 = [cv.KeyPoint(x, y, nfeatures) for y in range(dst_norm.shape[0]) for x in range(dst_norm.shape[1]) if np.array_equal(img1_aux[y, x], [0, 0, 255])]
    des1 = sift.compute(img1, kp1)[1]

    gray2 = np.float32(gray2)
    dst = cv.cornerHarris(gray2,2,3,0.04)
    dst_norm = np.empty_like(dst)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    img2_aux = img2.copy()
    img2_aux[dst>0.01*dst.max()]=[0,0,255]
    kp2 = [cv.KeyPoint(x, y, nfeatures) for y in range(dst_norm.shape[0]) for x in range(dst_norm.shape[1]) if np.array_equal(img2_aux[y, x], [0, 0, 255])]
    # Fin de detección
    t_deteccion = time.time() - inicio_deteccion
    
    des1 = sift.compute(img1, kp1)[1]
    des2 = sift.compute(img2, kp2)[1]
    
    

    inicio_emparejamiento = time.time()
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)
    good_matches = []
    ratio_threshold = 1
    for match in matches:
        # Obtener las dos distancias más cercanas para este match
        distances = [m.distance for m in matches]
        distances.sort()
        if distances[0] / distances[1] < ratio_threshold:
            # El match es válido, añadirlo a la lista de matches buenos
            good_matches.append(match)
    
    t_emparejamiento = time.time() - inicio_emparejamiento

    print(f'Característica detectadas en imagen 1: {len(kp1)}')
    print(f'Característica detectadas en imagen 2: {len(kp2)}')
    print(f'Número de emparejamientos: {len(matches)}')
    print(f'Tiempo de detección con ORB: {t_deteccion:.4f} segundos')
    print(f'Tiempo de emparejamiento por fuerza bruta: {t_emparejamiento:.4f} segundos')
    # Dibujar los emparejamientos encontrados
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:nfeatures], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar las imágenes con los emparejamientos
    cv.imshow("Matches", img_matches)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def orb():
    
    nfeatures = 50
    img1 = cv.imread("BuildingScene/building1.JPG", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("BuildingScene/building2.JPG", cv.IMREAD_GRAYSCALE)
    
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=nfeatures)
    
    inicio_deteccion = time.time()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    t_deteccion = time.time() - inicio_deteccion

    inicio_emparejamiento = time.time()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    # Match descriptors.
    matches = bf.match(des1,des2)

    matches = sorted(matches, key = lambda x:x.distance)
    good_matches = []
    ratio_threshold = 0.84
    for match in matches:
        # Obtener las dos distancias más cercanas para este match
        distances = [m.distance for m in matches]
        distances.sort()
        if distances[0] / distances[1] < ratio_threshold:
            # El match es válido, añadirlo a la lista de matches buenos
            good_matches.append(match)
    t_emparejamiento = time.time() - inicio_emparejamiento

    print(f'Característica detectadas en imagen 1: {len(kp1)}')
    print(f'Característica detectadas en imagen 2: {len(kp2)}')
    print(f'Número de emparejamientos: {len(matches)}')
    print(f'Tiempo de detección con ORB: {t_deteccion:.4f} segundos')
    print(f'Tiempo de emparejamiento por fuerza bruta: {t_emparejamiento:.4f} segundos')
    # Dibujar los emparejamientos encontrados
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:nfeatures], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #img_matches2 = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:nfeatures], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar las imágenes con los emparejamientos
    cv.imshow("Matches", img_matches)
    #cv.imshow("Matches 2", img_matches2)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

harris()