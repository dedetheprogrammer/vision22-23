import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Equivalente (solo en contraste): cv2.convertScaleAbs(image, alpha, beta).
def contrast_and_brightness(img, a, b):
    img_c = img.astype(int).copy()
    return np.clip(a*img_c + b, 0, 255).astype(np.uint8)

# https://www.codespeedy.com/skin-detection-using-opencv-in-python/
# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
min_range = np.array([0,133,77], np.uint8)
max_range = np.array([235,173,127], np.uint8)
def alien_skin(img, color):
    YCrCbimg = cv2.cvtColor(img ,cv2.COLOR_BGR2YCR_CB)
    YCrCbskin_region = cv2.inRange(YCrCbimg, min_range, max_range)
    mask_color = np.zeros_like(img, np.uint8)
    mask_color[YCrCbskin_region != 0] = color
    return cv2.addWeighted(img, 1, mask_color, 1, 0)

def histogram_equalization(img):
    pass

def addTrackbar(trackbar_name, window_name, pos, min, max):
    cv2.createTrackbar(trackbar_name, window_name, pos, max, lambda x : x) 
    cv2.setTrackbarMin(trackbar_name, window_name, min) 

def alien_bright_contr():
    window_name = 'Me voy a quedar puto calvo'
    settings_window = 'settings'
    cv2.namedWindow(window_name)
    cv2.namedWindow('Pero no mucho')
    cv2.namedWindow(settings_window)

    brightness_contrast_module = False
    addTrackbar('brightness', settings_window, 0, -2*256, 2*256)
    addTrackbar('contrast', settings_window, 10, -30, 30)

    alien_skin_module = False
    addTrackbar('R', settings_window, 0, 0, 255)
    addTrackbar('G', settings_window, 0, 0, 255)
    addTrackbar('B', settings_window, 0, 0, 255)
    cam = cv2.VideoCapture(0)

    while cv2.getWindowProperty(window_name, 0) >= 0:

        check, frame = cam.read()
        # Applying brigthness and contrast modifications:
        # Applying skin detection and color modification:
        #frame = alien_effect(frame, (0,255,0))
        if brightness_contrast_module:
            brightness = cv2.getTrackbarPos('brightness', settings_window)
            contrast   = cv2.getTrackbarPos('contrast', settings_window)/10
            frame = contrast_and_brightness(frame, contrast, brightness)
        if alien_skin_module:
            color = (
                cv2.getTrackbarPos('B', settings_window),
                cv2.getTrackbarPos('G', settings_window),
                cv2.getTrackbarPos('R', settings_window)
            )
            frame = alien_skin(frame, color)

        cv2.imshow(window_name, cv2.flip(frame,1))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(frame)
        # https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
        # Images: [frame].
        # Channels: [0] if grayscale, [1] if blue, [2] if green, [3] if red.
        # Mask: mask image.
        # HistSize: BIN count.
        # Ranges: histogram range.
        hist = cv2.calcHist([y], [0], None, [256], [0, 256])
        cdf  = hist.cumsum()
        # Para normalizar:
        # - Multiplicamos el cumsum por el valor de escala de grises más alto.
        # - Dividimos el cumsum por el número total de pixeles (que equivale al máximo del cumsum).
        #cdf_nor = cdf * float(hist.max())/ cdf.max()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        y = cdf[y]
        frame = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCR_CB2BGR)
        cv2.imshow('Pero no mucho', cv2.flip(frame,1))


        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

def barrelPincushion():
    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()
        # grab the dimensions of the image
        h, w, _ = img.shape

        # set up the x and y maps as float32
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        scale_x = 1
        scale_y = 1
        center_x = w/2
        center_y = h/2
        radius = w  # Radio de la distorsión
        #amount = -0.75   # negative values produce pincushion
        amount = 0.75   # positive values produce barrel

        # create map with the barrel pincushion distortion formula
        for y in range(h):
            delta_y = scale_y * (y - center_y)
            for x in range(w):
                # determine if pixel is within an ellipse
                delta_x = scale_x * (x - center_x)
                distance = delta_x * delta_x + delta_y * delta_y
                if distance >= (radius * radius):
                    map_x[y, x] = x
                    map_y[y, x] = y
                else:
                    factor = 1.0
                    if distance > 0.0:
                        factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
                    map_x[y, x] = factor * delta_x / scale_x + center_x
                    map_y[y, x] = factor * delta_y / scale_y + center_y
                    

        # do the remap
        dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        # show the result
        cv2.imshow('dst', dst)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

def manualPoster():
    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()

        rows, cols, channels = img.shape

        # Convertir la imagen a un array numpy
        img_array = np.float32(img.reshape(-1, 3))

        # Definir el número de clusters
        k = 150

        # Inicializar centroides aleatoriamente
        centroids = np.random.uniform(low=0, high=255, size=(k, 3))

        # Iterar hasta que no haya cambios en la asignación de puntos
        while True:
            # Asignar cada punto al centroide más cercano
            distances = np.sqrt(((img_array - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Guardar los centroides actuales
            old_centroids = centroids.copy()

            # Actualizar los centroides al promedio de los puntos asignados
            for i in range(k):
                cluster_points = img_array[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)

            # Si no hay cambios en la asignación de puntos, salir del bucle
            if (centroids == old_centroids).all():
                break

        # Reasignar cada punto al centroide final
        distances = np.sqrt(((img_array - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Convertir los centroides a enteros y crear la imagen resultante
        centroids = centroids.astype(np.uint8)
        result = centroids[labels].reshape(rows, cols, 3)

        # Mostrar la imagen resultante
        cv2.imshow('Resultado', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def fullOpencvPoster():
    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()

        # Convertir la imagen a un array numpy
        img_array = np.float32(img.reshape(-1, 3))

        # Definir los parámetros de K-Means
        k = 16
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Ejecutar K-Means
        ret, label, center = cv2.kmeans(img_array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convertir los valores de los píxeles al color más cercano
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        # Mostrar la imagen resultante
        cv2.imshow('Resultado', res2)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cv2.destroyAllWindows()