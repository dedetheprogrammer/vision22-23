from PIL import Image, ImageTk
import cv2 as cv
import math
import numpy as np
import tkinter as tk

def clip(frame):
    return np.clip(frame, 0, 255).astype(np.uint8)

def Sobel_Scharr(frame):
    frame = cv.cvtColor(clip(cv.GaussianBlur(frame,(3,3),0)), cv.COLOR_BGR2GRAY)
    gX    = cv.Sobel(frame,cv.CV_64F,1,0,ksize=3)
    gY    = cv.Sobel(frame,cv.CV_64F,0,1,ksize=3) * -1
    gXY   = cv.addWeighted(gX, 0.5, gY, 0.5, 0) 
    mod   = clip(np.sqrt(gX**2 + gY**2))
    ori   = np.arctan2(gY, gX)
    return gX, gY, gXY, mod, ori

# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def Canny(frame, low_threshold_ratio = 0.05, high_threshold_ratio = 0.09):
    _, _, _, mod, ori = Sobel_Scharr(frame)

    # Supresion de no maximos
    res = np.zeros(mod.shape, dtype=np.int32)
    ori_dgr = ori * 180. / np.pi
    ori_dgr[ori_dgr < 0] += 180
    q = np.where(
        (0 <= ori_dgr) & (ori_dgr < 22.5) | (157.5 <= ori_dgr) & (ori_dgr <= 180),
        np.roll(mod, shift=-1, axis=1),                 # q = mod[i, j+1]
        np.where(
            (22.5 <= ori_dgr) & (ori_dgr < 67.5),
            np.roll(mod, shift=(1, -1), axis=(0, 1)),   # q = mod[i+1, j-1]
            np.where(
                (67.5 <= ori_dgr) & (ori_dgr < 112.5),
                np.roll(mod, shift=-1, axis=0),         # q = mod[i+1, j]
                np.roll(mod, shift=(1,1), axis=(0,1))   # q = mod[i+1, j+1]
            )
        )
    )
    r = np.where(
        (0 <= ori_dgr) & (ori_dgr < 22.5) | (157.5 <= ori_dgr) & (ori_dgr <= 180),
        np.roll(mod, shift=1, axis=1),                  # r = mod[i, j-1]
        np.where(
            (22.5 <= ori_dgr) & (ori_dgr < 67.5),
            np.roll(mod, shift=(-1,1), axis=(0,1)),     # r = mod[i-1, j+1]
            np.where(
                (67.5 <= ori_dgr) & (ori_dgr < 112.5),
                np.roll(mod, shift=1, axis=0),          # r = mod[i-1, j]
                np.roll(mod, shift=(-1,-1), axis=(0,1)) # r = mod[i-1, j-1]
            )
        )
    )
    res[(mod >= q) & (mod >= r)] = mod[(mod >= q) & (mod >= r)]

    # Double threshold
    low_threshold  = res.min() * low_threshold_ratio
    high_threshold = res.max() * high_threshold_ratio
    weak   = np.int32(25) ; weak_i  , weak_j   = np.where((res <= high_threshold) & (res >= low_threshold))
    strong = np.int32(255); strong_i, strong_j = np.where(res >= high_threshold)

    res[weak_i  , weak_j] = weak
    res[strong_i, strong_j] = strong
    
    # Hysteresis
    #M, N = res.shape  
    #for i in range(1, M-1):
    #    for j in range(1, N-1):
    #        if (res[i,j] == weak):
    #            try:
    #                if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
    #                    or (res[i, j-1] == strong) or (res[i, j+1] == strong)
    #                    or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
    #                    res[i, j] = strong
    #                else:
    #                    res[i, j] = 0
    #            except IndexError as e:
    #                pass
    return res, ori

# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
def Hough_transform(gradient, orientation, threshold):

    M,N = gradient.shape; CM = int(M/2); CN = int(N/2)
    horizon = np.zeros(N)
    for i in range(M-1):
        for j in range(N-1):
            if (gradient[i,j] >= threshold):
                # Transformamos las coordenadas para que esten en el centro de 
                # la imagen:
                x = j - CN; y = CM - i
                # Calculamos las coordenadas polares:
                theta = orientation[i,j]
                rho   = x*np.cos(theta) + y*np.sin(theta)
                # Coordenadas polares del horizonte: (pi/2, N/2)
                if ((theta - np.pi/2) < 1e-6 and (rho - CN) < 1e-6):
                    horizon[x + CN] += 1

    return [np.argmax(horizon), CN]
                  
#frame = cv.imread('Contornos/poster.pgm')
frame     = cv.imread('Contornos/pasillo1.pgm')
#frame     = cv.imread('Contornos/pasillo2.pgm')
#frame     = cv.imread('Contornos/pasillo3.pgm')
frame_aux = frame.copy()

#cam = cv.VideoCapture(0)
#while True:
#    check, frame = cam.read()
#    frame = cv.flip(frame,1)
#    frame_aux = frame.copy()
#    cv.imshow('Source', frame)
#gX, gY, gXY, mod, ori = Sobel_Scharr(frame)
#cv.imshow('GX', clip(gX/2 + 128))
#cv.imshow('GY', clip(gY/2 + 128))
#cv.imshow('GXGY', clip(gXY/2 + 128))
#cv.imshow('Mod', clip(mod))
#cv.imshow('Orientacion', np.uint8(ori/np.pi * 128))

_,_,_, mod, ori = Sobel_Scharr(frame)
cv.imshow("Gradiente de Sobel", np.uint8(mod))
fuga = Hough_transform(mod, ori, 255)
cv.line(frame, (fuga[0], fuga[1]-10), (fuga[0], fuga[1]+10), (255,0,0), 2)
cv.line(frame, (fuga[0]-10, fuga[1]), (fuga[0]+10, fuga[1]), (255,0,0), 2)
cv.imshow("Hough", np.uint8(frame))

mod, ori = Canny(frame_aux, 0.05, 0.06)
cv.imshow("Gradiente de Canny", np.uint8(mod))
fuga = Hough_transform(mod, ori, 100)
cv.line(frame, (fuga[0], fuga[1]-10), (fuga[0], fuga[1]+10), (255,0,255), 2)
cv.line(frame, (fuga[0]-10, fuga[1]), (fuga[0]+10, fuga[1]), (255,0,255), 2)
cv.imshow("Hough", np.uint8(frame))

key = cv.waitKey(0)
    #if key == 27:
    #    break
