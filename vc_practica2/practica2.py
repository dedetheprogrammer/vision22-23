from PIL import Image, ImageTk
import cv2 as cv
import math
import numpy as np
import tkinter as tk

cam = cv.VideoCapture(0)

def clip(frame):
    return np.clip(frame, 0, 255).astype(np.uint8)

frame = cv.imread('Contornos/poster.pgm')


#while True:
#    check, frame = cam.read()
#    frame = cv.flip(frame,1)
cv.imshow('Source', frame)

frame  = cv.cvtColor(clip(cv.GaussianBlur(frame,(3,3),0)), cv.COLOR_BGR2GRAY)
gX = cv.Sobel(frame,cv.CV_64F,1,0,ksize=3)
gY = cv.Sobel(frame,cv.CV_64F,0,1,ksize=3) * -1
gXs = clip(gX/2 + 128)
gYs = clip(gY/2 + 128)
gXY = cv.addWeighted(gXs, 0.5, gYs, 0.5, 0)
cv.imshow('GX', gXs)
cv.imshow('GY', gYs)
cv.imshow('GXGY', gXY)
cv.imshow('Mod', clip(np.sqrt(gX**2 + gY**2)))
cv.imshow('Orientacion', np.uint8(np.arctan2(gY, gX)/np.pi * 128))

key = cv.waitKey(0)
    #if key == 27:
    #    break