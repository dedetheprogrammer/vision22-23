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