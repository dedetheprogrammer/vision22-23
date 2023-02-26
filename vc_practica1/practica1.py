import cv2
import numpy as np

def brightness_and_contrast(img, a, b):
    img_c = img.astype(int).copy()
    return np.clip(a*img_c + b, 0, 255).astype(np.uint8)

# https://www.codespeedy.com/skin-detection-using-opencv-in-python/
# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
min_range = np.array([0,133,77], np.uint8)
max_range = np.array([235,173,127], np.uint8)
def alien_effect(img, color):
    YCrCbimg = cv2.cvtColor(img ,cv2.COLOR_BGR2YCR_CB)
    YCrCbskin_region = cv2.inRange(YCrCbimg, min_range, max_range)
    mask_color = np.zeros_like(img, np.uint8)
    mask_color[YCrCbskin_region != 0] = color
    return cv2.addWeighted(img, 1, mask_color, 1, 0)

#def BrightnessContrast(brightness=0):
#    brightness = cv2.getTrackbarPos('Brightness', 'Me voy a quedar puto calvo')
#    contrast = cv2.getTrackbarPos('Contrast', 'Me voy a quedar puto calvo')

#cv2.convertScaleAbs(image, alpha, beta)
once = True
cam = cv2.VideoCapture(0)
#cv2.createTrackbar('Brightness', 'Me voy a quedar puto calvo', 255, 2 * 255, BrightnessContrast) 
#cv2.createTrackbar('Contrast', 'Me voy a quedar puto calvo',127, 2 * 127, BrightnessContrast)  
while True:
    check, frame = cam.read()

    # Applying brigthness and contrast modifications:
    #frame = brightness_and_contrast(frame, 1.5, 0)
    # Applying skin detection and color modification:
    frame = alien_effect(frame, (0,255,0))
    cv2.imshow('Me voy a quedar puto calvo', cv2.flip(frame,1))
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()