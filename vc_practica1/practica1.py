from PIL import Image, ImageTk
import cv2 as cv
import math
import numpy as np
import tkinter as tk
# =============================================================================
# Utils
# =============================================================================
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# =============================================================================
# Efectos
# =============================================================================
# Equivalente (solo en contraste): cv2.convertScaleAbs(image, alpha, beta).
def contrast_and_brightness(img, a, b):
    img_c = img.astype(int).copy()
    return np.clip(a*img_c + b, 0, 255).astype(np.uint8)

# Alien skin
# https://www.codespeedy.com/skin-detection-using-opencv-in-python/
# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
min_range = np.array([0,133,77], np.uint8)
max_range = np.array([235,173,127], np.uint8)
def alien_skin(img, color):
    YCrCbimg = cv.cvtColor(img ,cv.COLOR_BGR2YCR_CB)
    YCrCbskin_region = cv.inRange(YCrCbimg, min_range, max_range)
    mask_color = np.zeros_like(img, np.uint8)
    mask_color[YCrCbskin_region != 0] = color
    return cv.addWeighted(img, 1, mask_color, 1, 0)

# Poster
def poster(img, k):
    if (k > 0):
        rows, cols, _ = img.shape
        # Convertir la imagen a un array numpy
        img_array = np.float32(img.reshape(-1, 3))

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
        img = centroids[labels].reshape(rows, cols, 3)
    return img

# Implementacion directa con opencv:
def fullOpencvPoster(img, k):
    # Convertir la imagen a un array numpy
    img_array = np.float32(img.reshape(-1, 3))

    # Definir los parámetros de K-Means
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Ejecutar K-Means
    _, label, center = cv.kmeans(img_array, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Convertir los valores de los píxeles al color más cercano
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))

# Distorsion 
def barrelPincushion(img, amount):
    # Grab the dimensions of the image
    h, w, _ = img.shape

    # Set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    center_x = w/2
    center_y = h/2
    radius = w  # Radio de la distorsión
    #amount = -0.75   # negative values produce pincushion
    #amount = 0.75   # positive values produce barrel

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
    return cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

# =============================================================================
# Main
# =============================================================================
# https://www.tutorialspoint.com/how-to-show-webcam-in-tkinter-window#:~:text=Using%20the%20OpenCV%20module%2C%20we,webcam%20in%20a%20tkinter%20window.&text=Once%20the%20installation%20has%20been,(if%20possible)%20using%20OpenCV.
# http://omes-va.com/tkinter-opencv-imagen/
# https://www.tutorialspoint.com/python/tk_scale.htm
# GUI
# - The window:
root = tk.Tk()
root.title('Practica 1')
disabled_color = rgb_to_hex(240,240,240)
enabled_color  = rgb_to_hex(180,180,180)
# - Source image
source_image = tk.Label(root)
source_image.grid(column=0, row=2)
# - Output image
output_image = tk.Label(root)
output_image.grid(column=1, row=1, rowspan=6)
# - Options
# -- Ningun efecto:
previous_effect = 0
selected_effect = tk.IntVar()
none_option     = tk.Radiobutton(root, text='Ninguno', width=25,value=0, variable=selected_effect)
none_option.grid(column=0, row=3)
# -- Contraste
contrast_option = tk.Radiobutton(root, text='Contraste', width=25,value=1, variable=selected_effect)
contrast_option.grid(column=0, row=4)
bar_contrast    = tk.Scale(root, variable=tk.DoubleVar(value=1.0), from_=-3, to=3, resolution=0.1, length=255, orient=tk.HORIZONTAL, label='Contraste', state='disabled', troughcolor=disabled_color)
bar_contrast.grid(column=0, row=5)
bar_brightness  = tk.Scale(root, variable=tk.IntVar(value=0), from_=-512, to=512, length=255, orient=tk.HORIZONTAL, label='Brillo', state='disabled', troughcolor=disabled_color)
bar_brightness.grid(column=0, row=6)
# Ecualizacion de histograma
equalize_option = tk.Radiobutton(root, text='Ecualizacion de histograma',width=25, value=2, variable=selected_effect)
equalize_option.grid(column=0, row=7)
# -- Alien
alien_option = tk.Radiobutton(root, text='Alien', width=25,value=3, variable=selected_effect)
alien_option.grid(column=0, row=8)
bar_R = tk.Scale(root, from_=0, to=255, length=255, orient=tk.HORIZONTAL, label='Rojo (R)', state='disabled', troughcolor=disabled_color)
bar_R.grid(column=0, row=9)
bar_G = tk.Scale(root, from_=0, to=255, length=255, orient=tk.HORIZONTAL, label='Verde (G)', state='disabled', troughcolor=disabled_color)
bar_G.grid(column=0, row=10)
bar_B = tk.Scale(root, from_=0, to=255, length=255, orient=tk.HORIZONTAL, label='Azul (B)', state='disabled', troughcolor=disabled_color)
bar_B.grid(column=0, row=11)
# -- Poster
poster_option = tk.Radiobutton(root, text='Poster', width=25,value=4, variable=selected_effect)
poster_option.grid(column=0,row=12)
bar_clusters = tk.Scale(root, from_=0, to=150, length=255, orient=tk.HORIZONTAL, label='Clusters (k)', state='disabled', troughcolor=disabled_color)
bar_clusters.grid(column=0,row=13)
# -- Distorsion
distorsion_option = tk.Radiobutton(root, text='Distorsion', width=25,value=5, variable=selected_effect)
distorsion_option.grid(column=0,row=14)
bar_amount = tk.Scale(root, variable=tk.IntVar(value=0), from_=-1, to=1, length=255, resolution=0.05, orient=tk.HORIZONTAL, label='Cantidad', state='disabled', troughcolor=disabled_color)
bar_amount.grid(column=0,row=15)

# -- Meter GUI en la ventana.

# WEBCAM CAPTURE
cam = cv.VideoCapture(0)
last_frame = None

def disable_option(n):
    if (n == 1):
        bar_contrast.config(state='disabled', troughcolor=disabled_color)
        bar_brightness.config(state='disabled', troughcolor=disabled_color)
    elif (n == 3):
        bar_R.config(state='disabled', troughcolor=disabled_color)
        bar_G.config(state='disabled', troughcolor=disabled_color)
        bar_B.config(state='disabled', troughcolor=disabled_color)
    elif (n == 4):
        bar_clusters(state='disabled', troughcolor=disabled_color)
    elif (n == 5):
        bar_amount(state='disabled', troughtcolor=disabled_color)

def effects(frame):
    global previous_effect
    a = bar_contrast.get()
    b = bar_brightness.get()
    n = selected_effect.get()
    disable_option(previous_effect)
    if (n == 1):
        a = bar_contrast.get()
        b = bar_brightness.get()
        bar_contrast.config(state='normal', troughcolor=enabled_color)
        bar_brightness.config(state='normal', troughcolor=enabled_color)
        frame = contrast_and_brightness(frame, a, b)
    elif (n == 2):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(frame)
        # https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
        # Images: [frame].
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
        y = cdf[y]
        frame = cv.cvtColor(cv.merge((y, cr, cb)), cv.COLOR_YCR_CB2BGR)
    elif (n == 3):
        color = (
            bar_B.get(),
            bar_G.get(),
            bar_R.get()
        )
        bar_R.config(state='normal', troughcolor=rgb_to_hex(color[2],0,0))
        bar_G.config(state='normal', troughcolor=rgb_to_hex(0,color[1],0))
        bar_B.config(state='normal', troughcolor=rgb_to_hex(0,0,color[0]))
        frame = alien_skin(frame, color)
    elif (n == 4):
        bar_clusters.config(state='normal', troughcolor=enabled_color)
        frame = poster(frame, bar_clusters.get())
    elif (n == 5):
        print('Distorsion')
 
    previous_effect = n
    return frame

def update_view():
    check, frame = cam.read()
    if (check):
        frame       = cv.flip(frame, 1)
        # Source image:
        source      = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2RGB,1), (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)), interpolation=cv.INTER_AREA)
        img_frame   = Image.fromarray(source)
        imgtk_frame = ImageTk.PhotoImage(image=img_frame)
        source_image.imgtk = imgtk_frame
        source_image.configure(image=imgtk_frame)
        # Output image:
        output      = cv.cvtColor(effects(frame), cv.COLOR_BGR2RGB,1)
        img_frame   = Image.fromarray(output)
        imgtk_frame = ImageTk.PhotoImage(image=img_frame)
        output_image.imgtk = imgtk_frame
        output_image.configure(image=imgtk_frame)
    source_image.after(20, update_view)

# Main loop
update_view()
root.mainloop()
cam.release()
