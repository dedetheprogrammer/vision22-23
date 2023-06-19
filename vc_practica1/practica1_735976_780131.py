from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import tkinter as tk
# =============================================================================
# Utils
# =============================================================================
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def clip(frame):
    return np.clip(frame, 0, 255).astype(np.uint8)

# =============================================================================
# Efectos
# =============================================================================
# Equivalente (solo en contraste): cv2.convertScaleAbs(image, alpha, beta).
def contrast_and_brightness(img, a, b):
    img_c = img.astype(int).copy()
    return np.clip(a*img_c + b, 0, 255).astype(np.uint8)

# https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
# https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
# https://stackoverflow.com/questions/42651595/histogram-equalization-python-for-colored-image
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

# Posterizacion
# https://medium.com/analytics-vidhya/image-segmentation-using-k-means-clustering-from-scratch-1545c896e38e
# https://stackoverflow.com/questions/9575652/opencv-using-k-means-to-posterize-an-image
# https://github.com/joelgrus/posterization-pyladies
# https://flothesof.github.io/k-means-numpy.html
# https://mubaris.com/posts/kmeans-clustering/
def poster(img, k, more_precission):
    if (k > 0):
        rows, cols, _ = img.shape
        img_array = np.float32(img.reshape(-1,3))
        # Inicializar centroides aleatoriamente
        centroids = np.random.uniform(low = 0, high=255, size=(k,3))
        old_centroids = np.zeros(shape=centroids.shape)

        if (not more_precission):
            # Asignar cada punto al grupo más centroide
            distances = np.sqrt(((img_array - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            # Terminar cuando ya no se desplace ningún centroide
            while (centroids != old_centroids).all():
                old_centroids = centroids.copy()
                # Actualizar la posición de los centros según la posición de sus píxeles asignados
                for i in range(k):
                    cluster_points = img_array[labels == i]
                    if len(cluster_points) > 0:
                        centroids[i] = cluster_points.mean(axis=0)
            # Crear la imagen final asignando a cada punto de la imagen original su nuevo color
            distances = np.sqrt(((img_array - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
        else:
            # Terminar cuando ya no se desplace ningún centroide
            while (centroids != old_centroids).all():
                distances = np.sqrt(((img_array - centroids[:, np.newaxis])**2).sum(axis=2))
                labels    = np.argmin(distances, axis=0)
                old_centroids = centroids.copy()
                # Actualizar la posición de los centroides según la posición de sus píxeles asignados
                for i in range(k):
                    cluster_points = img_array[labels == i]
                    if len(cluster_points) > 0:
                        centroids[i] = cluster_points.mean(axis=0)
            # Crear la imagen final asignando a cada punto de la imagen original su nuevo color
            distances = np.sqrt(((img_array - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            centroids = centroids.astype(np.uint8)
        return centroids.astype(np.uint8)[labels].reshape(rows, cols, 3)
    else:
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
def barrel_pincussion(img, k1, center_x, center_y): #k2, center_x, center_y):
    # Grab the dimensions of the image
    h, w, _ = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    r2   = (x - center_x)**2 + (y - center_y)**2
    #r4   = (x - center_x)**4 + (y - center_y)**4 
    xd   = x + ((x - center_x) * k1 * r2) #+ ((x - center_x) * k2 * r4)
    yd   = y + ((y - center_y) * k1 * r2) #+ ((y - center_y) * k2 * r4)
    return cv.remap(img, xd.astype(np.float32), yd.astype(np.float32), cv.INTER_LINEAR)

# Pixelize
def pixelize(img, size):
    if (size == 0):
        return img

    height, width, _ = img.shape
    p_img = cv.resize(img, (width // size, height // size), interpolation=cv.INTER_NEAREST)
    return cv.resize(p_img, (width, height), interpolation=cv.INTER_NEAREST)

# Neon borders
def neon_borders(frame, sigma, threshold):
    sframe = cv.cvtColor(clip(cv.GaussianBlur(src=frame, ksize=(5,5), sigmaX=sigma)), cv.COLOR_BGR2GRAY)
    gX     = cv.Sobel(sframe, cv.CV_64F, 1, 0, ksize=3)
    gY     = cv.Sobel(sframe, cv.CV_64F, 0, 1, ksize=3) * -1
    _, mod = cv.threshold(np.sqrt(gX**2 + gY**2).astype(np.uint8), threshold, 255, cv.THRESH_BINARY)
    return cv.bitwise_and(frame, frame, mask=mod)

# =============================================================================
# Main
# =============================================================================
# https://www.tutorialspoint.com/how-to-show-webcam-in-tkinter-window#:~:text=Using%20the%20OpenCV%20module%2C%20we,webcam%20in%20a%20tkinter%20window.&text=Once%20the%20installation%20has%20been,(if%20possible)%20using%20OpenCV.
# http://omes-va.com/tkinter-opencv-imagen/
# https://www.tutorialspoint.com/python/tk_scale.htm
# WEBCAM CAPTURE
cam = cv.VideoCapture(0)
cam_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH) * 0.85)
cam_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT) * 0.85)

# GUI
# - The window:
root = tk.Tk()
root.title('Practica 1')
root.geometry(f'1300x{root.winfo_screenheight()}')
root.resizable(width=False, height=False)
root.rowconfigure(0, minsize=20)
root.rowconfigure(3, minsize=20)
root.columnconfigure(0, minsize=-10)
root.columnconfigure(3, minsize=20)

disabled_color = rgb_to_hex(240,240,240)
enabled_color  = rgb_to_hex(180,180,180)
# - Source image
images_frame = tk.Frame(root)
images_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

source_image = tk.Label(images_frame)
source_image.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
#source_image.grid(row=1,column=2,sticky='nswe')
# - Output image
#right_container = tk.Frame(root, bg="blue")
#right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#extra_space = tk.Label(right_container, text="Extra Space")
#extra_space.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
output_image = tk.Label(images_frame)
output_image.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
#output_image.grid(row=2,column=2,sticky='nswe')
#output_image.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
# - Options
options_canvas = tk.Canvas(root)
#options_canvas.grid(row=1,column=1, sticky='nswe', rowspan=2)
options_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
previous_effect = 0
selected_effect = tk.IntVar()
# -- Ningun efecto
none_option     = tk.Radiobutton(options_canvas, text='Ninguno', width=25,value=0, variable=selected_effect)
options_canvas.create_window(-50, 20, anchor=tk.NW, window=none_option)
# -- Contraste
contrast_option = tk.Radiobutton(options_canvas, text='Contraste y brillo', width=25,value=1, variable=selected_effect)
options_canvas.create_window(-29, 50, anchor=tk.NW, window=contrast_option)
bar_contrast    = tk.Scale(options_canvas, variable=tk.DoubleVar(value=1.0), from_=-3, to=3, resolution=0.1, length=245, orient=tk.HORIZONTAL,state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 70, anchor=tk.NW, window=bar_contrast)
bar_brightness  = tk.Scale(options_canvas, variable=tk.IntVar(value=0), from_=-512, to=512, length=245, orient=tk.HORIZONTAL, state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 125, anchor=tk.NW, window=bar_brightness)

# -- Pixelize
pixelize_option = tk.Radiobutton(options_canvas, text='Pixelizar', width=25, value=6, variable=selected_effect)
options_canvas.create_window(240, 50, anchor=tk.NW, window=pixelize_option)
bar_pixelize    = tk.Scale(options_canvas, variable=tk.IntVar(value=0.0), from_=0, to=50, label='Factor de pixelización', length=245, orient=tk.HORIZONTAL, state='disabled', troughcolor=disabled_color)
options_canvas.create_window(320, 85, anchor=tk.NW, window=bar_pixelize)

# -- Neon borders
neon_option = tk.Radiobutton(options_canvas, text='Bordes de neon', width=25,value=7, variable=selected_effect)
options_canvas.create_window(260, 180, anchor=tk.NW, window=neon_option)
bar_neon_sigma = tk.Scale(options_canvas, variable=tk.IntVar(value=0), from_=0, to=10, length=245, orient=tk.HORIZONTAL,state='disabled', troughcolor=disabled_color, label='Sigma')
options_canvas.create_window(320, 210, anchor=tk.NW, window=bar_neon_sigma)
bar_neon_threshold = tk.Scale(options_canvas, variable=tk.IntVar(value=0), from_=0, to=255, length=245, orient=tk.HORIZONTAL,state='disabled', troughcolor=disabled_color, label='Threshold')
options_canvas.create_window(320, 275, anchor=tk.NW, window=bar_neon_threshold)

#-- Ecualizacion de histograma
equalize_option = tk.Radiobutton(options_canvas, text='Ecualizacion de histograma',width=25, value=2, variable=selected_effect)
options_canvas.create_window(-2, 180, anchor=tk.NW, window=equalize_option)
# -- Alien
alien_option = tk.Radiobutton(options_canvas, text='Alien', width=25,value=3, variable=selected_effect)
options_canvas.create_window(-60, 210, anchor=tk.NW, window=alien_option)
bar_R = tk.Scale(options_canvas, from_=0, to=255, length=255, orient=tk.HORIZONTAL, label='Rojo (R)', state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 240, anchor=tk.NW, window=bar_R)
bar_G = tk.Scale(options_canvas, from_=0, to=255, length=255, orient=tk.HORIZONTAL, label='Verde (G)', state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 300, anchor=tk.NW, window=bar_G)
bar_B = tk.Scale(options_canvas, from_=0, to=255, length=255, orient=tk.HORIZONTAL, label='Azul (B)', state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 360, anchor=tk.NW, window=bar_B)
# -- Poster
more_precission = tk.BooleanVar()
poster_option = tk.Radiobutton(options_canvas, text='Poster', width=25,value=4, variable=selected_effect)
options_canvas.create_window(-57, 430, anchor=tk.NW, window=poster_option)
bar_clusters = tk.Scale(options_canvas, from_=0, to=150, length=255, orient=tk.HORIZONTAL, label='Clusters (k)', state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 460, anchor=tk.NW, window=bar_clusters)
checkbox_more_precission = tk.Checkbutton(root, text="Más precisión", variable=more_precission, state='disabled')
options_canvas.create_window(32, 530, anchor=tk.NW, window=checkbox_more_precission)

# -- Distorsion
distorsion_option = tk.Radiobutton(options_canvas, text='Distorsion', width=25,value=5, variable=selected_effect)
options_canvas.create_window(-50, 570, anchor=tk.NW, window=distorsion_option)
bar_coefficient = tk.Scale(options_canvas, variable=tk.DoubleVar(value=0), from_=-10e-6, to=10e-6, length=255, resolution=10e-7, orient=tk.HORIZONTAL, label='Primer coeficiente', state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 600, anchor=tk.NW, window=bar_coefficient)
bar_x_center_pos = tk.Scale(options_canvas, variable=tk.IntVar(value=cam_width/2), from_=0, to=cam_width, length=255, resolution=1, orient=tk.HORIZONTAL, label='Posicion del centro en X', state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 660, anchor=tk.NW, window=bar_x_center_pos)
bar_y_center_pos = tk.Scale(options_canvas, variable=tk.IntVar(value=cam_height/2), from_=0, to=cam_height, length=255, resolution=1, orient=tk.HORIZONTAL, label='Posicion del centro en Y', state='disabled', troughcolor=disabled_color)
options_canvas.create_window(34, 720, anchor=tk.NW, window=bar_y_center_pos)

def reset_values():
    selected_effect.set(0)
    bar_contrast.set(1.0)
    bar_brightness.set(0)
    bar_R.set(0)
    bar_G.set(0)
    bar_B.set(0)
    bar_clusters.set(0)
    more_precission.set(False)
    bar_coefficient.set(0)
    bar_x_center_pos.set(cam_width/2)
    bar_y_center_pos.set(cam_height/2)
    bar_pixelize.set(0)
    bar_neon_sigma.set(0)
    bar_neon_threshold.set(0)

button_reset = tk.Button(options_canvas, text="Reiniciar", command=reset_values)
options_canvas.create_window(265, 20, anchor=tk.NW, window=button_reset)

def disable_option(n):
    if (n == 1):
        bar_contrast.config(state='disabled', troughcolor=disabled_color)
        bar_brightness.config(state='disabled', troughcolor=disabled_color)
    elif (n == 3):
        bar_R.config(state='disabled', troughcolor=disabled_color)
        bar_G.config(state='disabled', troughcolor=disabled_color)
        bar_B.config(state='disabled', troughcolor=disabled_color)
    elif (n == 4):
        bar_clusters.config(state='disabled', troughcolor=disabled_color)
        checkbox_more_precission.config(state='disabled')
    elif (n == 5):
        bar_coefficient.config(state='disabled', troughcolor=disabled_color)
        bar_x_center_pos.config(state='disabled', troughcolor=disabled_color)
        bar_y_center_pos.config(state='disabled', troughcolor=disabled_color)
    elif (n == 6):
        bar_pixelize.config(state='disabled', troughcolor=disabled_color)
    elif (n == 7):
        bar_neon_sigma.config(state='disabled', troughcolor=disabled_color)
        bar_neon_threshold.config(state='disabled', troughcolor=disabled_color)
        
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
        frame = histogram_equalization(frame)
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
        checkbox_more_precission.config(state='normal')
        frame = poster(frame, bar_clusters.get(), more_precission.get())
    elif (n == 5):
        bar_coefficient.config(state='normal', troughcolor=enabled_color)
        bar_x_center_pos.config(state='normal', troughcolor=enabled_color)
        bar_y_center_pos.config(state='normal', troughcolor=enabled_color)
        frame = barrel_pincussion(frame, bar_coefficient.get(), bar_x_center_pos.get(), bar_y_center_pos.get())
    elif (n == 6):
        bar_pixelize.config(state='normal', troughcolor=enabled_color)
        frame = pixelize(frame, bar_pixelize.get())
    elif (n == 7):
        bar_neon_sigma.config(state='normal', troughcolor=enabled_color)
        bar_neon_threshold.config(state='normal', troughcolor=enabled_color)
        frame = neon_borders(frame, bar_neon_sigma.get(), bar_neon_threshold.get())
        
    previous_effect = n
    return frame

def update_view():
    check, frame = cam.read()
    if (check):
        frame       = cv.flip(cv.resize(frame, (int(frame.shape[1] * 0.85), int(frame.shape[0] * 0.85)), interpolation=cv.INTER_AREA), 1)
        # Source image:
        source      = cv.cvtColor(frame, cv.COLOR_BGR2RGB,1)
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
