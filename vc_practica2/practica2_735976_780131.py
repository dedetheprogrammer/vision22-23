from PIL import Image, ImageTk
import cv2 as cv
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog

def clip(frame):
    return np.clip(frame, 0, 255).astype(np.uint8)

def Sobel(frame):
    frame = cv.cvtColor(clip(cv.GaussianBlur(frame,(3,3),0)), cv.COLOR_BGR2GRAY)
    kernelx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3,3)
    kernely = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape(3,3)
    gX = np.zeros(frame.shape, dtype=np.int32)
    gY = np.zeros(frame.shape, dtype=np.int32)
    gXY = np.zeros(frame.shape, dtype=np.int32)
    height, width = frame.shape
    for i in range(height - 1) :
        for j in range(width - 1) :
            # Set output to 0 if the 3x3 receptive field is out of bound.
            if ((i < 1) | (i > height - 2) | (j < 1) | (j > width - 2)) :
                gX[i][j] = 0
                gY[i][j] = 0
                gXY[i][j] = 0
            else :

                # Apply the sobel filter at the current "receptive field".
                gx_sum = 0
                gy_sum = 0
                for m in range(-1, 2) :
                    for n in range(-1, 2) :
                        pixel = frame[i + m][j + n]
                        gx_sum += pixel * kernelx[m + 1][n + 1]
                        gy_sum += pixel * kernely[m + 1][n + 1]
                sum = gx_sum + gy_sum

                gX[i][j] = gx_sum
                gY[i][j] = gy_sum
                gXY[i][j] = sum

    #gXY   = cv.addWeighted(gX, 0.5, gY, 0.5, 0)
    print(np.min(gX), " ", np.max(gX))
    mod   = clip(np.sqrt(gX**2 + gY**2))
    ori   = np.arctan2(gY, gX)
    return gX, gY, gXY, mod, ori

# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def Canny(frame, low_threshold_ratio = 0.05, high_threshold_ratio = 0.09):
    _, _, _, mod, ori = Sobel(frame)

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

    N,M = gradient.shape; CM = int(M/2); CN = int(N/2)
    horizon = np.zeros(M)
    for i in range(N-1):
        for j in range(M-1):
            theta = orientation[i,j]
            if ((gradient[i,j] >= threshold) and ((np.abs(theta) > 1e-6) and (np.abs(theta - np.pi/2) > 1e-6) 
                                             and (np.abs(theta - np.pi) > 1e-6) and (np.abs(theta - (np.pi * (2/3)) > 1e-6)))):
                # Transformamos las coordenadas para que esten en el centro de 
                # la imagen:
                x = j - CN; y = CM - i
                # Calculamos las coordenadas polares:
                rho = x*np.cos(theta) + y*np.sin(theta)
                # Coordenadas polares del horizonte: (pi/2, N/2)
                ## Calcular intersección
                # horizon[x + CN] += 1 

    return [np.argmax(horizon), CN]

def pack_frame(root, side, fill, expand):
    new_frame = tk.Frame(root)
    new_frame.pack(side=side, fill=fill, expand=expand)
    return new_frame

def pack_label(root, side, fill, expand):
    new_label = tk.Label(root)
    new_label.pack(side=side, fill=fill, expand=expand)
    return new_label

def conf_label(label, frame):
    img_frame   = Image.fromarray(frame)
    imgtk_frame = ImageTk.PhotoImage(image=img_frame)
    label.imgtk = imgtk_frame
    label.configure(image=imgtk_frame)

def resize_frame(frame, rfactor):
    return cv.resize(frame, (int(frame.shape[1] * rfactor), int(frame.shape[0] * rfactor)), interpolation=cv.INTER_AREA)

def load_image():
    global using_camera
    global path_image
    using_camera = False
    path_image = filedialog.askopenfilename(
        filetypes = [
            ("image", ".jpeg"),
            ("image", ".png"),
            ("image", ".jpg"),
            ("image", ".pgm")
        ]
    )

def use_camera():
    global using_camera
    global path_image

    using_camera = True
    path_image   = None

# GUI
# - The window:
root = tk.Tk()
root.title('Practica 2')
camera_views = pack_frame(root, tk.RIGHT, tk.BOTH, True)
using_camera = True

# - Sources
sources_frame    = pack_frame(camera_views , tk.TOP, tk.BOTH, True)
sources_up_frame = pack_frame(sources_frame, tk.TOP, tk.BOTH, True)
sources_dw_frame = pack_frame(sources_frame, tk.TOP, tk.BOTH, True) 
sources = [
    pack_label(sources_up_frame, tk.LEFT, tk.BOTH, True),
    pack_label(sources_up_frame, tk.LEFT, tk.BOTH, True),
    pack_label(sources_up_frame, tk.LEFT, tk.BOTH, True),
    pack_label(sources_dw_frame, tk.LEFT, tk.BOTH, True),
    pack_label(sources_dw_frame, tk.LEFT, tk.BOTH, True),
    pack_label(sources_dw_frame, tk.LEFT, tk.BOTH, True),
]
# - Output image
outputs_frame = pack_frame(camera_views, tk.TOP, tk.BOTH, True)
outputs = [
    pack_label(outputs_frame, tk.LEFT, tk.BOTH, True)
]

# - Options
options_canvas = tk.Canvas(root)
options_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
btn = tk.Button(root, text="Elegir imagen", width=25, command=load_image)
options_canvas.create_window(90, 20, anchor=tk.NW, window=btn)
btn_1 = tk.Button(root, text="Poner camara", width=25, command=use_camera)
options_canvas.create_window(90, 80, anchor=tk.NW, window=btn_1)

gX_preview   = tk.BooleanVar()
gY_preview   = tk.BooleanVar()
g_XY_preview   = tk.BooleanVar()
rad_preview  = tk.BooleanVar()
smod_preview = tk.BooleanVar()
cmod_preview = tk.BooleanVar()

checkbox_gX_preview = tk.Checkbutton(root, text="Habilitar vista: gradiente en X", variable=gX_preview)
options_canvas.create_window(90, 140, anchor=tk.NW, window=checkbox_gX_preview)
checkbox_gY_preview = tk.Checkbutton(root, text="Habilitar vista: gradiente en Y", variable=gY_preview)
options_canvas.create_window(90, 200, anchor=tk.NW, window=checkbox_gY_preview)
checkbox_gY_preview = tk.Checkbutton(root, text="Habilitar vista: gradiente en XY", variable=g_XY_preview)
options_canvas.create_window(90, 260, anchor=tk.NW, window=checkbox_gY_preview)
checkbox_rad_preview = tk.Checkbutton(root, text="Habilitar vista: orientación", variable=rad_preview)
options_canvas.create_window(90, 320, anchor=tk.NW, window=checkbox_rad_preview)
checkbox_smod_preview = tk.Checkbutton(root, text="Habilitar vista: modulo de Sobel", variable=smod_preview)
options_canvas.create_window(90, 380, anchor=tk.NW, window=checkbox_smod_preview)
checkbox_cmod_preview = tk.Checkbutton(root, text="Habilitar modulo de Canny", variable=cmod_preview)
options_canvas.create_window(90, 440, anchor=tk.NW, window=checkbox_cmod_preview)

cam   = cv.VideoCapture(0)
frame = cam.read()[1]; path_image = None
blank = resize_frame(np.full(frame.shape, (200,200,200), dtype=np.uint8), 0.35)
def update_view():
    global using_camera
    global path_image
    global frame

    if (using_camera):
        _, frame = cam.read()
        frame = cv.flip(frame, 1)
    else:
        if (not path_image == None):
            frame = cv.imread(path_image)

    # Source image:
    if (using_camera or (not using_camera and not path_image == None)):
        source = cv.cvtColor(resize_frame(frame, 0.35), cv.COLOR_BGR2RGB,1)
        conf_label(sources[0], source)

        gX, gY, gXY, mod, rad = Sobel(frame)
        fuga_sobel = Hough_transform(mod, rad, 255)

        if (gX_preview.get()):
            conf_label(sources[0], resize_frame(clip(gX/2+128),0.35))
        else:
            conf_label(sources[0], blank)

        if (gY_preview.get()):
            conf_label(sources[1], resize_frame(clip(gY/2+128),0.35))
        else:
            conf_label(sources[1], blank)

        if (g_XY_preview.get()):
            conf_label(sources[2], resize_frame(clip(gXY/2+128),0.35))
        else:
            conf_label(sources[2], blank)

        if (rad_preview.get()):
            conf_label(sources[3], resize_frame(np.uint8(rad/np.pi*128),0.35))
        else:
            conf_label(sources[3], blank)

        if (smod_preview.get()):
            conf_label(sources[4], resize_frame(clip(mod),0.35))
        else:
            conf_label(sources[4], blank)
        # Output image:
        if (cmod_preview.get()):
            mod, rad = Canny(frame, 0.05, 0.06)
            fuga_canny = Hough_transform(mod, rad, 100)
            conf_label(sources[5], resize_frame(clip(mod),0.35))
            cv.line(frame, (fuga_canny[0], fuga_canny[1]-10), (fuga_canny[0], fuga_canny[1]+10), (255,0,255), 2)
            cv.line(frame, (fuga_canny[0]-10, fuga_canny[1]), (fuga_canny[0]+10, fuga_canny[1]), (255,0,255), 2)
        else:
            conf_label(sources[5], blank)

        cv.line(frame, (fuga_sobel[0], fuga_sobel[1]-10), (fuga_sobel[0], fuga_sobel[1]+10), (255,0,0), 2)
        cv.line(frame, (fuga_sobel[0]-10, fuga_sobel[1]), (fuga_sobel[0]+10, fuga_sobel[1]), (255,0,0), 2)
        output = cv.cvtColor(frame, cv.COLOR_BGR2RGB,1)
        for i in outputs:
            conf_label(i, np.uint8(output))
    sources[0].after(20, update_view)

update_view()
root.mainloop()
cam.release()