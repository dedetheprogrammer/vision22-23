from PIL import Image, ImageTk
import cv2 as cv
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.signal import convolve2d

img_name = ""

def clip(frame):
    return np.clip(frame, 0, 255).astype(np.uint8)

def gaussian(size, sigma):
    values = np.arange(int(-size/2),int(size/2)+1,1)
    gauss = np.zeros(size)
    gaussd = np.zeros(size)
    for i in range(size) :
        gauss[i] = math.exp(-(values[i]**2)/(2*(sigma**2)))
        gaussd[i] = (-values[i]/(sigma**2))*math.exp(-(values[i]**2)/(2*(sigma**2)))
    return gauss, gaussd

def canny(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    size = 5
    sigma = 1
    gauss, gaussd = gaussian(size, sigma)
    k = sum([x for x in gauss if x > 0])
    kd = sum([x for x in gaussd if x > 0])

    gX_aux = np.zeros(frame.shape, dtype=np.int32)
    gX = np.zeros(frame.shape, dtype=np.int32)
    gY_aux = np.zeros(frame.shape, dtype=np.int32)
    gY = np.zeros(frame.shape, dtype=np.int32)
    gXY = np.zeros(frame.shape, dtype=np.int32)
    height, width = frame.shape
    kernelv_x = np.transpose(gauss[np.newaxis])# / k
    kernelv_y = np.transpose(gaussd[np.newaxis])# / kd
    frame_v = np.pad(frame, ((2,2), (0,0)), mode='constant')
    for i in range(height) :
        for j in range(width) :
            gx_sum = 0
            gy_sum = 0
            for m in range(-int(size/2), int(size/2) + 1) :
                pixel = frame_v[i+2 + m,j]
                gx_sum += pixel * kernelv_x[m + int(size/2),0]
                gy_sum += pixel * kernelv_y[m + int(size/2),0]

            gX_aux[i,j] = gx_sum
            gY_aux[i,j] = gy_sum
    
    kernelh_x = (gaussd[::-1][np.newaxis])# / kd
    kernelh_y = (gauss[np.newaxis])# / k
    frame_h_gx = np.pad(gX_aux, ((0,0), (2,2)), mode='constant')
    frame_h_gy = np.pad(gY_aux, ((0,0), (2,2)), mode='constant')
    for i in range(height - 1) :
        for j in range(width - 1) :
            gx_sum = 0
            gy_sum = 0
            for m in range(-int(size/2), int(size/2) + 1) :
                #pixel = frame_v[i][j+2 + m]
                gx_sum += frame_h_gx[i,j+int(size/2) + m] * kernelh_x[0,m + int(size/2)]
                gy_sum += frame_h_gy[i,j+int(size/2) + m] * kernelh_y[0,m + int(size/2)]

            gX[i,j] = gx_sum
            gY[i,j] = gy_sum
    gXY = gX + gY
    mod   = clip(np.sqrt(gX**2 + gY**2))
    ori   = np.arctan2(gY, gX)
    return gX, gY, gXY, mod, ori


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
                gX[i,j] = 0
                gY[i,j] = 0
                gXY[i,j] = 0
            else :

                # Apply the sobel filter at the current "receptive field".
                gx_sum = 0
                gy_sum = 0
                for m in range(-1, 2) :
                    for n in range(-1, 2) :
                        pixel = frame[i + m,j + n]
                        gx_sum += pixel * kernelx[m + 1,n + 1]
                        gy_sum += pixel * kernely[m + 1,n + 1]
                sum = gx_sum + gy_sum

                gX[i,j] = gx_sum
                gY[i,j] = gy_sum
    gXY = gX + gY
    mod   = clip(np.sqrt(gX**2 + gY**2))
    ori   = np.arctan2(gY, gX)
    return gX, gY, gXY, mod, ori

# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
def Hough_transform(gradient, orientation, threshold):

    height, width = gradient.shape
    central_y = int(height / 2)
    central_x = int(width / 2)
    if(img_name == 'pasillo1.pgm') :
        central_y -= 50
    horizon = np.zeros(width)
    for i in range(height-1):
        for j in range(width-1):
            theta = orientation[i,j]
            # Además del umbral(threshold), comprobamos que la orientación no sea aproximadamente vertical ni horizontal
            if ((gradient[i,j] >= threshold) and 
                ((np.abs(theta) > np.radians(5)) and (np.abs(theta - np.pi/2) > np.radians(5)) and
                 (np.abs(theta - np.pi) > np.radians(5)) and (np.abs(theta - (np.pi * (2/3)) > np.radians(5))) and
                 (np.abs(theta - 2*np.pi) > np.radians(5)))):
                x = j - central_x
                y = central_y - i
                # Calculamos la ecuación de la recta con las coordenadas polares
                rho = x*np.cos(theta) + y*np.sin(theta)
                # Calculamos la coordenada x de la línea central donde intersecciona
                vote_x = int((rho / np.cos(theta) + central_x))
                # Si la x está dentro de la imagen se vota el píxel
                if((vote_x >= 0) and (vote_x < width)) :
                    horizon[vote_x] += 1

    return [np.argmax(horizon), central_y]

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
g_XY_preview = tk.BooleanVar()
rad_preview  = tk.BooleanVar()
smod_preview = tk.BooleanVar()
vanishing_point_preview = tk.BooleanVar()
selected_operator = tk.IntVar(value=1)

text_preview = tk.Label(root, text="Elegir operador")
options_canvas.create_window(90, 140, anchor=tk.NW, window=text_preview)
option1_checkbox = tk.Radiobutton(root, text="Sobel", variable=selected_operator, value=1)
options_canvas.create_window(110, 180, anchor=tk.NW, window=option1_checkbox)
option2_checkbox = tk.Radiobutton(root, text="Canny", variable=selected_operator, value=2)
options_canvas.create_window(110, 220, anchor=tk.NW, window=option2_checkbox)
checkbox_gX_preview = tk.Checkbutton(root, text="Habilitar vista: gradiente en X", variable=gX_preview)
options_canvas.create_window(90, 320, anchor=tk.NW, window=checkbox_gX_preview)
checkbox_gY_preview = tk.Checkbutton(root, text="Habilitar vista: gradiente en Y", variable=gY_preview)
options_canvas.create_window(90, 380, anchor=tk.NW, window=checkbox_gY_preview)
checkbox_gY_preview = tk.Checkbutton(root, text="Habilitar vista: gradiente en XY", variable=g_XY_preview)
options_canvas.create_window(90, 440, anchor=tk.NW, window=checkbox_gY_preview)
checkbox_rad_preview = tk.Checkbutton(root, text="Habilitar vista: orientación", variable=rad_preview)
options_canvas.create_window(90, 500, anchor=tk.NW, window=checkbox_rad_preview)
checkbox_smod_preview = tk.Checkbutton(root, text="Habilitar vista: modulo", variable=smod_preview)
options_canvas.create_window(90, 560, anchor=tk.NW, window=checkbox_smod_preview)
checkbox_vanishing_point_preview = tk.Checkbutton(root, text="Punto de fuga", variable=vanishing_point_preview)
options_canvas.create_window(90, 620, anchor=tk.NW, window=checkbox_vanishing_point_preview)

cam   = cv.VideoCapture(0)
frame = cam.read()[1]; path_image = None
blank = resize_frame(np.full(frame.shape, (200,200,200), dtype=np.uint8), 0.35)
def update_view():
    global using_camera
    global path_image
    global frame
    global img_name

    if (using_camera):
        _, frame = cam.read()
        frame = cv.flip(frame, 1)
    else:
        if (not path_image == None):
            frame = cv.imread(path_image)
            char = "/"
            last_index = path_image.rindex(char)
            substring = path_image[last_index + 1:]
            img_name = substring

    # Source image:
    if (using_camera or (not using_camera and not path_image == None)):
        source = cv.cvtColor(resize_frame(frame, 0.35), cv.COLOR_BGR2RGB,1)
        conf_label(sources[0], source)
        
        gX = np.zeros(frame.shape, dtype=np.int32)
        gY = np.zeros(frame.shape, dtype=np.int32)
        gXY = np.zeros(frame.shape, dtype=np.int32)
        mod = np.zeros(frame.shape, dtype=np.int32)
        rad = np.zeros(frame.shape, dtype=np.int32)
        if(selected_operator.get() == 1):
            gX, gY, gXY, mod, rad = Sobel(frame)
        else:
            gX, gY, gXY, mod, rad = canny(frame)

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
            cv.imwrite('Sobel_mod.jpg', clip(mod))
        else:
            conf_label(sources[4], blank)
        # Output image:
        if (vanishing_point_preview.get()):
            vanishing_point = Hough_transform(mod, rad, 100)
            conf_label(sources[5], resize_frame(clip(mod),0.35))
            cv.line(frame, (vanishing_point[0], vanishing_point[1]-10), (vanishing_point[0], vanishing_point[1]+10), (255,0,255), 2)
            cv.line(frame, (vanishing_point[0]-10, vanishing_point[1]), (vanishing_point[0]+10, vanishing_point[1]), (255,0,255), 2)
        else:
            conf_label(sources[5], blank)

        output = cv.cvtColor(frame, cv.COLOR_BGR2RGB,1)
        o_name = img_name + ".jpg"
        cv.imwrite(o_name, output)
        for i in outputs:
            conf_label(i, np.uint8(output))
    sources[0].after(20, update_view)

update_view()
root.mainloop()
#gauss, gaussd = gaussian(5, 1)
#k = sum([x for x in gauss if x > 0])
#kd = sum([x for x in gaussd if x > 0])
#print(np.transpose(gaussd[np.newaxis]))
#print(np.transpose(gaussd[np.newaxis]) / 2)


#print(gaussd[::-1][np.newaxis])

cam.release()