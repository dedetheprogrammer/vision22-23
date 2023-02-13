# 0. Loading libraries
import cv2 as cv                 # Import python-supported OpenCV functions.
#import matplotlib.pylab as plt  # Import matplotlib.
import numpy as np               # Import numpy.
#from skimage import io
###############################################################################
# 1. Read and show an image
# This is how you define a function. You don't need to define the type of the
# parameters, just write them, the python compiler will find out automatically
# their type.
def example_1():
    # Python list style.
    images = [
        # Loads the image from the given path with its default colors. I think that
        # without option and using IMREAD_COLOR has the same result.
        cv.imread('cook.jpg'),
        # cv.imread('cook.jpg', cv.IMREAD_COLOR),
        # Loads the image from the given path with grayscale colors.
        cv.imread('cook.jpg', cv.IMREAD_GRAYSCALE)
    ]
    cv.namedWindow('Example 1', cv.WINDOW_AUTOSIZE)
    # For-loop: loop through each element of images saving the value of
    # each iteration in img:
    for img in images:
        # Shows the image.
        cv.imshow('Example 1', img)
        # Shows the object type of the image. The fstrings (f'...') have the same
        # behavior as echo "$var", but instead of $var or ${var}, is {var}.
        print(f'Type: {type(img)}')
        # Shows the dimensions (Width, Height, Channels) of the image.
        print(f'Dimensions: {img.shape}') 
        if cv.waitKey(0) == 27:
            break
    cv.destroyWindow('Example 1') # cv2.destroyAllWindows()

# example_1()
###############################################################################
# 2. Import and show an image. Not working, needs skimage, and the installation
# gives an error.
#def example_2():
    # Loads image from a given url. The image is read in BGR format:
    #image = io.imread('https://i.kym-cdn.com/photos/images/newsfeed/002/488/664/964.jpg')
    # Shows the image:
    #   - With cvtColor, converts the image from BGR to RGB (option COLOR_BGR2RGB).
    #   - With hconcat, puts both images next to each other horizontally.
    #cv.namedWindow('Example 2', cv.WINDOW_AUTOSIZE)
    #cv.imshow('Example 2', cv.hconcat((image, cv.cvtColor(image, cv.COLOR_BGR2RGB))))
    # Shows the object type of the image.
    #print(f'Type: {type(image)}')
    # Shows the dimensions (Width, Height, Channels) of the image.
    #print(f'Dimensions: {image.shape}')
    # Waits until any key is pressed.
    #cv.waitKey(0)
    # Destroys the window.
    #cv.destroyWindow('Example 1') # cv2.destroyAllWindows()

# example_2()
###############################################################################
# 3. Images are matrices:
def imshow_properties(img, window):
    # Shows the image.
    cv.imshow(window, img)
    print('Properties of an image matrix:')
    # Shows the object type of the image.
    print(f'Type: {type(img)}')
    # Shows the dimensions (Width, Height, Channels) of the image.
    print(f'Dimensions (width, height, channels): {img.shape}')
    # Shows the number of pixels.
    print(f'Number of pixels: {img.size}')
    # Shows the image datatype.
    print(f'Image datatype: {img.dtype}')

def example_3():
    # An image is a matrix of values as we could see in previous examples:
    # Python list style.
    images = [
        # Loads the image from the given path with its default colors.
        cv.imread('cat.jpg'),
        # Loads the image from the given path with grayscale colors.
        cv.imread('cat.jpg', cv.IMREAD_GRAYSCALE)
    ]
    # Open a new window named 'Example 3'.
    cv.namedWindow('Example 3', cv.WINDOW_AUTOSIZE)
    # For-loop: loop through each element of images saving the value of
    # each iteration in img:
    for img in images:
        imshow_properties(img, 'Example 3')
        if cv.waitKey(0) == 27:
            break
    cv.destroyWindow('Example 3') # cv2.destroyAllWindows()

# example_3()
###############################################################################
# 4. Images matrices creation
def example_4():
   # Ways of initialize a matrix
    # BE CAREFUL with the sizes.
    width  = 480 # The width is the number of columns.
    height = 320 # The height is the number of rows.
    channels = 1 # 1 if grayscale, 3 if RGB or BGR.
    # GRAYSCALE IMAGES
    grayscale_images = [
        # Creates a 3rd-dim array and fills it with 0. When calling a function, you
        # can write the name of the parameter (dtype = 'uint8') or ignore it
        # ('uint8'). Sirve para llamar a la función unicamente con los parámetros 
        # que te interesen, a lo mejor solo quieres usar dos parámetros de una
        # función de 5 parámetros y ambos parámetros estan en los extremos. Al igual
        # que en C++, Python va a procesar los parámetros uno por uno y en orden,
        # pero si le pones el nombre, le dará el valor al parámetro que has
        # declarado y no al siguiente que debería tocarle.
        np.zeros((height, width, channels), dtype = 'uint8'),
        # Creates a 3rd-dim array and fills it with a given value (255).
        np.full((height, width, channels), 255, dtype = 'uint8'),
        # Creates a 3rd-dim array and fills it with random values (between 0-256).
        np.random.randint(0, 256, (height, width, channels), dtype = 'uint8')
    ]

    # RGB/BGR IMAGES
    channels = 3
    rgb_images = [
        # Creates a 3rd-dim array and fills it with 0.
        np.zeros((height, width, channels), dtype = 'uint8'),
        # Creates a 3rd-dim array and fills it with a given value (255).
        np.full((height, width, channels), 255, dtype = 'uint8'),
        # Creates a 3rd-dim array and fills it with random values (between 0-256).
        np.random.randint(0, 256, (height, width, channels), dtype = 'uint8')
    ]
    # Open a new window named 'Example 3'.
    cv.namedWindow('Example 4', cv.WINDOW_AUTOSIZE)
    for img_gs in grayscale_images:
        imshow_properties(img_gs, 'Example 4')
        if cv.waitKey(0) == 27:
            break

    for img_rgb in rgb_images:
        imshow_properties(img_rgb, 'Example 4')
        if cv.waitKey(0) == 27:
            break
    cv.destroyWindow('Example 4') # cv2.destroyAllWindows()

    # You can also modify individual elements or regions of the image (matrix):
    # Init only one channel (0 -> R, 1 -> G, 2 -> B):
    #   - [75,75,0]: each elements indicates the dimensions, ergo [row, column, channel].
    rgb_images[0][75,75,0] = 255
    # Init each channel:
    #   - [75,75]: dimensions can be omited but you'll need to initialize the previous 
    #     dimension with a list with its size equally to the omited dimension.
    rgb_images[0][75,75] = [255,255,255]
    # Init a region only in one channel:
    rgb_images[0][0:75, 0:75, 0] = 255
    # Init a region in each channel:
    rgb_images[0][45:105, 45:105] = [255,255,255]
    return rgb_images

# imgs = example_4()
###############################################################################
# 5. Image matrices operations
def example_5(imgs):
    # Creates a reference. If modified, will modify the original too.
    ref = imgs[2]
    # Creates a new whole variable.
    cpy = imgs[2].copy()
    # ADD, SUB, MUL
    cpy += 5 # Tambien vale con *=, /=. Aplica la operacion a toda la matriz.
    cpy = (imgs[0] - 1) * 3 - imgs[2] # Solo es un ejemplo.
    # Identity matrix
    idy = np.eye(3, dtype = 'float32')
    # One dimension array of 3 elements.
    x   = np.ones((3,1), dtype = 'float32')
    # Element to element product. Se multiplica elemento por elemento [0]*[0], etc.
    y_0 = idy * x
    print(f'{y_0}')
    # Matrician product. Producto matricial de toda la vida.
    y_1 = np.dot(idy,x)
    print(f'{y_1}')
    # Matrix transpose
    idy_t = idy.T
    # Matrix inverse
    idy_i = np.linalg.inv(idy)
    # Equation solution of A*X = Y
    cv.solve(idy, y_0, x)
    # Scalar and vector product
    v = np.random.randint(1, 10, (1, 3), dtype = 'uint8')
    w = np.random.randint(1, 10, (1, 3), dtype = 'uint8')
    s = np.dot(v,w.T) # Necesita invertirlo por el rollo de las dimensiones: (1,3) * (3,1) = (1,1)
    z = np.cross(v,w)
    # Boolean map. The operations will return a matrix with the result of the given
    # conditions for each element of the original matrices. Any condition can be
    # applied.
    cond = idy > np.random.randint(1, 10, (3, 3), dtype = 'uint8')

#example_5(imgs)
###############################################################################
# 6. La practica. Escribir y probar un programa que tome imágenes con una cámara
# y las muestre en pantalla. Si tu computador no lleva cámara integrada, 
# necesitarás una webcam (sirve una sencilla de 10 Euros).
# Inicializar el medio para hacer la foto.
cam = cv.VideoCapture(0)
# Hacer la foto
ret, frame = cam.read()
if ret:
    cv.namedWindow('Practica 0', cv.WINDOW_AUTOSIZE)
    imshow_properties(frame, 'Practica 0')
    cv.waitKey(0)
    cv.imwrite('captured.jpg', frame)
    cv.destroyWindow('Practica 0') # cv2.destroyAllWindows()
cam.release()