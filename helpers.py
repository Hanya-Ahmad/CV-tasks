import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter

def convolve(img, kernel):
    size = img.shape[0]
    k = kernel.shape[0]
    
    # Initiate an array of zeros for the resulting convolved image
    convolved_img = np.zeros(shape=(size, img.shape[1]))
    
    # Loop over the rows
    for i in range(img.shape[0]-2):
        # Loop over the columns
        for j in range(img.shape[1]-2):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
            
    return convolved_img

def gs_filter(img, sigma):
    return gaussian_filter(img, sigma)

def read_img(path):
    im = imageio.imread(path, as_gray=True)
    im=np.array(im)
    # im = im.astype('int32')
    return im
      # some lines of code
#   image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#   return  image

def plotting_filtered_images(original, x_filter, y_filter, total_filter, name):

    # if grayscale:
    plt.subplot(1,4,1)
    plt.imshow(original,cmap='gray',vmin=0,vmax=255)
    plt.title("Original")
    plt.subplot(1,4,2)
    plt.imshow(x_filter,cmap='gray', vmin=0,vmax=255)
    plt.title("{} X".format(name))
    plt.subplot(1,4,3)
    plt.imshow(y_filter,cmap='gray',vmin=0,vmax=255)
    plt.title("{} Y".format(name))

    plt.subplot(1,4,4)
    # prewitt_total_inverted=cv2.bitwise_not(prewitt_total)
    plt.imshow(total_filter,vmin=0,vmax=255,cmap='gray')
    plt.title("{} Total".format(name))
    plt.show()
    
def round_angle(angle):
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle
