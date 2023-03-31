import cv2
from helpers import convolve, read_img, plotting_filtered_images, round_angle, gs_filter
from Filters import gaussian_filter
import numpy as np
from scipy.ndimage import gaussian_filter as gs
import matplotlib.pyplot as plt


# --------------------------------------------------------- POINT 4 ---------------------------------------------------------#

# -------------------------------------------------------- Histogram --------------------------------------------------------#

def Histogram_Computation_RGB(img , format_image):
    image_Height = img.shape[0]
    image_Width = img.shape[1]
    image_Channels = img.shape[2]

    Histogram = np.zeros([256, image_Channels], np.int32)

    for x in range(0, image_Height):
        for y in range(0, image_Width):
            for c in range(0, image_Channels):
                Histogram[img[x, y, c], c] += 1

    plt.clf()
    plt.plot(Histogram)
    plt.savefig(f"images/{format_image}.jpg")
    return Histogram


def Histogram_Computation_Gray_Scale(img , format_image):
    height = img.shape[0]
    width = img.shape[1]
    # making an array of 256 which represent the max number of pixels
    # and every time we need to increase this pixel value we just go to the same index of it in the array
    # and just increase it by one and that for making it more fast and wasy to perform
    histogram = np.zeros(256)
    # a loop to get every element in the array and counting the number of pixels with different intensities
    # for h in range(height):
    for h in range(height):
        for w in range(width):
            index = img[h][w]
            histogram[index]=histogram[index]+1
    plt.clf()
    plt.plot(histogram)
    plt.savefig(f"images/{format_image}.jpg")
    return histogram

def commulative(img):
    # Calculate histogram and normalize
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    hist_norm = hist / float(img.size)

    # Calculate cumulative sum of normalized histogram
    cdf = np.cumsum(hist_norm)
    plt.clf()
    plt.plot(cdf)
    plt.savefig("images/comulative.jpg")

def comulative_RGB(img):
    cdf_vals = []
    for i in range(3):
        hist, bins = np.histogram(img[:,:,i].flatten(), 256, [0, 256])
        hist_norm = hist / float(img.size)
    # Calculate cumulative sum of normalized histogram
        cdf = np.cumsum(hist_norm)
        cdf_vals.append(cdf)

    plt.clf()
    plt.plot(cdf_vals[0], color='r')
    plt.plot(cdf_vals[1], color='g')
    plt.plot(cdf_vals[2], color='b')
    plt.savefig("images/comulative.jpg")
# ------------------------------------------------------ Distribution Curve ------------------------------------------------------#


# ------------------------------------------------------------ POINT 5 -----------------------------------------------------------#


# --------------------------------------------------------- Image Equalization --------------------------------------------------#
def histogram_equalization(image):
    # getting some initial values from the image and ease calculations
    maxVal = np.max(image)
    height = image.shape[0]
    width = image.shape[1]
    size = height * width
    # making an array of 256 which represent the max number of pixels
    # and every time we need to increase this pixel value we just go to the same index of it in the array
    # and just increase it by one and that for making it more fast and wasy to perform
    histogramArray = np.zeros(256)
    # a loop to get every element in the array and counting the number of pixels with different intensities
    for h in range(height):
        for w in range(width):
            index = image[h][w]
            histogramArray[index] = histogramArray[index] + 1
    # initial variable to make cdf function in same time as we looping in the image
    #  without need to loop on it many differen times
    add = 0  # here we add the past value of the array
    i = 0  # counter
    # making cdf and set the new values of the intensities in the array
    for element in histogramArray:
        histogramArray[i] = (element / size) * 255 + add
        add = histogramArray[i]
        i = i + 1
    # a loop for changing the last values of the image array and giving them the new intensities
    for h in range(height):
        for w in range(width):
            intensity = image[h][w]
            image[h][w] = histogramArray[intensity]
    plt.imsave('images/output.jpg', image)
    return image


# ------------------------------------------------------------ POINT 6 -----------------------------------------------------------#

# --------------------------------------------------------- Image Normalization --------------------------------------------------#

"""
---------------------------------------note-------------------------------
you need here to add this lines in the main code and
that to display the window of the image if you want  :
---------------------------------
cv2.waitkey(0)
cv2.destrotAllWindows()
---------------------------------
"""


def image_normalization(image):
    # making a  window to see the image before any changes
    # getting the max and minimum value in the array
    maxVal = np.max(image)
    minVal = np.min(image)
    # equation to make linear normalization as we are scaling it to the new points which are 255 and making
    # the smallest value in the array to be represented as zero and max value represented as one
    image = (image - minVal) / (maxVal - minVal) * 255
    # saving the image to save the changes and make it appropriate for opencv to read it again
    cv2.imwrite("normalized_image.png", image)
    # reading the image that we saved with the changes
    image = cv2.imread("normalized_image.png")
    # displaying this image to show the changes again
    plt.imsave('images/output.jpg', image)
    return image

# img = cv2.imread("images/lena.png",cv2.IMREAD_GRAYSCALE)
# image1= image_normalization(img)
# plt.imshow(image1,cmap='gray')
# plt.show()
# ------------------------------------------------------------ POINT 7 -----------------------------------------------------------#

# ------------------------------------------------------- Local Thresholding -----------------------------------------------------#


def local_threshold(img, block_size=12):
    h, w = img.shape
    img1 = np.copy(img)
    for i in range(h):
        for j in range(w):
            mean = np.mean(img[i: i + block_size, j: j + block_size])
            if img[i][j] >= mean:
                img1[i][j] = 255
            else:
                img1[i][j] = 0
    plt.imsave('images/output.jpg', img1,cmap='gray')

    return img1


# ------------------------------------------------------- Global Thresholding ----------------------------------------------------#


def global_threshold(img, threshold):
    h, w = img.shape
    img1 = np.copy(img)
    for i in range(h):
        for j in range(w):
            if img[i][j] >= threshold:
                img1[i][j] = 255
            else:
                img1[i][j] = 0
    plt.imsave('images/output.jpg', img1, cmap='gray')

    return img1


# plotting_filtered_images(noise_free_img, noise_free_img, global_threshold_img, thresh1,"")
# plotting_filtered_images(noise_free_img, noise_free_img, global_threshold_img, local_threshold_img,"")

# ------------------------------------------------------------ POINT 8 -----------------------------------------------------------#
def convert_to_grayscale(img):
    # Convert image to grayscale using luminosity method
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    # Convert back to uint8 format for display
    gray = gray.astype(np.uint8)
    return gray