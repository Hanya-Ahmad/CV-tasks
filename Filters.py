import cv2
import numpy as np
from helpers import convolve, read_img, plotting_filtered_images, round_angle
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


import imageio

# img = cv2.imread('images/lena.png', 0)
# row, col = img.shape
# -------------------------------------------------------- POINT 1 --------------------------------------------------------#

# -------------------------------------------------------- Uniform Noise --------------------------------------------------------#


def uni_noise(img, var):
    img = img / 255
    row, col = img.shape
    Uniform = np.zeros((row, col), dtype=np.float64)
    for i in range(row):
        for j in range(col):
            Uniform[i][j] = np.random.uniform(0, var)
    uniform_noise = img + Uniform
    uniform_noise = np.clip(uniform_noise, 0, 1) *255
    plt.imsave("images/output.jpg", uniform_noise, cmap="gray")
    return uniform_noise


# -------------------------------------------------------- Gaussian Noise --------------------------------------------------------#


def gauss_noise(img, var):
    mean = 0
    img = img /255
    n = np.random.normal(mean,var, img.shape)
    image = img + n
    image = image *255
    plt.imsave("images/output.jpg", image, cmap="gray")
    # cv2.imwrite("images/output.jpg",image)
    print("done saving ")
    return image


# ------------------------------------------------------ Salt & Pepper Noise -----------------------------------------------------#


def salt_pepper_noise(img):
    row, col = img.shape
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255

    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    plt.imsave("images/output.jpg", img, cmap="gray")

    return img


# -------------------------------------------------------- POINT 2 --------------------------------------------------------#


# -------------------------------------------------------- Median Filter --------------------------------------------------------#


def median_filter(img):
    row, col = img.shape
    img_new1 = np.zeros([row, col])

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            operator = [
                img[i - 1, j - 1],
                img[i - 1, j],
                img[i - 1, j + 1],
                img[i, j - 1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j - 1],
                img[i + 1, j],
                img[i + 1, j + 1],
            ]
            operator = sorted(operator)
            img_new1[i, j] = operator[4]

    img_new1 = img_new1.astype(np.uint8)
    plt.imsave("images/output.jpg", img_new1, cmap="gray")

    return img_new1


# -------------------------------------------------------- Mean Filter --------------------------------------------------------#


def mean_filter(img):
    row, col = img.shape
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9
    img_new = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            temp = img[i - 1, j - 1] * mask[0, 0]
            +img[i - 1, j] * mask[0, 1]
            +img[i - 1, j + 1] * mask[0, 2]
            +img[i, j - 1] * mask[1, 0]
            +img[i, j] * mask[1, 1]
            +img[i, j + 1] * mask[1, 2]
            +img[i + 1, j - 1] * mask[2, 0]
            +img[i + 1, j] * mask[2, 1]
            +img[i + 1, j + 1] * mask[2, 2]

            img_new[i, j] = temp

    img_new = img_new.astype(np.uint8)
    plt.imsave("images/output.jpg", img_new, cmap="gray")

    return img_new


# -------------------------------------------------------- Gaussian Filter --------------------------------------------------------#


def corr(img, mask):
    row, col = img.shape
    newRow, newCol = mask.shape
    new = np.zeros((row + newRow - 1, col + newCol - 1))
    newCol = newCol // 2
    newRow = newRow // 2
    filtered_img = np.zeros(img.shape)
    new[newRow : new.shape[0] - newRow, newCol : new.shape[1] - newCol] = img
    for i in range(newRow, new.shape[1] - newRow):
        for j in range(newCol, new.shape[1] - newCol):
            temp = new[i - newRow : i + newRow + 1, j - newRow : j + newRow + 1]
            result = temp * mask
            filtered_img[i - newRow, j - newCol] = result.sum()
    return filtered_img


def gaussian(img, newRow, newCol, sigma):
    gaussian_filter = np.zeros((newRow, newCol))
    newRow = newRow // 2
    newCol = newCol // 2
    for i in range(-newRow, newRow + 1):
        for j in range(-newCol, newCol + 1):
            x1 = sigma * (2 * np.pi) ** 2
            x2 = np.exp(-(i * 2 + j * 2) / (2 * sigma * 2))
            gaussian_filter[i + newRow, j + newCol] = (1 / x1) * x2
    plt.imsave("images/output.jpg", corr(img, gaussian_filter), cmap="gray")


# -------------------------------------------------------- POINT 3 --------------------------------------------------------#

# -------------------------------------------------------- Sobel Filter --------------------------------------------------------#


def sobel_filter(img):
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx_flipped = np.flip(Gx)
    Gy_flipped = np.flip(Gy)
    sobel_x = convolve(img, Gx_flipped)
    sobel_y = convolve(img, Gy_flipped)

    sobel_total = np.abs(sobel_x) + np.abs(sobel_y)

    angles = np.arctan2(sobel_y, sobel_x)
    plt.imsave("images/output.jpg", sobel_total, cmap="gray")

    return img, angles, sobel_x, sobel_y, sobel_total, "Sobel"


# ----------------------------------------------------- Roberts Cross Filter -----------------------------------------------------#


def roberts_cross_filter(img):
    Gx = np.array([[1, 0], [0, -1]])

    Gy = np.array([[0, 1], [-1, 0]])

    Gx_flipped = np.flip(Gx)
    Gy_flipped = np.flip(Gy)
    robert_x = convolve(img, Gx_flipped)
    robert_y = convolve(img, Gy_flipped)
    # robert_total = np.sqrt(np.square(robert_x) + np.square(robert_y))
    robert_total = np.abs(robert_x) + np.abs(robert_y)
    plt.imsave("images/output.jpg", robert_total)

    return img, robert_x, robert_y, robert_total, "Roberts"


# -------------------------------------------------------- Prewitt Filter --------------------------------------------------------#


def prewitt_filter(img):
    Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Gx_flipped = np.flip(Gx)
    Gy_flipped = np.flip(Gy)

    prewitt_x = convolve(img, Gx_flipped)
    prewitt_y = convolve(img, Gy_flipped)
    prewitt_total = np.abs(prewitt_x) + np.abs(prewitt_y)
    plt.imsave("images/output.jpg", prewitt_total, cmap="gray")

    return img, prewitt_x, prewitt_y, prewitt_total, "Prewitt"


# ----------------------------------------------------- Canny Edge Detector  -----------------------------------------------------#


def supression(img, D):
    h, w = img.shape
    z = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # find neighbour pixels to visit from the gradient directions
            loc = round_angle(D[i, j])
            try:
                if loc == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        z[i, j] = img[i, j]
                elif loc == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        z[i, j] = img[i, j]
                elif loc == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (
                        img[i, j] >= img[i + 1, j + 1]
                    ):
                        z[i, j] = img[i, j]
                elif loc == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (
                        img[i, j] >= img[i + 1, j - 1]
                    ):
                        z[i, j] = img[i, j]
            except IndexError as e:
                pass
    return z


def threshold(img, t, T):
    cf = {
        "WEAK": 70,
        "STRONG": 255,
    }

    strong_i, strong_j = np.where(img > T)

    weak_i, weak_j = np.where((img >= t) & (img <= T))

    zero_i, zero_j = np.where(img < t)

    img[strong_i, strong_j] = cf.get("STRONG")
    img[weak_i, weak_j] = cf.get("WEAK")
    img[zero_i, zero_j] = 0

    return (img, cf.get("WEAK"))


def tracking(img, weak, strong=255):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] == weak:
                try:
                    if (
                        (img[i + 1, j] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def canny_edge_detector(image, t, T):
    img = gaussian_filter(image,sigma=1.4)

    img, D, sobel_x, sobel_y, sobel_total, name = sobel_filter(img)
    img = supression(sobel_total, D)
    img, weak = threshold(img, t, T)
    img = tracking(img, weak)
    plt.imsave("images/output.jpg", img, cmap="gray")

    return img
