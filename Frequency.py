import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp
import cv2
import streamlit as st



def toFFT(img):
        F1 = fp.fft2((img).astype(float))
        F2 = fp.fftshift(F1)
        fft = (20 * np.log10(0.1 + F2)).astype(int)
        return F1, F2, fft



def lowPassFiltering(img, F2, size):
        img1= np.copy(img)
        (w, h) = img.shape
        half_w, half_h = int(w / 2), int(h / 2)
        Fblank = np.zeros((w, h))
        # select the first 30x30 frequencies
        Fblank[half_w - size:half_w + size + 1, half_h - size:half_h + size + 1] = 1
        F2 = Fblank * F2
        fft = (20 * np.log10(0.1 + F2)).astype(int)

        img2 = np.abs(fp.ifft2(fp.ifftshift(F2)))
        plt.imsave("images/output3.jpg", img2, cmap='gray')

        return img1, fft, F2


def highPassFiltering(img, F2, size):
        img1= np.copy(img)
        (w, h) = img.shape
        half_w, half_h = int(w / 2), int(h / 2)
        F2[half_w - size:half_w + size + 1,
           half_h - size:half_h + size + 1] = 0  # select all but the first 50x50 (low) frequencies
        fft = (20 * np.log10(0.1 + F2)).astype(int)
        img1 = np.abs(fp.ifft2(fp.ifftshift(F2)))
        plt.imsave("images/output2.jpg", img1, cmap='gray')
        return img1, fft, F2


def which_type(img1 , img2, size ,type ,col1 , col2):
        f1, f2 , fft = toFFT (img1)
        f3,f4 , fft =toFFT (img2)
        if type == "High Pass":
            highPassFiltering (img1, f2 , size)
            lowPassFiltering (img2 , f4 , size)
            with col1:
                st.image("images/output2.jpg" ,use_column_width=True)

            with col2 :
                st.image("images/output3.jpg" ,use_column_width=True)

        elif  type == "Low Pass":
            lowPassFiltering (img1, f2 , size)
            highPassFiltering(img2, f4 , size)
            with col1:
                st.image("images/output3.jpg" ,use_column_width=True)

            with col2 :
                st.image("images/output2.jpg" ,use_column_width=True)



# ------------------------------------------------------ point 10 ------------------------------------------#
def get_size(image):
    width = image.shape[1]
    height = image.shape[0]
    return width , height


def get_demension_ORI(image_1 , image_2):
        width_1 , height_1 = get_size(image_1)
        width_2 , height_2 = get_size(image_2)

        if (width_1 > width_2):
            width = width_2
        else:
            width = width_1

        if (height_1 > height_2):
            height = height_2
        else:
            height = height_1

        return width , height


def resizedImage(image_1, image_2):
    width , hieght =  get_demension_ORI(image_1,image_2 )
    resizedImg_1 =cv2.resize(image_1, (width , hieght))
    resizedImg_2 =cv2.resize(image_2 , (width , hieght))
    return resizedImg_1 , resizedImg_2


def gaussian_filter(img, row,col,sigma):
    gaussian_filter=np.zeros((row,col))
    row=row//2
    col=col//2
    for i in range(-row,row+1):
        for j in range (-col,col+1) :
            x1=sigma*(2*np.pi)**2
            x2=np.exp(-(i**2+j**2)/(2*sigma**2))
            gaussian_filter[i+row,j+col]=(1/x1)*x2
    print(gaussian_filter)
    return gaussian_filter

def add_gauss_filter(img):
    gaussian_filter_image = cv2.filter2D(img,-1,gaussian_filter(img, 3,3,0.2))
    return gaussian_filter_image

def hybrid_images(img1, img2):
    img1, img2 = resizedImage(img1, img2)

    filtered_img2= add_gauss_filter(img2)

    filtered_img1 = (img1 / 255) - add_gauss_filter(img1)
    plt.imshow(filtered_img2, cmap='gray')

    cv2.imwrite("images/low.jpg", filtered_img2 * 255)
    # Perform high pass filter and export.
    cv2.imwrite("images/high.jpg", (filtered_img1 + 0.5) * 255)

    hybdrid_image= filtered_img1 + filtered_img2
    plt.imsave("images/output_hybride.jpg", hybdrid_image, cmap='gray')

    return hybdrid_image