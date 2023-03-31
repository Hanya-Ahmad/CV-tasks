import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Filters
import Histograms
import Frequency


# set page layout to wide
st.set_page_config(layout="wide")
# upload css file
with open("style.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

st.markdown("", unsafe_allow_html=True)


def body():
    # dividing the page to 2 parts for output and input image
    col1, col2 = st.columns(2)
    # side bar that contains a select box for choosing your page option
    with st.sidebar:
        which_page = st.selectbox(
            "Choose Page", ["Filter Page", "Histogram Page", "Hybride Page"]
        )
        # uploading the file browser
        file = st.file_uploader("Upload file", type=["jpg", "png"])
    if file:
        # option to choose between the gray scale or the RGB
        with st.sidebar:
            check_type_of_image = st.checkbox(key="rgb_checkbox", label="To GrayScale")
        with col1:  # first part of the image for displaying the original image
            st.header("Original Images")
            # here we made a specific location for uploading the images and it is the relative folder images
            img_original_color = cv2.imread("images/{}".format(file.name))
            # converting the image to gray scale with out function
            img_original = Histograms.convert_to_grayscale(img_original_color)
            # checking the displaying of the original image (gray scale or RGB)
            if check_type_of_image:
                st.image(img_original, use_column_width=True)
            else:
                st.image(img_original_color, use_column_width=True)
        # the first option for the filter page which contains a lot of different filters
        if which_page == "Filter Page":
            with st.sidebar:
                # making chack box for checking if we want to add noise to the image
                add_noise = st.checkbox(key="noise_checkbox", label="Add Noise")
                if add_noise:
                    # choosing the type of noise and there we have 3 types of noises
                    which_noise_applied = st.selectbox(
                        "Select Noise Type",
                        [
                            " Uniform Noise ",
                            " Salt & Pepper Noise ",
                            " Gaussian Noise ",
                        ],
                    )
                    # if it was the uniform noise
                    if which_noise_applied == " Uniform Noise ":
                        # choosing the amount of the noise
                        var = st.slider("factor of Uniform noise ",min_value=0.1,max_value=1.0,step=0.1,)
                        # adding the noise
                        img_original = Filters.uni_noise(img_original, var)
                    # if it is salt and pepper
                    elif which_noise_applied == " Salt & Pepper Noise ":
                        img_original = Filters.salt_pepper_noise(img_original)
                    # if it was the Gaussian noise
                    elif which_noise_applied == " Gaussian Noise ":
                        # choosing the amount of noise to be applied
                        var = st.slider("factor of gaussian noise ",min_value=0.01,max_value=1.0,step=0.01,)
                        img_original = Filters.gauss_noise(img_original, var)
                    # choosing the type of the filter that will be applied
                which_filter_applied = st.selectbox(
                    "Select Filter",
                    [
                        "No Filter",
                        "Sobel Filter",
                        " Roberts Filter",
                        "Prewitt Filter",
                        "Canny Filter",
                        "Mean Filter",
                        " Gaussian Filter ",
                        "Median Filters",
                        "Global Thresholding",
                        "Local Thresholding",
                    ],
                )
                # if it was sobel filter
                if which_filter_applied == "Sobel Filter":
                    # get parameters for Sobel Function and call function
                    Filters.sobel_filter(img_original)
                    # if it was reobert
                elif which_filter_applied == " Roberts Filter":
                    # get parameters for Roberts Function and call function
                    Filters.roberts_cross_filter(img_original)
                    # if it was prewitt filter
                elif which_filter_applied == "Prewitt Filter":
                    # get parameters for Prewitt Function and call function
                    Filters.prewitt_filter(img_original)
                    # if the filter was canny filter
                elif which_filter_applied == "Canny Filter":
                    # choosing the parameters of the canny filter
                    canny_min = st.slider("Canny Lower Threshold",min_value=10,max_value=100,step=10,value=20,)
                    canny_max = st.slider("Canny Upper Threshold",min_value=20,max_value=100,step=10,value=40,)
                    # get parameters for Canny Function and call function
                    Filters.canny_edge_detector(img_original, canny_min, canny_max)
                    # if it was mean filter
                elif which_filter_applied == "Mean Filter":
                    # get parameters for Average Function and call function
                    Filters.mean_filter(img_original)
                    # if it was gaussian filter
                elif which_filter_applied == " Gaussian Filter ":
                    with st.sidebar:
                        # choosing the parameters of the filter
                        value = st.slider(
                            "Factor of Blur",
                            min_value=1,
                            max_value=101,
                            step=2,
                            value=3,
                        )
                        sigma = st.slider(
                            "Factor of Blur",
                            min_value=0,
                            max_value=100,
                            step=1,
                            value=2,
                        )
                    # get parameters for Gaussian Function and call function
                    Filters.gaussian(img_original, value, value, sigma)
                elif which_filter_applied == "Median Filters":
                    # get parameters for Gaussian Function and call function
                    Filters.median_filter(img_original)
                    # if it was the global threshold
                elif which_filter_applied == "Global Thresholding":
                    threshold_val = st.slider(
                        "Threshold", min_value=1, max_value=254, step=1, value=127
                    )
                    # if it was local threshold
                    Histograms.global_threshold(img_original, threshold_val)
                elif which_filter_applied == "Local Thresholding":
                    Histograms.local_threshold(img_original)
                elif not add_noise:
                    plt.imsave("images/output.jpg", img_original, cmap="gray")
            with col2:
                st.header("Output Images")
                st.image("images\output.jpg", use_column_width=True)

        elif which_page == "Histogram Page":
            with st.sidebar:
                which_filter_histo = st.selectbox(
                    "selct type of filter", ["Normalization", "Equalization"]
                )
                which_plot = st.selectbox(
                    "selct ype of plot", ["distribution plot", "histogram"]
                )
            col3, col4 = st.columns(2)
            if which_filter_histo == "Normalization":
                image_to_plot = Histograms.image_normalization(img_original.copy())
            else:
                image_to_plot = Histograms.histogram_equalization(img_original.copy())
            with col2:
                st.header("Output Images")
                st.image(cv2.imread("images\output.jpg", 0), use_column_width=True)
            if check_type_of_image:
                Histograms.Histogram_Computation_Gray_Scale(img_original, "test")
                Histograms.commulative(img_original.copy())
            else:
                Histograms.Histogram_Computation_RGB(img_original_color.copy(), "test")
                Histograms.comulative_RGB(img_original_color.copy())
            with col3:
                if which_plot == "distribution plot":
                    input_histogram = cv2.imread("images/comulative.jpg")
                else:
                    input_histogram = cv2.imread("images/test.jpg")
                st.image(input_histogram, use_column_width=True)
            with col4:
                Histograms.Histogram_Computation_Gray_Scale(
                    image_to_plot.copy(), "out_plotting"
                )
                Histograms.commulative(image_to_plot.copy())
                if which_plot == "distribution plot":
                    output_histogram = cv2.imread("images/comulative.jpg")
                else:
                    output_histogram = cv2.imread("images/out_plotting.jpg")
                st.image(output_histogram, use_column_width=True)

        else:
            file2 = st.sidebar.file_uploader("Upload file", type=["jpg", "png"], key=2)
            which_Frequency_applied = st.sidebar.selectbox(
                "Select Frequency Type Of first image", ["High Pass", "Low Pass"]
            )
            if file2:
                with col2:
                    st.header("Second Image")
                    img_original_color2 = cv2.imread("images/{}".format(file2.name))
                    img_original2 = Histograms.convert_to_grayscale(img_original_color2)
                    if check_type_of_image:
                        img_original2 = Histograms.convert_to_grayscale(
                            img_original_color2
                        )
                        st.image(img_original2, use_column_width=True)
                    else:
                        st.image(img_original_color2, use_column_width=True)
                if which_Frequency_applied == "High Pass":
                    size = st.sidebar.slider(
                        "Size",
                        min_value=10,
                        max_value=512,
                        step=10,
                        value=20,
                    )
                    Frequency.which_type(
                        img_original, img_original2, size, "High Pass", col1, col2
                    )
                    with col1:
                        Frequency.hybrid_images(img_original2, img_original)
                        st.image("images/output_hybride.jpg", use_column_width=True)
                    with col2:
                        Frequency.hybrid_images(img_original, img_original2)
                        st.image("images/output_hybride.jpg", use_column_width=True)
                elif which_Frequency_applied == "Low Pass":
                    size = st.sidebar.slider(
                        "Size",
                        min_value=10,
                        max_value=512,
                        step=10,
                        value=20,
                    )
                    Frequency.which_type(
                        img_original, img_original2, size, "Low Pass", col1, col2
                    )
                    with col1:
                        Frequency.hybrid_images(img_original, img_original2)
                        st.image("images/output_hybride.jpg", use_column_width=True)
                    with col2:
                        Frequency.hybrid_images(img_original2, img_original)
                        st.image("images/output_hybride.jpg", use_column_width=True)


if __name__ == "__main__":
    body()
