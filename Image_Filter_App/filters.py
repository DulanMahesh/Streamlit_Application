import cv2
import numpy as np
import streamlit as st


@st.cache_data # cache used by streamlit app
def bw_filter(img):   # Black and white filter.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


@st.cache_data
def vignette(img, level=2):     # Vignette Effect.
    height, width = img.shape[:2]

    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)

    # Generating resultant_kernel matrix.
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    img_vignette = np.copy(img)

    # Applying the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask
    return img_vignette

@st.cache_data
def sepia(img):  # Sepia / Vintage Filter.
    img_sepia = img.copy()
    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
    img_sepia = np.array(img_sepia, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to the range [0, 255].
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
    return img_sepia


@st.cache_data
def embossed_edges(img):   # Embossed Edges.
    kernel = np.array([[0, -3, -3],
                       [3, 0, -3],
                       [3, 3, 0]])

    img_emboss = cv2.filter2D(img, -1, kernel=kernel)
    return img_emboss


@st.cache_data
def pencil_sketch(img,ksize=5):     # Pencil Sketch Filter.
    img_sketch_bw = img.copy()
    img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0,0)
    img_sketch_bw, _ = cv2.pencilSketch(img_blur, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return img_sketch_bw

@st.cache_data
def stylization_filter(img):   # Stylization Filter.
    img_style =img.copy()
    img_blur = cv2.GaussianBlur(img, (5, 5), 0, 0)
    img_style = cv2.stylization(img_blur, sigma_s=40, sigma_r=0.1)
    return img_style





