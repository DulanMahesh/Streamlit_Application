import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import os

# Create application title and file uploader widget.
st.title("OpenCV Deep Learning based Image Classification")

@st.cache_resource
def load_model():
    """
    Loads the DNN model and class names for image classification.

    Returns:
        model: The loaded neural network model.
        class_names: List of class names corresponding to the model's outputs.
    """
    # Define the paths to the model and class names files.
    prototxt_path = "C:/Users/Administrator/PycharmProjects/streamlit_application/Deep_learning_network_streamlit_App/DenseNet_121.prototxt"
    caffemodel_path = "C:/Users/Administrator/PycharmProjects/streamlit_application/Deep_learning_network_streamlit_App/DenseNet_121.caffemodel"
    file_path = "C:/Users/Administrator/PycharmProjects/streamlit_application/Deep_learning_network_streamlit_App/classification_classes_ILSVRC2012.txt"

    # Check if the model files exist and report errors if they don't.
    if not os.path.exists(prototxt_path):
        st.error(f"Prototxt file '{prototxt_path}' not found. Please make sure it is in the correct directory.")
        st.stop()

    if not os.path.exists(caffemodel_path):
        st.error(f"Caffemodel file '{caffemodel_path}' not found. Please make sure it is in the correct directory.")
        st.stop()

    if not os.path.exists(file_path):
        st.error(f"Class names file '{file_path}' not found. Please make sure it is in the correct directory.")
        st.stop()

    # Read the ImageNet class names from the file.
    with open(file_path, "r") as f:
        image_net_names = f.read().split("\n")

    # Extract the class names, picking just the first name if there are multiple names in a line.
    class_names = [name.split(",")[0] for name in image_net_names]

    # Load the neural network model using OpenCV's DNN module.
    model = cv2.dnn.readNet(model=caffemodel_path, config=prototxt_path, framework="Caffe")
    return model, class_names

def classify(model, image, class_names):
    """
    Performs inference on the input image and returns the class with the highest confidence.

    Args:
        model: The loaded neural network model.
        image: The input image as a NumPy array.
        class_names: List of class names corresponding to the model's outputs.

    Returns:
        out_text: A string with the predicted class and its confidence.
    """
    # Remove alpha channel if present in the image.
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Create a blob from the image using the parameters expected by the model.
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.017, size=(224, 224), mean=(104, 117, 123))

    # Set the input blob for the neural network and perform forward pass.
    model.setInput(blob)
    outputs = model.forward()

    # Flatten the outputs to 1D and get the index of the highest confidence score.
    final_outputs = outputs[0].reshape(1000, 1)
    label_id = np.argmax(final_outputs)

    # Convert the output scores to softmax probabilities.
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

    # Get the highest probability and map it to the class label.
    final_prob = np.max(probs) * 100.0
    out_name = class_names[label_id]
    out_text = f"Class: {out_name}, Confidence: {final_prob:.1f}%"
    return out_text

def header(text):
    """
    Displays a header with a custom style.

    Args:
        text: The text to display in the header.
    """
    st.markdown(
        '<p style="background-color:#0066cc;color:#33ff33;font-size:24px;'
        f'border-radius:2%;" align="center">{text}</p>',
        unsafe_allow_html=True,
    )

# Load the model and class names when the script starts.
net, class_names = load_model()

# Create file uploader and URL input widgets for the user to upload an image or enter an image URL.
img_file_buffer = st.file_uploader("Choose a file or Camera", type=["jpg", "jpeg", "png"])
st.text("OR")
url = st.text_input("Enter URL")

if img_file_buffer is not None:
    # If a file is uploaded, read it and convert it to an OpenCV image format.
    image = np.array(Image.open(img_file_buffer))
    st.image(image)

    # Perform classification on the uploaded image and display the result.
    detections = classify(net, image, class_names)
    header(detections)

elif url != "":
    try:
        # If a URL is provided, fetch the image from the URL and convert it to an OpenCV image format.
        response = requests.get(url)
        image = np.array(Image.open(BytesIO(response.content)))
        st.image(image)

        # Perform classification on the fetched image and display the result.
        detections = classify(net, image, class_names)
        header(detections)
    except requests.exceptions.MissingSchema as err:
        # Handle invalid URL errors.
        st.header("Invalid URL, Try Again!")
        st.error(f"Invalid URL: {err}")
    except UnidentifiedImageError as err:
        # Handle cases where the URL does not point to an image.
        st.header("URL has no Image, Try Again!")
        st.error(f"Image not found in URL: {err}")
    except Exception as err:
        # Handle any other unexpected errors.
        st.header("An error occurred")
        st.error(f"Error: {err}")
