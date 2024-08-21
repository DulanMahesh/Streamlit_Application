

import streamlit as st  # Streamlit for creating web apps
import pathlib  # Path handling
from streamlit_drawable_canvas import st_canvas  # Streamlit component for drawing
import cv2
import numpy as np  # NumPy for numerical operations
import io  # Input/output operations for file handling
import base64  # Encoding/decoding for file download links
from PIL import Image  # PIL for image processing

# Function to create a download link for the output image
def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    # Create an in-memory buffer to store the image
    buffered = io.BytesIO()
    # Save the image in JPEG format to the buffer
    img.save(buffered, format='JPEG')
    # Encode the image in base64 format for embedding in an HTML link
    img_str = base64.b64encode(buffered.getvalue()).decode()
    # Generate an HTML link for downloading the image
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Set the title of the app in the sidebar
st.sidebar.title('Image Inpaint')

# File uploader in the sidebar for users to upload an image
uploaded_file = st.sidebar.file_uploader("Upload Image to restore:", type=["png", "jpg"])
image = None  # Placeholder for the uploaded image
res = None  # Placeholder for the result image

if uploaded_file is not None:
    # Convert the uploaded file into a NumPy array and decode it as an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Slider in the sidebar to adjust the stroke width for drawing
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 5)
    h, w = image.shape[:2]  # Get the height and width of the image
    if w > 800:
        # Resize the image if it's wider than 800 pixels
        h_, w_ = int(h * 800 / w), 800
    else:
        h_, w_ = h, w

    # Create a drawable canvas in the main app interface
    canvas_result = st_canvas(
        fill_color='white',  # Background color for filling
        stroke_width=stroke_width,  # Width of the stroke/line drawn
        stroke_color='black',  # Color of the stroke
        background_image=Image.open(uploaded_file).resize((h_, w_)),  # Display the uploaded image as the background
        update_streamlit=True,  # Enable updates to the Streamlit app
        height=h_,  # Height of the canvas
        width=w_,  # Width of the canvas
        drawing_mode='freedraw',  # Set the drawing mode to freehand drawing
        key="canvas",  # Unique key for the canvas element
    )
    stroke = canvas_result.image_data  # Get the image data from the canvas

    if stroke is not None:
        # Checkbox in the sidebar to show the mask created by the user
        if st.sidebar.checkbox('show mask'):
            st.image(stroke)  # Display the mask

        # Extract the alpha channel (transparency) from the mask to use as an inpainting mask
        mask = cv2.split(stroke)[3]
        mask = np.uint8(mask)  # Convert mask to unsigned 8-bit integer format
        mask = cv2.resize(mask, (w, h))  # Resize the mask to match the original image size

    # Sidebar caption asking if the user is satisfied with the selection
    st.sidebar.caption('Happy with the selection?')
    # Dropdown to select the inpainting mode (None, Telea, NS, Compare both)
    option = st.sidebar.selectbox('Mode', ['None', 'Telea', 'NS', 'Compare both'])

    if option == 'Telea':
        # If Telea's algorithm is selected, apply it and display the result
        st.subheader('Result of Telea')
        res = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
        st.image(res)  # Display the inpainted image

    elif option == 'Compare both':
        # If the comparison mode is selected, apply both algorithms and display the results side by side
        col1, col2 = st.columns(2)  # Create two columns for side-by-side display
        res1 = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
        res2 = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
        with col1:
            st.subheader('Result of Telea')
            st.image(res1)  # Display the result of Telea's algorithm
        with col2:
            st.subheader('Result of NS')
            st.image(res2)  # Display the result of the Navier-Stokes algorithm
        # Provide download links for the results if they exist
        if res1 is not None:
            result1 = Image.fromarray(res1)  # Convert the result to a PIL image
            st.sidebar.markdown(
                get_image_download_link(result1, 'telea.png', 'Download Output of Telea'),
                unsafe_allow_html=True)
        if res2 is not None:
            result2 = Image.fromarray(res2)  # Convert the result to a PIL image
            st.sidebar.markdown(
                get_image_download_link(result2, 'ns.png', 'Download Output of NS'),
                unsafe_allow_html=True)

    elif option == 'NS':
        # If Navier-Stokes algorithm is selected, apply it and display the result
        st.subheader('Result of NS')
        res = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
        st.image(res)  # Display the inpainted image
    else:
        pass  # Do nothing if 'None' is selected

    if res is not None:
        # If an inpainting result exists, provide a download link
        result = Image.fromarray(res)  # Convert the result to a PIL image
        st.sidebar.markdown(
            get_image_download_link(result, 'output.png', 'Download Output'),
            unsafe_allow_html=True)
