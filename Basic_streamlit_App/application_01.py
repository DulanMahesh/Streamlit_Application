
import streamlit as st


st.title("Streamlit Example")
st.header("This is header")

# File uploader
image_uploaded = st.file_uploader("Upload your files")

# Check if an image has been uploaded before displaying it
if image_uploaded is not None:
    st.image(image_uploaded)

# Implementation of selection box
selected_value = st.selectbox("Selection box for the selection", ["option 1", "option 2", "option 3"])

# Print the selected value
st.write(f"The selected value is: {selected_value}")

# Checkbox
checkbox_value = st.checkbox("option1")
st.write(checkbox_value)

#test#