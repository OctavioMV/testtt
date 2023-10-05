# Import necessary libraries
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Create a Streamlit app title and description
st.title("Image Captioning with Transformers")
st.write("Upload an image or use the default image to generate image captions.")

# Load pre-trained models
st.write("Loading pre-trained models...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
st.write("Models loaded successfully.")

# Add a file uploader widget for image input
st.write("Waiting for image upload...")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if the user wants to use the default image
use_default_image = st.checkbox("Use Default Image (burek1.jpg)")

# Default image URL
default_image_url = "https://raw.githubusercontent.com/OctavioMV/testtt/main/burek1.jpg"

if use_default_image:
    # Load and preprocess the default image
    st.write("Using default image...")
    response = requests.get(default_image_url)
    raw_image = Image.open(BytesIO(response.content)).convert("RGB")

    # Display the default image
    st.image(raw_image, caption="Default Image (burek1.jpg)", use_column_width=True)
else:
    # Display the uploaded image if provided
    if uploaded_image is not None:
        try:
            # Load and preprocess the uploaded image
            st.write("Processing uploaded image...")
            raw_image = Image.open(uploaded_image).convert("RGB")

            # Display the uploaded image
            st.image(raw_image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded image: {e}")
    else:
        st.warning("Please upload an image or use the default image.")

# Add a text input widget for the image caption
default_caption = "Enter a caption for the image"
text = st.text_input("Image Caption", default_caption)

# Conditional image captioning
if text != default_caption:
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs)
    caption_conditional = processor.decode(out[0], skip_special_tokens=True)
    st.write("Conditional Caption:", caption_conditional)

# Unconditional image captioning
st.subheader("Unconditional Caption:")
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
st.write(caption_unconditional)
