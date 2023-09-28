# Import necessary libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Create a Streamlit app title
st.title("Image Captioning with Transformers")

# Load pre-trained models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Add a file uploader widget for image input
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and preprocess the uploaded image
    raw_image = Image.open(uploaded_image).convert('RGB')

    # Add a text input widget for the image caption
    text = st.text_input("Enter a caption for the image", "a culinary dish from Montenegro consisting of")

    # Conditional image captioning
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    caption_conditional = processor.decode(out[0], skip_special_tokens=True)
    st.write("Conditional Caption:", caption_conditional)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
    st.write("Unconditional Caption:", caption_unconditional)
