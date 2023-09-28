# Import necessary libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Create a Streamlit app title and description
st.title("Image Captioning with Transformers")
st.write("Upload an image, and enter a caption to generate image captions.")

# Load pre-trained models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Add a file uploader widget for image input
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Load and preprocess the uploaded image
        raw_image = Image.open(uploaded_image).convert('RGB')

        # Display the uploaded image
        st.image(raw_image, caption="Uploaded Image", use_column_width=True)

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
    except Exception as e:
        st.error(f"An error occurred: {e}")


