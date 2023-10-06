# Import necessary libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Create a Streamlit app title and description
st.title("Image Captioning with Transformers")
st.write("Upload an image to generate image captions.")

# Load pre-trained models
st.write("Loading pre-trained models...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
st.write("Models loaded successfully.")

# Add a file uploader widget for image input
st.write("Waiting for image upload...")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_image is not None:
    try:
        # Load and preprocess the uploaded image
        st.write("Processing uploaded image...")
        raw_image = Image.open(uploaded_image).convert("RGB")

        # Display the uploaded image
        st.image(raw_image, caption="Uploaded Image", use_column_width=True)

        # Conditional image captioning
        conditional_prompt = st.text_input("Enter a conditional prompt (e.g., 'an Eastern European dish from Montenegro consisting of'):")
        if conditional_prompt:
            # Display the conditional text
            st.subheader("Conditional Text:")
            st.write(conditional_prompt)

            inputs = processor(raw_image, conditional_prompt, return_tensors="pt")
            out = model.generate(**inputs)
            caption_conditional = processor.decode(out[0], skip_special_tokens=True)
            st.subheader("Conditional Caption:")
            st.write(caption_conditional)

        # Unconditional image captioning
        st.subheader("Unconditional Caption:")
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
        st.write(caption_unconditional)

    except Exception as e:
        st.error(f"An error occurred while processing the uploaded image: {e}")
else:
    st.warning("Please upload an image.")
