# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pYO4Se_E4oocdajqgiUib0s1Qad_xbos
"""

# Install required libraries
#!pip install transformers
#!pip install Pillow
#!pip install flask

# Import necessary libraries
import os
import io
import base64
from PIL import Image
from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration

# Create a Flask web app
app = Flask(__name__)

# Load pre-trained models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image file
        uploaded_image = request.files["image"]

        if uploaded_image:
            # Load and preprocess the uploaded image
            raw_image = Image.open(io.BytesIO(uploaded_image.read())).convert("RGB")

            # Conditional image captioning
            text = request.form.get("text", "")
            inputs = processor(raw_image, text, return_tensors="pt")
            out = model.generate(**inputs)
            caption_conditional = processor.decode(out[0], skip_special_tokens=True)

            # Render the result page with the caption
            return render_template("result.html", caption=caption_conditional)

    # Render the main page with the image upload form
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
