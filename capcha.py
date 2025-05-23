import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import torch
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed', use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

def preprocess_image(image):
    try:
        rgb_image = image.convert("RGB")
        np_image = np.array(rgb_image)
        return np_image
    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None
        
def extract_text_from_image(image):
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        st.error(f"Error in extracting text from image: {e}")
        return None

def main():
    st.title("CAPTCHA Text Extractor")
    uploaded_file = st.file_uploader("Choose a CAPTCHA image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded CAPTCHA Image', use_column_width=True)
            
            st.write("Processing...")
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is not None:
                extracted_text = extract_text_from_image(preprocessed_image)
                if extracted_text is not None:
                    st.write(f"Extracted Text: {extracted_text}")
                else:
                    st.error("Failed to extract text from the image.")
            else:
                st.error("Failed to preprocess the image.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()