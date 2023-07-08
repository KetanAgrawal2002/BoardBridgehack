import streamlit as st
import requests
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

@st.cache(allow_output_mutation=True)
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')


def findans(file):
    processor, model = load_model() 
    image = Image.open(file).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def main():
    st.title("Handwritten Form Text Extraction")
    st.write("Upload an image of a handwritten form to extract the text.")
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        generated_text = findans(uploaded_file)
        
        st.write("Extracted Text:")
        st.write(generated_text)

if __name__ == "__main__":
    main()
