import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-base-handwritten"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def main():
    st.title("Handwritten Form Text Extraction")
    st.write("Upload an image of a handwritten form to extract the text.")
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        output = query(uploaded_file.name)
        
        st.write("Extracted Text:")
        st.write(output)

if __name__ == "__main__":
    main()
