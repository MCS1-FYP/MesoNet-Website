import io
from tkinter import Image
import streamlit as st
import pandas as pd

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to predict')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))

def mesonet_predict():
    pass


def main():
    st.title("Image DeepFake Detector")
    load_image()

if __name__ == '__main__':
    main()