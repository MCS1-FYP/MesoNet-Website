import io
import os
from PIL import Image
import streamlit as st
import pandas as pd

import numpy as np
from MesoNet.classifiers import *
from MesoNet.pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to predict')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))

def mesonet_predict(image):

    image_dir = "MesoNet/test_images/"

    os.rmdir(image_dir)
    os.mkdir(image_dir)
    image.save(f'{image_dir}/image.png')

    classifier = MesoInception4()
    classifier.load('MesoNet/weights/MesoInception_DF.h5')

    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
        'MesoNet/test_images/',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training'
    )

    X, y = generator.next()
    st.write(f'Predicted: {classifier.predict(X)}\nReal Class: {y}')


def main():
    st.title("Image DeepFake Detector")
    image = load_image()
    result = st.button("Run on Image")
    if result:
        st.write("Calculating results...")
        mesonet_predict(image)

if __name__ == '__main__':
    main()