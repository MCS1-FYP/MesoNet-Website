import io
import os
from PIL import Image
import cv2 as cv 
from skimage import io as oi
from PIL import Image 
import streamlit as st
import pandas as pd
import tensorflow as tf

import numpy as np
from classifiers import *
from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to predict')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))

def mesonet_predict(image):

    image_dir = "test_images"

    # os.rmdir(image_dir)
    # os.mkdir(image_dir)
    image.save(f'{image_dir}/df/image.jpg')

    mesoInc4 = MesoInception4()
    mesoInc4.load('weights/MesoInception_DF.h5')

    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
        "test_images/",
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training'
    )

    # print(generator)

    X, y = next(generator)
    # print(X, y)
    predicted = mesoInc4.predict(X)
    if round(predicted[0][0]) == 0:
        result = "DeepFake"
    elif round(predicted[0][0]) == 1:
        result = "Real"
    st.write(f'Prediction: {result} Image')
    st.write(f'Confidence: {predicted[0][0]}')

# def illuminate(image):

#     image_2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     gamma = 0.5

#     invGamma = 1 / gamma

#     table = [((i / 255) ** invGamma) * 255 for i in range(256)]
#     table = np.array(table, np.uint8)

#     return cv.LUT(image_2, table)


def main():
    st.title("Image DeepFake Detector")
    image = load_image()
    illuminate = st.button("Illuminate Image")  
    # if illuminate:
    #     pass
    result = st.button("Run on Image")
    if result:
        st.write("Calculating results...")
        mesonet_predict(image)

if __name__ == '__main__':
    main()