import io
import os
from skimage import io
import cv2 as cv 
from PIL import Image 
import streamlit as st
import pandas as pd
import tensorflow as tf

from MesoNet.classifiers import *
from MesoNet.pipeline import *

import numpy as np
# from classifiers import *
# from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_DIR = "./MesoNet"
DF_IMAGE_DIR = "./MesoNet/test_images/df/"
REAL_IMAGE_DIR = "./MesoNet/test_images/real/"
IMAGE_DIR = "./MesoNet/test_images/"
GAMMA_VALUE = 0.5

def upload_df_images():

    uploaded_files = st.file_uploader(label='Pick a DeepFake Image to Input Into the Folder', type=['jpg','jpeg','png'], accept_multiple_files=True)
    
    for file in uploaded_files:
        if file is not None:
            # image_data = file.getvalue()
            # bytes_data = file.read()
            file_details = {"FileName": file.name, "FileType": file.type}
            st.write(file_details)
            with open(os.path.join(DF_IMAGE_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
            st.success(f'Successfully uploaded File: {file.name}')
            # st.image(image_data)
            # return Image.open(io.BytesIO(image_data))

def upload_real_images():

    uploaded_files = st.file_uploader(label='Pick a Real Image to Input Into the Folder', type=['jpg','jpeg','png'], accept_multiple_files=True)
    
    for file in uploaded_files:
        if file is not None:
            # image_data = file.getvalue()
            # bytes_data = file.read()
            # file_details = {"FileName": file.name, "FileType": file.type}
            # st.write(file_details)
            with open(os.path.join(REAL_IMAGE_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
            st.success(f'Successfully uploaded File: {file.name}')
            # st.image(image_data)
            # return Image.open(io.BytesIO(image_data))

def illuminate_pictures():

    for image in os.listdir(DF_IMAGE_DIR):
        adjusted_image = illuminate_picture(os.path.join(DF_IMAGE_DIR, image))
        with open(os.path.join(DF_IMAGE_DIR, image), "wb") as f:
            f.write(adjusted_image)
            st.success(f'DeepFake Image {image} Perturbated with Illumination')

    for image in os.listdir(REAL_IMAGE_DIR):
        adjusted_image = illuminate_picture(os.path.join(REAL_IMAGE_DIR, image))
        with open(os.path.join(REAL_IMAGE_DIR, image), "wb") as f:
            f.write(adjusted_image)
            st.success(f'Real Image {image} Perturbated with Illumination')

def illuminate_picture(image):
    
    img = io.imread(image)
    img_colour = cv.cvtColor(img, cv.COLOR_BGR2RGB) 

    illuminated_img = gammaCorrection(img_colour, gamma=GAMMA_VALUE)

    return illuminated_img

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

def mesonet_predict():

    # os.rmdir(image_dir)
    # os.mkdir(image_dir)
    # image.save(f'{image_dir}/df/image.jpg')

    mesoInc4 = MesoInception4()
    mesoInc4.load(f'{MODEL_DIR}/weights/MesoInception_DF.h5')

    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
        IMAGE_DIR,
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training'
    )
    
    count = 0

    for path in os.listdir(DF_IMAGE_DIR):
        if os.path.isfile(os.path.join(DF_IMAGE_DIR, path)):
            count += 1

    for path in os.listdir(REAL_IMAGE_DIR):
        if os.path.isfile(os.path.join(REAL_IMAGE_DIR, path)):
            count += 1

    for _ in range(count):
        # print(generator)
        X, y = next(generator)
        # print(X, y)
        predicted = mesoInc4.predict(X)
        if round(predicted[0][0]) == 0:
            result = "DeepFake"
            prediction = 1 - predicted[0][0]
        elif round(predicted[0][0]) == 1:
            result = "Real"
            prediction = predicted[0][0]
        st.write(f'Prediction: {result} Image')
        st.write(f'Confidence: {predicted[0][0]}')


def main():
    st.title("Image DeepFake Detector")
    upload_df_images()
    upload_real_images()
    illuminate = st.button("Illuminate Image")  
    if illuminate:
        illuminate_pictures()
    result = st.button("Run on Image")
    if result:
        st.write("Calculating results...")
        mesonet_predict()

if __name__ == '__main__':
    main()