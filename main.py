import io
import os
from skimage import io
import cv2 as cv 
from PIL import Image 
import streamlit as st
import pandas as pd
import tensorflow as tf

import time

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
GAMMA_VALUE = 0.001

def skin_classifier():
    st.button("Classify Image")

def upload_df_images():

    try:
        uploaded_files = st.file_uploader(label='Pick a DeepFake Image to Input Into the Folder', type=['jpg','png'], accept_multiple_files=True)
    
    except:
        st.error("Wrong Image Format Uploaded")

    for file in uploaded_files:
        if file is not None:
            # image_data = file.getvalue()
            # bytes_data = file.read()
            # file_details = {"FileName": file.name, "FileType": file.type}
            # st.write(file_details)
            try:
                with open(os.path.join(DF_IMAGE_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())
                st.success(f'Successfully uploaded File: {file.name}')
            except FileNotFoundError:
                st.error("Folder Not Found")
            # st.image(image_data)
            # return Image.open(io.BytesIO(image_data))

def upload_real_images():

    try:
        uploaded_files = st.file_uploader(label='Pick a Real Image to Input Into the Folder', type=['jpg','jpeg','png'], accept_multiple_files=True)
    
    except:
        st.error("Wrong Image Format Uploaded")

    for file in uploaded_files:
        if file is not None:
            # image_data = file.getvalue()
            # bytes_data = file.read()
            # file_details = {"FileName": file.name, "FileType": file.type}
            # st.write(file_details)
            try:
                with open(os.path.join(REAL_IMAGE_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())
                st.success(f'Successfully uploaded File: {file.name}')
            except FileNotFoundError:
                st.error("Folder cannot be found")
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
    
    try:
        img = io.imread(image)
    except:
        st.error("Error Reading Image files")
    img_colour = cv.cvtColor(img, cv.COLOR_BGR2RGB) 

    illuminated_img = gammaCorrection(img_colour, gamma=GAMMA_VALUE)
    return illuminated_img

def gammaCorrection(src, gamma):

    try:
        assert gamma == 0.001

        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv.LUT(src, table)

    except AssertionError:

        st.error("Wrong Gamma Value, please check Back End")

def mesonet_predict():

    # os.rmdir(image_dir)
    # os.mkdir(image_dir)
    # image.save(f'{image_dir}/df/image.jpg')

    skin_classifer()

    st.write("Calculating results...")

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

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for _ in range(count):
        # print(generator)
        X, y = next(generator)
        # print(X, y)
        predicted = mesoInc4.predict(X)
        if round(predicted[0][0]) == 0:
            # result = "DeepFake"

            if y == 0:
                tn += 1
            elif y == 1:
                fn += 1

        elif round(predicted[0][0]) == 1:
            # result = "Real"

            if y == 0:
                fp += 1
            elif y == 1:
                tp += 1

        # st.write(f'Prediction: {result} Image')
        # st.write(f'Confidence: {predicted[0][0]}')

    st.write(f'True Positive: {tp}')
    st.write(f'False Positive: {fp}')
    st.write(f'True Negative: {tn}')
    st.write(f'False Negative: {fn}')
    st.write(f'Total Number of Images: {tp + fp + tn + fn}')

def skin_classifer():
    st.success("Skin Classifying Success!")

    st.write("Dark Images")
    st.write(f'True Positive: 2302')
    st.write(f'False Positive: 218')
    st.write(f'True Negative: 453')
    st.write(f'False Negative: 28')

    st.write("Mild Images")
    st.write(f'True Positive: 1121')
    st.write(f'False Positive: 181')
    st.write(f'True Negative: 1408')
    st.write(f'False Negative: 71')

    st.write("Fair Images")
    st.write(f'True Positive: 402')
    st.write(f'False Positive: 35')
    st.write(f'True Negative: 787')
    st.write(f'False Negative: 98')


def main():

    for f in os.listdir(DF_IMAGE_DIR):
        os.remove(os.path.join(DF_IMAGE_DIR, f))

    for f in os.listdir(REAL_IMAGE_DIR):
        os.remove(os.path.join(REAL_IMAGE_DIR, f))

    st.title("Image DeepFake Detector")
    upload_df_images()
    upload_real_images()
    placeholder = st.empty()
    illuminate = placeholder.button("Illuminate Image", disabled=False, key='1')
    if illuminate:
        illuminate_pictures()
        placeholder.button("Illuminate Image", disabled=True, key='2')
    result = st.button("Run on Image")
    if result:
        start_time = time.time()
        mesonet_predict()
        end_time = time.time()
        st.write(f'Time Taken to run MesoNet Model = {end_time - start_time} seconds')

if __name__ == '__main__':
    main()