# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:03:04 2021

@author: Abdiel W. Goni
"""

import streamlit as st
import tensorflow as tf
import numpy as np
#from keras.preprocessing import image
#import cv2
from PIL import Image, ImageOps

model=tf.keras.models.load_model('malaria_cell.h5') # load model from 

st.write("""
          # Malaria Cell Classification
          """
          )

def import_n_pred(image_data, model):
    size = (150,150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred

upload_file = st.sidebar.file_uploader("Upload Cell Images", type="png")

Generate_pred=st.sidebar.button("Predict")

if Generate_pred:
    image=Image.open(upload_file)
    with st.beta_expander('Cell Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    labels = ['Parasitized', 'Uninfected']
    st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))
