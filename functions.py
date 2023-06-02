import tensorflow as tf
from keras import layers, models
from keras.utils import load_img, img_to_array, array_to_img, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
import base64
import keras


import streamlit as st
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def getLabelAndImagesArray():
    # Buat simpan informasi label yang sudah dikonversi ke Angka
    st.session_state.label_data = {}
    labels_data = []
    images_data = []
    all_sessions = list(st.session_state)
    class_name_sessions = []
    images_sessions = []
    for session in all_sessions:
        if "data_class_name_" in session:
            class_name_sessions.append(session)
        elif "data_image_samples_" in session:
            images_sessions.append(session)

    class_name_sessions_sorted = sorted(class_name_sessions)
    images_sessions_sorted = sorted(images_sessions)

    label_number = 0
    for class_name_session, images_session in zip(
        class_name_sessions_sorted, images_sessions_sorted
    ):
        class_name = st.session_state[class_name_session]
        st.session_state.label_data[label_number] = class_name
        for image_session in st.session_state[images_session]:
            # Add Label
            labels_data.append(label_number)
            # Add Image
            img = load_img(image_session, target_size=(256, 256))
            img_arr = img_to_array(img)
            images_data.append(img_arr)
        label_number += 1
    # Di acak agar proses training maksimal
    images_data, labels_data = shuffle(images_data, labels_data)
    return np.array(images_data), np.array(labels_data)


st.cache(suppress_st_warning=True)


def trainingModel():
    # Hyperparams
    epochs = st.session_state.hyperparameter["epochs"]
    batch_size = st.session_state.hyperparameter["batch_size"]

    # Data Input
    images_data, labels_data = getLabelAndImagesArray()
    num_label = len(np.unique(labels_data))

    # One-Hot Encoding Labels Data
    labels_data = to_categorical(labels_data, num_label)

    # Model CNN
    model = keras.Sequential()

    model.add(
        layers.Conv2D(
            32, (3, 3), activation="relu", padding="SAME", input_shape=(256, 256, 3)
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="SAME"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="SAME"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="SAME"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="SAME"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="SAME"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="SAME"))
    # model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2, padding='SAME'))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='SAME'))
    # model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2, padding='SAME'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))

    # Compile model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.compile(optimizer='adam', loss='cross_', metrics=['accuracy'])

    model.fit(images_data, labels_data, epochs=epochs, batch_size=batch_size)

    # st.session_state.cm_results = cm_results
    st.session_state.isModelTrained = True
    model.save("simple_teachable_machine_model_trained.h5")
    return model


def getPredictedImage(model):
    img = load_img(st.session_state.data_image_predict, target_size=(256, 256))
    img_arr = img_to_array(img)[np.newaxis, ...]
    result = model.predict(img_arr)
    return result


def predictModel(model):
    # PROSES PREDICTION
    st.markdown(
        "<h1 style='text-align:center;'> -----Try to Predict Image----- </h1>",
        unsafe_allow_html=True,
    )
    image_predict = st.file_uploader(
        "Add image to predict",
        accept_multiple_files=False,
        key="data_image_predict",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    btn_predict_image = st.button("Predict", type="primary")
    if btn_predict_image:
        if image_predict:
            st.markdown("<h4>Image Predict</h4>", unsafe_allow_html=True)
            st.image(image_predict)
            st.markdown("<h4>Results</h4>", unsafe_allow_html=True)
            result = getPredictedImage(model)
            for probability, label in zip(
                result[0], list(st.session_state.label_data.values())
            ):
                st.write(
                    "Kelas: %s - Probabilitas : %.3f " % (label, probability * 100)
                )
        else:
            st.warning("Fill the image first", icon="⚠️")
