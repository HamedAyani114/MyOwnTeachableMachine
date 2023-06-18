import keras
from keras import layers
from keras.utils import load_img, img_to_array, to_categorical
from keras.callbacks import EarlyStopping

import streamlit as st
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# split data
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


le = LabelEncoder()


def get_ImagesClassForm():
    session_input = list(st.session_state)

    key_class_input = []
    key_images_input = []
    for input in session_input:
        if "class_input" in input:
            key_class_input.append(input)
        elif "image_input" in input:
            key_images_input.append(input)
    # st.write(key_class_input)
    # st.write(key_images_input)

    st.session_state.input_kelas = []
    data_images = []
    class_images = []

    for data_img, cls_img in zip(key_images_input, key_class_input):
        kelas = st.session_state[cls_img]
        image = st.session_state[data_img]
        st.session_state.input_kelas.append(kelas)
        st.write(st.session_state[cls_img])

        for img_input in image:
            # Add class
            class_images.append(kelas)
            # Add Image
            img = load_img(img_input, target_size=(256, 256))
            data_images.append(img_to_array(img))
    st.session_state.str_kelas = "-".join(st.session_state.input_kelas)
    X, y = shuffle(data_images, class_images)

    # normalize data
    X = np.array(X) / 255.0
    st.write(X.shape)
    return np.array(X), np.array(y)


def trainingModel(epoch, batch_size):
    epochs = epoch
    batch_size = batch_size
    # st.write(epochs, batch_size)

    # data input
    X, y = get_ImagesClassForm()
    y_num = le.fit_transform(y)

    # one hot encoding
    y_cat = to_categorical(y_num)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )

    # Model CNN
    model = keras.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="SAME",
            input_shape=X[0].shape,
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="SAME"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="SAME"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="SAME"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="SAME"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="SAME"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="SAME"))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(len(y_train[0]), activation="softmax"))

    # Compile model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    early_stop = EarlyStopping(monitor="loss", patience=1, min_delta=0.1)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    st.session_state.y_pred_class = le.inverse_transform(y_pred)
    st.session_state.y_test = le.inverse_transform(np.argmax(y_test, axis=1))
    # st.write(confusion_matrix(st.session_state.y_test, st.session_state.y_pred_class))

    model.save("models/teachable_machine_model_%s.h5" % (st.session_state.str_kelas))
    st.session_state.path_model = "models/teachable_machine_model_%s.h5" % (
        st.session_state.str_kelas
    )

    st.session_state.isModelTrained = True
    # return path_model


def sidebar():
    CM_fig = ConfusionMatrixDisplay.from_predictions(
        st.session_state.y_test, st.session_state.y_pred_class
    )
    st.sidebar.markdown(
        "<h2 style='text-align:center;'>Confusion Matrix</h2>", unsafe_allow_html=True
    )
    st.sidebar.pyplot(CM_fig.figure_)

    # download model
    st.sidebar.markdown(
        "<h2 style='text-align:center;'>Download Model</h2>", unsafe_allow_html=True
    )
    st.sidebar.download_button(
        label="Download Model",
        data=open(st.session_state.path_model, "rb").read(),
        file_name="teachable_machine_model_%s.h5" % (st.session_state.str_kelas),
    )


def get_ImagePredict():
    model = keras.models.load_model(st.session_state.path_model)
    img = load_img(st.session_state.data_image_predict, target_size=(256, 256))
    X_test = np.array([img_to_array(img)])
    X_test = X_test / 255.0
    result = model.predict(X_test)
    # st.write(result)
    return result


def predictModel():
    st.markdown(
        "<h1 style='text-align:center;'> -----Try to Predict Image----- </h1>",
        unsafe_allow_html=True,
    )
    image_predict = st.file_uploader(
        "Add image to predict",
        accept_multiple_files=False,
        key="data_image_predict",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )
    if image_predict:
        st.markdown("<h4>Image Predict</h4>", unsafe_allow_html=True)
        st.image(image_predict)
        st.markdown("<h4>Results</h4>", unsafe_allow_html=True)
        # st.write("Image Predict: ")
        result = get_ImagePredict()
        # st.write(result)
        y_pred = np.argmax(result, axis=1)
        # st.write(y_pred)
        # st.text((st.session_state.input_kelas))
        str_class = st.session_state.input_kelas[y_pred[0]]
        # st.write("Gambar ini termasuk ke dalam kelas: %s" % (str_class))
        st.write(
            "Gambar ini termasuk ke dalam kelas: %s - Probabilitas : %.3f"
            % (str_class, result[0][y_pred] * 100)
        )
        for probability, kelas in zip(result[0], list(st.session_state.input_kelas)):
            st.write("Kelas: %s - Probabilitas : %.3f " % (kelas, probability * 100))
    else:
        st.warning("Masukkan Gambar untuk prediksi", icon="⚠️")
