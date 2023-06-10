import keras
from keras import layers
from keras.utils import load_img, img_to_array, to_categorical

import streamlit as st
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping

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
        # st.write(st.session_state[class_img])

        for img_input in image:
            # Add class
            class_images.append(kelas)
            # Add Image
            img = load_img(img_input, target_size=(256, 256))
            data_images.append(img_to_array(img))

    # Di acak agar proses training maksimal
    X, y = shuffle(data_images, class_images)
    y = le.fit_transform(y)

    return np.array(X), np.array(y)


def trainingModel():
    # tuning parameter
    epochs = st.session_state.epochs_input
    batch_size = st.session_state.batch_size_input
    # st.write(epochs, batch_size)

    # data input
    X_train, y_train = get_ImagesClassForm()
    # num_label = len(np.unique(labels_data))

    # one hot encoding
    y_train = to_categorical(y_train)
    # st.write(len(labels_data[0]))

    # Model CNN
    model = keras.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="SAME",
            input_shape=X_train[0].shape,
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
    early_stop = EarlyStopping(monitor="loss", patience=2, min_delta=0.01)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
    )

    st.session_state.isModelTrained = True
    str_class = ""
    for i in st.session_state.input_kelas:
        str_class += i + "-"
    model.save("model/teachable_machine_model_%s.h5" % (str_class))
    return model


def get_ImagePredict(model):
    img = load_img(st.session_state.data_image_predict, target_size=(256, 256))
    X_test = np.array([img_to_array(img)])
    result = model.predict(X_test)
    # st.write(result)
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
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )
    btn_predict_image = st.button("Predict", type="primary")
    if btn_predict_image:
        if image_predict:
            st.markdown("<h4>Image Predict</h4>", unsafe_allow_html=True)
            st.image(image_predict)
            st.markdown("<h4>Results</h4>", unsafe_allow_html=True)

            # st.write("Image Predict: ")
            result = get_ImagePredict(model)
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
            for probability, kelas in zip(
                result[0], list(st.session_state.input_kelas)
            ):
                st.write(
                    "Kelas: %s - Probabilitas : %.3f " % (kelas, probability * 100)
                )
        else:
            st.warning("Masukkan Gambar untuk prediksi", icon="⚠️")
