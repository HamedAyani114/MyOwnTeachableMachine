import keras
from keras import layers
from keras.utils import load_img, img_to_array, to_categorical
from keras.applications.mobilenet_v2 import MobileNetV2
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder

# from keras.applications.mobilenet_v3 import MobileNetV3Large, MobileNetV3Small

# split data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import cv2
import time


le = LabelEncoder()


def record_video(class_name):
    st.markdown(
        "<h3>Recording for Class: {}</h3>".format(class_name),
        unsafe_allow_html=True,
    )

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("http://192.168.18.220:8080/video")
    fps = 12
    shut_speed = 1 / fps
    frames = []
    class_images = []

    cv2.namedWindow("Live Capture")
    start = True
    while start:
        ret, frame = cap.read()
        cv2.imshow("Live Capture", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
        frames.append(frame)
        class_images.append(class_name)
        time.sleep(shut_speed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            start = False
    cv2.destroyAllWindows()

    data_images = np.array(frames)
    class_images = np.array(class_images)

    st.markdown(
        "<h5> Total Sample: {}</h5>".format(data_images.shape[0]),
        unsafe_allow_html=True,
    )
    cap.release()
    st.success("Video recorded for Class: {}!".format(class_name), icon="✔️")

    # st.write(data_images.shape, class_images.shape)
    return data_images, class_images


def get_ImagesClassForm():
    session_input = list(st.session_state)

    key_class_input = []
    key_images_input = []
    for input in session_input:
        if "class_input" in input:
            key_class_input.append(input)
        elif "image_input" in input:
            key_images_input.append(input)
    key_class_input = sorted(key_class_input)
    key_images_input = sorted(key_images_input)

    data_images = []
    class_images = []

    for data_img, cls_img in zip(key_images_input, key_class_input):
        kelas = st.session_state[cls_img]
        image = st.session_state[data_img]

        if image:
            for img_input in image:
                # Add class
                class_images.append(kelas)
                # Add Image
                img = load_img(img_input, target_size=(224, 224))
                data_images.append(img_to_array(img))

    for i in range(len(key_class_input)):
        recorded_frames_key = "recorded_frames{}".format(i)
        recorded_classes_key = "recorded_class{}".format(i)
        # st.write(recorded_frames_key, recorded_classes_key)
        if (
            recorded_frames_key in st.session_state
            and recorded_classes_key in st.session_state
        ):
            recorded_frames = st.session_state[recorded_frames_key]
            recorded_classes = st.session_state[recorded_classes_key]

            data_images.extend(recorded_frames)
            class_images.extend(recorded_classes)

    data_images = np.array(data_images)
    class_images = np.array(class_images)
    # st.write("gabung: ", data_images.shape, class_images.shape)

    return data_images / 255.0, class_images


def trainingModel(epochs, batch_size):
    epochs = epochs
    batch_size = batch_size
    # st.write(epochs, batch_size)

    # data input
    X, y = get_ImagesClassForm()
    # st.write(y)
    y_num = le.fit_transform(y)
    st.session_state.input_kelas = le.classes_
    st.write(st.session_state.input_kelas)
    st.session_state.str_kelas = "-".join(st.session_state.input_kelas)

    # one hot encoding
    y_cat = to_categorical(y_num)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    # Model
    base_model = MobileNetV2(
        weights="imagenet", input_shape=X_train[0].shape, include_top=False
    )
    base_model.trainable = False
    inputs = keras.Input(shape=X_train[0].shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)

    outputs = layers.Dense(len(y_cat[0]), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Train model

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, y_test),
        )
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        # history model
        train_loss, train_acc = (
            model.history.history["loss"][0],
            model.history.history["accuracy"][0],
        )
        val_loss, val_acc = (
            model.history.history["val_loss"][0],
            model.history.history["val_accuracy"][0],
        )
        status_text.text(
            "Epoch: %d/%d - Training in progress... \ntrain loss: %.4f - train acc: %.3f \nval loss: %.4f - val acc: %.3f "
            % (epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)
        )
        if val_loss < 0.01:
            break

    # history model
    status_text.text(f"Epoch {epoch + 1}/{epochs} - Training completed!")
    st.write("train loss: %.4f - train acc: %.3f " % (train_loss, train_acc))
    st.write("val loss: %.4f - val acc: %.3f " % (val_loss, val_acc))

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


def sidebar():
    CM_fig = ConfusionMatrixDisplay.from_predictions(
        st.session_state.y_test, st.session_state.y_pred_class
    )
    st.sidebar.markdown(
        "<h2 style='text-align:center;'>Confusion Matrix from Trainning</h2>",
        unsafe_allow_html=True,
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
    img = load_img(st.session_state.data_image_predict, target_size=(224, 224))
    X_test = np.array([img_to_array(img)]) / 255.0
    # st.write(X_test)
    result = model.predict(X_test)
    # st.write(result)
    return result


def predictModel():
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;'> Prediksi Gambar </h1>",
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
        st.markdown("<h4>Hasil</h4>", unsafe_allow_html=True)
        result = get_ImagePredict()
        y_pred = np.argmax(result, axis=1)
        str_class = st.session_state.input_kelas[y_pred[0]]
        st.write(
            "Gambar ini termasuk ke dalam kelas: %s - Probabilitas : %.3f"
            % (str_class, result[0][y_pred] * 100)
        )
        for probability, kelas in zip(result[0], list(st.session_state.input_kelas)):
            st.write("Kelas: %s - Probabilitas : %.3f" % (kelas, probability * 100))
            st.progress(int(probability * 100))
    else:
        st.info("Masukkan Gambar untuk prediksi", icon="ℹ️")


st.cache(suppress_st_warning=True)
