import streamlit as st
from stqdm import stqdm

import keras
from keras import layers
from keras.utils import load_img, img_to_array, to_categorical
from keras.applications.mobilenet_v2 import MobileNetV2

import cv2
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

le = LabelEncoder()


def record_video(input_kelas, count_recordframe):
    st.markdown(
        "<h3>Capture Frame for Class: %s</h3>" % input_kelas,
        unsafe_allow_html=True,
    )

    cap = cv2.VideoCapture(0)  # webcam input
    # cap = cv2.VideoCapture("http://192.168.1.17:8080/video")
    # fps = 12

    shut_speed = 1 / 12  # untuk mengatur kecepatan pengambilan frame

    # list untuk menyimpan frame dan kelas
    frames = []
    class_images = []

    temp = 0  # counter frame
    cv2.namedWindow("Frame Capture")
    start = True
    while start:
        temp += 1  # counter frame
        ret, frame = cap.read()  # record frame
        frame = cv2.flip(frame, 1)  # mirror image
        cv2.putText(
            frame,
            " Kelas: %s - Sample: %.f" % (input_kelas, temp),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )  # menampilkan kelas dan sample frame
        cv2.imshow("Frame Capture", frame)  # menampilkan frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame ke RGB
        frame = cv2.resize(frame, (224, 224))  # resize frame

        frames.append(frame)  # simpan frame
        class_images.append(input_kelas)  # simpan kelas
        time.sleep(shut_speed)  # delay (shutter speed)

        if (
            cv2.waitKey(1) & 0xFF == ord("q") or temp == count_recordframe
        ):  # stop record jika menekan tombol q atau mencapai jumlah sample
            start = False

    cv2.destroyAllWindows()  # tutup window

    # convert list to array
    data_images = np.array(frames)
    class_images = np.array(class_images)

    # show first frame, middle frame, last frame
    st.markdown("<h3>Sample Frame</h3>", unsafe_allow_html=True)
    st.image(data_images[[0, len(data_images) // 2, -1]])
    st.markdown(
        "<h5> Total Sample Frame: {}</h5>".format(data_images.shape[0]),
        unsafe_allow_html=True,
    )
    cap.release()  # tutup webcam

    st.success(
        "Frame Capture for Class: {}!".format(input_kelas), icon="✔️"
    )  # notifikasi berhasil

    # st.write(data_images.shape, class_images.shape)
    return data_images, class_images


# get dataset
def get_ImagesClassForm():
    session_input = list(st.session_state)

    # init key untuk mengambil input class dan image dari input form
    key_class_input = []
    key_images_input = []

    for input in session_input:  # get key input
        if "class_input" in input:  # get key class
            key_class_input.append(input)  # add key class
        elif "image_input" in input:  # get key image
            key_images_input.append(input)  # add key image

    # sort key class dan image agar class sesuai dengan input frame
    key_class_input = sorted(key_class_input)
    key_images_input = sorted(key_images_input)  # sort key image

    # init list untuk menyimpan frame dan kelas
    data_images = []
    class_images = []

    for data_img, cls_img in zip(
        key_images_input, key_class_input
    ):  # get frame dan kelas dari key
        kelas = st.session_state[cls_img]  # get class
        image = st.session_state[data_img]  # get frame

        if image:  # jika frame tidak kosong
            for img_input in image:  # get frame
                # Add class
                class_images.append(kelas)
                # Add Image
                img = load_img(
                    img_input, target_size=(224, 224)
                )  # load image dan resize
                data_images.append(img_to_array(img))  # convert image ke array

    for i in range(len(key_class_input)):
        recorded_frames_key = "recorded_frames{}".format(i)
        recorded_classes_key = "recorded_class{}".format(i)
        # st.write(recorded_frames_key, recorded_classes_key)
        if (
            recorded_frames_key in st.session_state
            and recorded_classes_key in st.session_state
        ):  # jika ada key frame dan kelas yang tersimpan di session state
            recorded_frames = st.session_state[recorded_frames_key]  # get frame
            recorded_classes = st.session_state[recorded_classes_key]  # get kelas

            data_images.extend(recorded_frames)  # add frame
            class_images.extend(recorded_classes)  # add kelas

    # cek banyak class
    if len(np.unique(class_images)) < 2:
        return False

    data_images = np.array(data_images) / 255.0
    class_images = np.array(class_images)
    # st.write("gabung: ", data_images.shape, class_images.shape)

    return data_images, class_images


# training model
def trainingModel(epochs, batch_size):
    epochs = epochs
    batch_size = batch_size
    global glob_input_kelas, glob_path_model, glob_str_kelas, glob_y_test_pred_class, globy_y_test_class  # global variable

    # data input
    X, y = get_ImagesClassForm()

    # class label to numerik
    y_num = le.fit_transform(y)

    # get class
    glob_input_kelas = le.classes_
    # st.write(glob_input_kelas)
    glob_str_kelas = "-".join(glob_input_kelas)

    # one hot encoding
    y_cat = to_categorical(y_num)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )

    # Model
    base_model = MobileNetV2(
        weights="imagenet", input_shape=X_train[0].shape, include_top=False
    )  # base model mobilenetv2 dengan weight imagenet dan input shape dari data train, include top false untuk menghilangkan fully connected layer mobilenetv2
    base_model.trainable = False  # freeze base model

    inputs = keras.Input(shape=X_train[0].shape)  # input layer
    x = base_model(inputs, training=False)  # base model
    x = layers.GlobalAveragePooling2D()(x)  # global average pooling layer

    # fully connected layer
    x = layers.Dense(16, activation="relu")(x)  # hidden layer
    outputs = layers.Dense(len(y_cat[0]), activation="softmax")(
        x
    )  # output layer sesuai dengan jumlah class
    model = keras.Model(inputs, outputs)  # gabung model

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )  # compile model

    status_text = st.empty()  # init status text
    for epoch in stqdm(range(epochs), st_container=st.write()):  # training model, stqdm untuk progress bar
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, y_test),
        )

        # history model
        (train_loss, train_acc, val_loss, val_acc) = (
            model.history.history["loss"][0],
            model.history.history["accuracy"][0],
            model.history.history["val_loss"][0],
            model.history.history["val_accuracy"][0],
        )

        # status text
        status_text.text(
            "Epoch: %d/%d - Training in progress... \ntrain loss: %.4f - train acc: %.3f \nval loss: %.4f - val acc: %.3f "
            % (epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)
        )

        if train_loss < 0.005 or train_acc == 1.000:  # stopping condition
            break

    # history model
    status_text.text(f"Epoch {epoch + 1}/{epochs} - Training completed!")
    st.write("train loss: %.4f - train acc: %.3f " % (train_loss, train_acc))
    st.write("val loss: %.4f - val acc: %.3f " % (val_loss, val_acc))

    y_pred = model.predict(X_test)  # predict data test
    y_pred = np.argmax(
        y_pred, axis=1
    )  # get index terbesar dari hasil predict sebagai kelas

    glob_y_test_pred_class = le.inverse_transform(
        y_pred
    )  # convert kelas prediksi numerik ke string
    globy_y_test_class = le.inverse_transform(
        np.argmax(y_test, axis=1)
    )  # convert kelas tes numerik ke string
    # st.write(confusion_matrix(st.session_state.y_test, st.session_state.y_pred_class))

    glob_path_model = "models/teachable_machine_model_%s.h5" % (
        glob_str_kelas
    )  # path model
    model.save(glob_path_model)  # save model
    st.session_state.isModelTrained = 1  # set model trained true
    # return True


# sidebar
def sidebar():
    CM_fig = ConfusionMatrixDisplay.from_predictions(
        globy_y_test_class, glob_y_test_pred_class
    )  # confusion matrix
    st.sidebar.markdown(
        "<h2 style='text-align:center;'>Confusion Matrix from Trainning</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.pyplot(CM_fig.figure_)  # plot confusion matrix

    # download model
    st.sidebar.markdown(
        "<h2 style='text-align:center;'>Download Model</h2>", unsafe_allow_html=True
    )
    st.sidebar.download_button(
        label="Download Model",
        data=open(glob_path_model, "rb").read(),
        file_name=glob_path_model,
    )  # download model


def get_ImagePredict():
    model = keras.models.load_model(glob_path_model) # load model

    img_predict = st.session_state.data_image_predict  # get image predict
    img = load_img(img_predict, target_size=(224, 224))  # load image
    X_test = np.array([img_to_array(img)]) / 255.0  # convert image to array
    result = model.predict(X_test)  # predict image
    return result


def show_result():
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;'> Prediksi Gambar </h1>",
        unsafe_allow_html=True,
    )

    radiopredict = st.radio(
        "Pilih Salah Satu",
        ("Upload Gambar", "Ambil Gambar dari Webcam"),
        key="radiopredict",
    )  # radio button untuk pilih upload gambar atau ambil gambar dari webcam

    if radiopredict == "Upload Gambar":
        image_predict = st.file_uploader(
            "Upload Gambar",
            accept_multiple_files=False,
            key="data_image_predict",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )  # upload gambar

    elif radiopredict == "Ambil Gambar dari Webcam":
        image_predict = st.camera_input(
            "Ambil Gambar dari Webcam",
            key="data_image_predict",
        )  # ambil gambar dari webcam

    if image_predict:  # jika ada gambar
        st.markdown("<h4>Image Predict</h4>", unsafe_allow_html=True)
        st.image(image_predict)
        st.markdown("<h4>Hasil</h4>", unsafe_allow_html=True)

        result = get_ImagePredict()  # panggil fungsi predict
        y_pred = np.argmax(
            result, axis=1
        )  # get index terbesar dari hasil predict sebagai kelas
        y_pred_class = glob_input_kelas[y_pred[0]]  # get kelas dari index terbesar

        st.write(
            "Gambar ini termasuk ke dalam kelas: %s - Probabilitas : %.3f"
            % (y_pred_class, result[0][y_pred] * 100)
        )  # print hasil prediksi

        for probability, kelas in zip(
            result[0], list(glob_input_kelas)
        ):  # loop untuk print hasil prediksi setiap kelas
            st.write(
                "Kelas: %s - Probabilitas : %.3f" % (kelas, probability * 100)
            )  # probabilitas dari setiap kelas
            st.progress(
                int(probability * 100)
            )  # progress bar dari probability setiap kelas
    else:
        st.info("Masukkan Gambar untuk prediksi", icon="ℹ️")


st.cache_data(suppress_st_warning=True)
