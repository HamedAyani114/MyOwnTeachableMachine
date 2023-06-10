import streamlit as st
import numpy as np
import functions as f
import keras


# TITLE
st.markdown(
    "<h1 style='text-align:center'>Teachable Machine</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align:center'>Image Classification</h4>", unsafe_allow_html=True
)


def main():
    col1, col2, col3 = st.columns(3, gap="large")

    # form input numclass
    with col2:
        total_class_input = st.number_input(
            "Banyak Kelas",
            key="data_num_class",
            step=1,
            # min_value=2,
        )

    # form input class and image
    if total_class_input:
        try:
            for i in range(int(total_class_input)):
                st.text_input(
                    "Kelas {}".format(i + 1),
                    placeholder="Nama Kelas",
                    key="class_input{}".format(i),
                    # required=True,
                )
                st.file_uploader(
                    "Input Gambar",
                    label_visibility="hidden",
                    accept_multiple_files=True,
                    key="image_input{}".format(i),
                    type=["jpg", "jpeg", "png"],
                )
                st.markdown("<hr>", unsafe_allow_html=True)

            # training model
            col1, col2, col3 = st.columns(3, gap="large")
            btn_training = col2.button("Train Model", type="primary")
            tuning_param = col2.checkbox("tuning parameter")

            if tuning_param:
                st.session_state.epochs_input = col2.number_input(
                    "Epochs", min_value=1, value=15
                )
                st.session_state.batch_size_input = col2.number_input(
                    "Batch Size", min_value=1, value=8
                )

            if btn_training:
                if tuning_param == False:
                    st.session_state.epochs_input = 15
                    st.session_state.batch_size_input = 8
                model = f.trainingModel()

        except:
            st.warning("Form Tidak boleh Kosong!", icon="⚠️")

    try:
        if st.session_state.isModelTrained:
            model_file = ""
            for i in st.session_state.input_kelas:
                model_file += i + "-"
            model = keras.models.load_model(
                "models/teachable_machine_model_%s.h5" % (model_file)
            )
            f.predictModel(model)
    except:
        # langkah langkah how to use teachable machine mulai dari persiapan data, training, dan prediksi
        # new line
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align:justify'>How to Use Teachable Machine?</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align:justify'>1. Persiapan Data</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Masukkan jumlah kelas dari data (minimal 2 kelas)</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Masukkan nama kelas</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Masukkan gambar untuk setiap kelas</p>",
            unsafe_allow_html=True,
        )
        # st.markdown(
        #     "<p style='text-align:justify'>- Jumlah gambar minimal 10</p>",
        #     unsafe_allow_html=True,
        # )
        st.markdown(
            "<h4 style='text-align:justify'>2. Training</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tekan tombol Train Model</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tunggu hingga proses training selesai</p>",
            unsafe_allow_html=True,
        )
        # tuning parameter
        st.markdown(
            "<p style='text-align:justify'>- Centang Tuning Parameter untuk mengubah nilai Epochs dan Batch Size</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align:justify'>3. Prediksi</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tekan tombol Predict</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Pilih gambar yang akan diprediksi</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify'>- Tunggu hingga proses prediksi selesai</p>",
            unsafe_allow_html=True,
        )
        # hasil prediksi
        st.markdown(
            "<p style='text-align:justify'>- Hasil prediksi dan kemungkinan kelas dari input akan muncul </p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
