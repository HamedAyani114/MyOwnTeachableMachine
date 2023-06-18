# # import streamlit as st
# # import functions as f
# # import cv2
# # import os
# # from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


# # # Kelas VideoTransformer untuk merekam video dan menyimpan data tiap frame
# # # Kelas VideoRecorder untuk merekam video dan menyimpannya
# # class VideoRecorder(VideoTransformerBase):
# #     def __init__(self, class_name):
# #         self.class_name = class_name
# #         self.video_output_dir = "videos/{}".format(self.class_name)
# #         os.makedirs(self.video_output_dir, exist_ok=True)
# #         self.video_writer = None

# #     def transform(self, frame):
# #         if self.video_writer is None:
# #             frame_width = frame.shape[1]
# #             frame_height = frame.shape[0]
# #             self.video_writer = cv2.VideoWriter(
# #                 "{}/{}.avi".format(
# #                     self.video_output_dir, len(os.listdir(self.video_output_dir))
# #                 ),
# #                 cv2.VideoWriter_fourcc(*"MJPG"),
# #                 30,
# #                 (frame_width, frame_height),
# #             )
# #         # Simpan frame ke video
# #         self.video_writer.write(frame)
# #         return frame

# #     def close(self):
# #         if self.video_writer is not None:
# #             self.video_writer.release()


# # # TITLE
# # st.markdown(
# #     "<h1 style='text-align:center'>Teachable Machine</h1>",
# #     unsafe_allow_html=True,
# # )
# # st.markdown(
# #     "<h4 style='text-align:center'>Image Classification</h4>", unsafe_allow_html=True
# # )


# # def main():
# #     col1, col2, col3 = st.columns(3, gap="large")

# #     # form input numclass
# #     with col2:
# #         total_class_input = st.number_input(
# #             "Banyak Kelas",
# #             key="data_num_class",
# #             step=1,
# #             # min_value=2,
# #         )
# #     st.markdown("<br><br><hr>", unsafe_allow_html=True)

# #     if total_class_input > 1:
# #         # try:
# #         for i in range(total_class_input):
# #             input_kelas = st.text_input(
# #                 "Kelas {}".format(i + 1),
# #                 placeholder="Nama Kelas",
# #                 key="class_input{}".format(i),
# #             )
# #             # Tambahkan form untuk merekam video menggunakan streamlit-webrtc
# #             input_video = st.checkbox("Rekam Video", key="video_input{}".format(i))
# #             if input_video:
# #                 col1, col2 = st.columns([1, 4])
# #                 video_transformer = VideoRecorder(input_kelas)
# #                 webrtc_ctx = webrtc_streamer(
# #                     key="video-recorder-{}".format(i),
# #                     video_transformer_factory=video_transformer,
# #                     async_transform=True,
# #                 )
# #                 if webrtc_ctx.video_transformer:
# #                     col1.video(webrtc_ctx.video_transformer)
# #                 if not webrtc_ctx.state.playing:
# #                     video_transformer.close()
# #                     st.success(
# #                         "Video kelas {} telah direkam!".format(input_kelas),
# #                         icon="✔️",
# #                     )
# #                     # Mengirim video ke backend untuk diolah
# #                     # backend_result = f.process_video(video_transformer.video_output_dir)
# #                     st.success(
# #                         # "Hasil pengolahan video: {}".format(backend_result),
# #                         icon="✔️",
# #                     )
# #             input_image = st.file_uploader(
# #                 "Input Gambar",
# #                 label_visibility="hidden",
# #                 accept_multiple_files=True,
# #                 key="image_input{}".format(i),
# #                 type=["jpg", "jpeg", "png"],
# #             )
# #             st.markdown("<hr>", unsafe_allow_html=True)
# #         # ...

# #         # except:
# #         #     st.warning("Form Tidak boleh Kosong!", icon="⚠️")
# #     else:
# #         st.info("Minimal 2 Kelas", icon="ℹ️")

# #     try:
# #         if st.session_state.isModelTrained:
# #             f.sidebar()
# #             f.predictModel()
# #     except:
# #         # langkah langkah how to use teachable machine mulai dari persiapan data, training, dan prediksi
# #         st.markdown("<br><hr><br>", unsafe_allow_html=True)
# #         st.markdown(
# #             "<h3 style='text-align:justify'>How to Use Teachable Machine?</h3>",
# #             unsafe_allow_html=True,
# #         )


# # if __name__ == "__main__":
# #     main()

# import cv2
# import os
# import streamlit as st
# import numpy as np


# def record_videos(class_names, duration):
#     cap = cv2.VideoCapture(0)  # Membuka kamera dengan ID 0 (kamera default)

#     # Membuat direktori untuk menyimpan video
#     for class_name in class_names:
#         os.makedirs(class_name, exist_ok=True)

#     for class_name in class_names:
#         st.write("Merekam video untuk kelas:", class_name)

#         frames = []
#         start_time = cv2.getTickCount()
#         while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
#             ret, frame = cap.read()
#             frames.append(frame)

#             # Tampilkan frame pada Streamlit
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             st.image(frame_rgb, channels="RGB")

#         # Simpan video ke file
#         video_output_dir = os.path.join(class_name, f"{class_name}.avi")
#         height, width, _ = frames[0].shape
#         video_writer = cv2.VideoWriter(
#             video_output_dir, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height)
#         )
#         for frame in frames:
#             video_writer.write(frame)
#         video_writer.release()

#         st.write("Video kelas", class_name, "tersimpan:", video_output_dir)
#         st.markdown("---")

#     cap.release()


# # Contoh penggunaan
# class_names = ["Kelas 1", "Kelas 2", "Kelas 3"]
# record_videos(class_names, duration=5)


# import streamlit as st
# import functions as f
# import cv2
# import os

# # TITLE
# st.markdown(
#     "<h1 style='text-align:center'>Teachable Machine</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "<h4 style='text-align:center'>Image Classification</h4>", unsafe_allow_html=True
# )


# def record_videos(class_names, video_duration):
#     cap = cv2.VideoCapture(0)

#     for class_name in class_names:
#         st.markdown(
#             "<h3>Recording for Class: {}</h3>".format(class_name),
#             unsafe_allow_html=True,
#         )
#         st.write("Prepare to record video...")
#         st.info("Recording for {} seconds".format(video_duration))
#         frames = []

#         # Start recording
#         for _ in range(int(video_duration * 30)):
#             ret, frame = cap.read()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)

#         # Save video
#         video_dir = "recordings"
#         if not os.path.exists(video_dir):
#             os.makedirs(video_dir)

#         video_path = os.path.join(video_dir, class_name + ".avi")
#         out = cv2.VideoWriter(
#             video_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480)
#         )

#         for frame in frames:
#             out.write(frame)

#         out.release()

#         st.success("Video recorded for Class: {}!".format(class_name), icon="✔️")

#     cap.release()


# def main():
#     col1, col2, col3 = st.columns(3, gap="large")

#     # form input numclass
#     with col2:
#         total_class_input = st.number_input(
#             "Banyak Kelas",
#             key="data_num_class",
#             step=1,
#             # min_value=2,
#         )
#     st.markdown("<br><br><hr>", unsafe_allow_html=True)

#     # form input class and image
#     if total_class_input > 1:
#         try:
#             class_names = []
#             for i in range(total_class_input):
#                 input_kelas = st.text_input(
#                     "Kelas {}".format(i + 1),
#                     placeholder="Nama Kelas",
#                     key="class_input{}".format(i),
#                 )
#                 input_image = st.file_uploader(
#                     "Input Gambar",
#                     label_visibility="hidden",
#                     accept_multiple_files=True,
#                     key="image_input{}".format(i),
#                     type=["jpg", "jpeg", "png"],
#                 )
#                 video_duration = st.number_input(
#                     "Duration (in seconds)", min_value=1, value=5
#                 )
#                 btn_record = st.button("Record Videos", type="primary")

#                 if btn_record:
#                     record_videos([input_kelas], video_duration)
#                     st.success("Videos recorded for all classes!", icon="✔️")
#                 st.markdown("<hr>", unsafe_allow_html=True)

#             # training model
#             col1, col2, col3 = st.columns(3, gap="large")
#             tuning_param = col2.checkbox("tuning parameter")
#             btn_training = col2.button("Train Model", type="primary")

#             if tuning_param:
#                 epochs_input = col2.number_input("Epochs", min_value=1, value=15)
#                 batch_size_input = col2.number_input("Batch Size", min_value=1, value=8)
#             else:
#                 epochs_input = 10
#                 batch_size_input = 8
#             if btn_training:
#                 f.trainingModel(epochs_input, batch_size_input)
#                 st.success("Model Trained!", icon="✔️")

#             # Record videos
#             st.markdown("<br><hr><br>", unsafe_allow_html=True)
#             st.markdown("<h2>Record Videos for Each Class</h2>", unsafe_allow_html=True)
#             # video_duration = col2.number_input("Dur

#         except:
#             st.warning("Form must be filled!", icon="⚠️")
#     else:
#         st.info("Minimum 2 classes required", icon="ℹ️")

#     try:
#         if st.session_state.isModelTrained:
#             f.sidebar()
#             f.predictModel()
#     except:
#         # langkah langkah how to use teachable machine mulai dari persiapan data, training, dan prediksi
#         st.markdown("<br><hr><br>", unsafe_allow_html=True)
#         st.markdown(
#             "<h3 style='text-align:justify'>How to Use Teachable Machine?</h3>",
#             unsafe_allow_html=True,
#         )


# if __name__ == "__main__":
#     main()

import streamlit as st
import functions as f
import os
import cv2


# TITLE
st.markdown(
    "<h1 style='text-align:center'>Teachable Machine</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align:center'>Image Classification</h4>", unsafe_allow_html=True
)


def record_video(class_names, video_duration):
    cap = cv2.VideoCapture(0)

    for class_name in class_names:
        st.markdown(
            "<h3>Recording for Class: {}</h3>".format(class_name),
            unsafe_allow_html=True,
        )
        st.write("Prepare to record video...")
        st.info("Recording for {} seconds".format(video_duration))
        frames = []

        # Start recording
        for _ in range(int(video_duration * 30)):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Save video
        video_dir = "recordings"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(video_dir, class_name + ".avi")
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480)
        )

        for frame in frames:
            out.write(frame)

        out.release()

        st.success("Video recorded for Class: {}!".format(class_name), icon="✔️")

    cap.release()


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
    st.markdown("<br><br><hr>", unsafe_allow_html=True)

    # form input class and image
    if total_class_input > 1:
        # try:
        for i in range(total_class_input):
            input_kelas = st.text_input(
                "Kelas {}".format(i + 1),
                placeholder="Nama Kelas",
                key="class_input{}".format(i),
            )
            namakelas = []
            namakelas.append(input_kelas)
            input_image = st.file_uploader(
                "Input Gambar",
                label_visibility="hidden",
                accept_multiple_files=True,
                key="image_input{}".format(i),
                type=["jpg", "jpeg", "png"],
            )
            with st.form(key="record_form{}".format(i)):
                video_duration = st.number_input(
                    "Duration (in seconds)", min_value=1, value=5
                )
                submit_button = st.form_submit_button(label="Record")
            if submit_button:
                record_video(namakelas, video_duration)
            st.markdown("<hr>", unsafe_allow_html=True)
        # training model
        col1, col2, col3 = st.columns(3, gap="large")
        tuning_param = col2.checkbox("tuning parameter")
        btn_training = col2.button("Train Model", type="primary")
        if tuning_param:
            epochs_input = col2.number_input("Epochs", min_value=1, value=15)
            batch_size_input = col2.number_input("Batch Size", min_value=1, value=8)
        else:
            epochs_input = 10
            batch_size_input = 8
        if btn_training:
            f.trainingModel(epochs_input, batch_size_input)
            st.success("Model Trained!", icon="✔️")

        # except:
        #     st.warning("Form Tidak boleh Kosong!", icon="⚠️")
    else:
        st.info("Minimal 2 Kelas", icon="ℹ️")

    try:
        if st.session_state.isModelTrained:
            f.sidebar()
            f.predictModel()
    except:
        # langkah langkah how to use teachable machine mulai dari persiapan data, training, dan prediksi
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align:justify'>How to Use Teachable Machine?</h3>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
