import streamlit as st
import streamlit_webrtc as webrtc
import cv2


def record_video(video_stream):
    if video_stream.is_recording:
        # Convert frames to OpenCV format
        frame_images = [
            frame.to_ndarray(format="bgr24") for frame in video_stream.frames
        ]
        if len(frame_images) > 0:
            # Create a video writer to save the frames as an MP4 file
            height, width, _ = frame_images[0].shape
            video_writer = cv2.VideoWriter(
                "recorded_video.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (width, height),
            )
            # Write frames to the video writer
            for frame_image in frame_images:
                video_writer.write(frame_image)
            # Release the video writer
            video_writer.release()
            # Display a success message
            st.success("Video recording saved as recorded_video.mp4")


def main():
    st.title("MP4 Video Recorder")

    # Create a video recorder using the streamlit-webrtc module
    video_recorder = webrtc.VideoTransformer(
        callback=record_video, mimeType="video/mp4"
    )

    # Display the video recorder
    video_stream = video_recorder()
    st.write(video_stream)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


# TITLE
# st.markdown(
#     "<h1 style='text-align:center'>Teachable Machine</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "<h4 style='text-align:center'>Image Classification</h4>", unsafe_allow_html=True
# )
# webrtc to frame dataset
# class VideoTransformer(VideoTransformerBase):
#     def __init__(self) -> None:
#         self.threshold = 0.5

#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (224, 224))
#         img = img.astype("float32")
#         img /= 255.0
#         return img
# import streamlit as st
# import av
# import numpy as np
# import cv2
# from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# st.write("dataframe_video", dataframe_video)
# webrtc_streamer(key="sample", video_transformer_factory=VideoTransformer)


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

#     if total_class_input > 1:
#         # try:
#         for i in range(total_class_input):
#             input_kelas = st.text_input(
#                 "Kelas {}".format(i + 1),
#                 placeholder="Nama Kelas",
#                 key="class_input{}".format(i),
#             )
#             # Tambahkan form untuk merekam video menggunakan streamlit-webrtc
#             input_video = st.checkbox("Rekam Video", key="video_input{}".format(i))
#             if input_video:
#                 webrtc_streamer(
#                     key="sample {}".format(i),

#                     )

#             input_image = st.file_uploader(
#                 "Input Gambar",
#                 label_visibility="hidden",
#                 accept_multiple_files=True,
#                 key="image_input{}".format(i),
#                 type=["jpg", "jpeg", "png"],
#             )
#             st.markdown("<hr>", unsafe_allow_html=True)
#         # ...

#         # except:
#         #     st.warning("Form Tidak boleh Kosong!", icon="⚠️")
#     else:
#         st.info("Minimal 2 Kelas", icon="ℹ️")

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

# import streamlit as st
# import functions as f
# import os
# import cv2


# # TITLE
# st.markdown(
#     "<h1 style='text-align:center'>Teachable Machine</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "<h4 style='text-align:center'>Image Classification</h4>", unsafe_allow_html=True
# )


# def record_video(class_names, video_duration):
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
#         # try:
#         for i in range(total_class_input):
#             input_kelas = st.text_input(
#                 "Kelas {}".format(i + 1),
#                 placeholder="Nama Kelas",
#                 key="class_input{}".format(i),
#             )
#             namakelas = []
#             namakelas.append(input_kelas)
#             input_image = st.file_uploader(
#                 "Input Gambar",
#                 label_visibility="hidden",
#                 accept_multiple_files=True,
#                 key="image_input{}".format(i),
#                 type=["jpg", "jpeg", "png"],
#             )
#             with st.form(key="record_form{}".format(i)):
#                 video_duration = st.number_input(
#                     "Duration (in seconds)", min_value=1, value=5
#                 )
#                 submit_button = st.form_submit_button(label="Record")
#             if submit_button:
#                 record_video(namakelas, video_duration)
#             st.markdown("<hr>", unsafe_allow_html=True)
#         # training model
#         col1, col2, col3 = st.columns(3, gap="large")
#         tuning_param = col2.checkbox("tuning parameter")
#         btn_training = col2.button("Train Model", type="primary")
#         if tuning_param:
#             epochs_input = col2.number_input("Epochs", min_value=1, value=15)
#             batch_size_input = col2.number_input("Batch Size", min_value=1, value=8)
#         else:
#             epochs_input = 10
#             batch_size_input = 8
#         if btn_training:
#             f.trainingModel(epochs_input, batch_size_input)
#             st.success("Model Trained!", icon="✔️")

#         # except:
#         #     st.warning("Form Tidak boleh Kosong!", icon="⚠️")
#     else:
#         st.info("Minimal 2 Kelas", icon="ℹ️")

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
