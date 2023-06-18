# import streamlit as st
# import functions as f
# import os
# import cv2
# import time
# import numpy as np


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
#         fps = 12
#         frame_rate = 1 / fps
#         # Start recording
#         for _ in range(int(video_duration * fps)):
#             ret, frame = cap.read()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (224, 224))
#             frames.append(frame)
#             time.sleep(frame_rate)

#             # frame count
#             # st.write("Frame count: {}".format(len(frames)))
#             # st.image(frame, use_column_width=True)
#         dataset_array = np.array(frames)
#         st.write("Dataset array shape: {}".format(dataset_array.shape))
#         # Save video
#         video_dir = "recordings"
#         if not os.path.exists(video_dir):
#             os.makedirs(video_dir)

#         video_path = os.path.join(video_dir, class_name + ".mp4")
#         out = cv2.VideoWriter(
#             video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (224, 224)
#         )

#         for frame in frames:
#             out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

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

# import streamlit as st
# import cv2
# import os
# import numpy as np
# import time


# def record_video(class_name, video_duration):
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     st.markdown(
#         "<h3>Recording for Class: {}</h3>".format(class_name), unsafe_allow_html=True
#     )
#     st.write("Prepare to record video...")
#     st.info("Recording for {} seconds".format(video_duration))

#     frames = []
#     frame_rate = 12  # Kecepatan penangkapan frame per detik
#     delay = 1 / frame_rate

#     # Start recording
#     for _ in range(int(video_duration * frame_rate)):
#         ret, frame = cap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
#         frames.append(frame)
#         time.sleep(delay)  # Delay antara setiap penangkapan frame

#     # Save video
#     video_dir = "recordings"
#     if not os.path.exists(video_dir):
#         os.makedirs(video_dir)

#     video_path = os.path.join(video_dir, class_name + ".mp4")
#     out = cv2.VideoWriter(
#         video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (224, 224)
#     )

#     for frame in frames:
#         out.write(
#             cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         )  # Convert back to BGR before writing

#     out.release()

#     st.success("Video recorded for Class: {}!".format(class_name), icon="✔️")

#     cap.release()


# # Main function
# def main():
#     st.markdown(
#         "<h1 style='text-align:center'>Teachable Machine</h1>", unsafe_allow_html=True
#     )
#     st.markdown(
#         "<h4 style='text-align:center'>Image Classification</h4>",
#         unsafe_allow_html=True,
#     )

#     # Form input numclass
#     total_class_input = st.number_input("Banyak Kelas", key="data_num_class", step=1)

#     st.markdown("<br><br><hr>", unsafe_allow_html=True)

#     # Form input class and video duration
#     if total_class_input > 1:
#         try:
#             for i in range(total_class_input):
#                 input_kelas = st.text_input(
#                     "Kelas {}".format(i + 1),
#                     placeholder="Nama Kelas",
#                     key="class_input{}".format(i),
#                 )
#                 enable_camera = st.checkbox(
#                     "Aktifkan Kamera", key="enable_camera{}".format(i)
#                 )
#                 video_duration = st.number_input(
#                     "Durasi Video (detik)",
#                     min_value=1,
#                     value=5,
#                     key="video_duration{}".format(i),
#                 )

#                 if enable_camera:
#                     record_button = st.button("Record", key="record_button{}".format(i))
#                     stop_button = st.button("Stop", key="stop_button{}".format(i))

#                     if record_button:
#                         record_video(input_kelas, video_duration)

#                     if stop_button:
#                         st.warning("Perekaman dihentikan", icon="⚠️")

#                     cap = cv2.VideoCapture(0)

#                     if cap.isOpened():
#                         ret, frame = cap.read()
#                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                         frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224

#                         st.image(frame, channels="RGB", caption="Preview Camera")

#                     cap.release()
#                 else:
#                     st.info("Aktifkan kamera untuk melihat preview")

#                 st.markdown("<hr>", unsafe_allow_html=True)

#             # Training model and prediction code here

#         except:
#             st.warning("Form tidak boleh kosong!", icon="⚠️")
#     else:
#         st.info("Minimal 2 kelas", icon="ℹ️")


# if __name__ == "__main__":
#     main()


# import streamlit as st
# import cv2
# import os
# import numpy as np
# import time


# def record_video(class_name, video_duration):
#     cap = cv2.VideoCapture(0)

#     st.markdown(
#         "<h3>Recording for Class: {}</h3>".format(class_name), unsafe_allow_html=True
#     )
#     st.write("Prepare to record video...")
#     st.info("Recording for {} seconds".format(video_duration))

#     frames = []

#     frame_rate = 12  # Kecepatan penangkapan frame per detik
#     delay = 1 / frame_rate

#     # Start recording
#     for _ in range(int(video_duration * frame_rate)):
#         ret, frame = cap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
#         frames.append(frame)
#         time.sleep(delay)  # Delay antara setiap penangkapan frame
#     img_array = np.array(frames)
#     st.write(img_array.shape)
#     # st.image(frame, channels="RGB", use_column_width=True)  # Menampilkan live capture

#     # Save video
#     video_dir = "recordings"
#     if not os.path.exists(video_dir):
#         os.makedirs(video_dir)

#     video_path = os.path.join(video_dir, class_name + ".mp4")
#     out = cv2.VideoWriter(
#         video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (224, 224)
#     )

#     for frame in frames:
#         out.write(
#             cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         )  # Convert back to BGR before writing

#     out.release()

#     st.success("Video recorded for Class: {}!".format(class_name), icon="✔️")

#     cap.release()


# # Main function
# def main():
#     st.markdown(
#         "<h1 style='text-align:center'>Teachable Machine</h1>", unsafe_allow_html=True
#     )
#     st.markdown(
#         "<h4 style='text-align:center'>Image Classification</h4>",
#         unsafe_allow_html=True,
#     )

#     # Form input numclass
#     total_class_input = st.number_input("Banyak Kelas", key="data_num_class", step=1)

#     st.markdown("<br><br><hr>", unsafe_allow_html=True)

#     # Form input class and video duration
#     if total_class_input > 1:
#         try:
#             for i in range(total_class_input):
#                 input_kelas = st.text_input(
#                     "Kelas {}".format(i + 1),
#                     placeholder="Nama Kelas",
#                     key="class_input{}".format(i),
#                 )
#                 video_duration = st.number_input(
#                     "Durasi Video (detik)",
#                     min_value=1,
#                     value=5,
#                     key="video_duration{}".format(i),
#                 )
#                 if st.button("Record", key="record_button{}".format(i)):
#                     record_video(input_kelas, video_duration)
#         except:
#             st.warning("Form Tidak boleh Kosong!", icon="⚠️")
#     else:
#         st.info("Minimal 2 Kelas", icon="ℹ️")


# if __name__ == "__main__":
#     main()


import streamlit as st
import functions as f


# Main function
def main():
    st.markdown(
        "<h1 style='text-align:center'>Teachable Machine</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<h4 style='text-align:center'>Image Classification</h4>",
        unsafe_allow_html=True,
    )

    # Form input numclass
    total_class_input = st.number_input(
        "Banyak Kelas", key="data_num_class", step=1, value=2
    )

    st.markdown("<br><br><hr>", unsafe_allow_html=True)

    # Form input class and video duration
    if total_class_input > 1:
        try:
            for i in range(total_class_input):
                input_kelas = st.text_input(
                    "Kelas {}".format(i + 1),
                    placeholder="Nama Kelas",
                    key="class_input{}".format(i),
                )
                input_image = st.file_uploader(
                    "Input Gambar",
                    label_visibility="hidden",
                    accept_multiple_files=True,
                    key="image_input{}".format(i),
                    type=["jpg", "jpeg", "png"],
                )

                buttonvideo = st.button("Record", key="record_button{}".format(i))
                if buttonvideo:
                    recorded_frames = f.record_video(input_kelas)
                    st.session_state["recorded_frames{}".format(i)], st.session_state["recorded_class{}".format(i)] = recorded_frames
                st.markdown("<br><hr>", unsafe_allow_html=True)

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
        except:
            st.warning("Form Tidak boleh Kosong!", icon="⚠️")
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
