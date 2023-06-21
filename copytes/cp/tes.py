# import streamlit as st
# import cv2
# import numpy as np
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# class VideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.video_frames = []

#     def recv(self, frame):
#         print(frame)
#         frame_data = frame.to_ndarray(format="bgr24")
#         # self.video_frames.append(frame_data)
#         return frame


# def record_video():
#     video_transformer = VideoTransformer()

#     webrtc_streamer(
#         key="example", video_transformer_factory=VideoTransformer, async_transform=True
#     )

#     return video_transformer.video_frames


# def main():
#     st.title("MP4 Video Recorder")

#     record_video()


# if __name__ == "__main__":
#     main()

# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import av
# import numpy as np
# import cv2

# data = []


# def video_frame_callback(frame):
#     while True:
#         img = frame.to_ndarray(format="bgr24")
#         data.append(img)
# webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("copytes/cp/keras_model.h5", compile=False)

# Load the labels
class_names = open("copytes/cp/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window

    # Make the image a numpy array and reshape it to the models input shape.
    temp = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    temp = (temp / 127.5) - 1

    # Predicts the model
    prediction = model.predict(temp)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    cv2.putText(
        image,
        "Class: %s "
        % class_name[
            2:
        ],  # Remove the decimal point and the last two digits of the confidence score
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow("Webcam Image", image)

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
