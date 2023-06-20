# import cv2
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import numpy as np
# import keras


# def callback(frame: av.VideoFrame) -> av.VideoFrame:
#     img = frame.to_ndarray(format="bgr24")
#     return av.VideoFrame.from_ndarray(img, format="bgr24")


# webrtc_streamer(key="example", video_frame_callback=callback, sendback_audio=False)


import av
import cv2
import streamlit as st
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer

# Load the pre-trained Keras model
model = load_model("models\model.h5")


# Define a function to preprocess the input frame/image
def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0  # Normalize the pixel values
    frame = np.expand_dims(frame, axis=0)
    return frame


# Define a function to make predictions on the preprocessed image
def predict(image):
    predictions = model.predict(image)
    # Perform post-processing or extract the predicted class label
    # based on your model's output format and requirements
    return predictions


# Define the callback function for video frame processing
def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Make predictions
    predictions = predict(preprocessed_img)

    # Perform any desired visualizations or post-processing on the predictions

    # Display the frame and predictions in the Streamlit app
    st.image(img, channels="BGR", use_column_width=True)
    st.write("Predictions:", predictions)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("Live Prediction Demo")

    webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        sendback_audio=False,  # Disable audio
    )


if __name__ == "__main__":
    main()
