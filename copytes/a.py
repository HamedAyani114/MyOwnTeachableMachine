import streamlit as st
import cv2

vid = cv2.VideoCapture("http://192.168.18.220:8080/video")

st.title("Using Mobile Camera with Streamlit")
frame_window = st.image([])
take_picture_button = st.button("Take Picture")

while True:
    got_frame, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if got_frame:
        frame_window.image(frame)

    if take_picture_button:
        # Pass the frame to a model
        # And show the output here...
        break

vid.release()
