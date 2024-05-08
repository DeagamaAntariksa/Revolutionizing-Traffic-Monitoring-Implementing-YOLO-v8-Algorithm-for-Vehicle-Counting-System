import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import object_counter

def main():
    st.title("Object Counting using YOLO and Streamlit")

    # Load YOLO model
    model = YOLO("best.pt")

    # Define line points
    line_points = [(5, 600), (1500, 600)]

    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=line_points,
                     classes_names=model.names,
                     draw_tracks=True,
                     line_thickness=2)

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        # Read the video file
        video_bytes = uploaded_file.read()
        video_nparray = np.frombuffer(video_bytes, np.uint8)
        video = cv2.imdecode(video_nparray, cv2.IMREAD_UNCHANGED)

        if video is not None:
            # Process the video frame by frame
            with st.spinner("Processing video..."):
                for frame in video:
                    # Detect and track objects
                    tracks = model.track(frame, persist=True, show=False)

                    # Count objects and draw results
                    frame = counter.start_counting(frame, tracks)

                    # Display the processed frame
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="BGR")
        else:
            st.error("Failed to decode video. Please choose another file.")

if __name__ == "__main__":
    main()
