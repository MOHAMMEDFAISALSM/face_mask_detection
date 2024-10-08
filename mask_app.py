import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2

# Title for the App
st.title("Face Mask Detection App")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "Upload Image", "Upload Video", "How It Works"])

# Load the trained model
model_path = 'D:/faisal-VS/faisal project/self_driving_car_project/face mask/face_mask_model.h5'

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Function to predict mask usage
def predict_mask(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image_scaled = image / 255.0
    image_scaled = np.reshape(image_scaled, [1, 128, 128, 3])
    prediction = model.predict(image_scaled)
    return np.argmax(prediction)

# Home Page
if page == "Home":
    st.markdown("""
    Welcome to the **Face Mask Detection App**. This application allows you to upload an image or video to detect whether a person is wearing a mask.
    """)
    st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/mohammed-faisal-sm-266173297?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
    st.markdown("[GitHub Profile](https://github.com/)")

# "How It Works" Section with Images
elif page == "How It Works":
    st.header("How Face Mask Detection Works")
    st.markdown("""
    The face mask detection application uses a **Convolutional Neural Network (CNN)** to detect whether a person is wearing a mask or not. The process involves:
    1. **Image Input**: The user uploads an image or video.
    2. **Image Preprocessing**: The image is resized and normalized.
    3. **CNN Model**: The pre-trained CNN model predicts the probability of mask detection.
    4. **Result Output**: The app outputs whether the person in the image is wearing a mask.
    """)

    # Display example images (smaller size for professional display)
    st.image("D:/faisal-VS/faisal project/self_driving_car_project/face mask/mask wearing ai image.jpeg", caption="Mask Detection Example", width=300)  # Smaller display
    st.image("D:/faisal-VS/faisal project/self_driving_car_project/face mask/without wearing mask.jpeg", caption="No Mask Detection Example", width=300)  # Smaller display

# Upload an image and predict mask
elif page == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image to Predict", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        pred_label = predict_mask(image)
        if pred_label == 1:
            st.success("The person in the image is wearing a mask.")
        else:
            st.error("The person in the image is not wearing a mask.")

# Upload a video and predict mask
elif page == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video to Predict", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)

        # Process the video for predictions
        temp_video_path = f"temp_{uploaded_video.name}"  # Temporary name for the uploaded video

        # Save the uploaded video temporarily
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Read the video using OpenCV
        cap = cv2.VideoCapture(temp_video_path)

        mask_count = 0
        no_mask_count = 0
        frame_count = 0

        # Process each frame in the video (sampling every 30th frame to speed up)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 30th frame to avoid processing every frame
            if frame_count % 30 == 0:
                # Preprocess the frame for prediction
                frame_resized = cv2.resize(frame, (128, 128))
                frame_scaled = frame_resized / 255.0
                frame_scaled = np.reshape(frame_scaled, [1, 128, 128, 3])

                # Make a prediction
                prediction = model.predict(frame_scaled)
                pred_label = np.argmax(prediction)

                # Count mask and no mask
                if pred_label == 1:
                    mask_count += 1
                else:
                    no_mask_count += 1

            frame_count += 1

        # Release the video capture object
        cap.release()

        # Display the majority result
        if mask_count > no_mask_count:
            st.success("The person in the video is mostly wearing a mask.")
        else:
            st.error("The person in the video is mostly not wearing a mask.")

        st.info(f"Frames processed: {frame_count}, Mask frames: {mask_count}, No Mask frames: {no_mask_count}")

        st.success("Video processing complete!")

# Footer with contact info
st.markdown("""
    <footer>
        <p>Contact: <a href="mailto:mf5330766@gmail.com">mf5330766@gmail.com</a></p>
    </footer>
""", unsafe_allow_html=True)

# Custom CSS for styling and background image
st.markdown("""
    <style>
    body {
        background-image: url("https://www.example.com/background.jpg");  /* Add your background image URL */
        background-size: cover;
    }
    .main {
        background-color: #f0f0f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    h1, h2 {
        color: #4a4a4a;
    }
    footer {
        background-color: #4CAF50;
        padding: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)
