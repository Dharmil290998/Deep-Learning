import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Set title of the Streamlit app
st.title("Photo Label Prediction")

# Load the trained model (ensure 'photo_label_model.h5' is in the same directory)
model = tf.keras.models.load_model('C:/Users/Dell/OneDrive/Desktop/sem 3/Deep Learning/DNN/photo_label_model.h5')

# Define the label map for predictions
label_map = {0: 'food', 1: 'inside', 2: 'outside', 3: 'drink', 4: 'menu'}

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to 128x128
    img_array = img_to_array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Predict the label using the loaded model
    prediction = model.predict(img_array)
    predicted_label = label_map[np.argmax(prediction)]

    # Display the predicted label
    st.write(f"Predicted Label: {predicted_label}")
