from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (ensure 'photo_label_model.h5' is in the same directory)
model = tf.keras.models.load_model('C:/Users/Dell/OneDrive/Desktop/sem 3/Deep Learning/DNN/photo_label_model.h5')

# Define the label map for predictions
label_map = {0: 'food', 1: 'inside', 2: 'outside', 3: 'drink', 4: 'menu'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Debugging: Print file info and read file as bytes
        print(f"Received file: {image_file.filename}, Mimetype: {image_file.mimetype}")
        image_bytes = image_file.read()  # Read the file as bytes
        print(f"Image byte size: {len(image_bytes)}")  # Debug: Check byte size

        if len(image_bytes) == 0:
            return jsonify({"error": "Empty image file or upload failed"}), 400

        # Convert the bytes into an image
        image = Image.open(io.BytesIO(image_bytes))

        # Resize and preprocess the image (resize to 128x128 and normalize)
        image = image.resize((128, 128))
        img_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction using the loaded model
        prediction = model.predict(img_array)
        predicted_label = label_map[np.argmax(prediction)]

        # Return the predicted label as a JSON response
        return jsonify({'label': predicted_label})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
