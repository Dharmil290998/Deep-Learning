# Deep-Learning

# Image Label Classification with DNN Model

## Project Overview
This project aims to classify images into five categories: **inside**, **outside**, **food**, **drink**, and **menu**. A Deep Neural Network (DNN) model was developed for image classification, and the model is deployed using **Flask** and integrated with a **Streamlit** frontend for user interaction. The project demonstrates a full pipeline from model development to deployment, including a web interface for making predictions on new images.

## Key Features
- **Image Classification:** A DNN model is trained to classify images into one of the five categories.
- **Flask API:** The trained model is deployed using Flask to serve the classification results as an API.
- **Streamlit Frontend:** A simple user interface built with Streamlit allows users to upload images and receive predictions.
- **End-to-End Deployment:** The system is fully functional and deployed locally.

## Project Files
- **`DNN_model.ipynb`**: This Jupyter notebook contains the code for data preprocessing, model building, training, evaluation, and saving the trained DNN model.
- **`app.py`**: This file contains the Flask API for loading the trained model and predicting image labels.
- **`streamlit_app.py`**: This file includes the Streamlit frontend, allowing users to upload images and get predictions from the model.
- **`Question-4.mp4`**: A video demonstrating the functionality of the deployed model, showing how the model performs predictions in real-time via the Streamlit interface.

How to Run the Project

## 1. Clone the Repository
bash
git clone https://github.com/Dharmil290998/Deep-Learning.git
cd Deep-Learning


## 2. Install Dependencies
Make sure you have Python installed and set up. You can install the required dependencies using the requirements.txt file.

bash
Copy code
pip install -r requirements.txt

## 3. Run the Flask API
To start the Flask server, navigate to the directory and run the following command:

bash
Copy code
python app.py
The API will start running locally on http://localhost:5000.

## 4. Run the Streamlit Frontend
In a new terminal, navigate to the project directory and run:

bash
Copy code
streamlit run streamlit_app.py
The Streamlit application will open in your browser, and you can upload an image to get a prediction.

## Model Development
Dataset
The dataset consists of labeled images across five categories: inside, outside, food, drink, and menu. The images were preprocessed to resize them to a standard shape and normalized for optimal training.

## DNN Model
A deep neural network was designed using Keras and TensorFlow, with the following architecture:

Multiple convolutional layers to extract features.
Dense layers for classification.
Softmax activation for the final output layer, producing probabilities for each of the five classes.
Model Performance
The model was trained using cross-entropy loss and evaluated on a validation set. Key performance metrics include:

* Accuracy: 81%
<img src="https://github.com/Dharmil290998/Deep-Learning/blob/main/Images/Classification%20Report.png" alt="Classification Report" width="500" height="600">
* Confusion Matrix
<img src="https://github.com/Dharmil290998/Deep-Learning/blob/main/Images/Confusion%20Matrix.png" alt="Confusion Matrix" width="500" height="600"> 
* AUC-ROC Curve
<img src="https://github.com/Dharmil290998/Deep-Learning/blob/main/Images/AUC-ROC%20Curve.png" alt="AUC-ROC Curve" width="500" height="600"> 

# Hyperparameter Tuning
To optimize performance, several rounds of hyperparameter tuning were performed, adjusting parameters like learning rate, batch size, and the number of epochs.

# Deployment
Flask API: The trained model is loaded in app.py, where Flask serves the predictions as a RESTful API.
Streamlit Frontend: The web interface in streamlit_app.py allows users to upload an image, which is sent to the Flask API for classification.

# Usage
Upload an image through the Streamlit interface, and the system will classify it as one of the five categories. The result, along with the probability scores for each class, will be displayed on the page.

# Video Demo
A demonstration of the full functionality is provided in the Question-4.mp4 video, showing the step-by-step process of using the Streamlit frontend to classify images.

# Future Improvements
* Model Tuning: Further optimization of the DNN model's architecture and hyperparameters could improve accuracy.
* Scalability: Deploy the model to a cloud platform for wider accessibility.
* Additional Features: Add more classes and labels to improve the versatility of the image classifier.
