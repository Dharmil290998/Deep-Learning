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

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/.git
cd your-repo-name
