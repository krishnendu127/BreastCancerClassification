# Breast Cancer Classification

## Overview
This project focuses on classifying breast cancer tumors as either benign or malignant using machine learning techniques. It utilizes a neural network built with TensorFlow and Keras to predict the tumor type based on various features extracted from biopsy samples.

## Dataset
The dataset used for this project is sourced from the Breast Cancer Wisconsin (Diagnostic) dataset, available in the sklearn library. It contains features extracted from digitized images of breast cancer biopsies, such as tumor radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Features
- **Data Loading and Preprocessing**: The dataset is loaded from sklearn, converted into a pandas DataFrame, and preprocessed by standardizing the input features.
- **Neural Network Architecture**: A neural network model is built with an input layer, a hidden layer with 20 neurons using ReLU activation, and an output layer with 2 neurons using sigmoid activation.
- **Model Training**: The neural network is trained using both the raw and standardized input data. Training is performed over 10 epochs with a validation split of 10%.
- **Model Evaluation**: The model's accuracy is evaluated on the test data, and a plot is generated to visualize the training and validation accuracy over epochs.
- **Prediction**: The trained model is used to predict the tumor type (benign or malignant) for new biopsy samples.

## Dependencies
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## Usage
1. Ensure you have the necessary dependencies installed.
2. Run the provided code in a Python environment such as Jupyter Notebook or Google Colab.
3. The code will load, preprocess, train, evaluate, and make predictions using the breast cancer dataset.
4. You can modify the code, experiment with different neural network architectures or hyperparameters, and explore further improvements in the classification accuracy.
