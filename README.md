# Diabetes-ANN-Prediction
An artificial neural network-based model designed to predict diabetes risk based on key health indicators. This project leverages machine learning to analyze patterns in health data, providing insights that can support better decision-making and early risk assessment.

# Diabetes-ANN-Prediction

## Overview
**Diabetes-ANN-Prediction** is a deep learning project aimed at predicting the likelihood of diabetes using an Artificial Neural Network (ANN). The model leverages modern machine learning techniques to analyze health-related data, preprocess it efficiently and provide accurate predictions for diabetes diagnosis.

## Motivation
Diabetes is a chronic condition with severe complications if not diagnosed and managed in time. By utilizing neural networks, this project seeks to detect diabetes early and improve clinical decision-making. The model is designed to generalize well on unseen data by incorporating robust data preprocessing, class weighting and advanced regularization techniques.

## Project Structure
The project is developed as a Jupyter Notebook and is organized into the following cells:

1. **Import Libraries & Load Dataset**:  
   - Import necessary libraries (NumPy, Pandas, TensorFlow, Matplotlib, etc.).
   - Upload and read the dataset (CSV file).

2. **Data Preprocessing**:  
   - Handle missing values using `SimpleImputer` (median strategy).
   - Split the data into training and testing sets.
   - Scale features using `StandardScaler` to standardize the input data.

3. **Build & Train the ANN Model**:  
   - Define the neural network architecture using multiple Dense layers with L2 regularization and Dropout to prevent overfitting.
   - Use a custom learning rate scheduler to decay the learning rate over epochs.
   - Apply class weighting to counterbalance the data imbalance.
   - Train the model using a combination of EarlyStopping (or alternative strategies) and ModelCheckpoint (if desired).

4. **Model Evaluation**:  
   - Generate predictions and compute evaluation metrics such as Classification Report, ROC-AUC, and Confusion Matrix.
   - Plot training history (Accuracy & Loss curves), ROC Curve, and Normalized Confusion Matrix with proper dimensions and formatting.

## Model Architecture
The ANN model is structured as follows:
- **Input Layer**: Accepts standardized features.
- **Dense Layers**:  
  - A 128-neuron layer with ReLU activation and L2 regularization.
  - Followed by a Dropout layer (15% rate) to reduce overfitting.
  - A 64-neuron layer with ReLU activation and L2 regularization, followed by a 15% Dropout.
  - Subsequent layers include a 32-neuron layer (with 10% Dropout), a 16-neuron layer, an 8-neuron layer, and a 4-neuron layer.
- **Output Layer**:  
  - A single neuron with Sigmoid activation that outputs a probability value (0 to 1) representing the likelihood of diabetes.

## Training Configuration
- **Loss Function**: Binary crossentropy.
- **Optimizer**: Adam with an initial learning rate of 0.001.
- **Learning Rate Scheduler**: The learning rate decays exponentially over epochs.
- **Class Weighting**:  
  - Weight for class 0: 1.0  
  - Weight for class 1: 1.5  
- **Batch Size**: 16  
- **Epochs**: 200 (with early stopping if the validation loss does not improve over 15 epochs).

## Evaluation Metrics
- **Accuracy, Precision, Recall, and F1-Score**: Detailed in the classification report.
- **ROC-AUC**: Indicates the model's ability to distinguish between diabetic and non-diabetic cases.
- **Confusion Matrix**: Shows the distribution of true positives, true negatives, false positives, and false negatives.

## License
This project is licensed under the MIT License.

