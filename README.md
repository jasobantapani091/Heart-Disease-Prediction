# Heart Disease Prediction using Machine Learning

## Overview

This project focuses on predicting the presence of heart disease using machine learning classification algorithms. The system analyzes various medical parameters such as age, blood pressure, cholesterol level, chest pain type, and maximum heart rate to determine whether a patient is likely to have heart disease.

The primary objective of this project is to assist in early diagnosis through data-driven predictive analysis.

---

## Objectives

- Predict the likelihood of heart disease in patients
- Identify the most influential medical features
- Compare the performance of multiple machine learning algorithms
- Evaluate model accuracy using standard classification metrics

---

## Dataset Information

- **Dataset:** Heart Disease Dataset  
- **Source:** UCI Machine Learning Repository
- **Total Records:** 303
- **Total Features:** 14

### Target Variable

- `0` → No Heart Disease
- `1` → Heart Disease Present

---

## Dataset Features

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia |
| target | Heart disease prediction (0 or 1) |

---

## Technologies Used

### Programming Language
- Python

### Libraries
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Development Environment
- Jupyter Notebook
- Google Colab

---

## Machine Learning Models

The following classification algorithms were implemented and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

---

## Project Workflow

1. Data Collection  
2. Data Preprocessing  
3. Exploratory Data Analysis  
4. Feature Selection  
5. Model Training  
6. Model Evaluation  
7. Prediction and Result Analysis  

---

## Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Results

The implemented machine learning models successfully predicted the likelihood of heart disease using patient health data. Comparative analysis showed that ensemble-based models and advanced classifiers achieved better predictive performance.

This project demonstrates the practical use of machine learning in healthcare prediction systems and highlights its potential for supporting early diagnosis.

---

## Future Improvements

- Implement deep learning models
- Deploy the application using Flask or Streamlit
- Add real-time patient data input
- Improve performance using feature engineering and hyperparameter tuning

---

## Project Structure

```bash
Heart-Disease-Prediction/
│
├── dataset/
│   └── heart.csv
│
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── images/
│   └── results.png
│
├── requirements.txt
├── README.md
└── app.py
