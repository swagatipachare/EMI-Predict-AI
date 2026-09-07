# EMIPredict AI – Intelligent Financial Risk Assessment Platform

EMIPredict AI is a machine learning-based financial risk assessment platform that predicts **EMI eligibility** and estimates the **maximum affordable monthly EMI** for an applicant.

The project combines data preprocessing, feature engineering, machine learning, model evaluation, and a Streamlit-based interactive web application to provide an easy-to-use financial assessment system.

---

🚀 **Live Demo:** [EMIPredict AI](https://emi-predict-ai-wgvbjc86cy7pbkqljrn7ja.streamlit.app/)

📂 **GitHub Repository:** [EMIPredict AI](https://github.com/swagatipachare/EMI-Predict-AI)


## 🚀 Project Overview

The system provides two major predictions:

1. **EMI Eligibility Prediction**
   - Eligible
   - High Risk
   - Not Eligible

2. **Maximum Monthly EMI Prediction**
   - Estimates the maximum EMI amount an applicant can reasonably afford based on their financial profile.

The application is designed as a FinTech-oriented machine learning solution for automated financial risk assessment.

---

## 🎯 Project Objectives

- Analyze applicant financial information.
- Perform data cleaning and preprocessing.
- Engineer meaningful financial features.
- Predict EMI eligibility using classification models.
- Predict maximum affordable EMI using regression models.
- Compare multiple machine learning algorithms.
- Save trained models for deployment.
- Provide predictions through an interactive Streamlit application.
- Track machine learning experiments using MLflow during model development.

---

## 🛠️ Technologies Used

### Programming Language

- Python 3.9+

### Machine Learning

- Scikit-learn
- XGBoost
- Random Forest
- Logistic Regression
- Linear Regression

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### Web Application

- Streamlit

### Experiment Tracking

- MLflow

### Version Control

- Git
- GitHub

---

## 📂 Project Structure

```text
EMI PREDICT/
│
├── data/
│   └── emi_prediction_dataset.csv
│
├── models/
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   └── classification_label_encoder.pkl
│
├── notebooks/
│   └── EMI.ipynb
│
├── src/
│   └── bg.png
│
├── venv/
│
├── app.py
├── requirements.txt
└── README.md
