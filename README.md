# Investor-Profiling-Risk-Prediction-System
This project is a machine learning application designed to classify investors based on their financial behavior, investment preferences, and risk appetite. The model predicts whether an investor is more likely to be Conservative or Risk-Oriented using demographic and behavioral indicators collected from survey-based financial data.

The application was built using Scikit-learn and deployed with Streamlit as an interactive web app.

## Project Overview
Investor decisions are influenced by several factors such as:
expected returns
investment duration
savings goals
preference for low-risk or high-risk assets
financial objectives

The goal of this project is to analyze these patterns and build a predictive system capable of identifying investor risk profiles. The project also explores challenges associated with small datasets and class imbalance in machine learning workflows.

## Problem Statement
The original dataset contained a very small number of observations, with the conservative investor class heavily underrepresented.
An initial multiclass approach was tested using SMOTE to balance the minority class. However, due to the limited dataset size, the model became unstable:
overall accuracy dropped significantly
predictions became inconsistent
in some cases, the conservative class disappeared entirely during prediction

To improve model stability and interpretability, the problem was reframed as a binary classification task:
Conservative Investor
Risk-Oriented Investor

This produced more reliable and explainable results.

## Features Used
The model was trained using the following features:
Gender
Age
Investment Avenues
Mutual Fund Preference
Debenture Preference
Government Bond Preference
PPF Preference
Gold Preference
Investment Factor
Financial Objective
Investment Purpose
Investment Duration
Investment Monitoring Frequency
Expected Returns
Preferred Investment Avenue
Savings Objectives
Information Source

These features collectively capture demographic, behavioral, and financial preference patterns relevant to investor decision-making.

## Machine Learning Pipeline
The project uses a complete preprocessing and modeling pipeline built with Scikit-learn and Imbalanced-Learn.
Preprocessing
Numerical scaling with StandardScaler
Categorical encoding with OneHotEncoder
Column transformation using ColumnTransformer
Modeling
SMOTE for handling class imbalance
LogisticRegression for classification

Using a pipeline ensured that preprocessing and prediction remained consistent during deployment.

## Model Evaluation

The model was evaluated using:
Accuracy
Precision
Recall
F1-score
Cross-validation

Example evaluation output:
```bash
precision    recall  f1-score   support

0       0.60      0.75      0.67         4
1       0.67      0.50      0.57         4

accuracy                           0.62         8
```

Given the small size of the dataset, the focus of the project was placed more on:
pipeline consistency
deployment workflow
explainability
handling imbalance issues, rather than maximizing predictive accuracy.

## Feature Importance
Feature importance analysis was used to understand which variables had the strongest influence on investor classification.
Some of the most influential features included:
expected return range
savings objectives
investment purpose
PPF preference
debenture preference
age

This helped improve transparency and interpretability within the model.

## Streamlit Application
The project includes a Streamlit web application where users can:
enter investor profile information
receive a predicted investor type
view prediction confidence
explore the behavioral drivers behind predictions
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Imbalanced-Learn
Matplotlib
Seaborn
Streamlit
Joblib

## Project Structure
```bash
Investor-Profiling-Risk-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── training_data.csv
│
├── models/
│   ├── investor_pipeline.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   └── investor_analysis.ipynb
│
└── visuals/
    └── feature_importance.png
```

## Installation

Clone the repository:
```bash
git clone <repository-url>
cd Investor-Profiling-Risk-Prediction-System
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the application:
```bash
streamlit run app.py
```

## Challenges Encountered
A major part of the project involved debugging deployment and preprocessing issues.
Some of the challenges included:
instability caused by oversampling on a very small dataset
schema mismatch between training data and Streamlit inputs
OneHotEncoder category inconsistencies during deployment
handling preprocessing inside a deployment-safe pipeline

Resolving these issues helped improve understanding of real-world ML deployment workflows.

## Future Improvements
Possible improvements for future versions include:
larger training datasets
SHAP-based explainability
API integration with FastAPI
personalized investment recommendation engine
advanced ensemble models
dashboard analytics integration
Conclusion

This project demonstrates an end-to-end machine learning workflow covering:

preprocessing
class imbalance handling
pipeline construction
model evaluation
deployment with Streamlit
explainability and debugging

It also highlights the practical challenges that arise when deploying machine learning systems with limited and imbalanced data.
