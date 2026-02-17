# Loan Eligibility Prediction using Logistic Regression

## Overview
This project implements a **Loan Eligibility Prediction System** using **Logistic Regression** to classify whether a loan application will be approved or rejected. The model is trained on historical applicant data and evaluated using standard classification metrics and ROC-AUC analysis.

---

## Objective
To build an end-to-end machine learning pipeline that:
- Handles missing data effectively
- Encodes categorical variables
- Scales numerical features
- Trains a robust classification model
- Evaluates performance using multiple metrics

---

## Dataset
- **train_data.csv** – Used for training and evaluation  
- **test_data.csv** – Used for generating predictions  

### Target Variable
- `Loan_Status`
  - `1` → Loan Approved  
  - `0` → Loan Rejected  

---

## Methodology

### 1. Data Loading & Exploration
- Loaded datasets using Pandas
- Analyzed class distribution of `Loan_Status`
- Identified missing values and feature types
- Performed crosstab analysis on `Credit_History`

---

### 2. Data Preprocessing
- Dropped non-informative column (`Loan_ID`)
- Missing value treatment:
  - **Categorical features** → Mode
  - **Numerical features** → Median
- Binary categorical encoding:
  - Gender, Married, Self_Employed, Education
- Converted `Dependents` into numeric format
- One-Hot Encoding applied to `Property_Area`
- Ensured train-test feature alignment

---

### 3. Feature Scaling
- Standardized numerical features using **StandardScaler**
- Prevented data leakage by fitting scaler only on training data

---

### 4. Model Training
- Algorithm: **Logistic Regression**
- Handled class imbalance using `class_weight='balanced'`
- Increased iteration limit to ensure convergence

---

### 5. Model Evaluation
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC Curve and AUC Score

---

## Results
- Model successfully learned decision boundaries
- ROC-AUC score indicates good class separation
- Credit history emerged as a strong predictive feature

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## How to Run
```bash
pip install pandas scikit-learn matplotlib
Project Structure
├── train_data.csv
├── test_data.csv
├── model.py
└── README.md
Author

Anshu Sharma
B.Tech (CSE) | Machine Learning & Data Science
