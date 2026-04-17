###Breast Cancer Classification Model
This project implements a Logistic Regression model to classify tumors as Malignant (1) or Benign (0) using medical diagnostic features.

###Dataset
File: data.csv
Target column: target
0 → Benign
1 → Malignant

###Approach
Data preprocessing and feature scaling using StandardScaler
Train-test split (80-20, stratified)
Logistic Regression with:
class_weight='balanced'
max_iter=5000

###Model evaluation using:
Accuracy: 0.95   
After Adjusting threshold ;Accuracy = 0.98
Recall (primary metric):0.94
AfterAdjusting the threshold; Recall=0.98
Classification Report
                   precision    recall     f1-score   support

           0          0.91      0.98      0.94        42
           1          0.99      0.94      0.96        72
Confusion Matrix
5-Fold Cross-Validation (Recall): 0.96

###Objective
Since this is a healthcare-related problem, the model prioritizes high recall to reduce false negatives and correctly detect malignant tumors.

###Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

Anshu Sharma
Machine Learning Intern | B.Tech Student