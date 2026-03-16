# ================================
# 1. Import Libraries
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score
)

# ================================
# 2. Load Dataset
# ================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Target Classes:", data.target_names)
print("Dataset Shape:", X.shape)

# ================================
# 3. Convert to DataFrame (Optional)
# ================================

df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

print("\nFirst 5 Rows:")
print(df.head())

# ================================
# 4. Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# 5. Feature Scaling
# ================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 6. Train Logistic Regression Model
# (Balanced for better recall)
# ================================

model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# ================================
# 7. Predictions
# ================================

y_pred = model.predict(X_test)

# ================================
# 8. Evaluation
# ================================

print("\nModel Performance")
print("------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ================================
# 9. Confusion Matrix
# ================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 10. Improve Recall using Custom Threshold
# ================================

y_probs = model.predict_proba(X_test)[:, 1]

# Lower threshold → higher recall
threshold = 0.4
y_pred_custom = (y_probs > threshold).astype(int)

print("\nAfter Adjusting Threshold to", threshold)
print("Recall :", recall_score(y_test, y_pred_custom))
print("Accuracy :", accuracy_score(y_test, y_pred_custom))

# ================================
# 11. Cross Validation (5-Fold)
# ================================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='recall'
)

print("\nCross Validation Recall Scores:", cv_scores)
print("Average CV Recall:", cv_scores.mean())

# ================================
# 12. Feature Importance (Optional)
# ================================

importance = pd.Series(model.coef_[0], index=data.feature_names)
importance = importance.sort_values(key=abs, ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))