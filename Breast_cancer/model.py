# ─── Imports 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, accuracy_score, roc_auc_score, RocCurveDisplay
)

# ─── Load & Clean Data ─────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")

# Drop empty and irrelevant columns
df.drop(columns=["Unnamed: 32", "id"], inplace=True, errors="ignore")
df.dropna(inplace=True)

print("Shape    :", df.shape)
print("Nulls    :", df.isnull().sum().sum())
print(df.head())

# ─── Features & Target ────────────────────────────────────────────────────────
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"].map({"M": 1, "B": 0})

print("\nClass distribution:\n", y.value_counts())

# ─── Train/Test Split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ─── Build Pipeline (prevents data leakage) ───────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=5000, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred      = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nModel Performance")
print("---------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("AUC-ROC  :", roc_auc_score(y_test, y_pred_prob))   # ← added
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ─── ROC Curve ────────────────────────────────────────────────────────────────
RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# ─── Cross Validation (no leakage — pipeline handles scaling inside CV) ───────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="recall")

print("\nCV Recall Scores :", cv_scores)
print("Mean CV Recall   :", cv_scores.mean().round(4))
print("Std  CV Recall   :", cv_scores.std().round(4))     # ← added: check stability

# ─── Save Pipeline (scaler + model together) 
joblib.dump(pipeline,             "pipeline.pkl")
joblib.dump(X.columns.tolist(),   "feature_names.pkl")
print("\nPipeline and feature names saved.")

# ─── Load & Predict on New Data ───────────────────────────────────────────────
pipeline      = joblib.load("pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

new_data = pd.read_csv("data.csv")          # ← change as needed
new_data = new_data[feature_names]              # ensure correct column order

predictions  = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]

print("\nPredictions   :", predictions)           # 0 = Benign, 1 = Malignant
print("Probabilities :", probabilities.round(3))