# ==================================
# One-vs-Rest Classifier (CSV)
# ==================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.data

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data.csv")

X = df.drop("target", axis=1)   # CHANGE target column if needed
y = df["target"]

# -------------------------
# Train / Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train Model
# Base estimator: Logistic Regression
# -------------------------
base_model = LogisticRegression(max_iter=1000)

model = OneVsRestClassifier(base_model)
model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# -------------------------
# MLflow Logging
# -------------------------
mlflow.set_experiment("csv_based_classification_models")

input_example = X_train.iloc[:5]

with mlflow.start_run(run_name="One_vs_Rest_Classifier"):
    mlflow.log_param("model_type", "OneVsRestClassifier")
    mlflow.log_param("base_estimator", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision_weighted", precision)
    mlflow.log_metric("Recall_weighted", recall)
    mlflow.log_metric("F1_weighted", f1)

    mlflow.log_artifact("data.csv", artifact_path="data")

    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=input_example
    )

print("One-vs-Rest Classifier training & logging complete.")
