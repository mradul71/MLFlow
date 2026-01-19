# ==================================
# Linear SVC (CSV)
# ==================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
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
# -------------------------
model = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# -------------------------
# MLflow Logging
# -------------------------
mlflow.set_experiment("csv_based_classification_models")

input_example = X_train.iloc[:5]

with mlflow.start_run(run_name="Linear_SVC"):
    mlflow.log_param("model_type", "LinearSVC")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 5000)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1", f1)

    mlflow.log_artifact("data.csv", artifact_path="data")

    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=input_example
    )

print("Linear SVC training & logging complete.")
