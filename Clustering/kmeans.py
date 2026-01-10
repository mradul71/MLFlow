# ==================================
# K-Means Clustering (CSV)
# ==================================

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.data

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data.csv")

# Clustering → NO target column
X = df.select_dtypes(include=[np.number])

# -------------------------
# Scaling (IMPORTANT)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Train Model
# -------------------------
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# -------------------------
# Evaluation
# -------------------------
labels = kmeans.labels_
silhouette = silhouette_score(X_scaled, labels)

# -------------------------
# MLflow Logging
# -------------------------
mlflow.set_experiment("Clustering")

input_example = X.iloc[:5]

with mlflow.start_run(run_name="KMeans"):
    mlflow.log_param("model_type", "KMeans")
    mlflow.log_param("n_clusters", 3)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_init", 10)

    mlflow.log_metric("silhouette_score", silhouette)

    mlflow.log_artifact("data.csv", artifact_path="data")

    mlflow.sklearn.log_model(
        kmeans,
        name="model",
        input_example=input_example
    )

print("K-Means clustering & logging complete.")
