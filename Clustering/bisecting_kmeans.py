# ==================================
# Bisecting K-Means (CSV - sklearn style)
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

# Clustering → numeric features only
X = df.select_dtypes(include=[np.number])

# -------------------------
# Scaling
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Bisecting K-Means Logic
# -------------------------
def bisecting_kmeans(X, n_clusters=3, random_state=42):
    clusters = {0: X}

    for i in range(1, n_clusters):
        # Pick the largest cluster to split
        largest_cluster_key = max(clusters, key=lambda k: len(clusters[k]))
        data_to_split = clusters.pop(largest_cluster_key)

        kmeans = KMeans(
            n_clusters=2,
            random_state=random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(data_to_split)

        clusters[f"{largest_cluster_key}_0"] = data_to_split[labels == 0]
        clusters[f"{largest_cluster_key}_1"] = data_to_split[labels == 1]

    return clusters

# -------------------------
# Train Model
# -------------------------
clusters = bisecting_kmeans(X_scaled, n_clusters=3)

# Build labels array
labels = np.zeros(len(X_scaled), dtype=int)
current_label = 0
start_idx = 0

for cluster_data in clusters.values():
    size = len(cluster_data)
    labels[start_idx:start_idx + size] = current_label
    start_idx += size
    current_label += 1

# -------------------------
# Evaluation
# -------------------------
silhouette = silhouette_score(X_scaled, labels)

# -------------------------
# MLflow Logging
# -------------------------
mlflow.set_experiment("Clustering")

with mlflow.start_run(run_name="Bisecting_KMeans"):
    mlflow.log_param("model_type", "BisectingKMeans")
    mlflow.log_param("n_clusters", 3)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("silhouette_score", silhouette)

    mlflow.log_artifact("data.csv", artifact_path="data")

print("Bisecting K-Means clustering & logging complete.")
