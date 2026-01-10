# ==================================
# Gaussian Mixture Clustering (CSV)
# ==================================

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
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
# Train Model
# -------------------------
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42
)

gmm.fit(X_scaled)

# -------------------------
# Evaluation
# -------------------------
labels = gmm.predict(X_scaled)
silhouette = silhouette_score(X_scaled, labels)

# -------------------------
# MLflow Logging
# -------------------------
mlflow.set_experiment("Clustering")

input_example = X.iloc[:5]

with mlflow.start_run(run_name="Gaussian_Mixture"):
    mlflow.log_param("model_type", "GaussianMixture")
    mlflow.log_param("n_components", 3)
    mlflow.log_param("covariance_type", "full")
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("silhouette_score", silhouette)

    mlflow.log_artifact("data.csv", artifact_path="data")

    mlflow.sklearn.log_model(
        gmm,
        name="model",
        input_example=input_example
    )

print("Gaussian Mixture clustering & logging complete.")
