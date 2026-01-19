# ==================================
# LDA (Latent Dirichlet Allocation)
# ==================================

import pandas as pd
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.data

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data.csv")

# CHANGE THIS to your text column
TEXT_COLUMN = "text"

documents = df[TEXT_COLUMN].astype(str).tolist()

# -------------------------
# Vectorization
# -------------------------
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    stop_words="english"
)

X = vectorizer.fit_transform(documents)

# -------------------------
# Train Model
# -------------------------
lda = LatentDirichletAllocation(
    n_components=5,
    learning_method="batch",
    random_state=42
)

lda.fit(X)

# -------------------------
# Evaluation
# -------------------------
perplexity = lda.perplexity(X)

# -------------------------
# MLflow Logging
# -------------------------
mlflow.set_experiment("csv_based_clustering_models")

with mlflow.start_run(run_name="LDA"):
    mlflow.log_param("model_type", "LDA")
    mlflow.log_param("n_topics", 5)
    mlflow.log_param("learning_method", "batch")
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("perplexity", perplexity)

    mlflow.log_artifact("data.csv", artifact_path="data")

    mlflow.sklearn.log_model(
        lda,
        name="model",
        input_example=X[:5]
    )

print("LDA topic modeling & logging complete.")
