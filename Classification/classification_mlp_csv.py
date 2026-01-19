from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# MLFLOW
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Load data
# =========================
df = pd.read_csv("data.csv")

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Model pipeline
# (scaling is IMPORTANT for MLP)
# =========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# =========================
# MLflow
# =========================
mlflow.set_experiment("csv_based_mlp_classifier")
input_example = X_train.iloc[:5]

with mlflow.start_run():
    mlflow.log_param("model_type", "MultilayerPerceptronClassifier")
    mlflow.log_param("hidden_layers", "(100, 50)")
    mlflow.log_param("activation", "relu")
    mlflow.log_param("solver", "adam")
    mlflow.log_param("max_iter", 500)

    preds = pipeline.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds, average="weighted"))
    mlflow.log_metric("recall", recall_score(y_test, preds, average="weighted"))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))

    mlflow.log_artifact("data.csv", artifact_path="data")

    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        input_example=input_example
    )
