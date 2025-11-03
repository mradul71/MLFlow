import os
import tempfile
import warnings

import mlflow
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# --- Streamlit Dashboard ---
st.set_page_config(page_title="MLflow + SHAP Explainability", layout="wide")
st.title("📊 MLflow Model Dashboard with SHAP Explainability")

# --- Sidebar: MLflow settings ---
st.sidebar.header("MLflow Settings")
mlflow_tracking_uri = st.sidebar.text_input("Tracking URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

experiment_name = st.sidebar.text_input("Experiment Name")
artifact_subpath = st.sidebar.text_input("Model Artifact Path", "model")
max_rows = st.sidebar.number_input("Max rows to explain (sampling)", min_value=50, max_value=10000, value=1000, step=50)
summary_bar = st.sidebar.checkbox("Use SHAP summary bar plot", value=False)
sample_force = st.sidebar.number_input("Force plot sample index", min_value=0, value=0, step=1)

runs = []
if experiment_name:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(experiment.experiment_id)
    else:
        st.error("Experiment not found")

if isinstance(runs, pd.DataFrame) and not runs.empty:
    run_id = st.sidebar.selectbox("Select Run", runs["run_id"])
    model_uri = f"runs:/{run_id}/{artifact_subpath}"

    # Try sklearn flavor first, fallback to generic loader
    model = None
    load_error = None
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e1:
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e2:
            load_error = (e1, e2)

    if model is None:
        st.error(f"Could not load model from {model_uri}. Check artifact path/flavor. Errors: {load_error}")
        st.stop()

    st.success(f"Loaded Model from Run ID: {run_id}")

    # --- Load Sample Data ---
    st.header("Step 1: Provide Sample Data")
    uploaded_file = st.file_uploader("Upload a CSV file for explanations", type=["csv"])
    if uploaded_file:
        X_full = pd.read_csv(uploaded_file)
        st.write("Sample Data Preview:", X_full.head())

        # Keep only numeric/object columns compatible with model input where possible
        # If your pipeline includes encoders, the model may accept raw features.
        # Otherwise, adapt preprocessing to match training pipeline.
        X = X_full.copy()

        # Safety: sample rows for speed
        if len(X) > max_rows:
            X = X.sample(n=max_rows, random_state=42).reset_index(drop=True)

        st.caption(f"Explaining {len(X)} rows (sampled if original > {max_rows}).")

        # --- SHAP Explainability ---
        st.header("Step 2: SHAP Explainability")

        # Try TreeExplainer first (fast and exact for tree models)
        explainer = None
        shap_values = None
        explanation = None
        tree_ok = False
        try:
            explainer = shap.TreeExplainer(model)  # works for XGBoost/LightGBM/CatBoost/sklearn-tree
            # Optional: provide background data for better performance/stability
            background = shap.sample(X, min(200, len(X))) if len(X) > 200 else X
            shap_values = explainer.shap_values(X)  # legacy array API
            tree_ok = True
        except Exception:
            # General fallback: shap.Explainer with model callable
            try:
                # If mlflow.pyfunc model, wrap prediction function
                def predict_fn(data):
                    try:
                        return model.predict(data)
                    except Exception:
                        # Some pyfunc models return DataFrame/Series or need numpy
                        return model.predict(pd.DataFrame(data, columns=X.columns))

                explainer = shap.Explainer(predict_fn, X)  # auto selects partitioning
                explanation = explainer(X)  # returns shap.Explanation
            except Exception as e:
                st.error(f"Failed to create SHAP explainer: {e}")
                st.stop()

        # --- SHAP Summary Plot ---
        st.subheader("SHAP Summary Plot")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.figure(figsize=(9, 5))
            try:
                if tree_ok and shap_values is not None:
                    # summary_plot with array API
                    if summary_bar:
                        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                    else:
                        shap.summary_plot(shap_values, X, show=False)
                else:
                    # explanation object API
                    if summary_bar:
                        shap.summary_plot(explanation, X, plot_type="bar", show=False)
                    else:
                        shap.summary_plot(explanation, X, show=False)
                plt.tight_layout()
                plt.savefig(tmpfile.name, dpi=160, bbox_inches="tight")
                st.image(tmpfile.name, caption="SHAP Summary Plot")
            finally:
                plt.close()
                os.unlink(tmpfile.name)

        # --- SHAP Force Plot (first sample) ---
        st.subheader("SHAP Force Plot (Single Prediction)")
        shap.initjs()  # load JS assets
        try:
            if tree_ok and shap_values is not None:
                # Handle binary/multiclass vs regression shapes
                # Binary/multiclass: shap_values may be [n_classes][n_samples, n_features]
                if isinstance(shap_values, list):
                    class_idx = 0
                    base_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value
                    contrib = shap_values[class_idx][sample_force, :]
                else:
                    base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value[0]
                    contrib = shap_values[sample_force, :]

                fp = shap.force_plot(
                    base_value,
                    contrib,
                    X.iloc[sample_force, :],
                    matplotlib=False
                )
                components.html(shap.getjs() + fp.html(), height=300)
            else:
                # Explanation object API
                # explanation[sample] yields a single-instance explanation
                e_row = explanation[sample_force]
                fp = shap.plots.force(e_row, matplotlib=False, show=False)
                # Newer shap returns a plot object with .html() potentially unavailable; capture via html() if present
                try:
                    html_str = shap.getjs() + fp.html()
                    components.html(html_str, height=300)
                except Exception:
                    # Fallback: render a static matplotlib force plot image
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        plt.figure(figsize=(10, 2.5))
                        shap.plots.force(e_row, matplotlib=True, show=False)
                        plt.tight_layout()
                        plt.savefig(tmpfile.name, dpi=160, bbox_inches="tight")
                        st.image(tmpfile.name, caption="SHAP Force Plot (static)")
                        plt.close()
                        os.unlink(tmpfile.name)
        except Exception as e:
            st.warning(f"Interactive force plot unavailable, showing static plot. Reason: {e}")
            # Static fallback
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.figure(figsize=(10, 2.5))
                try:
                    if tree_ok and shap_values is not None:
                        if isinstance(shap_values, list):
                            class_idx = 0
                            sv = shap_values[class_idx][sample_force, :]
                        else:
                            sv = shap_values[sample_force, :]
                        shap.force_plot(
                            explainer.expected_value if not isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value[0],
                            sv, X.iloc[sample_force, :], matplotlib=True, show=False
                        )
                    else:
                        e_row = explanation[sample_force]
                        shap.plots.force(e_row, matplotlib=True, show=False)
                    plt.tight_layout()
                    plt.savefig(tmpfile.name, dpi=160, bbox_inches="tight")
                    st.image(tmpfile.name, caption="SHAP Force Plot (static)")

                finally:
                    plt.close()
                    os.unlink(tmpfile.name)

else:
    st.info("Please input a valid experiment name to load MLflow runs.")

st.markdown("---")
st.caption("Built with MLflow, SHAP, and Streamlit 🧪")
