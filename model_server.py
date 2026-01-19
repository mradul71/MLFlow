from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import subprocess
import tempfile
import pandas as pd
import shutil
from pathlib import Path
from Models import regression_run_linear, regression_gbt_regressor, regression_decision_tree, regression_random_forest, clustering_bisecting_kmeans, clustering_kmeans, clustering_gaussian_mixture, clustering_lda, classification_decision_tree, classification_random_forest, classification_LogisticRegression, classification_GBT_Classifier, classification_LinearSVC, classification_MLPC, classification_NaiveBayes, classification_OVR

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/run-model', methods=['POST'])
def run_model():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type')
    experiment_type = model_type.split('_')[0]
    run_type = model_type.split('_')[1]
    
    if not model_type:
        return jsonify({'success': False, 'error': 'No model type specified'}), 400
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File must be CSV format'}), 400
    
    run_dir = tempfile.mkdtemp()
    
    file_path = os.path.join('', 'data.csv')
    file.save(file_path)

    df = pd.read_csv(file_path)
    dataset_level = determine_dataset_level(df)
    
    if experiment_type=="Regression" and run_type=="LinearRegression":
        result = run_linear(experiment_type, run_type, dataset_level)
    if experiment_type=="Regression" and run_type=="GBTRegressor":
        result = gbt_regressor(experiment_type, run_type, dataset_level)
    if experiment_type=="Regression" and run_type=="RandomForest":
        result = regression_random_forest(experiment_type, run_type, dataset_level)
    if experiment_type=="Regression" and run_type=="DecisionTree":
        result = regression_decision_tree(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="BisectingKMeans":
        result = clustering_bisecting_kmeans(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="GaussianMixture":
        result = clustering_gaussian_mixture(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="KMeans":
        result = clustering_kmeans(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="LDA":
        result = clustering_lda(experiment_type, "Latent_Dirichlet_Allocation", dataset_level)
    if experiment_type=="Classification" and run_type=="LogisticRegression":
        result = classification_LogisticRegression(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="DecisionTree":
        result = decision_tree(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="RandomForest":
        result = classification_random_forest(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="GBTClassifier":
        result = classification_GBT_Classifier(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="LinearSVC":
        result = classification_LinearSVC(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="MLPC":
        result = classification_MLPC(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="NaiveBayes":
        result = classification_NaiveBayes(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="OVR":
        result = classification_OVR(experiment_type, run_type, dataset_level)
    
    return jsonify({
        'success': True,
        'message': 'Model training completed successfully',
    }), 200

def determine_dataset_level(df):
    has_numeric = False
    has_string = False
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'int32', 'float64', 'float32']:
            has_numeric = True
        elif df[column].dtype == 'object':
            if df[column].astype(str).str.isalpha().any():
                has_string = True
    
    if has_numeric and not has_string:
        return "LEVEL 1"
    elif has_numeric and has_string:
        return "LEVEL 2"
    else:
        return "LEVEL 2"

if __name__ == '__main__':
    print("🚀 Flask Model Training Server starting on http://localhost:5001")
    print("Make sure MLflow is running on http://localhost:8080")
    app.run(debug=True, port=5001)