// Application Data - Using the provided experiment data with Ryan instead of Alice
const experimentsData = {
  "experiments": [
    {"id": "run_001", "name": "LogisticRegression_Iris_1", "status": "FINISHED", "user": "Ryan", "model_type": "LogisticRegression", "dataset": "Iris", "start_time": "2025-08-20 08:47:30", "duration": "87s", "parameters": {"C": 0.71}, "metrics": {"accuracy": 0.7623, "precision": 0.778, "recall": 0.7375, "f1_score": 0.7584, "auc_roc": 0.7538}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_002", "name": "DecisionTree_Diabetes_2", "status": "FINISHED", "user": "Bob", "model_type": "DecisionTree", "dataset": "Diabetes", "start_time": "2025-08-26 11:23:14", "duration": "183s", "parameters": {"max_depth": 8}, "metrics": {"accuracy": 0.8324, "precision": 0.8156, "recall": 0.8654, "f1_score": 0.8226, "auc_roc": 0.8551}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png"]},
    {"id": "run_003", "name": "RandomForest_Titanic_3", "status": "FINISHED", "user": "Charlie", "model_type": "RandomForest", "dataset": "Titanic", "start_time": "2025-08-28 15:39:58", "duration": "244s", "parameters": {"learning_rate": 0.131, "n_estimators": 156, "max_depth": 9}, "metrics": {"accuracy": 0.9018, "precision": 0.9095, "recall": 0.8723, "f1_score": 0.899, "auc_roc": 0.9156}, "fairness_metrics": {"demographic_parity_difference": 0.0742, "equalized_odds_difference": -0.0231, "equal_opportunity_difference": 0.0856}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png", "fairness_report.html"]},
    {"id": "run_004", "name": "XGBoost_Wine Quality_4", "status": "FINISHED", "user": "Diana", "model_type": "XGBoost", "dataset": "Wine Quality", "start_time": "2025-09-01 09:15:42", "duration": "156s", "parameters": {"learning_rate": 0.087, "n_estimators": 312, "max_depth": 11}, "metrics": {"accuracy": 0.8913, "precision": 0.8734, "recall": 0.9156, "f1_score": 0.8876, "auc_roc": 0.8945}, "fairness_metrics": {"demographic_parity_difference": -0.1102, "equalized_odds_difference": 0.0398, "equal_opportunity_difference": -0.0234}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_005", "name": "SVM_Iris_5", "status": "FINISHED", "user": "Eva", "model_type": "SVM", "dataset": "Iris", "start_time": "2025-09-03 14:28:17", "duration": "91s", "parameters": {"C": 4.23, "kernel": "rbf"}, "metrics": {"accuracy": 0.8241, "precision": 0.8456, "recall": 0.7923, "f1_score": 0.8167, "auc_roc": 0.8376}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_006", "name": "LogisticRegression_Diabetes_6", "status": "FINISHED", "user": "Ryan", "model_type": "LogisticRegression", "dataset": "Diabetes", "start_time": "2025-09-05 12:10:33", "duration": "76s", "parameters": {"C": 2.89}, "metrics": {"accuracy": 0.7923, "precision": 0.8134, "recall": 0.7656, "f1_score": 0.7845, "auc_roc": 0.8123}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png"]},
    {"id": "run_007", "name": "RandomForest_Wine Quality_7", "status": "FINISHED", "user": "Bob", "model_type": "RandomForest", "dataset": "Wine Quality", "start_time": "2025-09-07 16:45:22", "duration": "203s", "parameters": {"learning_rate": 0.234, "n_estimators": 287, "max_depth": 6}, "metrics": {"accuracy": 0.8567, "precision": 0.8789, "recall": 0.8345, "f1_score": 0.8523, "auc_roc": 0.8678}, "fairness_metrics": {"demographic_parity_difference": 0.1234, "equalized_odds_difference": -0.0567, "equal_opportunity_difference": 0.0789}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png", "fairness_report.html"]},
    {"id": "run_016", "name": "RandomForest_Diabetes_16", "status": "FINISHED", "user": "Ryan", "model_type": "RandomForest", "dataset": "Diabetes", "start_time": "2025-09-17 20:08:29", "duration": "211s", "parameters": {"learning_rate": 0.089, "n_estimators": 356, "max_depth": 10}, "metrics": {"accuracy": 0.8567, "precision": 0.8723, "recall": 0.8412, "f1_score": 0.8556, "auc_roc": 0.8678}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_018", "name": "LogisticRegression_Titanic_18", "status": "RUNNING", "user": "Charlie", "model_type": "LogisticRegression", "dataset": "Titanic", "start_time": "2025-09-17 23:47:15", "duration": "N/A", "parameters": {"C": 3.45}, "metrics": {"accuracy": null, "precision": null, "recall": null, "f1_score": null, "auc_roc": null}, "fairness_metrics": {}, "artifacts": []},
    {"id": "run_019", "name": "DecisionTree_Wine Quality_19", "status": "FAILED", "user": "Diana", "model_type": "DecisionTree", "dataset": "Wine Quality", "start_time": "2025-09-17 23:58:42", "duration": "N/A", "parameters": {"max_depth": 3}, "metrics": {"accuracy": null, "precision": null, "recall": null, "f1_score": null, "auc_roc": null}, "fairness_metrics": {}, "artifacts": []},
    {"id": "run_020", "name": "SVM_Diabetes_20", "status": "RUNNING", "user": "Eva", "model_type": "SVM", "dataset": "Diabetes", "start_time": "2025-09-17 23:59:18", "duration": "N/A", "parameters": {"C": 5.67, "kernel": "rbf"}, "metrics": {"accuracy": null, "precision": null, "recall": null, "f1_score": null, "auc_roc": null}, "fairness_metrics": {}, "artifacts": []}
  ],
  "feature_importance": {
    "run_001": {"feature_importance": {"petal_width": 0.5636, "sepal_length": 0.3756, "sepal_width": 0.0335, "petal_length": 0.0273}, "top_features": [["petal_width", 0.5636], ["sepal_length", 0.3756], ["sepal_width", 0.0335], ["petal_length", 0.0273]], "shap_values": {"petal_width": 0.4234, "sepal_length": -0.2145, "sepal_width": 0.1876, "petal_length": 0.3456}},
    "run_002": {"feature_importance": {"glucose": 0.6234, "bmi": 0.1892, "age": 0.0956, "blood_pressure": 0.0535, "skin_thickness": 0.0246, "insulin": 0.0093, "pregnancies": 0.0044}, "top_features": [["glucose", 0.6234], ["bmi", 0.1892], ["age", 0.0956], ["blood_pressure", 0.0535], ["skin_thickness", 0.0246]], "shap_values": {"glucose": 0.5123, "bmi": -0.1456, "age": 0.2134, "blood_pressure": -0.0876, "skin_thickness": 0.1234}},
    "run_003": {"feature_importance": {"Sex": 0.4567, "Fare": 0.2341, "Age": 0.1456, "Pclass": 0.0823, "SibSp": 0.0456, "Parch": 0.0234, "Embarked": 0.0123}, "top_features": [["Sex", 0.4567], ["Fare", 0.2341], ["Age", 0.1456], ["Pclass", 0.0823], ["SibSp", 0.0456]], "shap_values": {"Sex": 0.3876, "Fare": -0.1987, "Age": 0.2567, "Pclass": -0.1234, "SibSp": 0.0987}},
    "run_004": {"feature_importance": {"alcohol": 0.3456, "volatile_acidity": 0.2123, "citric_acid": 0.1789, "density": 0.1234, "sulphates": 0.0789, "pH": 0.0456, "residual_sugar": 0.0153}, "top_features": [["alcohol", 0.3456], ["volatile_acidity", 0.2123], ["citric_acid", 0.1789], ["density", 0.1234], ["sulphates", 0.0789]], "shap_values": {"alcohol": 0.2876, "volatile_acidity": -0.1654, "citric_acid": 0.1987, "density": -0.0987, "sulphates": 0.1456}},
    "run_005": {"feature_importance": {"petal_length": 0.4123, "petal_width": 0.3456, "sepal_length": 0.1789, "sepal_width": 0.0632}, "top_features": [["petal_length", 0.4123], ["petal_width", 0.3456], ["sepal_length", 0.1789], ["sepal_width", 0.0632]], "shap_values": {"petal_length": 0.3654, "petal_width": -0.2876, "sepal_length": 0.1987, "sepal_width": 0.0876}}
  },
  "shap_data": {
    "base_value": 0.5234,
    "sample_predictions": [
      {"features": {"petal_width": 2.4, "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}, "shap_values": {"petal_width": 0.4234, "sepal_length": -0.2145, "sepal_width": 0.1876, "petal_length": 0.3456}, "prediction": 0.8765},
      {"features": {"glucose": 148, "bmi": 33.6, "age": 50, "blood_pressure": 72, "skin_thickness": 35}, "shap_values": {"glucose": 0.5123, "bmi": -0.1456, "age": 0.2134, "blood_pressure": -0.0876, "skin_thickness": 0.1234}, "prediction": 0.7234}
    ]
  }
};

// Global state
let currentView = 'dashboard';
let filteredExperiments = [...experimentsData.experiments];
let selectedExperiments = [];
let compareMode = false;
let charts = {};
let currentModel = null;
let uploadedData = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM Content Loaded - initializing application');
  initializeNavigation();
  initializeDashboard();
  initializeExperiments();
  setupEventListeners();
  
  // Make functions globally accessible
  window.showView = showView;
  window.showExperimentDetail = showExperimentDetail;
  window.showComparison = showComparison;
  window.clearSelection = clearSelection;
  window.handleExperimentSelection = handleExperimentSelection;
  window.updateCompareMode = updateCompareMode;
  window.loadModel = loadModel;
  window.generateShapAnalysis = generateShapAnalysis;
  window.switchFeatureTab = switchFeatureTab;
  
  console.log('Application initialized successfully');
});

// Navigation
function initializeNavigation() {
  console.log('Initializing navigation');
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    // Remove any existing event listeners
    link.removeEventListener('click', handleNavClick);
    // Add new event listener
    link.addEventListener('click', handleNavClick);
  });
  
  console.log(`Added event listeners to ${navLinks.length} navigation links`);
}

function handleNavClick(e) {
  e.preventDefault();
  e.stopPropagation();
  
  const view = e.currentTarget.getAttribute('data-view');
  console.log('Navigation clicked:', view, e.currentTarget);
  
  if (view) {
    showView(view);
  } else {
    console.error('No view attribute found on navigation link');
  }
}

function showView(viewName) {
  console.log('Showing view:', viewName);
  
  // Hide all views
  document.querySelectorAll('.view').forEach(view => {
    view.classList.remove('active');
  });
  
  // Show selected view
  const targetView = document.getElementById(`${viewName}-view`);
  if (targetView) {
    targetView.classList.add('active');
    console.log(`Successfully showed view: ${viewName}`);
  } else {
    console.error(`Could not find view element: ${viewName}-view`);
    return;
  }
  
  // Update navigation active state
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
  });
  
  const activeNavLink = document.querySelector(`[data-view="${viewName}"]`);
  if (activeNavLink) {
    activeNavLink.classList.add('active');
  } else {
    console.error(`Could not find nav link for view: ${viewName}`);
  }
  
  currentView = viewName;
  
  // Initialize view-specific content
  if (viewName === 'dashboard') {
    initializeDashboard();
  } else if (viewName === 'experiments') {
    renderExperimentsTable();
  } else if (viewName === 'architecture') {
    initializeArchitecture();
  } else if (viewName === 'fairness') {
    initializeFairness();
  } else if (viewName === 'shap') {
    initializeShap();
  }
}

// SHAP Analysis functionality
function initializeShap() {
  console.log('Initializing SHAP view');
  setupShapEventListeners();
  populateRunSelector();
  generateDefaultShapVisualizations();
}

function setupShapEventListeners() {
  console.log('Setting up SHAP event listeners');
  
  // File upload
  const fileInput = document.getElementById('file-input');
  const fileUploadArea = document.getElementById('file-upload-area');
  
  if (fileUploadArea && fileInput) {
    // Remove existing listeners first
    fileUploadArea.removeEventListener('click', handleUploadClick);
    fileUploadArea.removeEventListener('dragover', handleDragOver);
    fileUploadArea.removeEventListener('drop', handleFileDrop);
    fileUploadArea.removeEventListener('dragleave', handleDragLeave);
    fileInput.removeEventListener('change', handleFileSelect);
    
    // Add new listeners
    fileUploadArea.addEventListener('click', handleUploadClick);
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('drop', handleFileDrop);
    fileUploadArea.addEventListener('dragleave', handleDragLeave);
    fileInput.addEventListener('change', handleFileSelect);
    
    console.log('File upload event listeners added');
  } else {
    console.error('File upload elements not found');
  }

  // Experiment name input
  const experimentNameInput = document.getElementById('experiment-name-input');
  if (experimentNameInput) {
    experimentNameInput.removeEventListener('input', updateRunSelector);
    experimentNameInput.addEventListener('input', updateRunSelector);
    console.log('Experiment name input listener added');
  }
}

function handleUploadClick(e) {
  e.preventDefault();
  console.log('Upload area clicked');
  const fileInput = document.getElementById('file-input');
  if (fileInput) {
    fileInput.click();
  }
}

function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
}

function handleFileSelect(e) {
  console.log('File selected:', e.target.files);
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
}

function handleFile(file) {
  console.log('Processing file:', file.name);
  
  if (!file.name.endsWith('.csv')) {
    alert('Please upload a CSV file');
    return;
  }
  
  // Mock file processing
  const reader = new FileReader();
  reader.onload = function(e) {
    const csv = e.target.result;
    const lines = csv.split('\n');
    const headers = lines[0].split(',');
    const rows = lines.slice(1, 6).map(line => line.split(',')); // Show first 5 rows
    
    displayDataPreview(headers, rows);
    uploadedData = { headers, rows: lines.slice(1).map(line => line.split(',')) };
    console.log('File processed successfully');
  };
  reader.readAsText(file);
}

function displayDataPreview(headers, rows) {
  const dataPreview = document.getElementById('data-preview');
  const table = document.getElementById('data-preview-table');
  
  if (dataPreview && table) {
    dataPreview.classList.remove('hidden');
    
    table.innerHTML = `
      <thead>
        <tr>
          ${headers.map(header => `<th>${header.trim()}</th>`).join('')}
        </tr>
      </thead>
      <tbody>
        ${rows.map(row => `
          <tr>
            ${row.map(cell => `<td>${cell.trim()}</td>`).join('')}
          </tr>
        `).join('')}
      </tbody>
    `;
    
    console.log('Data preview displayed');
  }
}

function updateRunSelector() {
  const experimentName = document.getElementById('experiment-name-input')?.value;
  const runSelector = document.getElementById('run-selector');
  
  if (!runSelector) return;
  
  if (!experimentName) {
    runSelector.innerHTML = '<option value="">Select a run...</option>';
    return;
  }
  
  // Mock filtering based on experiment name
  const matchingRuns = experimentsData.experiments.filter(exp => 
    exp.name.toLowerCase().includes(experimentName.toLowerCase()) ||
    exp.dataset.toLowerCase().includes(experimentName.toLowerCase()) ||
    exp.model_type.toLowerCase().includes(experimentName.toLowerCase())
  );
  
  runSelector.innerHTML = '<option value="">Select a run...</option>' +
    matchingRuns.map(exp => 
      `<option value="${exp.id}">${exp.name} (${exp.id})</option>`
    ).join('');
    
  console.log('Run selector updated with', matchingRuns.length, 'matching runs');
}

function populateRunSelector() {
  const runSelector = document.getElementById('run-selector');
  if (!runSelector) return;
  
  runSelector.innerHTML = '<option value="">Select a run...</option>' +
    experimentsData.experiments
      .filter(exp => exp.status === 'FINISHED')
      .map(exp => `<option value="${exp.id}">${exp.name} (${exp.id})</option>`)
      .join('');
      
  console.log('Run selector populated');
}

function loadModel() {
  console.log('Load model button clicked');
  const runId = document.getElementById('run-selector')?.value;
  const modelStatus = document.getElementById('model-status');
  
  if (!runId) {
    alert('Please select a run first');
    return;
  }
  
  // Mock model loading
  currentModel = experimentsData.experiments.find(exp => exp.id === runId);
  
  if (modelStatus) {
    modelStatus.classList.remove('hidden');
    modelStatus.innerHTML = `
      <div class="status-indicator">
        <span class="status-icon">✅</span>
        <span class="status-text">Model loaded successfully: ${currentModel.name}</span>
      </div>
    `;
    console.log('Model loaded:', currentModel.name);
  }
}

function generateShapAnalysis() {
  console.log('Generate SHAP analysis button clicked');
  
  if (!currentModel) {
    alert('Please load a model first');
    return;
  }
  
  if (!uploadedData) {
    alert('Please upload sample data first');
    return;
  }
  
  const analysisType = document.getElementById('shap-analysis-type')?.value || 'summary';
  console.log('Generating SHAP analysis of type:', analysisType);
  
  // Generate SHAP visualizations based on analysis type
  setTimeout(() => {
    generateShapSummaryChart();
    generateShapForceChart();
    generateShapValuesTable();
    console.log('SHAP analysis completed');
  }, 500);
}

function generateDefaultShapVisualizations() {
  setTimeout(() => {
    generateShapSummaryChart();
    generateShapForceChart();
    generateShapValuesTable();
    console.log('Default SHAP visualizations generated');
  }, 500);
}

function generateShapSummaryChart() {
  const ctx = document.getElementById('shap-summary-chart');
  if (!ctx) return;
  
  // Mock SHAP summary data
  const features = ['petal_width', 'sepal_length', 'sepal_width', 'petal_length'];
  const shapValues = [0.4234, -0.2145, 0.1876, 0.3456];
  
  if (charts.shapSummary) {
    charts.shapSummary.destroy();
  }
  
  charts.shapSummary = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: features,
      datasets: [{
        label: 'SHAP Values',
        data: shapValues.map(Math.abs),
        backgroundColor: shapValues.map(val => val >= 0 ? '#1FB8CD' : '#B4413C'),
        borderColor: shapValues.map(val => val >= 0 ? '#1FB8CD' : '#B4413C'),
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Mean |SHAP Value|'
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: 'SHAP Summary Plot - Feature Importance'
        }
      }
    }
  });
}

function generateShapForceChart() {
  const ctx = document.getElementById('shap-force-chart');
  if (!ctx) return;
  
  // Mock SHAP force plot data (waterfall chart)
  const features = ['Base Value', 'petal_width', 'sepal_length', 'sepal_width', 'petal_length', 'Prediction'];
  const values = [0.5234, 0.4234, -0.2145, 0.1876, 0.3456, 0.8765];
  
  if (charts.shapForce) {
    charts.shapForce.destroy();
  }
  
  charts.shapForce = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: features,
      datasets: [{
        label: 'SHAP Force Plot',
        data: values,
        backgroundColor: values.map((val, idx) => {
          if (idx === 0 || idx === values.length - 1) return '#ECEBD5';
          return val >= 0 ? '#1FB8CD' : '#B4413C';
        }),
        borderColor: values.map((val, idx) => {
          if (idx === 0 || idx === values.length - 1) return '#ECEBD5';
          return val >= 0 ? '#1FB8CD' : '#B4413C';
        }),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          title: {
            display: true,
            text: 'Output'
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: 'SHAP Force Plot - Individual Prediction Explanation'
        }
      }
    }
  });
}

function generateShapValuesTable() {
  const table = document.getElementById('shap-values-table');
  if (!table) return;
  
  // Mock SHAP values data
  const shapData = [
    { feature: 'petal_width', shap_value: 0.4234, feature_value: 2.4, contribution: 'High Positive' },
    { feature: 'sepal_length', shap_value: -0.2145, feature_value: 5.1, contribution: 'Negative' },
    { feature: 'sepal_width', shap_value: 0.1876, feature_value: 3.5, contribution: 'Positive' },
    { feature: 'petal_length', shap_value: 0.3456, feature_value: 1.4, contribution: 'Positive' }
  ];
  
  const tbody = table.querySelector('tbody');
  if (tbody) {
    tbody.innerHTML = shapData.map(row => `
      <tr>
        <td>${row.feature}</td>
        <td class="${row.shap_value >= 0 ? 'shap-value-positive' : 'shap-value-negative'}">
          ${row.shap_value.toFixed(4)}
        </td>
        <td>${row.feature_value}</td>
        <td>
          ${row.contribution}
          <div class="contribution-bar">
            <div class="${row.shap_value >= 0 ? 'contribution-positive' : 'contribution-negative'}" 
                 style="width: ${Math.abs(row.shap_value) * 100}%"></div>
          </div>
        </td>
      </tr>
    `).join('');
  }
}

// Architecture functionality
function initializeArchitecture() {
  console.log('Initializing architecture view');
  renderArchitectureDiagram();
  setTimeout(() => {
    createDataFlowChart();
    createTechStackChart();
  }, 100);
}

function renderArchitectureDiagram() {
  const diagramContainer = document.getElementById('architecture-diagram');
  if (!diagramContainer) {
    console.log('Architecture diagram container not found');
    return;
  }
  
  diagramContainer.innerHTML = `
    <div class="diagram-layer">
      <div class="diagram-tier">
        <div class="diagram-component ui">
          <span class="component-emoji">🖥️</span>
          <div class="component-name">Enhanced UI</div>
        </div>
      </div>
      
      <div class="diagram-tier">
        <div class="diagram-component api">
          <span class="component-emoji">🔗</span>
          <div class="component-name">API Gateway</div>
        </div>
        <div class="diagram-component explainability">
          <span class="component-emoji">🔍</span>
          <div class="component-name">Explainability Engine</div>
        </div>
        <div class="diagram-component fairness">
          <span class="component-emoji">⚖️</span>
          <div class="component-name">Fairness Monitor</div>
        </div>
      </div>
      
      <div class="diagram-tier">
        <div class="diagram-component mlflow">
          <span class="component-emoji">🎯</span>
          <div class="component-name">MLflow Core</div>
        </div>
        <div class="diagram-component storage">
          <span class="component-emoji">🗄️</span>
          <div class="component-name">Data Storage</div>
        </div>
      </div>
    </div>
  `;
}

function createDataFlowChart() {
  const ctx = document.getElementById('dataFlowChart');
  if (!ctx) return;
  
  if (charts.dataFlow) {
    charts.dataFlow.destroy();
  }
  
  charts.dataFlow = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Model Training', 'Feature Analysis', 'SHAP Computation', 'Fairness Evaluation', 'Result Storage', 'UI Rendering'],
      datasets: [{
        data: [22, 18, 15, 13, 20, 12],
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom'
        },
        title: {
          display: true,
          text: 'Enhanced Data Processing Distribution'
        }
      }
    }
  });
}

function createTechStackChart() {
  const ctx = document.getElementById('techStackChart');
  if (!ctx) return;
  
  if (charts.techStack) {
    charts.techStack.destroy();
  }
  
  charts.techStack = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Frontend', 'Backend', 'ML/AI', 'Explainability', 'Storage', 'Analytics'],
      datasets: [{
        label: 'Technology Components',
        data: [8, 12, 15, 10, 10, 6],
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Number of Components'
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

// Dashboard functionality
function initializeDashboard() {
  renderRecentExperiments();
  setTimeout(() => {
    createSuccessRateChart();
    createPerformanceChart();
  }, 100);
}

function renderRecentExperiments() {
  const tbody = document.getElementById('recent-experiments-tbody');
  if (!tbody) return;
  
  const recentExperiments = experimentsData.experiments.slice(0, 5);
  
  tbody.innerHTML = recentExperiments.map(exp => `
    <tr>
      <td>
        <a href="#" class="experiment-name" onclick="showExperimentDetail('${exp.id}'); return false;">${exp.name}</a>
      </td>
      <td><span class="status-badge ${exp.status.toLowerCase()}">${exp.status}</span></td>
      <td>${exp.model_type}</td>
      <td>${exp.dataset}</td>
      <td>${exp.metrics.accuracy ? (exp.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
      <td>${exp.duration}</td>
      <td>${exp.user}</td>
    </tr>
  `).join('');
}

function createSuccessRateChart() {
  const ctx = document.getElementById('successRateChart');
  if (!ctx) return;
  
  // Generate sample data for success rate over time
  const dates = ['Sep 10', 'Sep 11', 'Sep 12', 'Sep 13', 'Sep 14', 'Sep 15', 'Sep 16', 'Sep 17'];
  const successRates = [82, 85, 87, 83, 89, 91, 88, 85];
  
  if (charts.successRate) {
    charts.successRate.destroy();
  }
  
  charts.successRate = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Success Rate (%)',
        data: successRates,
        borderColor: '#1FB8CD',
        backgroundColor: 'rgba(31, 184, 205, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

function createPerformanceChart() {
  const ctx = document.getElementById('performanceChart');
  if (!ctx) return;
  
  const finishedExperiments = experimentsData.experiments.filter(exp => exp.status === 'FINISHED');
  const modelTypes = [...new Set(finishedExperiments.map(exp => exp.model_type))];
  const avgAccuracy = modelTypes.map(model => {
    const modelExps = finishedExperiments.filter(exp => exp.model_type === model);
    const avgAcc = modelExps.reduce((sum, exp) => sum + exp.metrics.accuracy, 0) / modelExps.length;
    return avgAcc * 100;
  });
  
  if (charts.performance) {
    charts.performance.destroy();
  }
  
  charts.performance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: modelTypes,
      datasets: [{
        label: 'Average Accuracy (%)',
        data: avgAccuracy,
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

// Experiments functionality
function initializeExperiments() {
  filteredExperiments = [...experimentsData.experiments];
  renderExperimentsTable();
}

function renderExperimentsTable() {
  const tbody = document.getElementById('experiments-tbody');
  if (!tbody) return;
  
  tbody.innerHTML = filteredExperiments.map(exp => `
    <tr>
      ${compareMode ? `<td class="compare-column"><input type="checkbox" class="compare-checkbox" value="${exp.id}" onchange="handleExperimentSelection(this)"></td>` : ''}
      <td>
        <a href="#" class="experiment-name" onclick="showExperimentDetail('${exp.id}'); return false;">${exp.name}</a>
      </td>
      <td><span class="status-badge ${exp.status.toLowerCase()}">${exp.status}</span></td>
      <td>${exp.model_type}</td>
      <td>${exp.dataset}</td>
      <td>${exp.metrics.accuracy ? (exp.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
      <td>${exp.metrics.f1_score ? exp.metrics.f1_score.toFixed(3) : 'N/A'}</td>
      <td>${exp.duration}</td>
      <td>${exp.user}</td>
      <td>
        <button class="btn btn--sm btn--outline" onclick="showExperimentDetail('${exp.id}')">View</button>
      </td>
    </tr>
  `).join('');
}

function setupEventListeners() {
  // Search functionality
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('input', handleSearch);
  }
  
  // Filter functionality
  ['status-filter', 'model-filter', 'dataset-filter'].forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      element.addEventListener('change', handleFilters);
    }
  });
}

function handleSearch(e) {
  applyFilters();
}

function handleFilters() {
  applyFilters();
}

function applyFilters() {
  const searchTerm = document.getElementById('search-input')?.value.toLowerCase() || '';
  const statusFilter = document.getElementById('status-filter')?.value || '';
  const modelFilter = document.getElementById('model-filter')?.value || '';
  const datasetFilter = document.getElementById('dataset-filter')?.value || '';
  
  filteredExperiments = experimentsData.experiments.filter(exp => {
    const matchesSearch = exp.name.toLowerCase().includes(searchTerm) || 
                         exp.user.toLowerCase().includes(searchTerm);
    const matchesStatus = !statusFilter || exp.status === statusFilter;
    const matchesModel = !modelFilter || exp.model_type === modelFilter;
    const matchesDataset = !datasetFilter || exp.dataset === datasetFilter;
    
    return matchesSearch && matchesStatus && matchesModel && matchesDataset;
  });
  
  renderExperimentsTable();
}

// Compare mode functionality
function updateCompareMode() {
  const compareBanner = document.getElementById('compare-banner');
  const compareHeader = document.getElementById('compare-header');
  
  if (compareMode) {
    compareBanner?.classList.remove('hidden');
    compareHeader?.classList.remove('hidden');
  } else {
    compareBanner?.classList.add('hidden');
    compareHeader?.classList.add('hidden');
    selectedExperiments = [];
  }
  
  renderExperimentsTable();
  updateSelectedCount();
}

function handleExperimentSelection(checkbox) {
  if (checkbox.checked) {
    if (selectedExperiments.length < 4) {
      selectedExperiments.push(checkbox.value);
    } else {
      checkbox.checked = false;
      alert('You can compare up to 4 experiments at once.');
    }
  } else {
    selectedExperiments = selectedExperiments.filter(id => id !== checkbox.value);
  }
  updateSelectedCount();
}

function updateSelectedCount() {
  const countElement = document.getElementById('selected-count');
  if (countElement) {
    countElement.textContent = `${selectedExperiments.length} selected`;
  }
}

function clearSelection() {
  selectedExperiments = [];
  document.querySelectorAll('.compare-checkbox').forEach(cb => cb.checked = false);
  updateSelectedCount();
}

function showComparison() {
  if (selectedExperiments.length < 2) {
    alert('Please select at least 2 experiments to compare.');
    return;
  }
  
  renderComparisonView();
  showView('comparison');
}

// Feature analysis tabs functionality
function switchFeatureTab(tabName, experimentId) {
  console.log('Switching feature tab:', tabName, 'for experiment:', experimentId);
  
  // Hide all tab contents for this experiment
  document.querySelectorAll(`#experiment-${experimentId} .feature-tab-content`).forEach(content => {
    content.classList.remove('active');
  });
  
  // Show selected tab content
  const targetContent = document.getElementById(`${tabName}-content-${experimentId}`);
  if (targetContent) {
    targetContent.classList.add('active');
  }
  
  // Update tab buttons
  document.querySelectorAll(`#experiment-${experimentId} .feature-tab`).forEach(tab => {
    tab.classList.remove('active');
  });
  
  const targetTab = document.querySelector(`#experiment-${experimentId} .feature-tab[onclick*="${tabName}"]`);
  if (targetTab) {
    targetTab.classList.add('active');
  }
}

// Experiment detail view
function showExperimentDetail(experimentId) {
  console.log('Showing experiment detail for:', experimentId);
  const experiment = experimentsData.experiments.find(exp => exp.id === experimentId);
  if (!experiment) {
    console.error('Experiment not found:', experimentId);
    return;
  }
  
  document.getElementById('experiment-detail-title').textContent = experiment.name;
  
  const content = document.getElementById('experiment-detail-content');
  content.innerHTML = `
    <div class="grid-2">
      <div class="detail-section">
        <h3>Overview</h3>
        <div class="detail-grid">
          <div class="detail-item">
            <span class="detail-label">Run ID</span>
            <span class="detail-value">${experiment.id}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Status</span>
            <span class="detail-value">
              <span class="status-badge ${experiment.status.toLowerCase()}">${experiment.status}</span>
            </span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Model Type</span>
            <span class="detail-value">${experiment.model_type}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Dataset</span>
            <span class="detail-value">${experiment.dataset}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">User</span>
            <span class="detail-value">${experiment.user}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Duration</span>
            <span class="detail-value">${experiment.duration}</span>
          </div>
        </div>
      </div>
      
      <div class="detail-section">
        <h3>Parameters</h3>
        <div class="detail-grid">
          ${Object.entries(experiment.parameters).map(([key, value]) => `
            <div class="detail-item">
              <span class="detail-label">${key}</span>
              <span class="detail-value">${value}</span>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
    
    ${experiment.status === 'FINISHED' ? `
    <div class="detail-section">
      <h3>Performance Metrics</h3>
      <div class="detail-grid">
        ${Object.entries(experiment.metrics).map(([key, value]) => `
          <div class="detail-item">
            <span class="detail-label">${key.replace('_', ' ').toUpperCase()}</span>
            <span class="detail-value metric-value-large">${value ? (value * 100).toFixed(2) + '%' : 'N/A'}</span>
            <div class="metric-progress">
              <div class="metric-progress-fill" style="width: ${value ? value * 100 : 0}%"></div>
            </div>
          </div>
        `).join('')}
      </div>
    </div>` : ''}
    
    ${experimentsData.feature_importance[experimentId] ? `
    <div class="detail-section" id="experiment-${experimentId}">
      <h3>🔍 Enhanced Feature Analysis</h3>
      <p style="color: var(--color-text-secondary); margin-bottom: var(--space-16);">
        Comprehensive feature analysis using traditional importance methods and SHAP explainability
      </p>
      
      <div class="feature-analysis-tabs">
        <button class="feature-tab active" onclick="switchFeatureTab('importance', '${experimentId}')">Feature Importance</button>
        <button class="feature-tab" onclick="switchFeatureTab('shap', '${experimentId}')">SHAP Analysis</button>
      </div>
      
      <div id="importance-content-${experimentId}" class="feature-tab-content active">
        <div class="feature-importance-chart" style="position: relative; height: 300px; margin-bottom: var(--space-24);">
          <canvas id="feature-importance-chart-${experimentId}"></canvas>
        </div>
        <div class="feature-importance-list">
          ${experimentsData.feature_importance[experimentId].top_features.map(([feature, importance]) => `
            <div class="feature-item">
              <span class="feature-name">${feature}</span>
              <div class="feature-bar">
                <div class="feature-bar-fill" style="width: ${importance * 100}%"></div>
              </div>
              <span class="feature-score">${(importance * 100).toFixed(1)}%</span>
            </div>
          `).join('')}
        </div>
      </div>
      
      <div id="shap-content-${experimentId}" class="feature-tab-content">
        <div class="shap-visualizations grid-2">
          <div>
            <h4>SHAP Summary Chart</h4>
            <div style="position: relative; height: 300px;">
              <canvas id="shap-summary-detail-${experimentId}"></canvas>
            </div>
          </div>
          <div>
            <h4>SHAP Force Plot</h4>
            <div style="position: relative; height: 300px;">
              <canvas id="shap-force-detail-${experimentId}"></canvas>
            </div>
          </div>
        </div>
        
        ${experimentsData.feature_importance[experimentId].shap_values ? `
        <div style="margin-top: var(--space-24);">
          <h4>SHAP Values Breakdown</h4>
          <div class="shap-values-list">
            ${Object.entries(experimentsData.feature_importance[experimentId].shap_values).map(([feature, value]) => `
              <div class="feature-item">
                <span class="feature-name">${feature}</span>
                <div class="feature-bar">
                  <div class="feature-bar-fill ${value >= 0 ? '' : 'negative'}" style="width: ${Math.abs(value) * 100}%; background-color: ${value >= 0 ? '#1FB8CD' : '#B4413C'}"></div>
                </div>
                <span class="feature-score ${value >= 0 ? 'shap-value-positive' : 'shap-value-negative'}">${value.toFixed(4)}</span>
              </div>
            `).join('')}
          </div>
        </div>` : ''}
      </div>
    </div>` : ''}
    
    ${Object.keys(experiment.fairness_metrics).length > 0 && experiment.status === 'FINISHED' ? `
    <div class="detail-section">
      <h3>⚖️ Fairness Metrics</h3>
      <div class="detail-grid">
        ${Object.entries(experiment.fairness_metrics).map(([key, value]) => `
          <div class="detail-item">
            <span class="detail-label">${key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
            <span class="detail-value ${Math.abs(value) > 0.1 ? 'metric-change negative' : 'metric-change positive'}">${value ? value.toFixed(4) : 'N/A'}</span>
          </div>
        `).join('')}
      </div>
    </div>` : ''}
    
    <div class="detail-section">
      <h3>Artifacts</h3>
      <div class="artifacts-list">
        ${experiment.artifacts.map(artifact => `
          <a href="#" class="artifact-item" onclick="alert('Download: ${artifact}'); return false;">
            📄 ${artifact}
          </a>
        `).join('')}
      </div>
    </div>
  `;
  
  showView('experiment-detail');
  
  // Create feature importance chart if data exists
  if (experimentsData.feature_importance[experimentId]) {
    setTimeout(() => {
      createFeatureImportanceChart(experimentId);
      createDetailShapCharts(experimentId);
    }, 100);
  }
  
  console.log('Experiment detail view created successfully');
}

function createFeatureImportanceChart(experimentId) {
  const ctx = document.getElementById(`feature-importance-chart-${experimentId}`);
  if (!ctx) return;
  
  const featureData = experimentsData.feature_importance[experimentId];
  const features = featureData.top_features.map(f => f[0]);
  const importances = featureData.top_features.map(f => f[1] * 100);
  
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: features,
      datasets: [{
        label: 'Importance (%)',
        data: importances,
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'],
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `Importance: ${context.parsed.x.toFixed(2)}%`;
            }
          }
        }
      }
    }
  });
}

function createDetailShapCharts(experimentId) {
  const featureData = experimentsData.feature_importance[experimentId];
  if (!featureData.shap_values) return;
  
  // SHAP Summary Chart
  const summaryCtx = document.getElementById(`shap-summary-detail-${experimentId}`);
  if (summaryCtx) {
    const features = Object.keys(featureData.shap_values);
    const shapValues = Object.values(featureData.shap_values);
    
    new Chart(summaryCtx, {
      type: 'bar',
      data: {
        labels: features,
        datasets: [{
          label: 'SHAP Values',
          data: shapValues.map(Math.abs),
          backgroundColor: shapValues.map(val => val >= 0 ? '#1FB8CD' : '#B4413C'),
          borderColor: shapValues.map(val => val >= 0 ? '#1FB8CD' : '#B4413C'),
          borderWidth: 1
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            beginAtZero: true,
            title: { display: true, text: 'Mean |SHAP Value|' }
          }
        },
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'SHAP Feature Importance' }
        }
      }
    });
  }
  
  // SHAP Force Chart
  const forceCtx = document.getElementById(`shap-force-detail-${experimentId}`);
  if (forceCtx) {
    const features = Object.keys(featureData.shap_values);
    const shapValues = Object.values(featureData.shap_values);
    const baseValue = 0.5;
    
    // Create waterfall data
    const waterfallLabels = ['Base Value', ...features, 'Prediction'];
    const waterfallValues = [baseValue, ...shapValues, baseValue + shapValues.reduce((sum, val) => sum + val, 0)];
    
    new Chart(forceCtx, {
      type: 'bar',
      data: {
        labels: waterfallLabels,
        datasets: [{
          label: 'SHAP Force Plot',
          data: waterfallValues,
          backgroundColor: waterfallValues.map((val, idx) => {
            if (idx === 0 || idx === waterfallValues.length - 1) return '#ECEBD5';
            return val >= 0 ? '#1FB8CD' : '#B4413C';
          }),
          borderColor: waterfallValues.map((val, idx) => {
            if (idx === 0 || idx === waterfallValues.length - 1) return '#ECEBD5';
            return val >= 0 ? '#1FB8CD' : '#B4413C';
          }),
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: false, title: { display: true, text: 'Output' } }
        },
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'SHAP Force Plot - Prediction Breakdown' }
        }
      }
    });
  }
}

// Comparison view
function renderComparisonView() {
  const selectedExps = experimentsData.experiments.filter(exp => selectedExperiments.includes(exp.id));
  
  const content = document.getElementById('comparison-content');
  content.innerHTML = `
    <div class="comparison-grid">
      ${selectedExps.map(exp => `
        <div class="comparison-experiment">
          <h3>${exp.name}</h3>
          <div class="comparison-metrics">
            ${renderComparisonMetrics(selectedExps, exp)}
          </div>
        </div>
      `).join('')}
    </div>
    
    <div class="card" style="margin-top: var(--space-32);">
      <div class="card__body">
        <h3>Performance Comparison</h3>
        <div class="chart-container" style="position: relative; height: 400px;">
          <canvas id="comparisonChart"></canvas>
        </div>
      </div>
    </div>
  `;
  
  setTimeout(() => createComparisonChart(selectedExps), 100);
}

function renderComparisonMetrics(allExps, currentExp) {
  if (currentExp.status !== 'FINISHED') {
    return '<p>Experiment not finished</p>';
  }
  
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
  return metrics.map(metric => {
    const value = currentExp.metrics[metric];
    const allValues = allExps
      .filter(exp => exp.status === 'FINISHED')
      .map(exp => exp.metrics[metric])
      .filter(v => v !== null);
    const isBest = allValues.length > 0 && value === Math.max(...allValues);
    
    return `
      <div class="comparison-metric ${isBest ? 'best' : ''}">
        <span class="comparison-metric-name">${metric.replace('_', ' ').toUpperCase()}</span>
        <span class="comparison-metric-value">${value ? (value * 100).toFixed(2) + '%' : 'N/A'}</span>
      </div>
    `;
  }).join('');
}

function createComparisonChart(experiments) {
  const ctx = document.getElementById('comparisonChart');
  if (!ctx) return;
  
  const finishedExps = experiments.filter(exp => exp.status === 'FINISHED');
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
  
  const datasets = finishedExps.map((exp, index) => ({
    label: exp.name,
    data: metrics.map(metric => exp.metrics[metric] * 100),
    borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'][index % 4],
    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'][index % 4] + '20',
    borderWidth: 2,
    fill: false
  }));
  
  new Chart(ctx, {
    type: 'radar',
    data: {
      labels: metrics.map(m => m.replace('_', ' ').toUpperCase()),
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      }
    }
  });
}

// Fairness analysis
function initializeFairness() {
  setTimeout(() => createFairnessChart(), 100);
}

function createFairnessChart() {
  const ctx = document.getElementById('fairnessChart');
  if (!ctx) return;
  
  const experimentsWithFairness = experimentsData.experiments.filter(exp => 
    Object.keys(exp.fairness_metrics).length > 0 && exp.status === 'FINISHED'
  );
  
  if (experimentsWithFairness.length === 0) return;
  
  const metrics = ['demographic_parity_difference', 'equalized_odds_difference', 'equal_opportunity_difference'];
  
  const datasets = metrics.map((metric, index) => ({
    label: metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
    data: experimentsWithFairness.map(exp => Math.abs(exp.fairness_metrics[metric] || 0)),
    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C'][index],
    borderColor: ['#1FB8CD', '#FFC185', '#B4413C'][index],
    borderWidth: 1
  }));
  
  if (charts.fairness) {
    charts.fairness.destroy();
  }
  
  charts.fairness = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: experimentsWithFairness.map(exp => exp.name.split('_')[0]),
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Absolute Difference'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Fairness Metrics Comparison Across Models'
        }
      }
    }
  });
}