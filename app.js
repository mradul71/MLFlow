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
let uploadedShapFile = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
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
});

// Navigation
function initializeNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    // Remove any existing event listeners
    link.removeEventListener('click', handleNavClick);
    // Add new event listener
    link.addEventListener('click', handleNavClick);
  });
}

function handleNavClick(e) {
  e.preventDefault();
  e.stopPropagation();
  
  const view = e.currentTarget.getAttribute('data-view');
  
  if (view) {
    showView(view);
  } else {
    console.error('No view attribute found on navigation link');
  }
}

function showView(viewName) {
  
  // Hide all views
  document.querySelectorAll('.view').forEach(view => {
    view.classList.remove('active');
  });
  
  // Show selected view
  const targetView = document.getElementById(`${viewName}-view`);
  if (targetView) {
    targetView.classList.add('active');
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
  setupShapEventListeners();
  populateRunSelector();
  generateDefaultShapVisualizations();
}

function setupShapEventListeners() {
  
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
  } else {
    console.error('File upload elements not found');
  }

  // Experiment name input
  const experimentNameInput = document.getElementById('experiment-name-input');
  if (experimentNameInput) {
    experimentNameInput.removeEventListener('input', updateRunSelector);
    experimentNameInput.addEventListener('input', updateRunSelector);
  }
}

function handleUploadClick(e) {
  e.preventDefault();
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
    handleFileUpload(files[0]);
  }
}

function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
}

function handleFile(file) {
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
}

function populateRunSelector() {
  const runSelector = document.getElementById('run-selector');
  if (!runSelector) return;
  
  runSelector.innerHTML = '<option value="">Select a run...</option>' +
    experimentsData.experiments
      .filter(exp => exp.status === 'FINISHED')
      .map(exp => `<option value="${exp.id}">${exp.name} (${exp.id})</option>`)
      .join('');
}

function loadModel() {
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
  }
}

function generateShapAnalysis() {
  if (!currentModel) {
    alert('Please load a model first');
    return;
  }
  
  if (!uploadedShapFile) {
    alert('Please upload sample data first');
    return;
  }
  
  const analysisType = document.getElementById('shap-analysis-type')?.value || 'summary';
  
  // Generate SHAP visualizations using the uploaded file
  setTimeout(() => {
    generateShapSummaryChart(uploadedShapFile);
    generateShapDecisionChart(uploadedShapFile);
    generateShapForceChart(uploadedShapFile);
    generateShapValuesTable(uploadedShapFile);
  }, 500);
}

function generateDefaultShapVisualizations() {
  setTimeout(() => {
    // Check if a file has been uploaded
    if (uploadedShapFile) {
      // Use the uploaded file
      generateShapSummaryChart(uploadedShapFile);
      generateShapDecisionChart(uploadedShapFile);
      generateShapForceChart(uploadedShapFile);
      generateShapValuesTable(uploadedShapFile);
    } else {
      // Try to fetch from path (if you have a default file)
      // Or show a message to upload a file
      console.log('No file uploaded yet. Please upload a CSV file.');
      
      // Optionally, you can create empty/placeholder charts here
      // or just leave them empty until a file is uploaded
    }
  }, 500);
}

// SHAP Summary Chart - Standard SHAP Summary Plot
async function generateShapSummaryChart(file) {
  const container = document.getElementById('shap-summary-chart');
  if (!container) return;
  
  try {
    let fileData;
    
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric feature columns (exclude 'species')
    const features = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Calculate SHAP values (using normalized deviation from mean as proxy)
    const shapData = {};
    features.forEach(feature => {
      const values = data.map(row => parseFloat(row[feature])).filter(v => !isNaN(v));
      const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
      const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length);
      
      shapData[feature] = data.map(row => {
        const val = parseFloat(row[feature]);
        const shapValue = std > 0 ? (val - mean) / std * 0.1 : 0; // Scaled SHAP value
        const featureValue = val;
        const species = row.species ? row.species.toLowerCase() : 'setosa';
        
        return { shapValue, featureValue, species };
      });
    });
    
    // Calculate mean absolute SHAP value for sorting
    const featureImportance = features.map(f => {
      const meanAbsShap = shapData[f].reduce((sum, d) => sum + Math.abs(d.shapValue), 0) / shapData[f].length;
      return { feature: f, importance: meanAbsShap };
    });
    
    // Sort by importance (descending)
    featureImportance.sort((a, b) => b.importance - a.importance);
    const sortedFeatures = featureImportance.map(f => f.feature);
    
    // Color mapping for species (representing feature value - high to low)
    const getColor = (featureValue, feature) => {
      const values = data.map(row => parseFloat(row[feature])).filter(v => !isNaN(v));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const normalized = (featureValue - min) / (max - min);
      
      // Color gradient: low (blue) to high (pink/red)
      if (normalized > 0.66) return '#E91E63'; // High - Pink
      if (normalized > 0.33) return '#9C27B0'; // Medium - Purple
      return '#2196F3'; // Low - Blue
    };
    
    // Create scatter datasets for each feature with vertical jitter
    const datasets = [];
    sortedFeatures.forEach((feature, idx) => {
      const points = shapData[feature].map(d => {
        // Add random vertical jitter to prevent overlap (-0.3 to +0.3)
        const jitter = (Math.random() - 0.5) * 0.6;
        
        return {
          x: d.shapValue,
          y: idx + jitter,
          backgroundColor: getColor(d.featureValue, feature),
          borderColor: getColor(d.featureValue, feature)
        };
      });
      
      datasets.push({
        label: feature,
        data: points,
        backgroundColor: points.map(p => p.backgroundColor),
        borderColor: points.map(p => p.borderColor),
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false
      });
    });
    
    // Destroy existing chart
    if (charts.shapSummary) {
      charts.shapSummary.destroy();
    }
    
    // Get or create canvas
    let canvas = container;
    if (container.tagName !== 'CANVAS') {
      canvas = container.querySelector('canvas') || document.createElement('canvas');
      if (!canvas.parentElement) {
        container.innerHTML = '';
        container.appendChild(canvas);
      }
    }
    
    // Create the chart
    charts.shapSummary = new Chart(canvas, {
      type: 'scatter',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: {
              display: true,
              text: 'SHAP value (impact on model output)',
              font: { size: 12 }
            },
            grid: { color: '#e0e0e0' }
          },
          y: {
            type: 'linear',
            min: -0.5,
            max: sortedFeatures.length - 0.5,
            ticks: {
              stepSize: 1,
              callback: function(value) {
                const idx = Math.round(value);
                return sortedFeatures[idx] || '';
              },
              font: { size: 11 }
            },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'SHAP Summary Plot',
            font: { size: 16, weight: 'bold' }
          },
          tooltip: {
            callbacks: {
              title: function(context) {
                const idx = Math.round(context[0].parsed.y);
                return sortedFeatures[idx] || '';
              },
              label: function(context) {
                return `SHAP value: ${context.parsed.x.toFixed(4)}`;
              }
            }
          }
        }
      }
    });
    
    return charts.shapSummary;
    
  } catch (error) {
    console.error('Error generating SHAP summary chart:', error);
    throw error;
  }
}

async function generateShapDecisionChart(file) {
  const ctx = document.getElementById('shap-decision-chart');
  
  if (!ctx) {
    console.error('shap-decision-chart element not found');
    return;
  }
  
  // Ensure canvas is visible and has proper display
  ctx.style.display = 'block';
  ctx.style.width = '100%';
  ctx.style.height = '100%';
  
  if (ctx.offsetWidth === 0 || ctx.offsetHeight === 0) {
    console.error('Canvas has zero dimensions even after styling!');
    console.error('Width:', ctx.offsetWidth, 'Height:', ctx.offsetHeight);
    // Wait a tick for layout
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  try {
    let fileData;
    
    // Handle different input types
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    // Parse CSV using PapaParse
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric columns (excluding 'species' or other categorical columns)
    const numericColumns = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Map species to class numbers
    const speciesMap = {
      'setosa': 0,
      'versicolor': 1,
      'virginica': 2
    };
    
    // Calculate mean absolute SHAP values per feature to sort them
    const featureMeans = {};
    numericColumns.forEach(feature => {
      const values = data.map(row => Math.abs(parseFloat(row[feature]) || 0));
      featureMeans[feature] = values.reduce((sum, v) => sum + v, 0) / values.length;
    });
    
    // Sort features by importance (descending)
    const sortedFeatures = numericColumns.sort((a, b) => featureMeans[b] - featureMeans[a]);
    
    // Create decision plot using canvas
    const canvas = ctx;
    // Use explicit fallback sizes if offsetWidth/Height are 0
    const containerWidth = canvas.offsetWidth || canvas.parentElement?.offsetWidth || 800;
    const containerHeight = canvas.offsetHeight || canvas.parentElement?.offsetHeight || 600;
    
    canvas.width = containerWidth;
    canvas.height = containerHeight;
    
    const context = canvas.getContext('2d');
    
    if (!context) {
      console.error('Could not get 2d context!');
      return;
    }
    
    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // Chart dimensions
    const margin = { top: 40, right: 150, bottom: 60, left: 120 };
    const width = canvas.width - margin.left - margin.right;
    const height = canvas.height - margin.top - margin.bottom;
    
    // Calculate cumulative SHAP values for each sample
    const samples = data.map(row => {
      const species = row.species ? row.species.toLowerCase() : 'setosa';
      const classNum = speciesMap[species] || 0;
      
      let cumulative = 1.0; // Base value
      const path = [{ feature: 'base', value: cumulative, class: classNum }];
      
      sortedFeatures.forEach(feature => {
        const shapValue = parseFloat(row[feature]) || 0;
        cumulative += shapValue;
        path.push({ feature, value: cumulative, class: classNum });
      });
      
      return { path, class: classNum };
    });
    
    // Determine value range
    const allValues = samples.flatMap(s => s.path.map(p => p.value));
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    
    // Scales
    const xScale = (val) => margin.left + ((val - minValue) / (maxValue - minValue)) * width;
    const yScale = (idx) => margin.top + (idx / sortedFeatures.length) * height;
    
    // Colors for classes
    const classColors = ['#1E88E5', '#9C27B0', '#E91E63'];
    
    // Draw horizontal grid lines and feature labels
    context.strokeStyle = '#e0e0e0';
    context.lineWidth = 1;
    context.setLineDash([2, 2]);
    context.font = '14px Arial';
    context.fillStyle = '#e0e0e0';
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    
    sortedFeatures.forEach((feature, idx) => {
      const y = yScale(idx);
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(canvas.width - margin.right, y);
      context.stroke();
      
      context.fillText(feature, margin.left - 10, y);
    });
    
    context.setLineDash([]);
    
    // Draw x-axis
    context.strokeStyle = '#e0e0e0';
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(margin.left, canvas.height - margin.bottom);
    context.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
    context.stroke();
    
    // X-axis label
    context.font = '13px Arial';
    context.fillStyle = '#e0e0e0';
    context.textAlign = 'center';
    context.fillText('Model output value', margin.left + width / 2, canvas.height - margin.bottom + 35);
    
    // X-axis ticks
    context.font = '12px Arial';
    context.textBaseline = 'top';
    for (let i = 0; i <= 8; i++) {
      const val = minValue + (i / 8) * (maxValue - minValue);
      const x = xScale(val);
      context.fillText(val.toFixed(2), x, canvas.height - margin.bottom + 5);
    }
    
    // Color scale bar at top
    const barHeight = 20;
    const barY = 10;
    const gradient = context.createLinearGradient(margin.left, 0, canvas.width - margin.right, 0);
    gradient.addColorStop(0, classColors[0]);
    gradient.addColorStop(0.5, classColors[1]);
    gradient.addColorStop(1, classColors[2]);
    
    context.fillStyle = gradient;
    context.fillRect(margin.left, barY, width, barHeight);
    
    // Draw sample paths
    samples.forEach(sample => {
      const color = classColors[sample.class];
      context.strokeStyle = color;
      context.lineWidth = 1.5;
      context.globalAlpha = 0.6;
      
      context.beginPath();
      sample.path.forEach((point, idx) => {
        const x = xScale(point.value);
        const y = idx === 0 ? yScale(-0.5) : yScale(idx - 1);
        
        if (idx === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      });
      context.stroke();
    });
    
    context.globalAlpha = 1.0;
    
    // Store for cleanup
    charts.shapDecision = { 
      destroy: () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
    };
    return charts.shapDecision;
    
  } catch (error) {
    console.error('Error generating SHAP decision chart:', error);
    throw error;
  }
}

async function generateShapForceChart(file) {
  let container = document.getElementById('shap-force-chart');
  if (!container) {
    console.error('Force chart container not found');
    return;
  }
  
  // If it's a canvas, we need to replace it with a div
  if (container.tagName === 'CANVAS') {
    const parent = container.parentElement;
    const newContainer = document.createElement('div');
    newContainer.id = 'shap-force-chart';
    newContainer.style.width = '100%';
    newContainer.style.height = '100%';
    parent.replaceChild(newContainer, container);
    container = newContainer;
  }
  
  try {
    let fileData;
    
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric columns (excluding 'species')
    const numericColumns = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Calculate means
    const means = {};
    numericColumns.forEach(col => {
      const vals = data.map(row => parseFloat(row[col])).filter(v => !isNaN(v));
      means[col] = vals.reduce((sum, v) => sum + v, 0) / vals.length;
    });
    
    // Use first data row
    const firstRow = data[0];
    const baseValue = 1.0; // Starting base value
    
    // Calculate SHAP contributions
    const contributions = numericColumns.map(col => {
      const value = parseFloat(firstRow[col]) || 0;
      const contribution = (value - means[col]) * 0.1; // Scale factor
      return { feature: col, value: contribution, featureValue: value };
    });
    
    // Sort by contribution magnitude
    contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    
    // Build cumulative values for waterfall
    let cumulative = baseValue;
    const positions = [{ x: baseValue, label: `base value`, isBase: true }];
    
    contributions.forEach(contrib => {
      const start = cumulative;
      cumulative += contrib.value;
      positions.push({
        x: cumulative,
        label: `${contrib.feature} = ${contrib.featureValue.toFixed(1)}`,
        start: start,
        end: cumulative,
        value: contrib.value,
        isPositive: contrib.value >= 0
      });
    });
    
    const finalValue = cumulative;
    
    // Create custom HTML visualization
    container.innerHTML = '';
    container.style.position = 'relative';
    container.style.height = '180px';
    container.style.padding = '50px 30px 30px 30px';
    container.style.backgroundColor = '#f8f9fa';
    container.style.borderRadius = '8px';
    
    // Create SVG
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.position = 'absolute';
    svg.style.top = '50px';
    svg.style.left = '30px';
    svg.style.right = '30px';
    svg.style.bottom = '30px';
    
    // Calculate scale with better spacing
    const minVal = Math.min(baseValue, finalValue, ...positions.map(p => p.x));
    const maxVal = Math.max(baseValue, finalValue, ...positions.map(p => p.x));
    const range = maxVal - minVal || 1;
    const padding = range * 0.15;
    const scale = (val) => {
      const normalized = (val - minVal + padding) / (range + 2 * padding);
      return 5 + normalized * 90; // Use 5-95% of width for better spacing
    };
    
    // Draw axis
    const axisY = 70;
    const axisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    axisLine.setAttribute('x1', '5%');
    axisLine.setAttribute('x2', '95%');
    axisLine.setAttribute('y1', axisY);
    axisLine.setAttribute('y2', axisY);
    axisLine.setAttribute('stroke', '#aaa');
    axisLine.setAttribute('stroke-width', '1.5');
    svg.appendChild(axisLine);
    
    // Draw fewer, cleaner axis ticks
    const numTicks = 6;
    for (let i = 0; i <= numTicks; i++) {
      const val = minVal - padding + (range + 2 * padding) * (i / numTicks);
      const x = 5 + (90 * i / numTicks);
      
      const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tick.setAttribute('x1', `${x}%`);
      tick.setAttribute('x2', `${x}%`);
      tick.setAttribute('y1', axisY);
      tick.setAttribute('y2', axisY + 4);
      tick.setAttribute('stroke', '#aaa');
      tick.setAttribute('stroke-width', '1');
      svg.appendChild(tick);
      
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', `${x}%`);
      text.setAttribute('y', axisY + 16);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#666');
      text.setAttribute('font-size', '11');
      text.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
      text.textContent = val.toFixed(2);
      svg.appendChild(text);
    }
    
    // Draw force plot arrows with improved styling
    let prevX = scale(baseValue);
    
    positions.forEach((pos, idx) => {
      if (idx === 0) {
        // Base value marker - cleaner style
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', `${prevX}%`);
        label.setAttribute('y', axisY - 45);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', '#555');
        label.setAttribute('font-size', '12');
        label.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
        label.textContent = 'base value';
        svg.appendChild(label);
        
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        marker.setAttribute('x1', `${prevX}%`);
        marker.setAttribute('x2', `${prevX}%`);
        marker.setAttribute('y1', axisY - 38);
        marker.setAttribute('y2', axisY);
        marker.setAttribute('stroke', '#888');
        marker.setAttribute('stroke-width', '2');
        marker.setAttribute('stroke-dasharray', '2,2');
        svg.appendChild(marker);
      } else {
        const currentX = scale(pos.x);
        const color = pos.isPositive ? '#ff0d57' : '#1e88e5';
        const width = Math.abs(currentX - prevX);
        
        // Only show arrow if width is significant
        if (width > 2) {
          // Arrow body with gradient
          const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
          const arrowHeight = 22;
          const arrowHeadWidth = 6;
          
          const points = pos.isPositive
            ? `${prevX},${axisY - arrowHeight} ${currentX - arrowHeadWidth},${axisY - arrowHeight} ${currentX},${axisY - arrowHeight / 2} ${currentX - arrowHeadWidth},${axisY} ${prevX},${axisY}`
            : `${prevX},${axisY - arrowHeight} ${currentX + arrowHeadWidth},${axisY - arrowHeight} ${currentX},${axisY - arrowHeight / 2} ${currentX + arrowHeadWidth},${axisY} ${prevX},${axisY}`;
          
          arrow.setAttribute('points', points);
          arrow.setAttribute('fill', color);
          arrow.setAttribute('opacity', '0.85');
          arrow.setAttribute('stroke', color);
          arrow.setAttribute('stroke-width', '0.5');
          svg.appendChild(arrow);
          
          // Feature label - only if there's enough space
          if (width > 8) {
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', `${(prevX + currentX) / 2}%`);
            label.setAttribute('y', axisY - arrowHeight - 3);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('fill', '#444');
            label.setAttribute('font-size', '10');
            label.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
            label.textContent = pos.label;
            svg.appendChild(label);
          }
        }
        
        prevX = currentX;
      }
    });
    
    // Final prediction marker with better styling
    const finalX = scale(finalValue);
    
    const finalLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    finalLabel.setAttribute('x', `${finalX}%`);
    finalLabel.setAttribute('y', axisY - 45);
    finalLabel.setAttribute('text-anchor', 'middle');
    finalLabel.setAttribute('fill', '#222');
    finalLabel.setAttribute('font-size', '13');
    finalLabel.setAttribute('font-weight', 'bold');
    finalLabel.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
    finalLabel.textContent = `f(x) = ${finalValue.toFixed(2)}`;
    svg.appendChild(finalLabel);
    
    const finalMarker = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    finalMarker.setAttribute('x1', `${finalX}%`);
    finalMarker.setAttribute('x2', `${finalX}%`);
    finalMarker.setAttribute('y1', axisY - 38);
    finalMarker.setAttribute('y2', axisY);
    finalMarker.setAttribute('stroke', '#222');
    finalMarker.setAttribute('stroke-width', '2.5');
    svg.appendChild(finalMarker);
    
    // Legend with improved styling
    const legend = document.createElement('div');
    legend.style.position = 'absolute';
    legend.style.top = '15px';
    legend.style.right = '30px';
    legend.style.fontSize = '12px';
    legend.style.fontFamily = 'system-ui, -apple-system, sans-serif';
    legend.innerHTML = `
      <div style="display: flex; gap: 12px; align-items: center; background: white; padding: 6px 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div><span style="color: #ff0d57; font-weight: 600;">higher</span></div>
        <div style="display: flex; align-items: center; gap: 6px;">
          <div style="width: 24px; height: 10px; background: linear-gradient(to right, #1e88e5, #ff0d57); border-radius: 2px;"></div>
          <span style="color: #555; font-weight: 500;">f(x)</span>
        </div>
        <div><span style="color: #1e88e5; font-weight: 600;">lower</span></div>
      </div>
    `;
    
    container.appendChild(svg);
    container.appendChild(legend);
    
    return true;
    
  } catch (error) {
    console.error('Error generating SHAP force chart:', error);
    throw error;
  }
}

async function generateShapBarChart(file) {
  const ctx = document.getElementById('shap-bar-chart');
  if (!ctx) return;
  
  try {
    let fileData;
    
    // Handle different input types
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    // Parse CSV using PapaParse
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric columns (excluding 'species' or other categorical columns)
    const numericColumns = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Map species to class numbers
    const speciesMap = {
      'setosa': 0,
      'versicolor': 1,
      'virginica': 2
    };
    
    // Calculate mean absolute SHAP values per feature per class
    const featureClassValues = {};
    
    numericColumns.forEach(feature => {
      featureClassValues[feature] = { 0: [], 1: [], 2: [] };
      
      data.forEach(row => {
        const species = row.species ? row.species.toLowerCase() : 'setosa';
        const classNum = speciesMap[species] || 0;
        const value = parseFloat(row[feature]);
        
        if (!isNaN(value)) {
          featureClassValues[feature][classNum].push(Math.abs(value));
        }
      });
    });
    
    // Calculate mean for each feature-class combination
    const featureMeans = {};
    numericColumns.forEach(feature => {
      featureMeans[feature] = {};
      for (let c = 0; c < 3; c++) {
        const values = featureClassValues[feature][c];
        const mean = values.length > 0 
          ? values.reduce((sum, v) => sum + v, 0) / values.length 
          : 0;
        featureMeans[feature][c] = mean;
      }
    });
    
    // Sort features by total impact (sum across all classes)
    const sortedFeatures = numericColumns.sort((a, b) => {
      const sumA = featureMeans[a][0] + featureMeans[a][1] + featureMeans[a][2];
      const sumB = featureMeans[b][0] + featureMeans[b][1] + featureMeans[b][2];
      return sumB - sumA; // Descending order
    });
    
    // Create datasets for each class
    const datasets = [
      {
        label: 'Class 0',
        data: sortedFeatures.map(f => featureMeans[f][0]),
        backgroundColor: '#1E88E5', // Blue
        borderColor: '#1E88E5',
        borderWidth: 0
      },
      {
        label: 'Class 1',
        data: sortedFeatures.map(f => featureMeans[f][1]),
        backgroundColor: '#E91E63', // Pink/Magenta
        borderColor: '#E91E63',
        borderWidth: 0
      },
      {
        label: 'Class 2',
        data: sortedFeatures.map(f => featureMeans[f][2]),
        backgroundColor: '#9E9D24', // Olive/Yellow-green
        borderColor: '#9E9D24',
        borderWidth: 0
      }
    ];
    
    // Destroy existing chart if it exists
    if (charts.shapBar) {
      charts.shapBar.destroy();
    }
    
    // Create stacked horizontal bar chart
    charts.shapBar = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: sortedFeatures,
        datasets: datasets
      },
      options: {
        indexAxis: 'y', // Horizontal bars
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            stacked: true,
            beginAtZero: true,
            title: {
              display: true,
              text: 'mean(|SHAP value|) (average impact on model output magnitude)',
              font: { size: 12 }
            },
            grid: {
              color: '#e0e0e0'
            }
          },
          y: {
            stacked: true,
            grid: {
              display: false
            },
            ticks: {
              font: { size: 13 }
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'right',
            labels: {
              font: { size: 12 },
              boxWidth: 40,
              padding: 15
            }
          },
          title: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.dataset.label || '';
                const value = context.parsed.x.toFixed(4);
                return `${label}: ${value}`;
              }
            }
          }
        }
      }
    });
    
    return charts.shapBar;
    
  } catch (error) {
    console.error('Error generating SHAP bar chart:', error);
    throw error;
  }
}

async function generateShapValuesTable(file) {
  const table = document.getElementById('shap-values-table');
  if (!table) return;
  
  try {
    let fileData;
    
    // Handle different input types
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    // Parse CSV using PapaParse
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Check if this is pre-formatted SHAP data or raw data
    const hasShapColumns = headers.some(h => 
      (h.toLowerCase() === 'feature' || h.toLowerCase() === 'shap_value')
    );
    
    let shapData;
    
    if (hasShapColumns) {
      // Pre-formatted SHAP data with feature, shap_value, feature_value columns
      shapData = data.map(row => {
        const shapValue = parseFloat(row.shap_value || row['SHAP Value'] || row.shap || 0);
        const featureValue = parseFloat(row.feature_value || row['Feature Value'] || row.value || 0);
        
        // Determine contribution category
        let contribution;
        const absValue = Math.abs(shapValue);
        if (shapValue >= 0.3) {
          contribution = 'High Positive';
        } else if (shapValue > 0) {
          contribution = 'Positive';
        } else if (shapValue <= -0.3) {
          contribution = 'High Negative';
        } else {
          contribution = 'Negative';
        }
        
        return {
          feature: row.feature || row.Feature || '',
          shap_value: shapValue,
          feature_value: featureValue,
          contribution: row.contribution || contribution
        };
      });
    } else {
      // Raw data format - calculate SHAP-like values from first row
      const numericColumns = headers.filter(h => {
        const firstValue = data[0][h];
        return typeof firstValue === 'number' || !isNaN(parseFloat(firstValue));
      });
      
      // Calculate means and standard deviations
      const stats = {};
      numericColumns.forEach(col => {
        const vals = data.map(row => parseFloat(row[col])).filter(v => !isNaN(v));
        const mean = vals.reduce((sum, v) => sum + v, 0) / vals.length;
        const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
        const stdDev = Math.sqrt(variance);
        stats[col] = { mean, stdDev };
      });
      
      // Use first row to generate SHAP-like values
      const firstRow = data[0];
      
      shapData = numericColumns.map(col => {
        const featureValue = parseFloat(firstRow[col]) || 0;
        const { mean, stdDev } = stats[col];
        
        // Calculate normalized deviation as SHAP value
        const shapValue = stdDev > 0 ? (featureValue - mean) / stdDev : 0;
        
        // Determine contribution category
        let contribution;
        if (shapValue >= 0.5) {
          contribution = 'High Positive';
        } else if (shapValue > 0) {
          contribution = 'Positive';
        } else if (shapValue <= -0.5) {
          contribution = 'High Negative';
        } else {
          contribution = 'Negative';
        }
        
        return {
          feature: col,
          shap_value: shapValue,
          feature_value: featureValue,
          contribution: contribution
        };
      });
      
      // Sort by absolute SHAP value (descending)
      shapData.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    }
    
    // Find or create tbody
    let tbody = table.querySelector('tbody');
    if (!tbody) {
      tbody = document.createElement('tbody');
      table.appendChild(tbody);
    }
    
    // Find max absolute SHAP value for scaling the bars
    const maxAbsShap = Math.max(...shapData.map(row => Math.abs(row.shap_value)));
    
    // Populate table
    tbody.innerHTML = shapData.map(row => `
      <tr>
        <td>${row.feature}</td>
        <td class="${row.shap_value >= 0 ? 'shap-value-positive' : 'shap-value-negative'}">
          ${row.shap_value.toFixed(4)}
        </td>
        <td>${row.feature_value.toFixed(2)}</td>
        <td>
          ${row.contribution}
          <div class="contribution-bar">
            <div class="${row.shap_value >= 0 ? 'contribution-positive' : 'contribution-negative'}" 
                 style="width: ${maxAbsShap > 0 ? (Math.abs(row.shap_value) / maxAbsShap * 100) : 0}%"></div>
          </div>
        </td>
      </tr>
    `).join('');
    
    return shapData;
    
  } catch (error) {
    console.error('Error generating SHAP values table:', error);
    throw error;
  }
}

// Architecture functionality
function initializeArchitecture() {
  renderArchitectureDiagram();
  setTimeout(() => {
    createDataFlowChart();
    createTechStackChart();
  }, 100);
}

function renderArchitectureDiagram() {
  const diagramContainer = document.getElementById('architecture-diagram');
  if (!diagramContainer) {
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


// file upload code
// Simple direct approach - no fancy stuff
function initializeFileUpload() {
  const fileUploadArea = document.getElementById('file-upload-area');
  const fileInput = document.getElementById('file-input');
  
  if (!fileUploadArea || !fileInput) {
    alert('ERROR: Elements not found!');
    return;
  }
  
  // Direct click handler - simplest possible
  fileUploadArea.onclick = function() {
    fileInput.click();
  };
  
  // File selected handler
  fileInput.onchange = function(e) {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };
}

async function handleFileUpload(file) {
  
  if (!file.name.endsWith('.csv')) {
    alert('Please upload a CSV file');
    return;
  }
  
  try {
    const fileUploadArea = document.getElementById('file-upload-area');
    
    // Show loading
    fileUploadArea.innerHTML = `
      <div class="upload-icon">⏳</div>
      <div class="upload-text">
        <p class="upload-title">Processing file...</p>
      </div>
    `;
    
    // STORE THE FILE GLOBALLY
    uploadedShapFile = file;

    await generateShapSummaryChart(file);
    await generateShapDecisionChart(file);
    await generateShapForceChart(file);
    await generateShapBarChart(file);
    await generateShapValuesTable(file);
    
    // Show success AFTER charts are generated
    fileUploadArea.innerHTML = `
      <div class="upload-icon">✅</div>
      <div class="upload-text">
        <p class="upload-title">File uploaded successfully!</p>
        <p class="upload-subtitle">${file.name}</p>
        <p class="upload-subtitle" style="color: var(--color-success); margin-top: 8px;">✓ All 5 charts generated</p>
      </div>
    `;
    
  } catch (error) {
    console.error('=== ERROR IN FILE UPLOAD ===');
    console.error('Error:', error);
    console.error('Error stack:', error.stack);
    alert('Failed to process file: ' + error.message);
    
    fileUploadArea.innerHTML = `
      <div class="upload-icon">📤</div>
      <div class="upload-text">
        <p class="upload-title">Upload CSV file for SHAP analysis</p>
        <p class="upload-subtitle">Drag and drop your file here or click to browse</p>
      </div>
    `;
    initializeFileUpload();
  }
}

// IMPORTANT: Only call this once!
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeFileUpload);
} else {
  initializeFileUpload();
}