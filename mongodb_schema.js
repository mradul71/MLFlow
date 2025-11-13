// MongoDB Schema Design for Enhanced MLflow System

// ============================================
// 1. USERS Collection
// ============================================
db.users.insertOne({
  _id: ObjectId("..."),
  user_id: "ryan_001",
  email: "ryan@example.com",
  name: "Ryan",
  created_at: ISODate("2025-09-01T10:00:00Z"),
  last_login: ISODate("2025-11-14T08:30:00Z"),
  preferences: {
    default_mlflow_uri: "http://localhost:5000",
    theme: "light",
    notifications_enabled: true
  },
  roles: ["data_scientist", "ml_engineer"]
});

// ============================================
// 2. MODELS Collection
// Stores metadata about models uploaded/tracked via MLflow
// ============================================
db.models.insertOne({
  _id: ObjectId("..."),
  model_id: "model_12345",
  user_id: "ryan_001",
  
  // MLflow Connection
  mlflow_run_id: "run_001",
  mlflow_experiment_id: "exp_123",
  mlflow_tracking_uri: "http://localhost:5000",
  artifact_path: "model",
  
  // Model Metadata
  model_name: "RandomForest_Titanic_Final",
  model_type: "RandomForest",
  model_version: "v1.2.0",
  framework: "sklearn",
  dataset_name: "Titanic",
  
  // Training Info
  trained_at: ISODate("2025-09-17T14:30:00Z"),
  training_duration_seconds: 244,
  
  // Model Parameters
  parameters: {
    n_estimators: 156,
    max_depth: 9,
    learning_rate: 0.131,
    random_state: 42
  },
  
  // Performance Metrics
  metrics: {
    accuracy: 0.9018,
    precision: 0.9095,
    recall: 0.8723,
    f1_score: 0.899,
    auc_roc: 0.9156
  },
  
  // Feature Information
  features: {
    feature_names: ["Sex", "Fare", "Age", "Pclass", "SibSp", "Parch", "Embarked"],
    feature_count: 7,
    feature_importance: {
      "Sex": 0.4567,
      "Fare": 0.2341,
      "Age": 0.1456,
      "Pclass": 0.0823,
      "SibSp": 0.0456,
      "Parch": 0.0234,
      "Embarked": 0.0123
    }
  },
  
  // SHAP Analysis Results
  shap_analysis: {
    available: true,
    computed_at: ISODate("2025-09-17T15:00:00Z"),
    base_value: 0.5234,
    shap_values: {
      "Sex": 0.3876,
      "Fare": -0.1987,
      "Age": 0.2567,
      "Pclass": -0.1234,
      "SibSp": 0.0987
    },
    summary_plot_url: "s3://mlflow-artifacts/shap/model_12345/summary.png",
    force_plot_url: "s3://mlflow-artifacts/shap/model_12345/force.html"
  },
  
  // Fairness Metrics (if computed)
  fairness_metrics: {
    available: true,
    demographic_parity_difference: 0.0742,
    equalized_odds_difference: -0.0231,
    equal_opportunity_difference: 0.0856,
    sensitive_features: ["Sex", "Age"],
    computed_at: ISODate("2025-09-17T15:05:00Z")
  },
  
  // Artifacts
  artifacts: [
    {
      name: "model.pkl",
      type: "model_file",
      size_bytes: 1024000,
      url: "s3://mlflow-artifacts/exp_123/run_001/model.pkl"
    },
    {
      name: "confusion_matrix.png",
      type: "visualization",
      size_bytes: 45000,
      url: "s3://mlflow-artifacts/exp_123/run_001/confusion_matrix.png"
    }
  ],
  
  // Status and Tags
  status: "active", // active, archived, deleted
  tags: ["production", "titanic", "classification"],
  notes: "Best performing model for Titanic dataset",
  
  // Metadata
  created_at: ISODate("2025-09-17T14:30:00Z"),
  updated_at: ISODate("2025-09-17T15:05:00Z"),
  is_favorite: true,
  view_count: 45,
  comparison_count: 12
});

// ============================================
// 3. COMPARISONS Collection
// Stores model comparison sessions and results
// ============================================
db.comparisons.insertOne({
  _id: ObjectId("..."),
  comparison_id: "comp_67890",
  user_id: "ryan_001",
  
  // Comparison Metadata
  comparison_name: "RandomForest vs XGBoost - Wine Quality",
  description: "Comparing tree-based models on Wine Quality dataset",
  created_at: ISODate("2025-09-20T10:15:00Z"),
  
  // Models Being Compared
  models: [
    {
      model_id: "model_12345",
      model_name: "RandomForest_WineQuality_7",
      mlflow_run_id: "run_007"
    },
    {
      model_id: "model_12346",
      model_name: "XGBoost_WineQuality_4",
      mlflow_run_id: "run_004"
    }
  ],
  
  // Comparison Results - Performance Metrics
  performance_comparison: {
    metrics: ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
    results: {
      "model_12345": {
        accuracy: 0.8567,
        precision: 0.8789,
        recall: 0.8345,
        f1_score: 0.8523,
        auc_roc: 0.8678
      },
      "model_12346": {
        accuracy: 0.8913,
        precision: 0.8734,
        recall: 0.9156,
        f1_score: 0.8876,
        auc_roc: 0.8945
      }
    },
    winner: {
      metric: "accuracy",
      model_id: "model_12346",
      model_name: "XGBoost_WineQuality_4"
    }
  },
  
  // Feature Importance Comparison
  feature_comparison: {
    common_features: ["alcohol", "volatile_acidity", "citric_acid", "density"],
    importance_delta: {
      "alcohol": {
        "model_12345": 0.35,
        "model_12346": 0.3456,
        difference: 0.0044
      },
      "volatile_acidity": {
        "model_12345": 0.25,
        "model_12346": 0.2123,
        difference: 0.0377
      }
    }
  },
  
  // SHAP Comparison (if both models have SHAP analysis)
  shap_comparison: {
    available: true,
    feature_agreement_score: 0.87, // How similarly models weight features
    divergent_features: ["citric_acid", "pH"], // Features with very different SHAP values
    comparison_plot_url: "s3://mlflow-artifacts/comparisons/comp_67890/shap_comparison.png"
  },
  
  // Fairness Comparison
  fairness_comparison: {
    available: true,
    metrics_compared: ["demographic_parity_difference", "equalized_odds_difference"],
    results: {
      "model_12345": {
        demographic_parity_difference: 0.1234,
        equalized_odds_difference: -0.0567
      },
      "model_12346": {
        demographic_parity_difference: -0.1102,
        equalized_odds_difference: 0.0398
      }
    },
    fairer_model: "model_12345",
    fairness_score: {
      "model_12345": 8.5,
      "model_12346": 7.2
    }
  },
  
  // Training Efficiency Comparison
  efficiency_comparison: {
    training_time: {
      "model_12345": 203,  // seconds
      "model_12346": 156
    },
    model_size: {
      "model_12345": 1024000,  // bytes
      "model_12346": 856000
    },
    inference_time_ms: {
      "model_12345": 45,
      "model_12346": 38
    }
  },
  
  // Visualizations Generated
  visualizations: [
    {
      type: "radar_chart",
      title: "Performance Metrics Comparison",
      url: "s3://mlflow-artifacts/comparisons/comp_67890/radar_comparison.png"
    },
    {
      type: "bar_chart",
      title: "Feature Importance Comparison",
      url: "s3://mlflow-artifacts/comparisons/comp_67890/feature_comparison.png"
    }
  ],
  
  // User Analysis Notes
  user_notes: "XGBoost performs better but RandomForest shows better fairness metrics",
  
  // Comparison Statistics
  stats: {
    total_metrics_compared: 5,
    models_count: 2,
    computation_time_seconds: 12.5
  },
  
  // Status
  status: "completed", // pending, completed, failed
  is_shared: false,
  shared_with: [], // Array of user_ids if shared
  
  // Metadata
  updated_at: ISODate("2025-09-20T10:20:00Z"),
  view_count: 8
});

// ============================================
// 4. SHAP_ANALYSES Collection
// Detailed SHAP analysis results (separate for scalability)
// ============================================
db.shap_analyses.insertOne({
  _id: ObjectId("..."),
  analysis_id: "shap_abc123",
  model_id: "model_12345",
  user_id: "ryan_001",
  mlflow_run_id: "run_001",
  
  // Analysis Configuration
  analysis_config: {
    analysis_type: "summary", // summary, force, waterfall, bar
    max_features: 10,
    sample_size: 1000,
    background_samples: 200
  },
  
  // SHAP Values (can be large, consider GridFS for very large datasets)
  shap_values: {
    base_value: 0.5234,
    feature_values: {
      "Sex": [0.3876, 0.2341, -0.1567, 0.4123, ...], // Array of SHAP values per sample
      "Fare": [-0.1987, 0.1234, 0.0456, -0.2341, ...],
      "Age": [0.2567, -0.1234, 0.3456, 0.1789, ...]
    }
  },
  
  // Aggregated Statistics
  feature_importance_summary: {
    "Sex": {
      mean_abs_shap: 0.3876,
      std_shap: 0.1234,
      min_shap: -0.5432,
      max_shap: 0.6789
    },
    "Fare": {
      mean_abs_shap: 0.1987,
      std_shap: 0.0987,
      min_shap: -0.4321,
      max_shap: 0.4567
    }
  },
  
  // Sample-level Explanations (for force plots)
  sample_explanations: [
    {
      sample_index: 0,
      prediction: 0.8765,
      feature_contributions: {
        "Sex": 0.3876,
        "Fare": -0.1987,
        "Age": 0.2567
      }
    },
    {
      sample_index: 1,
      prediction: 0.6543,
      feature_contributions: {
        "Sex": 0.2341,
        "Fare": 0.1234,
        "Age": -0.1234
      }
    }
  ],
  
  // Generated Plots (URLs or base64 if small)
  plots: {
    summary_plot: "s3://mlflow-artifacts/shap/shap_abc123/summary.png",
    force_plot_interactive: "s3://mlflow-artifacts/shap/shap_abc123/force.html",
    waterfall_plot: "s3://mlflow-artifacts/shap/shap_abc123/waterfall.png"
  },
  
  // Computation Metadata
  computed_at: ISODate("2025-09-17T15:00:00Z"),
  computation_time_seconds: 45.2,
  explainer_type: "TreeExplainer", // TreeExplainer, KernelExplainer, etc.
  
  // Status
  status: "completed"
});

// ============================================
// 5. FAIRNESS_ANALYSES Collection
// Fairness and bias analysis results
// ============================================
db.fairness_analyses.insertOne({
  _id: ObjectId("..."),
  analysis_id: "fair_xyz789",
  model_id: "model_12345",
  user_id: "ryan_001",
  mlflow_run_id: "run_001",
  
  // Sensitive Attributes Analyzed
  sensitive_attributes: ["Sex", "Age_Group"],
  
  // Fairness Metrics
  metrics: {
    demographic_parity: {
      difference: 0.0742,
      ratio: 0.89,
      by_group: {
        "Male": 0.85,
        "Female": 0.78
      }
    },
    equalized_odds: {
      difference: -0.0231,
      true_positive_rate_by_group: {
        "Male": 0.87,
        "Female": 0.89
      },
      false_positive_rate_by_group: {
        "Male": 0.12,
        "Female": 0.14
      }
    },
    equal_opportunity: {
      difference: 0.0856,
      by_group: {
        "Male": 0.87,
        "Female": 0.78
      }
    }
  },
  
  // Bias Detection
  bias_detected: true,
  bias_severity: "moderate", // low, moderate, high
  problematic_features: ["Sex", "Age_Group"],
  
  // Recommendations
  recommendations: [
    {
      issue: "High demographic parity difference",
      severity: "moderate",
      suggestion: "Consider rebalancing training data or using fairness-aware algorithms",
      estimated_impact: "Could improve fairness score by 15-20%"
    }
  ],
  
  // Mitigation Strategies Applied (if any)
  mitigation_applied: false,
  mitigation_strategies: [],
  
  // Fairness Score (0-10)
  overall_fairness_score: 7.2,
  
  // Analysis Configuration
  test_dataset_size: 5000,
  confidence_level: 0.95,
  
  // Visualizations
  plots: {
    fairness_comparison: "s3://mlflow-artifacts/fairness/fair_xyz789/comparison.png",
    confusion_matrix_by_group: "s3://mlflow-artifacts/fairness/fair_xyz789/confusion_by_group.png"
  },
  
  computed_at: ISODate("2025-09-17T15:05:00Z"),
  computation_time_seconds: 23.7,
  status: "completed"
});

// ============================================
// 6. USER_ACTIVITIES Collection
// Track user interactions for analytics
// ============================================
db.user_activities.insertOne({
  _id: ObjectId("..."),
  user_id: "ryan_001",
  activity_type: "model_comparison", // model_upload, comparison_created, shap_generated, etc.
  activity_details: {
    comparison_id: "comp_67890",
    models_compared: ["model_12345", "model_12346"]
  },
  timestamp: ISODate("2025-09-20T10:15:00Z"),
  ip_address: "192.168.1.100",
  user_agent: "Mozilla/5.0..."
});

// ============================================
// INDEXES for Performance Optimization
// ============================================

// Users Collection
db.users.createIndex({ "user_id": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });

// Models Collection
db.models.createIndex({ "model_id": 1 }, { unique: true });
db.models.createIndex({ "user_id": 1 });
db.models.createIndex({ "mlflow_run_id": 1 });
db.models.createIndex({ "mlflow_experiment_id": 1 });
db.models.createIndex({ "model_type": 1 });
db.models.createIndex({ "dataset_name": 1 });
db.models.createIndex({ "status": 1 });
db.models.createIndex({ "created_at": -1 });
db.models.createIndex({ "tags": 1 });
db.models.createIndex({ "metrics.accuracy": -1 });

// Comparisons Collection
db.comparisons.createIndex({ "comparison_id": 1 }, { unique: true });
db.comparisons.createIndex({ "user_id": 1 });
db.comparisons.createIndex({ "models.model_id": 1 });
db.comparisons.createIndex({ "created_at": -1 });
db.comparisons.createIndex({ "status": 1 });

// SHAP Analyses Collection
db.shap_analyses.createIndex({ "analysis_id": 1 }, { unique: true });
db.shap_analyses.createIndex({ "model_id": 1 });
db.shap_analyses.createIndex({ "user_id": 1 });
db.shap_analyses.createIndex({ "computed_at": -1 });

// Fairness Analyses Collection
db.fairness_analyses.createIndex({ "analysis_id": 1 }, { unique: true });
db.fairness_analyses.createIndex({ "model_id": 1 });
db.fairness_analyses.createIndex({ "user_id": 1 });
db.fairness_analyses.createIndex({ "computed_at": -1 });

// User Activities Collection
db.user_activities.createIndex({ "user_id": 1, "timestamp": -1 });
db.user_activities.createIndex({ "activity_type": 1 });

// ============================================
// SAMPLE QUERIES
// ============================================

// Get all models for a user
db.models.find({ user_id: "ryan_001" }).sort({ created_at: -1 });

// Get top performing models by accuracy
db.models.find({ status: "active" }).sort({ "metrics.accuracy": -1 }).limit(10);

// Find all comparisons involving a specific model
db.comparisons.find({ "models.model_id": "model_12345" });

// Get models with SHAP analysis available
db.models.find({ "shap_analysis.available": true });

// Get models with fairness concerns
db.models.find({ 
  "fairness_metrics.available": true,
  "fairness_metrics.demographic_parity_difference": { $gt: 0.1 }
});

// Get recent user activities
db.user_activities.find({ user_id: "ryan_001" })
  .sort({ timestamp: -1 })
  .limit(20);

// Aggregate: Count models by type
db.models.aggregate([
  { $match: { status: "active" } },
  { $group: { 
      _id: "$model_type", 
      count: { $sum: 1 },
      avg_accuracy: { $avg: "$metrics.accuracy" }
  }},
  { $sort: { count: -1 } }
]);

// Aggregate: Get comparison summary for a user
db.comparisons.aggregate([
  { $match: { user_id: "ryan_001", status: "completed" } },
  { $group: {
      _id: "$user_id",
      total_comparisons: { $sum: 1 },
      avg_computation_time: { $avg: "$stats.computation_time_seconds" }
  }}
]);
