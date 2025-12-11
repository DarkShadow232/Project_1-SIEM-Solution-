# AI Models Used in Network Traffic Anomaly Detection Project

## Overview

This project implements **multiple machine learning models** for network traffic anomaly detection and threat classification. The models are organized into different categories based on their use case.

---

## 1. One-Class SVM (OCSVM) ⭐ **Recommended for SIEM Integration**

**Location:** `Models/AnomalyDetection-OCSVM/` and `siem_integration/`

**Status:** ✅ **Primary Model for SIEM Integration** (Recommended by Eng Mariam)

### Description
- **Type:** Unsupervised Anomaly Detection
- **Algorithm:** One-Class Support Vector Machine
- **Kernel:** RBF (Radial Basis Function)
- **Training:** Trains only on normal traffic (no labeled attack data needed)

### Key Features
- ✅ Ideal for detecting unknown attack patterns
- ✅ Works with unlabeled data (only normal traffic required)
- ✅ Effective for large-scale log analysis
- ✅ Suitable for real-time SIEM integration

### Parameters Used
- `nu = 0.1` (Expected fraction of outliers: 10%)
- `gamma = 'scale'` (Kernel coefficient)
- `kernel = 'rbf'` (Radial Basis Function)

### Use Cases
- SIEM Security Dashboard integration
- Real-time anomaly detection
- Unknown attack pattern detection
- Large-scale network log analysis

### Files
- `train_ocsvm.py` - Training script
- `SIEM_Integration_Notebook.ipynb` - Complete integration notebook
- `run_siem_notebook.py` - Executable Python script
- `siem_connector.py` - SIEM connector class

---

## 2. Isolation Forest

**Location:** `Models/AnomalyDetection-IsolationForest/`

**Status:** ✅ Implemented

### Description
- **Type:** Unsupervised Anomaly Detection
- **Algorithm:** Isolation Forest (Tree-based)
- **Method:** Random partitioning to isolate anomalies

### Key Features
- ✅ Fast training and prediction
- ✅ Handles high-dimensional data well
- ✅ Effective for detecting outliers
- ✅ No need for labeled data

### Use Cases
- Outlier detection in network traffic
- Anomaly identification
- Feature analysis

### Output Files
- `isolation_forest_model.pkl`
- `robust_scaler.pkl`
- `simple_imputer.pkl`
- Classification reports and confusion matrices

---

## 3. Gaussian Mixture Model (GMM)

**Location:** `Models/AnomalyDetection-GMM/`

**Status:** ✅ Implemented

### Description
- **Type:** Unsupervised Anomaly Detection
- **Algorithm:** Gaussian Mixture Model
- **Method:** Probabilistic clustering and density estimation

### Key Features
- ✅ Probabilistic approach
- ✅ Can model complex distributions
- ✅ Provides probability scores
- ✅ Good for multi-modal data

### Use Cases
- Probabilistic anomaly detection
- Density-based outlier detection
- Clustering analysis

### Output Files
- `gmm_model.pkl`
- `scaler.pkl`
- Classification reports (training and testing)
- Confusion matrices

---

## 4. XGBoost - Binary Classification

**Location:** `Models/BinaryClassification-xgboost/`

**Status:** ✅ Implemented

### Description
- **Type:** Supervised Binary Classification
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Task:** Classify traffic as Normal (0) or Anomaly (1)

### Key Features
- ✅ High accuracy
- ✅ Feature importance analysis
- ✅ Handles class imbalance (scale_pos_weight)
- ✅ Hyperparameter tuning support
- ✅ Fast prediction

### Parameters
- Handles class imbalance automatically
- Supports GridSearchCV for hyperparameter tuning
- Uses feature scaling (StandardScaler)

### Use Cases
- Binary anomaly classification
- Feature importance analysis
- High-accuracy detection
- Production deployment

### Output Files
- `xgboost_model.pkl`
- `scaler.pkl`
- `feature_importance.csv`
- `feature_importance.png`
- Classification reports

---

## 5. XGBoost - Multi-Class Classification

**Location:** `Models/MultiClassification-xgboost/`

**Status:** ✅ Implemented

### Description
- **Type:** Supervised Multi-Class Classification
- **Algorithm:** XGBoost Classifier
- **Task:** Classify attacks into specific categories/types

### Key Features
- ✅ Multi-class attack classification
- ✅ Identifies specific attack types
- ✅ Label encoding for categorical targets
- ✅ High accuracy for classification

### Use Cases
- Attack type classification
- Threat categorization
- Multi-class anomaly detection
- Detailed threat analysis

### Output Files
- `xgbclass_model.pkl`
- `label_encoder.pkl`
- `scaler.pkl`
- Classification reports
- Confusion matrices

---

## Model Comparison Summary

| Model | Type | Training Data | Use Case | Best For |
|-------|------|---------------|----------|----------|
| **One-Class SVM** | Unsupervised | Normal traffic only | SIEM Integration | Unknown attacks, Real-time detection |
| **Isolation Forest** | Unsupervised | Unlabeled data | Outlier Detection | Fast detection, High-dimensional data |
| **GMM** | Unsupervised | Unlabeled data | Probabilistic Detection | Density-based anomalies |
| **XGBoost Binary** | Supervised | Labeled data | Binary Classification | High accuracy, Known attack patterns |
| **XGBoost Multi-Class** | Supervised | Labeled data | Attack Classification | Specific attack type identification |

---

## Recommended Model Selection

### For SIEM Integration (Production)
**✅ One-Class SVM** - Recommended by Eng Mariam
- No need for labeled attack data
- Detects unknown attack patterns
- Suitable for real-time monitoring
- Works with large-scale logs

### For High Accuracy (When Labels Available)
**✅ XGBoost Binary/Multi-Class**
- Requires labeled training data
- Higher accuracy on known attack types
- Provides feature importance
- Best for classification tasks

### For Fast Outlier Detection
**✅ Isolation Forest**
- Fast training and prediction
- Good for initial analysis
- Effective outlier detection

---

## Model Performance Metrics

All models generate:
- ✅ Classification Reports (Precision, Recall, F1-Score)
- ✅ Confusion Matrices
- ✅ ROC-AUC Scores (where applicable)
- ✅ Feature Importance (for XGBoost)
- ✅ Visualizations and plots

---

## Integration Status

### ✅ Fully Integrated
- **One-Class SVM** - Complete SIEM integration pipeline
- **XGBoost Binary** - Streamlit web application

### ✅ Implemented
- **Isolation Forest** - Model trained and saved
- **GMM** - Model trained and saved
- **XGBoost Multi-Class** - Model trained and saved

---

## Files Structure

```
project 1/
├── Models/
│   ├── AnomalyDetection-OCSVM/        # One-Class SVM
│   ├── AnomalyDetection-IsolationForest/  # Isolation Forest
│   ├── AnomalyDetection-GMM/          # Gaussian Mixture Model
│   ├── BinaryClassification-xgboost/  # XGBoost Binary
│   └── MultiClassification-xgboost/   # XGBoost Multi-Class
│
└── siem_integration/                   # SIEM Integration (uses OCSVM)
    ├── SIEM_Integration_Notebook.ipynb
    ├── run_siem_notebook.py
    └── siem_connector.py
```

---

## Summary

This project implements **5 different AI/ML models** for network anomaly detection:

1. **One-Class SVM** ⭐ - Primary model for SIEM (Recommended by Eng Mariam)
2. **Isolation Forest** - Fast outlier detection
3. **Gaussian Mixture Model** - Probabilistic anomaly detection
4. **XGBoost Binary** - High-accuracy binary classification
5. **XGBoost Multi-Class** - Attack type classification

**Primary Model:** One-Class SVM is the main model used for SIEM integration and real-time anomaly detection, as it doesn't require labeled attack data and can detect unknown threats.

