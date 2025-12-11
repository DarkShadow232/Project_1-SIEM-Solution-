# SIEM Security Solution Integration

This module provides a complete solution for network anomaly detection using One-Class SVM (recommended by Eng Mariam) and integration with SIEM Security Dashboard.

## Overview

The solution consists of:
1. **One-Class SVM Training** - Train model on large-scale network logs
2. **Attack Simulation** - Generate realistic attack scenarios for testing
3. **Real-time Detection** - Detect anomalies in network traffic
4. **SIEM Integration** - Send alerts to SIEM Security Dashboard

## Components

### 1. Dataset Loader (`dataset_loader.py`)
- Loads large-scale network log datasets from Kaggle or local files
- Supports chunked loading for very large datasets
- Recommended datasets:
  - NF-CSE-CIC-IDS2018 (188K+ samples)
  - CIC-IDS2017 (2.8M+ samples)
  - UNSW-NB15 (257K+ samples)
  - KDD-CUP-99 (494K+ samples)

### 2. One-Class SVM Trainer (`train_ocsvm.py`)
- Trains One-Class SVM model on normal traffic only
- Handles large datasets with efficient preprocessing
- Saves trained model and scaler for deployment

### 3. Attack Simulator (`attack_simulator.py`)
- Simulates various attack types:
  - DDoS attacks
  - Port scanning
  - Brute force attacks
  - Data exfiltration
  - Malware communication
  - DNS tunneling
  - SQL injection
  - ICMP floods
- Generates realistic attack logs for testing

### 4. SIEM Connector (`siem_connector.py`)
- Real-time anomaly detection
- Alert generation and formatting
- REST API integration with SIEM dashboard
- Alert severity classification
- Security recommendations

### 5. Pipeline Runner (`run_siem_pipeline.py`)
- Complete end-to-end pipeline
- Trains model, simulates attacks, detects anomalies, sends alerts

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Additional dependencies for SIEM integration
pip install kagglehub requests
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
cd siem_integration
python run_siem_pipeline.py
```

### Option 2: Step-by-Step Execution

#### Step 1: Load Dataset

```python
from dataset_loader import DatasetLoader

# List recommended datasets
DatasetLoader.list_recommended_datasets()

# Load dataset
df = DatasetLoader.load_recommended_dataset(
    'NF-CSE-CIC-IDS2018',
    sample_size=50000  # Optional: sample for faster training
)
```

#### Step 2: Train One-Class SVM Model

```python
from Models.AnomalyDetection-OCSVM.train_ocsvm import OCSVMTrainer

trainer = OCSVMTrainer(output_dir="outputs")
df_train, _ = trainer.load_data(
    kaggle_dataset="mohamedelrifai/network-anomaly-detection-dataset",
    kaggle_filename="sampled_NF-CSE-CIC-IDS2018-v2.csv",
    use_normal_only=True  # Train only on normal traffic
)

X_train = trainer.preprocess_data(df_train)
trainer.train(X_train, nu=0.1, gamma='scale', kernel='rbf')
trainer.save_model()
```

#### Step 3: Simulate Attacks

```python
from attack_simulator import AttackSimulator, save_attack_logs

simulator = AttackSimulator()

# Simulate mixed attack scenario
attacks = simulator.generate_mixed_attack_scenario(
    attack_types=['ddos', 'port_scan', 'brute_force'],
    duration_seconds=60,
    attack_rate=2.0
)

save_attack_logs(attacks, "simulated_attacks.csv")
```

#### Step 4: Detect Anomalies and Send to SIEM

```python
from siem_connector import SIEMConnector
import pandas as pd

# Load model
connector = SIEMConnector(
    model_path="Models/AnomalyDetection-OCSVM/outputs/one_class_svm_model.pkl",
    scaler_path="Models/AnomalyDetection-OCSVM/outputs/standard_scaler.pkl",
    siem_api_url="http://your-siem-dashboard.com/api/alerts",  # Your SIEM API endpoint
    api_key="your-api-key"  # If required
)

# Load attack logs
attack_logs = pd.read_csv("simulated_attacks.csv")

# Process and detect anomalies
alerts = connector.process_log_stream(attack_logs, send_to_siem=True)
```

## Configuration

### Model Training Parameters

- `nu`: Outlier fraction (0.01-0.5). Lower = more sensitive
- `gamma`: Kernel coefficient ('scale', 'auto', or float)
- `kernel`: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')

### Attack Simulation Parameters

- `attack_types`: List of attack types to simulate
- `duration_seconds`: Duration of attack simulation
- `attack_rate`: Attacks per second

### SIEM Integration

Set the following in `run_siem_pipeline.py`:

```python
CONFIG = {
    'siem_api_url': 'http://your-siem-dashboard.com/api/alerts',
    'siem_api_key': 'your-api-key',  # If required
    'alert_threshold': 0.5  # Decision score threshold
}
```

## SIEM Dashboard Integration

The SIEM connector sends alerts in the following format:

```json
{
    "alert_id": "ALERT-20240101120000000",
    "timestamp": "2024-01-01T12:00:00",
    "severity": "high",
    "status": "new",
    "source": "One-Class SVM Anomaly Detection",
    "anomaly_score": 8.5,
    "network_info": {
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.1",
        "src_port": 12345,
        "dst_port": 80,
        "protocol": 6
    },
    "attack_info": {
        "attack_type": "DDOS",
        "attack_category": "DoS"
    },
    "recommendations": [
        "Block source IP addresses",
        "Enable rate limiting"
    ]
}
```

## Output Files

- `one_class_svm_model.pkl` - Trained model
- `standard_scaler.pkl` - Feature scaler
- `feature_names.json` - Feature names for consistency
- `simulated_attacks.csv` - Generated attack logs
- `siem_alerts.json` - Generated alerts

## Troubleshooting

### Dataset Loading Issues
- Ensure `kagglehub` is installed: `pip install kagglehub`
- For large datasets, use `sample_size` parameter
- Check Kaggle credentials if using Kaggle datasets

### Model Training Issues
- Ensure sufficient normal traffic samples (recommended: 10K+)
- Adjust `nu` parameter based on expected anomaly rate
- Use 'scale' or 'auto' for gamma with large feature sets

### SIEM Integration Issues
- Verify SIEM API URL is correct
- Check API authentication (API key if required)
- Ensure network connectivity to SIEM dashboard
- Review alert format matches SIEM requirements

## Next Steps

1. **Customize Attack Patterns**: Modify `attack_simulator.py` to add custom attack types
2. **Fine-tune Model**: Adjust hyperparameters based on your network traffic patterns
3. **SIEM Dashboard**: Configure your SIEM dashboard to receive alerts
4. **Real-time Integration**: Integrate with live network traffic streams
5. **Monitoring**: Set up monitoring and alerting for the detection system

## Support

For issues or questions:
- Check the main project README
- Review error messages and logs
- Verify all dependencies are installed
- Ensure dataset format matches expected structure

