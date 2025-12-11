# SIEM Integration Guide - Complete Solution

## Overview

This guide provides step-by-step instructions for implementing the SIEM Security Solution using One-Class SVM (recommended by Eng Mariam) for network anomaly detection.

## Requirements

### 1. Dataset with Large Amount of Logs

**Recommended Datasets:**

1. **NF-CSE-CIC-IDS2018** (Recommended)
   - Source: Kaggle (`mohamedelrifai/network-anomaly-detection-dataset`)
   - Size: 188K+ samples
   - Features: Network flow features
   - Attacks: DDoS, Brute Force, Port Scan, SQL Injection, XSS

2. **CIC-IDS2017**
   - Source: Kaggle (`cranix/ids2017`)
   - Size: 2.8M+ samples (Very Large)
   - Features: Network flow features
   - Attacks: DDoS, Brute Force, Port Scan, Botnet, Infiltration

3. **UNSW-NB15**
   - Source: Kaggle (`mrwellsdavid/unsw-nb15`)
   - Size: 257K+ samples
   - Features: Network flow features
   - Attacks: Fuzzers, Analysis, Backdoors, DoS, Exploits, etc.

### 2. One-Class SVM Model (Recommended by Eng Mariam)

One-Class SVM is ideal for anomaly detection because:
- Trains only on normal traffic (no need for labeled attacks)
- Detects deviations from normal behavior
- Effective for detecting unknown attack patterns
- Suitable for large-scale log analysis

## Implementation Steps

### Step 1: Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install additional dependencies for SIEM integration
pip install kagglehub requests
```

### Step 2: Load Large-Scale Dataset

**Option A: Using Kaggle Dataset (Recommended)**

```python
from siem_integration.dataset_loader import DatasetLoader

# List available datasets
DatasetLoader.list_recommended_datasets()

# Load NF-CSE-CIC-IDS2018 dataset
df = DatasetLoader.load_recommended_dataset(
    'NF-CSE-CIC-IDS2018',
    sample_size=100000  # Adjust based on your needs
)
```

**Option B: Using Local File**

```python
from siem_integration.dataset_loader import DatasetLoader

df = DatasetLoader.load_from_local(
    'path/to/your/network_logs.csv',
    sample_size=100000  # Optional: sample for faster processing
)
```

### Step 3: Train One-Class SVM Model

```python
from Models.AnomalyDetection-OCSVM.train_ocsvm import OCSVMTrainer

# Initialize trainer
trainer = OCSVMTrainer(
    output_dir="Models/AnomalyDetection-OCSVM/outputs",
    random_state=42
)

# Load data (normal traffic only for training)
df_train, df_full = trainer.load_data(
    kaggle_dataset="mohamedelrifai/network-anomaly-detection-dataset",
    kaggle_filename="sampled_NF-CSE-CIC-IDS2018-v2.csv",
    sample_size=50000,  # Adjust based on available resources
    use_normal_only=True  # IMPORTANT: Train only on normal traffic
)

# Preprocess data
X_train = trainer.preprocess_data(df_train)

# Train model
trainer.train(
    X_train,
    nu=0.1,        # Outlier fraction (0.01-0.5). Lower = more sensitive
    gamma='scale', # Kernel coefficient
    kernel='rbf'   # Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
)

# Save model
trainer.save_model()
```

**Model Parameters Explained:**
- `nu`: Expected fraction of outliers (0.1 = 10% outliers expected)
- `gamma`: Kernel coefficient ('scale' recommended for large feature sets)
- `kernel`: 'rbf' (Radial Basis Function) recommended for non-linear patterns

### Step 4: Simulate Attack Scenarios

```python
from siem_integration.attack_simulator import AttackSimulator, save_attack_logs

# Initialize simulator
simulator = AttackSimulator()

# Simulate individual attack
ddos_attacks = simulator.simulate_attack_scenario(
    attack_type='ddos',
    duration_seconds=60,
    attack_rate=2.0  # Attacks per second
)

# Simulate mixed attack scenario (multiple attack types)
mixed_attacks = simulator.generate_mixed_attack_scenario(
    attack_types=['ddos', 'port_scan', 'brute_force', 'data_exfiltration'],
    duration_seconds=300,  # 5 minutes
    attack_rate=1.0  # Total attacks per second
)

# Save attack logs
save_attack_logs(mixed_attacks, "simulated_attacks.csv")
```

**Available Attack Types:**
- `ddos` - Distributed Denial of Service
- `port_scan` - Port Scanning
- `brute_force` - Brute Force Login Attempts
- `data_exfiltration` - Data Exfiltration
- `malware_communication` - Malware C&C Communication
- `dns_tunneling` - DNS Tunneling
- `sql_injection` - SQL Injection
- `icmp_flood` - ICMP Flood Attack

### Step 5: Real-time Anomaly Detection & SIEM Integration

```python
from siem_integration.siem_connector import SIEMConnector
import pandas as pd

# Initialize SIEM connector
connector = SIEMConnector(
    model_path="Models/AnomalyDetection-OCSVM/outputs/one_class_svm_model.pkl",
    scaler_path="Models/AnomalyDetection-OCSVM/outputs/standard_scaler.pkl",
    siem_api_url="http://your-siem-dashboard.com/api/alerts",  # Your SIEM API endpoint
    api_key="your-api-key",  # If authentication required
    alert_threshold=0.5  # Decision score threshold
)

# Load attack logs (or real-time network traffic)
attack_logs = pd.read_csv("simulated_attacks.csv")

# Process logs and detect anomalies
alerts = connector.process_log_stream(
    attack_logs,
    send_to_siem=True,  # Send alerts to SIEM dashboard
    batch_size=100
)

# Save alerts to file (backup)
connector.save_alerts_to_file(alerts, "siem_alerts.json")
```

### Step 6: Run Complete Pipeline

For automated execution, use the complete pipeline:

```bash
cd siem_integration
python run_siem_pipeline.py
```

Edit `run_siem_pipeline.py` to configure:
- Dataset source and size
- Model training parameters
- Attack simulation parameters
- SIEM API endpoint

## SIEM Dashboard Integration

### Alert Format

The system sends alerts in JSON format:

```json
{
    "alert_id": "ALERT-20240101120000000",
    "timestamp": "2024-01-01T12:00:00",
    "severity": "high",
    "status": "new",
    "source": "One-Class SVM Anomaly Detection",
    "anomaly_score": 8.5,
    "decision_score": -8.5,
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
        "Enable rate limiting",
        "Scale up resources to handle traffic"
    ],
    "log_data": { /* Full log entry */ }
}
```

### Severity Levels

- **Critical**: Decision score > 10
- **High**: Decision score > 5
- **Medium**: Decision score > 2
- **Low**: Decision score ≤ 2

### SIEM API Endpoint Requirements

Your SIEM dashboard should accept POST requests to:
- Endpoint: `/api/alerts` (or your custom endpoint)
- Method: POST
- Content-Type: application/json
- Authentication: Bearer token (if required)

Example SIEM API handler (for testing):

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/alerts', methods=['POST'])
def receive_alert():
    alert = request.json
    # Process alert in your SIEM system
    print(f"Received alert: {alert['alert_id']}")
    return jsonify({"status": "received", "alert_id": alert['alert_id']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

## Testing the Solution

### 1. Test Model Training

```bash
cd Models/AnomalyDetection-OCSVM
python train_ocsvm.py
```

### 2. Test Attack Simulation

```bash
cd siem_integration
python attack_simulator.py
```

### 3. Test Detection

```bash
cd siem_integration
python siem_connector.py
```

### 4. Test Complete Pipeline

```bash
cd siem_integration
python run_siem_pipeline.py
```

## Performance Optimization

### For Large Datasets:

1. **Use Sampling**: Set `sample_size` parameter when loading datasets
2. **Chunked Processing**: Process logs in batches
3. **Parallel Processing**: Use multiple cores for model training
4. **Model Caching**: Save trained models to avoid retraining

### Example Configuration for Large Datasets:

```python
# Load 100K samples for training
df = DatasetLoader.load_recommended_dataset(
    'NF-CSE-CIC-IDS2018',
    sample_size=100000
)

# Train with optimized parameters
trainer.train(
    X_train,
    nu=0.05,  # Lower nu for large datasets
    gamma='scale',
    kernel='rbf'
)
```

## Troubleshooting

### Issue: Dataset Loading Fails

**Solution:**
- Ensure `kagglehub` is installed: `pip install kagglehub`
- Check Kaggle credentials
- Use local file path as alternative

### Issue: Model Training Takes Too Long

**Solution:**
- Reduce `sample_size` parameter
- Use smaller `nu` value (0.05 instead of 0.1)
- Consider using 'linear' kernel for faster training

### Issue: Too Many False Positives

**Solution:**
- Increase `nu` parameter (0.2-0.3)
- Adjust `alert_threshold` in SIEM connector
- Retrain with more diverse normal traffic

### Issue: SIEM API Connection Fails

**Solution:**
- Verify SIEM API URL is correct
- Check network connectivity
- Verify API authentication credentials
- Review firewall rules

## Next Steps

1. **Deploy to Production**: Set up continuous monitoring
2. **Fine-tune Model**: Adjust parameters based on your network patterns
3. **Customize Attacks**: Add domain-specific attack patterns
4. **Dashboard Integration**: Connect with your SIEM dashboard
5. **Monitoring**: Set up alerting for the detection system itself

## Support

For questions or issues:
- Review the code comments
- Check error messages and logs
- Verify all dependencies are installed
- Ensure dataset format matches expected structure

---

**Note**: This solution uses One-Class SVM as recommended by Eng Mariam for detecting anomalies in large-scale network logs without requiring labeled attack data.
