# Quick Start Guide - SIEM Integration

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
pip install kagglehub requests
```

### Step 2: Run Complete Pipeline

```bash
cd siem_integration
python run_siem_pipeline.py
```

This will:
1. ✅ Load dataset (NF-CSE-CIC-IDS2018)
2. ✅ Train One-Class SVM model
3. ✅ Simulate attack scenarios
4. ✅ Detect anomalies
5. ✅ Generate alerts

### Step 3: Configure SIEM Dashboard (Optional)

Edit `siem_integration/run_siem_pipeline.py`:

```python
CONFIG = {
    'siem_api_url': 'http://your-siem-dashboard.com/api/alerts',
    'siem_api_key': 'your-api-key',
    # ... other settings
}
```

## 📋 What You Get

After running the pipeline:

- **Trained Model**: `Models/AnomalyDetection-OCSVM/outputs/one_class_svm_model.pkl`
- **Attack Logs**: `simulated_attacks.csv`
- **SIEM Alerts**: `siem_alerts.json`

## 🎯 Key Features

✅ **One-Class SVM** (Recommended by Eng Mariam)
- Trains on normal traffic only
- Detects unknown attack patterns
- Suitable for large-scale logs

✅ **Attack Simulation**
- 8+ attack types
- Realistic attack patterns
- Configurable intensity

✅ **SIEM Integration**
- REST API support
- Alert formatting
- Severity classification

## 📚 Detailed Documentation

- **Complete Guide**: See `SIEM_INTEGRATION_GUIDE.md`
- **Module Documentation**: See `siem_integration/README.md`

## 🔧 Customization

### Use Your Own Dataset

```python
# In run_siem_pipeline.py
CONFIG = {
    'dataset_file_path': 'path/to/your/logs.csv',
    'dataset_name': None,  # Set to None when using local file
}
```

### Adjust Model Sensitivity

```python
CONFIG = {
    'nu': 0.1,  # Lower = more sensitive (0.01-0.5)
    'gamma': 'scale',
    'kernel': 'rbf',
}
```

### Customize Attack Simulation

```python
CONFIG = {
    'attack_types': ['ddos', 'port_scan', 'brute_force'],
    'attack_duration_seconds': 60,
    'attack_rate': 2.0,  # Attacks per second
}
```

## 🆘 Troubleshooting

**Dataset not loading?**
- Install kagglehub: `pip install kagglehub`
- Or use local file path

**Model training slow?**
- Reduce `sample_size` in CONFIG
- Use smaller dataset

**SIEM API errors?**
- Check API URL is correct
- Verify network connectivity
- Check authentication credentials

## 📞 Need Help?

1. Check `SIEM_INTEGRATION_GUIDE.md` for detailed instructions
2. Review error messages in console output
3. Verify all dependencies are installed

---

**Ready to detect anomalies!** 🎉

