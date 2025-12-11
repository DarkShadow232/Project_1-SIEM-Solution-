"""
Complete SIEM Pipeline Runner
Trains One-Class SVM model and simulates attack detection

This script:
1. Loads large-scale network log dataset
2. Trains One-Class SVM model (recommended by Eng Mariam)
3. Simulates attack scenarios
4. Detects anomalies in real-time
5. Sends alerts to SIEM dashboard
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import OCSVM trainer - adjust path as needed
# Note: Directory name uses hyphen, so we need to import differently
ocsvm_dir = os.path.join(parent_dir, 'Models', 'AnomalyDetection-OCSVM')
if os.path.exists(ocsvm_dir):
    sys.path.insert(0, ocsvm_dir)
    from train_ocsvm import OCSVMTrainer
else:
    raise ImportError(f"OCSVM training module not found at {ocsvm_dir}")

# Import SIEM integration modules
from siem_integration.attack_simulator import AttackSimulator, save_attack_logs
from siem_integration.siem_connector import SIEMConnector, simulate_realtime_detection
from siem_integration.dataset_loader import DatasetLoader


def main():
    """Main pipeline execution"""
    print("=" * 80)
    print("SIEM Security Solution - Complete Pipeline")
    print("One-Class SVM Anomaly Detection (Recommended by Eng Mariam)")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        # Dataset configuration
        'dataset_name': 'NF-CSE-CIC-IDS2018',  # or use 'local' with file_path
        'dataset_file_path': None,  # Set if using local file
        'sample_size': 50000,  # None for full dataset, or number for sampling
        
        # Model training configuration
        'train_model': True,  # Set False to use existing model
        'model_output_dir': 'Models/AnomalyDetection-OCSVM/outputs',
        'nu': 0.1,  # Outlier fraction (adjust based on expected anomaly rate)
        'gamma': 'scale',
        'kernel': 'rbf',
        
        # Attack simulation configuration
        'simulate_attacks': True,
        'attack_types': ['ddos', 'port_scan', 'brute_force', 'data_exfiltration'],
        'attack_duration_seconds': 60,
        'attack_rate': 2.0,  # Attacks per second
        
        # SIEM integration configuration
        'siem_api_url': None,  # Set to your SIEM API endpoint
        'siem_api_key': None,  # Set if authentication required
        'process_attacks': True,  # Detect anomalies in simulated attacks
    }
    
    # Step 1: Load Dataset
    print("\n" + "=" * 80)
    print("STEP 1: Loading Dataset")
    print("=" * 80)
    
    try:
        if CONFIG['dataset_file_path']:
            df_train, df_full = DatasetLoader.load_from_local(
                CONFIG['dataset_file_path'],
                sample_size=CONFIG['sample_size']
            ), None
        else:
            df_train, df_full = DatasetLoader.load_recommended_dataset(
                CONFIG['dataset_name'],
                sample_size=CONFIG['sample_size']
            ), None
        
        print(f"✅ Dataset loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTrying alternative: Loading from local sample file...")
        try:
            df_train = pd.read_csv('sample_for_testing.csv')
            df_full = None
            print("✅ Using sample file")
        except:
            print("❌ Could not load dataset. Please check configuration.")
            return
    
    # Step 2: Train One-Class SVM Model
    print("\n" + "=" * 80)
    print("STEP 2: Training One-Class SVM Model")
    print("=" * 80)
    
    model_path = os.path.join(CONFIG['model_output_dir'], 'one_class_svm_model.pkl')
    scaler_path = os.path.join(CONFIG['model_output_dir'], 'standard_scaler.pkl')
    
    if CONFIG['train_model'] and not os.path.exists(model_path):
        trainer = OCSVMTrainer(
            output_dir=CONFIG['model_output_dir'],
            random_state=42
        )
        
        # Preprocess data
        X_train = trainer.preprocess_data(df_train)
        
        # Train model
        trainer.train(
            X_train,
            nu=CONFIG['nu'],
            gamma=CONFIG['gamma'],
            kernel=CONFIG['kernel']
        )
        
        # Save model
        trainer.save_model()
        
        print("✅ Model training completed!")
    else:
        if os.path.exists(model_path):
            print(f"✅ Using existing model from {model_path}")
        else:
            print("❌ Model not found and training disabled. Please train model first.")
            return
    
    # Step 3: Simulate Attack Scenarios
    print("\n" + "=" * 80)
    print("STEP 3: Simulating Attack Scenarios")
    print("=" * 80)
    
    attack_logs = None
    
    if CONFIG['simulate_attacks']:
        simulator = AttackSimulator()
        
        # Generate mixed attack scenario
        attack_logs = simulator.generate_mixed_attack_scenario(
            attack_types=CONFIG['attack_types'],
            duration_seconds=CONFIG['attack_duration_seconds'],
            attack_rate=CONFIG['attack_rate']
        )
        
        # Save attack logs
        attack_file = 'simulated_attacks.csv'
        save_attack_logs(attack_logs, attack_file)
        print(f"✅ Attack simulation completed. Logs saved to {attack_file}")
    
    # Step 4: Real-time Anomaly Detection
    print("\n" + "=" * 80)
    print("STEP 4: Real-time Anomaly Detection")
    print("=" * 80)
    
    if CONFIG['process_attacks'] and attack_logs is not None:
        alerts = simulate_realtime_detection(
            model_path=model_path,
            scaler_path=scaler_path,
            attack_logs=attack_logs,
            siem_api_url=CONFIG['siem_api_url']
        )
        
        print(f"\n✅ Detection completed. Generated {len(alerts)} alerts")
        
        # Display summary
        if alerts:
            print("\nAlert Summary:")
            attack_types = {}
            for alert in alerts:
                attack_type = alert.get('attack_info', {}).get('attack_type', 'unknown')
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
            
            for attack_type, count in attack_types.items():
                print(f"  {attack_type}: {count} alerts")
    
    # Step 5: SIEM Integration (if configured)
    if CONFIG['siem_api_url']:
        print("\n" + "=" * 80)
        print("STEP 5: SIEM Dashboard Integration")
        print("=" * 80)
        print(f"✅ Alerts sent to SIEM dashboard: {CONFIG['siem_api_url']}")
    else:
        print("\n" + "=" * 80)
        print("STEP 5: SIEM Dashboard Integration")
        print("=" * 80)
        print("⚠️ SIEM API URL not configured. Alerts saved to file only.")
        print("   To enable SIEM integration, set 'siem_api_url' in CONFIG")
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Execution Completed!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Review generated alerts in 'siem_alerts.json'")
    print("2. Configure SIEM API URL to send alerts to dashboard")
    print("3. Integrate with your SIEM Security Solution")
    print("=" * 80)


if __name__ == "__main__":
    main()

