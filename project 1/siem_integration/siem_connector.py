"""
SIEM Integration Module
Connects anomaly detection system with SIEM Security Dashboard

This module handles real-time anomaly detection, alert generation,
and communication with SIEM dashboard via REST API or message queue.
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
import requests
from queue import Queue
import threading
import joblib
import os


class SIEMConnector:
    """Connector for SIEM Security Dashboard"""
    
    def __init__(self, 
                 model_path: str = None,
                 scaler_path: str = None,
                 siem_api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 alert_threshold: float = 0.5):
        """
        Initialize SIEM Connector.
        
        Parameters:
        -----------
        model_path : str
            Path to trained One-Class SVM model
        scaler_path : str
            Path to StandardScaler
        siem_api_url : str, optional
            SIEM dashboard API endpoint URL
        api_key : str, optional
            API key for authentication
        alert_threshold : float
            Threshold for alert generation (decision score)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.siem_api_url = siem_api_url
        self.api_key = api_key
        self.alert_threshold = alert_threshold
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Alert queue for batch processing
        self.alert_queue = Queue()
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'alerts_sent': 0,
            'alerts_failed': 0
        }
        
        # Load model if paths provided
        if model_path and scaler_path:
            self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load feature names if available
            features_path = os.path.join(os.path.dirname(model_path), 'feature_names.json')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
            
            print(f"✅ Model loaded from {model_path}")
            print(f"✅ Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_log(self, log: pd.Series) -> np.ndarray:
        """
        Preprocess a single log entry for prediction.
        
        Parameters:
        -----------
        log : pd.Series
            Network log entry
        
        Returns:
        --------
        np.ndarray: Preprocessed features
        """
        # Remove categorical columns (same as training)
        categorical_cols = [
            'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT',
            'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS',
            'SERVER_TCP_FLAGS', 'ICMP_TYPE', 'ICMP_IPV4_TYPE',
            'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE',
            'SRC_IP_CLASS', 'DST_IP_CLASS', 'ICMP_TYPE_LABEL',
            'ICMP_IPV4_TYPE_LABEL', 'DNS_QUERY_TYPE_LABEL',
            'FTP_RET_CATEGORY', 'PROTOCOL_LABEL', 'L7_PROTO_LABEL',
            'SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY', 'DST_SERVICE',
            'SRC_SERVICE', 'Label', 'Attack', 'Attack_Category', 'timestamp',
            'attack_description'
        ]
        
        # Convert to DataFrame for easier processing
        log_df = pd.DataFrame([log])
        log_cleaned = log_df.drop(columns=categorical_cols, errors='ignore')
        
        # Handle missing values
        log_cleaned = log_cleaned.fillna(0)
        
        # Ensure feature order matches training
        if self.feature_names:
            # Reorder and fill missing features
            for feature in self.feature_names:
                if feature not in log_cleaned.columns:
                    log_cleaned[feature] = 0
            log_cleaned = log_cleaned[self.feature_names]
        
        return log_cleaned.values
    
    def detect_anomaly(self, log: pd.Series) -> Dict:
        """
        Detect anomaly in a single log entry.
        
        Parameters:
        -----------
        log : pd.Series
            Network log entry
        
        Returns:
        --------
        dict: Detection results with anomaly status, score, and metadata
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess log
            X = self.preprocess_log(log)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict (1 = normal, -1 = anomaly)
            prediction = self.model.predict(X_scaled)[0]
            
            # Get decision score (distance from hyperplane)
            decision_score = self.model.decision_function(X_scaled)[0]
            
            # Determine if anomaly (negative score = anomaly)
            is_anomaly = prediction == -1 or decision_score < -self.alert_threshold
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': bool(is_anomaly),
                'prediction': int(prediction),
                'decision_score': float(decision_score),
                'anomaly_score': float(abs(decision_score)) if is_anomaly else 0.0,
                'src_ip': log.get('IPV4_SRC_ADDR', 'unknown'),
                'dst_ip': log.get('IPV4_DST_ADDR', 'unknown'),
                'src_port': log.get('L4_SRC_PORT', 0),
                'dst_port': log.get('L4_DST_PORT', 0),
                'protocol': log.get('PROTOCOL', 0),
                'attack_type': log.get('Attack', 'unknown'),
                'attack_category': log.get('Attack_Category', 'unknown'),
            }
            
            self.stats['total_processed'] += 1
            if is_anomaly:
                self.stats['anomalies_detected'] += 1
            
            return result
            
        except Exception as e:
            print(f"Error detecting anomaly: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': False,
                'error': str(e)
            }
    
    def create_alert(self, detection_result: Dict, log: pd.Series) -> Dict:
        """
        Create SIEM alert from detection result.
        
        Parameters:
        -----------
        detection_result : dict
            Result from detect_anomaly()
        log : pd.Series
            Original log entry
        
        Returns:
        --------
        dict: SIEM alert format
        """
        alert = {
            'alert_id': f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            'timestamp': detection_result['timestamp'],
            'severity': self._calculate_severity(detection_result),
            'status': 'new',
            'source': 'One-Class SVM Anomaly Detection',
            'anomaly_score': detection_result['anomaly_score'],
            'decision_score': detection_result['decision_score'],
            'network_info': {
                'src_ip': detection_result['src_ip'],
                'dst_ip': detection_result['dst_ip'],
                'src_port': detection_result['src_port'],
                'dst_port': detection_result['dst_port'],
                'protocol': detection_result['protocol'],
            },
            'attack_info': {
                'attack_type': detection_result['attack_type'],
                'attack_category': detection_result['attack_category'],
            },
            'log_data': log.to_dict() if isinstance(log, pd.Series) else log,
            'recommendations': self._generate_recommendations(detection_result)
        }
        
        return alert
    
    def _calculate_severity(self, detection_result: Dict) -> str:
        """Calculate alert severity based on anomaly score"""
        score = abs(detection_result['decision_score'])
        
        if score > 10:
            return 'critical'
        elif score > 5:
            return 'high'
        elif score > 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, detection_result: Dict) -> List[str]:
        """Generate security recommendations based on detection"""
        recommendations = []
        
        attack_type = detection_result.get('attack_type', '').lower()
        
        if 'ddos' in attack_type or 'flood' in attack_type:
            recommendations.extend([
                "Block source IP addresses",
                "Enable rate limiting",
                "Scale up resources to handle traffic"
            ])
        elif 'scan' in attack_type:
            recommendations.extend([
                "Investigate source IP for reconnaissance activity",
                "Consider blocking source IP if persistent",
                "Monitor for follow-up attacks"
            ])
        elif 'brute' in attack_type:
            recommendations.extend([
                "Implement account lockout policies",
                "Enable multi-factor authentication",
                "Block source IP after multiple failed attempts"
            ])
        else:
            recommendations.append("Investigate anomalous network behavior")
        
        return recommendations
    
    def send_alert_to_siem(self, alert: Dict) -> bool:
        """
        Send alert to SIEM dashboard via REST API.
        
        Parameters:
        -----------
        alert : dict
            Alert dictionary
        
        Returns:
        --------
        bool: True if successful, False otherwise
        """
        if not self.siem_api_url:
            print("⚠️ SIEM API URL not configured. Alert not sent.")
            return False
        
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.post(
                self.siem_api_url,
                json=alert,
                headers=headers,
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                self.stats['alerts_sent'] += 1
                print(f"✅ Alert sent to SIEM: {alert['alert_id']}")
                return True
            else:
                print(f"❌ Failed to send alert: {response.status_code} - {response.text}")
                self.stats['alerts_failed'] += 1
                return False
                
        except Exception as e:
            print(f"❌ Error sending alert to SIEM: {e}")
            self.stats['alerts_failed'] += 1
            return False
    
    def process_log_stream(self, log_stream: pd.DataFrame, 
                          send_to_siem: bool = True,
                          batch_size: int = 100):
        """
        Process a stream of logs and detect anomalies.
        
        Parameters:
        -----------
        log_stream : pd.DataFrame
            DataFrame of network logs
        send_to_siem : bool
            Whether to send alerts to SIEM
        batch_size : int
            Batch size for processing
        """
        print(f"Processing {len(log_stream)} logs...")
        
        alerts = []
        
        for idx, log in log_stream.iterrows():
            # Detect anomaly
            detection_result = self.detect_anomaly(log)
            
            if detection_result['is_anomaly']:
                # Create alert
                alert = self.create_alert(detection_result, log)
                alerts.append(alert)
                
                # Send to SIEM if enabled
                if send_to_siem:
                    self.send_alert_to_siem(alert)
                
                # Print detection
                print(f"🚨 Anomaly detected: {detection_result['src_ip']} -> {detection_result['dst_ip']} "
                      f"(Score: {detection_result['decision_score']:.2f})")
        
        print(f"\n✅ Processing complete:")
        print(f"   Total processed: {self.stats['total_processed']}")
        print(f"   Anomalies detected: {self.stats['anomalies_detected']}")
        print(f"   Alerts sent: {self.stats['alerts_sent']}")
        
        return alerts
    
    def save_alerts_to_file(self, alerts: List[Dict], filename: str = "siem_alerts.json"):
        """Save alerts to JSON file"""
        with open(filename, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        print(f"✅ Alerts saved to {filename}")


def simulate_realtime_detection(model_path: str, scaler_path: str, 
                                attack_logs: pd.DataFrame,
                                siem_api_url: Optional[str] = None):
    """
    Simulate real-time anomaly detection on attack logs.
    
    Parameters:
    -----------
    model_path : str
        Path to trained model
    scaler_path : str
        Path to scaler
    attack_logs : pd.DataFrame
        Attack logs to process
    siem_api_url : str, optional
        SIEM API URL
    """
    print("=" * 60)
    print("Real-time Anomaly Detection Simulation")
    print("=" * 60)
    
    # Initialize connector
    connector = SIEMConnector(
        model_path=model_path,
        scaler_path=scaler_path,
        siem_api_url=siem_api_url
    )
    
    # Process logs
    alerts = connector.process_log_stream(attack_logs, send_to_siem=(siem_api_url is not None))
    
    # Save alerts
    if alerts:
        connector.save_alerts_to_file(alerts)
    
    return alerts


if __name__ == "__main__":
    # Example usage
    print("SIEM Connector Example")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "Models/AnomalyDetection-OCSVM/outputs/one_class_svm_model.pkl"
    SCALER_PATH = "Models/AnomalyDetection-OCSVM/outputs/standard_scaler.pkl"
    SIEM_API_URL = None  # Set to your SIEM API endpoint, e.g., "http://localhost:8000/api/alerts"
    
    # Load attack logs (from simulator)
    try:
        attack_logs = pd.read_csv("simulated_attacks.csv")
        print(f"Loaded {len(attack_logs)} attack logs")
        
        # Simulate real-time detection
        alerts = simulate_realtime_detection(
            MODEL_PATH,
            SCALER_PATH,
            attack_logs,
            SIEM_API_URL
        )
        
        print(f"\n✅ Generated {len(alerts)} alerts")
        
    except FileNotFoundError:
        print("⚠️ simulated_attacks.csv not found. Run attack_simulator.py first.")

