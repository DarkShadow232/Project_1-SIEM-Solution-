"""
SIEM Integration Package
Network Anomaly Detection using One-Class SVM for SIEM Security Solution
"""

from .attack_simulator import AttackSimulator, save_attack_logs
from .siem_connector import SIEMConnector, simulate_realtime_detection
from .dataset_loader import DatasetLoader

__all__ = [
    'AttackSimulator',
    'save_attack_logs',
    'SIEMConnector',
    'simulate_realtime_detection',
    'DatasetLoader'
]

__version__ = '1.0.0'

