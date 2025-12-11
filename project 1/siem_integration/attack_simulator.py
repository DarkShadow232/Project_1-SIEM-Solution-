"""
Attack Simulation Module for SIEM Security Solution
Simulates realistic network attacks to test anomaly detection system

This module generates various attack scenarios that can be detected
by the One-Class SVM model and sent to SIEM dashboard.
"""

import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class AttackSimulator:
    """Simulates various network attack scenarios"""
    
    def __init__(self, base_normal_profile: Optional[pd.DataFrame] = None):
        """
        Initialize attack simulator.
        
        Parameters:
        -----------
        base_normal_profile : pd.DataFrame, optional
            Normal traffic profile to base attacks on
        """
        self.base_profile = base_normal_profile
        self.attack_logs = []
        
        # Common attack patterns
        self.attack_patterns = {
            'ddos': {
                'description': 'Distributed Denial of Service Attack',
                'features': {
                    'IN_PKTS': lambda x: x * 100,  # Massive packet increase
                    'OUT_PKTS': lambda x: x * 100,
                    'FLOW_DURATION_MILLISECONDS': lambda x: x * 0.1,  # Very short duration
                    'L4_DST_PORT': lambda x: random.choice([80, 443, 53]),  # Common ports
                }
            },
            'port_scan': {
                'description': 'Port Scanning Attack',
                'features': {
                    'L4_DST_PORT': lambda x: random.randint(1, 65535),  # Random ports
                    'TCP_FLAGS': lambda x: 2,  # SYN flag
                    'FLOW_DURATION_MILLISECONDS': lambda x: x * 0.01,  # Very short
                    'IN_PKTS': lambda x: 1,  # Single packet
                    'OUT_PKTS': lambda x: 1,
                }
            },
            'brute_force': {
                'description': 'Brute Force Login Attempt',
                'features': {
                    'L4_DST_PORT': lambda x: random.choice([22, 23, 3389, 3306]),  # SSH, Telnet, RDP, MySQL
                    'TCP_FLAGS': lambda x: 24,  # PSH+ACK
                    'IN_PKTS': lambda x: random.randint(5, 20),
                    'OUT_PKTS': lambda x: random.randint(5, 20),
                    'FLOW_DURATION_MILLISECONDS': lambda x: random.randint(1000, 5000),
                }
            },
            'data_exfiltration': {
                'description': 'Data Exfiltration Attack',
                'features': {
                    'OUT_BYTES': lambda x: x * 1000,  # Large outbound data
                    'OUT_PKTS': lambda x: x * 50,
                    'L4_DST_PORT': lambda x: random.choice([443, 80, 53]),  # HTTPS, HTTP, DNS
                    'FLOW_DURATION_MILLISECONDS': lambda x: x * 10,  # Long duration
                }
            },
            'malware_communication': {
                'description': 'Malware Command and Control Communication',
                'features': {
                    'L4_DST_PORT': lambda x: random.choice([4444, 5555, 6666, 8080]),  # Uncommon ports
                    'IN_BYTES': lambda x: random.randint(100, 1000),
                    'OUT_BYTES': lambda x: random.randint(100, 1000),
                    'FLOW_DURATION_MILLISECONDS': lambda x: random.randint(30000, 300000),  # Long connection
                }
            },
            'dns_tunneling': {
                'description': 'DNS Tunneling Attack',
                'features': {
                    'L4_DST_PORT': lambda x: 53,  # DNS port
                    'DNS_QUERY_ID': lambda x: random.randint(1, 65535),
                    'DNS_QUERY_TYPE': lambda x: random.choice([1, 12, 16, 28]),
                    'IN_PKTS': lambda x: random.randint(10, 100),
                    'OUT_PKTS': lambda x: random.randint(10, 100),
                }
            },
            'sql_injection': {
                'description': 'SQL Injection Attack',
                'features': {
                    'L4_DST_PORT': lambda x: random.choice([3306, 1433, 5432]),  # MySQL, MSSQL, PostgreSQL
                    'IN_BYTES': lambda x: x * 5,  # Larger input
                    'TCP_FLAGS': lambda x: 24,  # PSH+ACK
                    'FLOW_DURATION_MILLISECONDS': lambda x: random.randint(5000, 15000),
                }
            },
            'icmp_flood': {
                'description': 'ICMP Flood Attack',
                'features': {
                    'PROTOCOL': lambda x: 1,  # ICMP
                    'ICMP_TYPE': lambda x: 8,  # Echo request
                    'IN_PKTS': lambda x: x * 1000,
                    'OUT_PKTS': lambda x: x * 1000,
                    'FLOW_DURATION_MILLISECONDS': lambda x: x * 0.001,
                }
            }
        }
    
    def generate_attack_log(self, attack_type: str, base_log: Optional[pd.Series] = None, 
                           timestamp: Optional[datetime] = None) -> Dict:
        """
        Generate a single attack log entry.
        
        Parameters:
        -----------
        attack_type : str
            Type of attack ('ddos', 'port_scan', 'brute_force', etc.)
        base_log : pd.Series, optional
            Base log entry to modify (if None, creates from scratch)
        timestamp : datetime, optional
            Timestamp for the log entry
        
        Returns:
        --------
        dict: Attack log entry
        """
        if attack_type not in self.attack_patterns:
            raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(self.attack_patterns.keys())}")
        
        pattern = self.attack_patterns[attack_type]
        
        # Start with base log or create default
        if base_log is not None:
            log = base_log.to_dict()
        else:
            log = self._create_default_log()
        
        # Apply attack pattern modifications
        for feature, transform in pattern['features'].items():
            if feature in log:
                original_value = log[feature]
                if isinstance(original_value, (int, float)) and not np.isnan(original_value):
                    log[feature] = transform(original_value)
                else:
                    log[feature] = transform(1)  # Default value
        
        # Add attack metadata
        log['Attack'] = attack_type.upper()
        log['Attack_Category'] = self._categorize_attack(attack_type)
        log['Label'] = 1  # Mark as anomaly
        log['timestamp'] = timestamp or datetime.now()
        log['attack_description'] = pattern['description']
        
        return log
    
    def _create_default_log(self) -> Dict:
        """Create a default network log entry"""
        return {
            'IPV4_SRC_ADDR': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            'IPV4_DST_ADDR': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            'L4_SRC_PORT': random.randint(1024, 65535),
            'L4_DST_PORT': random.randint(1, 65535),
            'PROTOCOL': random.choice([6, 17, 1]),  # TCP, UDP, ICMP
            'L7_PROTO': random.choice([8.0, 9.0, 7.0]),  # HTTP, HTTPS, DNS
            'IN_BYTES': random.randint(100, 10000),
            'IN_PKTS': random.randint(1, 100),
            'OUT_BYTES': random.randint(100, 10000),
            'OUT_PKTS': random.randint(1, 100),
            'TCP_FLAGS': random.choice([16, 24, 2, 18]),  # ACK, PSH+ACK, SYN, SYN+ACK
            'CLIENT_TCP_FLAGS': random.choice([2, 16, 24]),
            'SERVER_TCP_FLAGS': random.choice([18, 24]),
            'FLOW_DURATION_MILLISECONDS': random.randint(100, 10000),
            'DURATION_IN': random.randint(50, 5000),
            'DURATION_OUT': random.randint(50, 5000),
            'MIN_TTL': random.randint(32, 255),
            'MAX_TTL': random.randint(32, 255),
            'LONGEST_FLOW_PKT': random.randint(64, 1514),
            'SHORTEST_FLOW_PKT': random.randint(64, 512),
            'MIN_IP_PKT_LEN': random.randint(64, 512),
            'MAX_IP_PKT_LEN': random.randint(512, 1514),
            'SRC_TO_DST_SECOND_BYTES': random.randint(0, 10000),
            'DST_TO_SRC_SECOND_BYTES': random.randint(0, 10000),
            'RETRANSMITTED_IN_BYTES': random.randint(0, 1000),
            'RETRANSMITTED_IN_PKTS': random.randint(0, 10),
            'RETRANSMITTED_OUT_BYTES': random.randint(0, 1000),
            'RETRANSMITTED_OUT_PKTS': random.randint(0, 10),
            'SRC_TO_DST_AVG_THROUGHPUT': random.randint(100, 1000000),
            'DST_TO_SRC_AVG_THROUGHPUT': random.randint(100, 1000000),
            'NUM_PKTS_UP_TO_128_BYTES': random.randint(0, 50),
            'NUM_PKTS_128_TO_256_BYTES': random.randint(0, 50),
            'NUM_PKTS_256_TO_512_BYTES': random.randint(0, 50),
            'NUM_PKTS_512_TO_1024_BYTES': random.randint(0, 50),
            'NUM_PKTS_1024_TO_1514_BYTES': random.randint(0, 50),
            'TCP_WIN_MAX_IN': random.randint(65535, 65535),
            'TCP_WIN_MAX_OUT': random.randint(65535, 65535),
            'ICMP_TYPE': 0,
            'ICMP_IPV4_TYPE': 0,
            'DNS_QUERY_ID': 0,
            'DNS_QUERY_TYPE': 0,
            'DNS_TTL_ANSWER': 0,
            'FTP_COMMAND_RET_CODE': 0,
            'Label': 0,  # Will be changed to 1 for attacks
        }
    
    def _categorize_attack(self, attack_type: str) -> str:
        """Categorize attack type"""
        categories = {
            'ddos': 'DoS',
            'icmp_flood': 'DoS',
            'port_scan': 'Reconnaissance',
            'brute_force': 'Brute Force',
            'data_exfiltration': 'Data Exfiltration',
            'malware_communication': 'Malware',
            'dns_tunneling': 'Tunneling',
            'sql_injection': 'Web Attack',
        }
        return categories.get(attack_type, 'Unknown')
    
    def simulate_attack_scenario(self, attack_type: str, duration_seconds: int = 60,
                                attack_rate: float = 1.0, base_logs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Simulate an attack scenario over time.
        
        Parameters:
        -----------
        attack_type : str
            Type of attack to simulate
        duration_seconds : int
            Duration of attack simulation in seconds
        attack_rate : float
            Attacks per second
        base_logs : pd.DataFrame, optional
            Base normal logs to modify
        
        Returns:
        --------
        pd.DataFrame: DataFrame of attack logs
        """
        print(f"Simulating {attack_type} attack for {duration_seconds} seconds...")
        
        attack_logs = []
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration_seconds)
        
        attack_count = 0
        total_attacks = int(duration_seconds * attack_rate)
        
        while datetime.now() < end_time and attack_count < total_attacks:
            # Select base log if available
            base_log = None
            if base_logs is not None and len(base_logs) > 0:
                base_log = base_logs.sample(n=1).iloc[0]
            
            # Generate attack log
            timestamp = datetime.now()
            attack_log = self.generate_attack_log(attack_type, base_log, timestamp)
            attack_logs.append(attack_log)
            
            attack_count += 1
            
            # Sleep to maintain attack rate
            if attack_rate > 0:
                time.sleep(1.0 / attack_rate)
        
        print(f"Generated {len(attack_logs)} attack logs")
        return pd.DataFrame(attack_logs)
    
    def generate_mixed_attack_scenario(self, attack_types: List[str], 
                                      duration_seconds: int = 300,
                                      attack_rate: float = 0.5) -> pd.DataFrame:
        """
        Generate a mixed attack scenario with multiple attack types.
        
        Parameters:
        -----------
        attack_types : List[str]
            List of attack types to include
        duration_seconds : int
            Total duration
        attack_rate : float
            Attacks per second (total across all types)
        
        Returns:
        --------
        pd.DataFrame: DataFrame of mixed attack logs
        """
        print(f"Simulating mixed attack scenario with {len(attack_types)} attack types...")
        
        all_attacks = []
        attacks_per_type = int((duration_seconds * attack_rate) / len(attack_types))
        
        for attack_type in attack_types:
            attacks = self.simulate_attack_scenario(
                attack_type, 
                duration_seconds=duration_seconds // len(attack_types),
                attack_rate=attack_rate / len(attack_types)
            )
            all_attacks.append(attacks)
        
        combined = pd.concat(all_attacks, ignore_index=True)
        combined = combined.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        print(f"Total attack logs generated: {len(combined)}")
        return combined


def save_attack_logs(attack_logs: pd.DataFrame, filename: str = "simulated_attacks.csv"):
    """Save attack logs to CSV file"""
    attack_logs.to_csv(filename, index=False)
    print(f"Attack logs saved to {filename}")


if __name__ == "__main__":
    # Example usage
    simulator = AttackSimulator()
    
    # Simulate individual attacks
    print("=" * 60)
    print("Attack Simulation Examples")
    print("=" * 60)
    
    # Example 1: DDoS attack
    ddos_attacks = simulator.simulate_attack_scenario('ddos', duration_seconds=10, attack_rate=2.0)
    print(f"\nDDoS Attack Logs:\n{ddos_attacks[['Attack', 'IN_PKTS', 'OUT_PKTS', 'timestamp']].head()}")
    
    # Example 2: Port scan
    port_scan_attacks = simulator.simulate_attack_scenario('port_scan', duration_seconds=10, attack_rate=5.0)
    print(f"\nPort Scan Attack Logs:\n{port_scan_attacks[['Attack', 'L4_DST_PORT', 'TCP_FLAGS', 'timestamp']].head()}")
    
    # Example 3: Mixed attack scenario
    mixed_attacks = simulator.generate_mixed_attack_scenario(
        ['ddos', 'port_scan', 'brute_force'],
        duration_seconds=30,
        attack_rate=1.0
    )
    print(f"\nMixed Attack Scenario:\n{mixed_attacks['Attack'].value_counts()}")
    
    # Save to file
    save_attack_logs(mixed_attacks, "simulated_attacks.csv")

