"""
Network Anomaly Detection - Feature Engineering Module
Modular functions for processing network traffic data
"""

import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import OrdinalEncoder
from typing import Dict, List, Tuple, Any


class NetworkFeatureEngineer:
    """Main class for network traffic feature engineering"""
    
    def __init__(self):
        self.encoders = {}
        self.mappings = {}
        self._initialize_mappings()
    
    def _initialize_mappings(self):
        """Initialize static mappings for protocols and services"""
        self.mappings = {
            'protocol': {
                1: 'ICMP', 2: 'IGMP', 6: 'TCP', 17: 'UDP', 
                47: 'GRE', 58: 'ICMPv6', 255: 'RAW'
            },
            'l7_proto': {
                0.000: 'Unknown', 1.000: 'ICMP', 5.119: 'TLS',
                5.120: 'SSLv3/TLS', 5.124: 'TLSv1.2', 5.126: 'TLSv1.3/HTTPS',
                5.178: 'TLS Handshake', 5.212: 'TLS Encrypted', 5.233: 'TLS Certificate',
                5.239: 'TLS Record', 5.240: 'TLS Alert', 7.000: 'DNS',
                7.126: 'DNS-over-HTTPS', 7.178: 'DNS-over-TLS', 8: 'HTTP',
                9: 'HTTPS', 10: 'SMTP', 11: 'POP3', 12: 'IMAP', 13: 'FTP',
                41.000: 'QUIC/HTTP', 77.000: 'Facebook/GQUIC', 88.000: 'Facebook/Telegram',
                91.000: 'QUIC/TLS-JA3', 91.119: 'HTTPS-JA3', 91.120: 'HTTPS-JA3-Alt',
                91.126: 'HTTP/2', 91.140: 'HTTP3/Encrypted Web', 91.178: 'HTTP/3/HTTP-JA3',
                91.212: 'Cloudflare QUIC', 91.220: 'Google QUIC', 91.239: 'Cloud Apps QUIC',
                91.240: 'QUIC Alt/Fallback', 92.000: 'Streaming/SSL-Fallback',
                131.000: 'Zoom/Encrypted P2P', 131.700: 'Zoom-Encrypted'
            },
            'icmp_type': {
                0: 'Echo Reply', 3: 'Destination Unreachable', 4: 'Source Quench',
                5: 'Redirect', 8: 'Echo Request', 9: 'Router Advertisement',
                10: 'Router Solicitation', 11: 'Time Exceeded', 12: 'Parameter Problem',
                13: 'Timestamp Request', 14: 'Timestamp Reply', 15: 'Information Request',
                16: 'Information Reply', 17: 'Address Mask Request', 18: 'Address Mask Reply',
                30: 'Traceroute', 42: 'Extended Echo Request', 43: 'Extended Echo Reply',
                128: 'Echo Request (ICMPv6)', 129: 'Echo Reply (ICMPv6)',
                130: 'Multicast Listener Query', 131: 'Multicast Listener Report',
                132: 'Multicast Listener Done', 133: 'Router Solicitation (ICMPv6)',
                134: 'Router Advertisement (ICMPv6)', 135: 'Neighbor Solicitation',
                136: 'Neighbor Advertisement', 137: 'Redirect Message'
            },
            'dns_query_type': {
                0: 'A (IPv4)', 1: 'A (IPv4)', 12: 'PTR (Reverse DNS)',
                16: 'TXT (Text)', 28: 'AAAA (IPv6)', 33: 'SRV (Service)',
                255: 'ANY (Wildcard)'
            },
            'port_service': {
                20: 'ftp', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
                53: 'dns', 80: 'http', 110: 'pop3', 123: 'ntp', 135: 'msrpc',
                137: 'netbios-ns', 139: 'netbios-ssn', 143: 'imap', 443: 'https',
                445: 'smb', 3306: 'mysql', 3389: 'rdp', 5060: 'sip', 5355: 'llmnr',
                5900: 'vnc', 8080: 'http-proxy', 8443: 'https-alt', 11211: 'memcached',
                1433: 'ms-sql-s', 3128: 'squid-http', 1900: 'ssdp', 6000: 'X11'
            }
        }
        
        self.common_ports = [20, 21, 22, 23, 25, 53, 80, 110, 123, 135, 137, 139, 
                           143, 443, 445, 3306, 3389, 5060, 5355, 5900, 6000, 
                           8080, 8443, 11211, 1433, 1900, 3128]
        
        self.tcp_flags = {
            'FIN': 1, 'SYN': 2, 'RST': 4, 'PSH': 8,
            'ACK': 16, 'URG': 32, 'ECE': 64, 'CWR': 128
        }
    
    def engineer_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract IP-based features"""
        df = df.copy()
        
        # Private IP detection
        df['SRC_IS_PRIVATE'] = df['IPV4_SRC_ADDR'].apply(self._is_private_ip)
        df['DST_IS_PRIVATE'] = df['IPV4_DST_ADDR'].apply(self._is_private_ip)
        
        # IP class detection
        df['SRC_IP_CLASS'] = df['IPV4_SRC_ADDR'].apply(self._get_ip_class)
        df['DST_IP_CLASS'] = df['IPV4_DST_ADDR'].apply(self._get_ip_class)
        
        # Encode IP classes
        if 'ip_class_encoder' not in self.encoders:
            self.encoders['ip_class_encoder'] = OrdinalEncoder()
            df[['SRC_IP_CLASS_ENC', 'DST_IP_CLASS_ENC']] = self.encoders['ip_class_encoder'].fit_transform(
                df[['SRC_IP_CLASS', 'DST_IP_CLASS']]
            )
        else:
            df[['SRC_IP_CLASS_ENC', 'DST_IP_CLASS_ENC']] = self.encoders['ip_class_encoder'].transform(
                df[['SRC_IP_CLASS', 'DST_IP_CLASS']]
            )
        
        # Internal traffic detection
        df['INTERNAL_TRAFFIC'] = (
            (df['SRC_IP_CLASS'] == df['DST_IP_CLASS']) &
            df['SRC_IS_PRIVATE'] & df['DST_IS_PRIVATE']
        )
        
        return df
    
    def engineer_protocol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract protocol-based features"""
        df = df.copy()
        
        # Map protocol labels
        df['PROTOCOL_LABEL'] = df['PROTOCOL'].map(self.mappings['protocol']).fillna('OTHER')
        df['L7_PROTO_LABEL'] = df['L7_PROTO'].map(self.mappings['l7_proto']).fillna('UNKNOWN')
        
        # Encode protocols
        if 'protocol_encoder' not in self.encoders:
            self.encoders['protocol_encoder'] = OrdinalEncoder()
            df['PROTOCOL_ENC'] = self.encoders['protocol_encoder'].fit_transform(df[['PROTOCOL_LABEL']])
        else:
            df['PROTOCOL_ENC'] = self.encoders['protocol_encoder'].transform(df[['PROTOCOL_LABEL']])
            
        if 'l7_proto_encoder' not in self.encoders:
            self.encoders['l7_proto_encoder'] = OrdinalEncoder()
            df['L7_PROTO_ENC'] = self.encoders['l7_proto_encoder'].fit_transform(df[['L7_PROTO_LABEL']])
        else:
            df['L7_PROTO_ENC'] = self.encoders['l7_proto_encoder'].transform(df[['L7_PROTO_LABEL']])
        
        return df
    
    def engineer_tcp_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract TCP flag features"""
        df = df.copy()
        
        # Extract individual TCP flags for all flag columns
        for flag_name, bit in self.tcp_flags.items():
            df[f'TCP_HAS_{flag_name}'] = df['TCP_FLAGS'].apply(lambda x: self._has_flag(x, bit))
            df[f'CLIENT_HAS_{flag_name}'] = df['CLIENT_TCP_FLAGS'].apply(lambda x: self._has_flag(x, bit))
            df[f'SERVER_HAS_{flag_name}'] = df['SERVER_TCP_FLAGS'].apply(lambda x: self._has_flag(x, bit))
        
        return df
    
    def engineer_port_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract port-based features"""
        df = df.copy()
        
        # Port categories
        df['SRC_PORT_CATEGORY'] = df['L4_SRC_PORT'].apply(self._port_category)
        df['DST_PORT_CATEGORY'] = df['L4_DST_PORT'].apply(self._port_category)
        
        # Encode port categories
        if 'port_category_encoder' not in self.encoders:
            self.encoders['port_category_encoder'] = OrdinalEncoder()
            df[['SRC_PORT_CATEGORY_ENC', 'DST_PORT_CATEGORY_ENC']] = self.encoders['port_category_encoder'].fit_transform(
                df[['SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY']]
            )
        else:
            df[['SRC_PORT_CATEGORY_ENC', 'DST_PORT_CATEGORY_ENC']] = self.encoders['port_category_encoder'].transform(
                df[['SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY']]
            )
        
        # Common ports detection
        df['SRC_PORT_COMMON'] = df['L4_SRC_PORT'].isin(self.common_ports).astype(int)
        df['DST_PORT_COMMON'] = df['L4_DST_PORT'].isin(self.common_ports).astype(int)
        
        # Port difference
        df['PORT_DIFF'] = abs(df['L4_SRC_PORT'] - df['L4_DST_PORT'])
        
        # Service mapping
        df['SRC_SERVICE'] = df['L4_SRC_PORT'].map(self.mappings['port_service']).fillna('other')
        df['DST_SERVICE'] = df['L4_DST_PORT'].map(self.mappings['port_service']).fillna('other')
        
        # Encode services
        if 'service_encoder' not in self.encoders:
            self.encoders['service_encoder'] = OrdinalEncoder()
            df[['SRC_SERVICE_ENC', 'DST_SERVICE_ENC']] = self.encoders['service_encoder'].fit_transform(
                df[['SRC_SERVICE', 'DST_SERVICE']]
            )
        else:
            df[['SRC_SERVICE_ENC', 'DST_SERVICE_ENC']] = self.encoders['service_encoder'].transform(
                df[['SRC_SERVICE', 'DST_SERVICE']]
            )
        
        return df
    
    def engineer_icmp_dns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ICMP and DNS features"""
        df = df.copy()
        
        # ICMP type mapping
        df['ICMP_TYPE_LABEL'] = df['ICMP_TYPE'].map(self.mappings['icmp_type']).fillna('Other')
        df['ICMP_IPV4_TYPE_LABEL'] = df['ICMP_IPV4_TYPE'].map(self.mappings['icmp_type']).fillna('Other')
        
        # DNS features
        df['HAS_DNS_QUERY'] = (df['DNS_QUERY_ID'] != 0).astype(int)
        df['DNS_QUERY_TYPE_LABEL'] = df['DNS_QUERY_TYPE'].map(self.mappings['dns_query_type']).fillna('Other')
        
        # Encode ICMP and DNS features
        for feature in ['ICMP_TYPE_LABEL', 'ICMP_IPV4_TYPE_LABEL', 'DNS_QUERY_TYPE_LABEL']:
            encoder_name = f"{feature.lower()}_encoder"
            if encoder_name not in self.encoders:
                self.encoders[encoder_name] = OrdinalEncoder()
                df[f'{feature}_ENC'] = self.encoders[encoder_name].fit_transform(df[[feature]])
            else:
                df[f'{feature}_ENC'] = self.encoders[encoder_name].transform(df[[feature]])
        
        return df
    
    def process_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("🔧 Starting feature engineering...")
        
        # Apply all feature engineering steps
        df = self.engineer_ip_features(df)
        print("✅ IP features engineered")
        
        df = self.engineer_protocol_features(df)
        print("✅ Protocol features engineered")
        
        df = self.engineer_tcp_flags(df)
        print("✅ TCP flag features engineered")
        
        df = self.engineer_port_features(df)
        print("✅ Port features engineered")
        
        df = self.engineer_icmp_dns_features(df)
        print("✅ ICMP/DNS features engineered")
        
        print(f"🎉 Feature engineering complete! Shape: {df.shape}")
        return df
    
    # Helper methods
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private"""
        try:
            return ipaddress.ip_address(ip).is_private
        except ValueError:
            return False
    
    def _get_ip_class(self, ip: str) -> str:
        """Get IP class (A, B, C, D, E)"""
        try:
            first_octet = int(ip.split('.')[0])
            if 1 <= first_octet <= 126:
                return 'A'
            elif 128 <= first_octet <= 191:
                return 'B'
            elif 192 <= first_octet <= 223:
                return 'C'
            elif 224 <= first_octet <= 239:
                return 'D'
            elif 240 <= first_octet <= 254:
                return 'E'
            else:
                return 'Unknown'
        except:
            return 'Invalid'
    
    def _has_flag(self, value: int, flag_bit: int) -> int:
        """Check if TCP flag is set"""
        try:
            return int((int(value) & flag_bit) > 0)
        except:
            return 0
    
    def _port_category(self, port: int) -> str:
        """Categorize port number"""
        if port <= 1023:
            return 'well_known'
        elif port <= 49151:
            return 'registered'
        else:
            return 'dynamic'


def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load CSV data and apply all feature engineering"""
    try:
        print(f"📂 Loading data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        
        # Initialize feature engineer
        engineer = NetworkFeatureEngineer()
        
        # Process all features
        df_processed = engineer.process_all_features(df)
        
        return df_processed
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        raise


def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics of engineered features"""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'ip_classes': df['DST_IP_CLASS'].value_counts().to_dict(),
        'protocols': df['PROTOCOL_LABEL'].value_counts().head(10).to_dict(),
        'l7_protocols': df['L7_PROTO_LABEL'].value_counts().head(10).to_dict(),
        'services': df['DST_SERVICE'].value_counts().head(10).to_dict(),
        'internal_traffic_pct': (df['INTERNAL_TRAFFIC'].sum() / len(df) * 100),
        'private_src_pct': (df['SRC_IS_PRIVATE'].sum() / len(df) * 100),
        'private_dst_pct': (df['DST_IS_PRIVATE'].sum() / len(df) * 100)
    }
    return summary


# Main execution function for testing
if __name__ == "__main__":
    # Example usage
    sample_data = {
        'IPV4_SRC_ADDR': ['192.168.1.1', '10.0.0.1', '8.8.8.8'],
        'IPV4_DST_ADDR': ['192.168.1.2', '10.0.0.2', '1.1.1.1'],
        'L4_SRC_PORT': [12345, 80, 443],
        'L4_DST_PORT': [80, 12346, 53],
        'PROTOCOL': [6, 6, 17],
        'L7_PROTO': [41.000, 7.000, 5.126],
        'TCP_FLAGS': [24, 16, 0],
        'CLIENT_TCP_FLAGS': [2, 16, 0],
        'SERVER_TCP_FLAGS': [18, 24, 0],
        'ICMP_TYPE': [0, 0, 8],
        'ICMP_IPV4_TYPE': [0, 0, 0],
        'DNS_QUERY_ID': [0, 12345, 0],
        'DNS_QUERY_TYPE': [0, 1, 0],
        'FTP_COMMAND_RET_CODE': [0, 0, 0]
    }
    
    df_sample = pd.DataFrame(sample_data)
    engineer = NetworkFeatureEngineer()
    result = engineer.process_all_features(df_sample)
    print("\n📊 Sample processed data:")
    print(result.head())