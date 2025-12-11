"""
Dataset Loading Utilities for Large-Scale Network Log Analysis
Provides functions to load datasets from various sources for training One-Class SVM
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️ kagglehub not available. Install with: pip install kagglehub")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class DatasetLoader:
    """Utility class for loading large-scale network log datasets"""
    
    # Recommended datasets for network anomaly detection
    RECOMMENDED_DATASETS = {
        'NF-CSE-CIC-IDS2018': {
            'source': 'kaggle',
            'identifier': 'mohamedelrifai/network-anomaly-detection-dataset',
            'filename': 'sampled_NF-CSE-CIC-IDS2018-v2.csv',
            'description': 'Network Flow dataset with various attack types',
            'size': 'Large (188K+ samples)',
            'features': 'Network flow features',
            'attacks': ['DDoS', 'Brute Force', 'Port Scan', 'SQL Injection', 'XSS']
        },
        'CIC-IDS2017': {
            'source': 'kaggle',
            'identifier': 'cranix/ids2017',
            'description': 'Intrusion Detection System dataset from 2017',
            'size': 'Very Large (2.8M+ samples)',
            'features': 'Network flow features',
            'attacks': ['DDoS', 'Brute Force', 'Port Scan', 'Botnet', 'Infiltration']
        },
        'UNSW-NB15': {
            'source': 'kaggle',
            'identifier': 'mrwellsdavid/unsw-nb15',
            'description': 'Network-based intrusion detection dataset',
            'size': 'Large (257K+ samples)',
            'features': 'Network flow features',
            'attacks': ['Fuzzers', 'Analysis', 'Backdoors', 'DoS', 'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
        },
        'KDD-CUP-99': {
            'source': 'kaggle',
            'identifier': 'datasets/cesarcasado/kddcup99',
            'description': 'Classic intrusion detection dataset',
            'size': 'Large (494K+ samples)',
            'features': 'Network connection features',
            'attacks': ['DoS', 'Probe', 'R2L', 'U2R']
        }
    }
    
    @staticmethod
    def list_recommended_datasets():
        """List all recommended datasets"""
        print("=" * 80)
        print("Recommended Datasets for Network Anomaly Detection")
        print("=" * 80)
        
        for name, info in DatasetLoader.RECOMMENDED_DATASETS.items():
            print(f"\n[Dataset] {name}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Source: {info['source']}")
            if 'identifier' in info:
                print(f"   Identifier: {info['identifier']}")
            if 'filename' in info:
                print(f"   Filename: {info['filename']}")
            print(f"   Attack Types: {', '.join(info['attacks'])}")
        
        print("\n" + "=" * 80)
    
    @staticmethod
    def load_from_kaggle(dataset_identifier: str, filename: Optional[str] = None,
                        sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load dataset from Kaggle.
        
        Parameters:
        -----------
        dataset_identifier : str
            Kaggle dataset identifier (e.g., 'mohamedelrifai/network-anomaly-detection-dataset')
        filename : str, optional
            Specific filename within dataset
        sample_size : int, optional
            Number of samples to load (for large datasets)
        
        Returns:
        --------
        pd.DataFrame: Loaded dataset
        """
        if not KAGGLE_AVAILABLE:
            raise ImportError("kagglehub not available. Install with: pip install kagglehub")
        
        print(f"Loading dataset from Kaggle: {dataset_identifier}")
        
        # Download dataset
        path = kagglehub.dataset_download(dataset_identifier)
        
        if filename:
            file_path = os.path.join(path, filename)
        else:
            # Find CSV files in dataset
            files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if not files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            file_path = os.path.join(path, files[0])
            print(f"Using file: {files[0]}")
        
        # Load dataset
        print(f"Loading from: {file_path}")
        
        if sample_size:
            # Load in chunks for large files
            print(f"Loading {sample_size} samples...")
            chunk_list = []
            chunk_size = min(100000, sample_size)
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk_list.append(chunk)
                if len(chunk_list) * chunk_size >= sample_size:
                    break
            
            df = pd.concat(chunk_list, ignore_index=True)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
        else:
            df = pd.read_csv(file_path)
        
        print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    @staticmethod
    def load_from_local(file_path: str, sample_size: Optional[int] = None,
                       chunksize: int = 100000) -> pd.DataFrame:
        """
        Load dataset from local file.
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
        sample_size : int, optional
            Number of samples to load
        chunksize : int
            Chunk size for reading large files
        
        Returns:
        --------
        pd.DataFrame: Loaded dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading dataset from local file: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        if sample_size:
            # Load in chunks
            print(f"Loading {sample_size} samples in chunks...")
            chunk_list = []
            
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                chunk_list.append(chunk)
                if len(chunk_list) * chunksize >= sample_size:
                    break
            
            df = pd.concat(chunk_list, ignore_index=True)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
        else:
            # Load full file
            if file_size > 500:  # If > 500 MB, use chunks
                print("Large file detected. Loading in chunks...")
                chunk_list = []
                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    chunk_list.append(chunk)
                df = pd.concat(chunk_list, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
        
        print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    @staticmethod
    def load_recommended_dataset(dataset_name: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load a recommended dataset by name.
        
        Parameters:
        -----------
        dataset_name : str
            Name of recommended dataset
        sample_size : int, optional
            Number of samples to load
        
        Returns:
        --------
        pd.DataFrame: Loaded dataset
        """
        if dataset_name not in DatasetLoader.RECOMMENDED_DATASETS:
            available = ', '.join(DatasetLoader.RECOMMENDED_DATASETS.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        info = DatasetLoader.RECOMMENDED_DATASETS[dataset_name]
        
        if info['source'] == 'kaggle':
            filename = info.get('filename')
            return DatasetLoader.load_from_kaggle(
                info['identifier'],
                filename=filename,
                sample_size=sample_size
            )
        else:
            raise ValueError(f"Unsupported source: {info['source']}")
    
    @staticmethod
    def get_dataset_info(dataset_name: str) -> dict:
        """Get information about a recommended dataset"""
        if dataset_name not in DatasetLoader.RECOMMENDED_DATASETS:
            available = ', '.join(DatasetLoader.RECOMMENDED_DATASETS.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        return DatasetLoader.RECOMMENDED_DATASETS[dataset_name]


if __name__ == "__main__":
    # List recommended datasets
    DatasetLoader.list_recommended_datasets()
    
    # Example: Load NF-CSE-CIC-IDS2018 dataset
    print("\n" + "=" * 80)
    print("Example: Loading NF-CSE-CIC-IDS2018 dataset")
    print("=" * 80)
    
    try:
        # Load recommended dataset (sampled for demo)
        df = DatasetLoader.load_recommended_dataset('NF-CSE-CIC-IDS2018', sample_size=10000)
        
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {list(df.columns[:10])}...")
        
        if 'Label' in df.columns:
            print(f"\nLabel distribution:")
            print(df['Label'].value_counts())
        
        if 'Attack' in df.columns:
            print(f"\nAttack types:")
            print(df['Attack'].value_counts().head(10))
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

