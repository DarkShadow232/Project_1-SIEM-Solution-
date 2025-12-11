"""
One-Class SVM Training Script for Large-Scale Network Log Analysis
Recommended by Eng Mariam for SIEM Security Solution

This script trains a One-Class SVM model on normal network traffic logs
to detect anomalies and potential attacks.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# For large datasets - use chunking
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: kagglehub not available. Use local dataset loading.")


class OCSVMTrainer:
    """One-Class SVM Trainer for Network Anomaly Detection"""
    
    def __init__(self, output_dir="outputs", random_state=42):
        self.output_dir = output_dir
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, file_path=None, kaggle_dataset=None, kaggle_filename=None, 
                  sample_size=None, use_normal_only=True):
        """
        Load dataset for training One-Class SVM.
        
        Parameters:
        -----------
        file_path : str, optional
            Local file path to CSV dataset
        kaggle_dataset : str, optional
            Kaggle dataset identifier (e.g., 'mohamedelrifai/network-anomaly-detection-dataset')
        kaggle_filename : str, optional
            Filename within Kaggle dataset
        sample_size : int, optional
            Number of samples to use (for large datasets)
        use_normal_only : bool, default=True
            For One-Class SVM, train only on normal traffic (Label=0)
        
        Returns:
        --------
        pd.DataFrame: Loaded and filtered dataset
        """
        print("=" * 60)
        print("Loading Dataset for One-Class SVM Training")
        print("=" * 60)
        
        if file_path and os.path.exists(file_path):
            print(f"Loading from local file: {file_path}")
            if sample_size:
                # Load in chunks for large files
                chunk_list = []
                chunk_size = 100000
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunk_list.append(chunk)
                    if len(chunk_list) * chunk_size >= sample_size:
                        break
                df = pd.concat(chunk_list, ignore_index=True)
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=self.random_state)
            else:
                df = pd.read_csv(file_path)
        
        elif kaggle_dataset and KAGGLE_AVAILABLE:
            print(f"Loading from Kaggle: {kaggle_dataset}")
            path = kagglehub.dataset_download(kaggle_dataset)
            file_path = os.path.join(path, kaggle_filename or 'sampled_NF-CSE-CIC-IDS2018-v2.csv')
            df = pd.read_csv(file_path)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=self.random_state)
        
        else:
            raise ValueError("Please provide either file_path or kaggle_dataset")
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # For One-Class SVM, we typically train only on normal data
        if use_normal_only and 'Label' in df.columns:
            normal_count = (df['Label'] == 0).sum()
            anomaly_count = (df['Label'] == 1).sum()
            print(f"\nLabel distribution:")
            print(f"  Normal (0): {normal_count} ({normal_count/len(df)*100:.2f}%)")
            print(f"  Anomaly (1): {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")
            
            # Filter to normal traffic only for training
            df_normal = df[df['Label'] == 0].copy()
            print(f"\nUsing {len(df_normal)} normal samples for training")
            
            return df_normal, df  # Return training data and full dataset for evaluation
        
        return df, df
    
    def preprocess_data(self, df, categorical_cols=None):
        """
        Preprocess data by removing categorical columns and preparing features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        categorical_cols : list, optional
            List of categorical columns to drop
        
        Returns:
        --------
        pd.DataFrame: Preprocessed features
        """
        print("\n" + "=" * 60)
        print("Preprocessing Data")
        print("=" * 60)
        
        if categorical_cols is None:
            categorical_cols = [
                'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 
                'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 
                'SERVER_TCP_FLAGS', 'ICMP_TYPE', 'ICMP_IPV4_TYPE',
                'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE',
                'SRC_IP_CLASS', 'DST_IP_CLASS', 'ICMP_TYPE_LABEL', 
                'ICMP_IPV4_TYPE_LABEL', 'DNS_QUERY_TYPE_LABEL', 
                'FTP_RET_CATEGORY', 'PROTOCOL_LABEL', 'L7_PROTO_LABEL',
                'SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY', 'DST_SERVICE', 
                'SRC_SERVICE', 'Label', 'Attack', 'Attack_Category'
            ]
        
        # Drop categorical columns
        df_cleaned = df.drop(columns=categorical_cols, errors='ignore')
        
        # Store feature names
        self.feature_names = df_cleaned.columns.tolist()
        
        print(f"Features after preprocessing: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names[:10]}..." if len(self.feature_names) > 10 else f"Feature names: {self.feature_names}")
        
        # Handle missing values
        if df_cleaned.isnull().sum().sum() > 0:
            print(f"Filling {df_cleaned.isnull().sum().sum()} missing values...")
            df_cleaned = df_cleaned.fillna(df_cleaned.median())
        
        return df_cleaned
    
    def train(self, X_train, nu=0.1, gamma='scale', kernel='rbf'):
        """
        Train One-Class SVM model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features (normal traffic only)
        nu : float, default=0.1
            An upper bound on the fraction of training errors and a lower bound
            of the fraction of support vectors. Should be in (0, 1].
        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
            Specifies the kernel type to be used in the algorithm
        """
        print("\n" + "=" * 60)
        print("Training One-Class SVM Model")
        print("=" * 60)
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"Training on {X_train_scaled.shape[0]} samples with {X_train_scaled.shape[1]} features")
        
        # Initialize and train One-Class SVM
        print(f"\nModel parameters:")
        print(f"  nu (outlier fraction): {nu}")
        print(f"  gamma: {gamma}")
        print(f"  kernel: {kernel}")
        
        self.model = OneClassSVM(
            nu=nu,
            gamma=gamma,
            kernel=kernel,
            random_state=self.random_state
        )
        
        print("\nTraining model... (this may take a while for large datasets)")
        self.model.fit(X_train_scaled)
        
        print("✅ Model training completed!")
        
        # Get training predictions
        train_predictions = self.model.predict(X_train_scaled)
        n_support_vectors = self.model.n_support_[0]
        
        print(f"\nModel Statistics:")
        print(f"  Support vectors: {n_support_vectors}")
        print(f"  Training samples predicted as normal: {(train_predictions == 1).sum()}")
        print(f"  Training samples predicted as anomaly: {(train_predictions == -1).sum()}")
        
        return self.model
    
    def evaluate(self, X_test, y_test=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray, optional
            True labels (0=normal, 1=anomaly)
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions: 1 = normal, -1 = anomaly
        predictions = self.model.predict(X_test_scaled)
        
        # Convert to binary: -1 -> 1 (anomaly), 1 -> 0 (normal)
        y_pred = (predictions == -1).astype(int)
        
        # Get decision scores (distance from hyperplane)
        decision_scores = self.model.decision_function(X_test_scaled)
        
        print(f"\nPredictions:")
        print(f"  Normal: {(y_pred == 0).sum()} ({(y_pred == 0).sum()/len(y_pred)*100:.2f}%)")
        print(f"  Anomaly: {(y_pred == 1).sum()} ({(y_pred == 1).sum()/len(y_pred)*100:.2f}%)")
        
        results = {
            'predictions': y_pred,
            'decision_scores': decision_scores,
            'raw_predictions': predictions
        }
        
        # If true labels available, calculate metrics
        if y_test is not None:
            # Convert labels if needed (assuming 0=normal, 1=anomaly)
            if isinstance(y_test, pd.Series):
                y_true = y_test.values
            else:
                y_true = y_test
            
            # Calculate metrics
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Normal  Anomaly")
            print(f"Actual Normal   {cm[0][0]:5d}   {cm[0][1]:5d}")
            print(f"       Anomaly  {cm[1][0]:5d}   {cm[1][1]:5d}")
            
            print(f"\nClassification Report:")
            print(classification_report(y_true, y_pred))
            
            # Calculate ROC-AUC (using decision scores)
            try:
                # For ROC-AUC, we need to invert decision scores (lower = more anomalous)
                roc_auc = roc_auc_score(y_true, -decision_scores)
                print(f"\nROC-AUC Score: {roc_auc:.4f}")
                results['roc_auc'] = roc_auc
            except Exception as e:
                print(f"Could not calculate ROC-AUC: {e}")
            
            results['confusion_matrix'] = cm
            results['classification_report'] = report
            results['accuracy'] = report['accuracy']
            results['precision'] = report['1']['precision'] if '1' in report else 0
            results['recall'] = report['1']['recall'] if '1' in report else 0
            results['f1_score'] = report['1']['f1-score'] if '1' in report else 0
            
            # Plot confusion matrix
            self._plot_confusion_matrix(cm)
            
        return results
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - One-Class SVM')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved to {self.output_dir}/confusion_matrix.jpg")
    
    def save_model(self):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        model_path = os.path.join(self.output_dir, 'one_class_svm_model.pkl')
        scaler_path = os.path.join(self.output_dir, 'standard_scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\n✅ Model saved to: {model_path}")
        print(f"✅ Scaler saved to: {scaler_path}")
        
        # Save feature names
        if self.feature_names:
            import json
            features_path = os.path.join(self.output_dir, 'feature_names.json')
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f)
            print(f"✅ Feature names saved to: {features_path}")
    
    def load_model(self, model_path=None, scaler_path=None):
        """Load trained model and scaler"""
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'one_class_svm_model.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(self.output_dir, 'standard_scaler.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load feature names if available
        features_path = os.path.join(self.output_dir, 'feature_names.json')
        if os.path.exists(features_path):
            import json
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Scaler loaded from: {scaler_path}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("One-Class SVM Training Pipeline")
    print("For SIEM Security Solution - Recommended by Eng Mariam")
    print("=" * 60)
    
    # Initialize trainer
    trainer = OCSVMTrainer(output_dir="outputs", random_state=42)
    
    # Configuration
    # Option 1: Use local file
    FILE_PATH = None  # Set to your dataset path, e.g., "data/network_logs.csv"
    
    # Option 2: Use Kaggle dataset
    KAGGLE_DATASET = "mohamedelrifai/network-anomaly-detection-dataset"
    KAGGLE_FILENAME = "sampled_NF-CSE-CIC-IDS2018-v2.csv"
    
    # For large datasets, you can sample
    SAMPLE_SIZE = None  # Set to number like 100000 for faster training
    
    # Model hyperparameters
    NU = 0.1  # Fraction of outliers (adjust based on expected anomaly rate)
    GAMMA = 'scale'  # or 'auto' or float value
    KERNEL = 'rbf'  # 'rbf', 'linear', 'poly', 'sigmoid'
    
    try:
        # Load data
        df_train, df_full = trainer.load_data(
            file_path=FILE_PATH,
            kaggle_dataset=KAGGLE_DATASET if FILE_PATH is None else None,
            kaggle_filename=KAGGLE_FILENAME,
            sample_size=SAMPLE_SIZE,
            use_normal_only=True
        )
        
        # Preprocess training data
        X_train = trainer.preprocess_data(df_train)
        
        # Train model
        trainer.train(X_train, nu=NU, gamma=GAMMA, kernel=KERNEL)
        
        # Evaluate on full dataset (if labels available)
        if 'Label' in df_full.columns:
            X_test = trainer.preprocess_data(df_full)
            y_test = df_full['Label']
            results = trainer.evaluate(X_test, y_test)
        
        # Save model
        trainer.save_model()
        
        print("\n" + "=" * 60)
        print("✅ Training Pipeline Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

