import kagglehub
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_score, recall_score, accuracy_score, roc_auc_score
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

def load_data_local(filename) -> pd.DataFrame:
    """Load dataset from local file system."""
    df = pd.read_csv(filename)
    return df    

def load_data_kaggle(filename: str = 'sampled_NF-CSE-CIC-IDS2018-v2.csv') -> pd.DataFrame:
    """Load dataset from Kaggle."""
    path = kagglehub.dataset_download("mohamedelrifai/network-anomaly-detection-dataset")
    file_path = os.path.join(path, filename)
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Remove categorical columns and split into features and target.
    Returns X features and y labels.
    """
    categorical_cols = [
        'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
        'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'ICMP_TYPE', 'ICMP_IPV4_TYPE',
        'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE', 'SRC_IP_CLASS', 'DST_IP_CLASS',
        'ICMP_TYPE_LABEL', 'ICMP_IPV4_TYPE_LABEL', 'DNS_QUERY_TYPE_LABEL', 'FTP_RET_CATEGORY',
        'PROTOCOL_LABEL', 'L7_PROTO_LABEL', 'SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY',
        'DST_SERVICE', 'SRC_SERVICE'
    ]

    df_cleaned = df.drop(columns=categorical_cols, errors='ignore')
    X = df_cleaned.drop(columns=['Label', 'Attack', 'Attack_Category'], errors='ignore')
    y = df_cleaned['Label']
    
    # Print data shape information
    print(f"Dataset shape after preprocessing: {df_cleaned.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution: \n{y.value_counts(normalize=True).apply(lambda x: f'{x:.2%}')}")
    
    return X, y

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Returns scaled training data, scaled test data, and the scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, tune_hyperparams=False) -> XGBClassifier:
    """
    Train XGBoost model with appropriate scale_pos_weight for class imbalance.
    Optional hyperparameter tuning with GridSearchCV.
    """
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    if tune_hyperparams:
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        model = XGBClassifier(scale_pos_weight=scale_weight, random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        # Use default parameters with scale_pos_weight
        model = XGBClassifier(scale_pos_weight=scale_weight, random_state=42)
        model.fit(X_train, y_train)
    
    return model

def plot_confusion_matrix(y_true, y_pred, output_dir="outputs"):
    """
    Plot and save confusion matrix as an image.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to '{output_dir}/confusion_matrix.png'")

def plot_feature_importance(model, X_train_df, output_dir="outputs"):
    """
    Plot and save feature importance as an image.
    Uses actual feature names for better interpretability.
    
    Parameters:
    -----------
    model : XGBClassifier
        The trained XGBoost model
    X_train_df : pd.DataFrame
        DataFrame containing the training features with column names
    output_dir : str
        Directory to save output files
    """
    # Ensure X_train_df is a DataFrame with column names
    if not isinstance(X_train_df, pd.DataFrame):
        raise TypeError("X_train_df must be a pandas DataFrame with column names")
    
    # Get feature names and importance values
    feature_names = X_train_df.columns.tolist()
    
    # Check if feature_names is empty or None
    if not feature_names:
        print("Warning: No feature names found. Using generic feature names.")
        feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
    
    # Print debug information
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of importance values: {len(model.feature_importances_)}")
    
    # Create a DataFrame for better handling
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Debug: Show top 5 features and their importance
    print("Top 5 features by importance:")
    print(feature_importance_df.sort_values('Importance', ascending=False).head(5))
    
    # Sort by importance value (descending)
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Get top 20 features (or fewer if less than 20 features)
    top_n = min(20, len(feature_importance_df))
    top_features = feature_importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    
    # Add value labels to the bars
    for i, v in enumerate(top_features['Importance']):
        ax.text(v + 0.001, i, f"{v:.4f}", va='center')
    
    plt.title('Most Important Features', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature Name', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save feature importance as CSV
    feature_importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    print(f"Feature importance plot saved to '{output_dir}/feature_importance.png'")
    print(f"Feature importance data saved to '{output_dir}/feature_importance.csv'")

def evaluate_model(model: XGBClassifier, X_test, y_test, X_train, y_train, output_dir="outputs") -> dict:
    """
    Comprehensive model evaluation with multiple metrics:
    - Classification report (precision, recall, f1-score)
    - Confusion matrix
    - ROC-AUC score
    - Feature importance analysis
    
    Parameters:
    -----------
    model : XGBClassifier
        The trained XGBoost model
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        Test labels
    X_train : pd.DataFrame or np.ndarray
        Training features (for feature importance)
    y_train : pd.Series or np.ndarray
        Training labels
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Predictions on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except:
        roc_auc = None
        print("Warning: Could not calculate ROC-AUC score")
    
    # Print evaluation results
    print("\n===== Model Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report to text file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        if roc_auc:
            f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, output_dir)
    
    # For feature importance, ensure we have a DataFrame with column names
    print("\nPreparing feature importance plot...")
    
    # Debug info
    print(f"X_train type: {type(X_train)}")
    if hasattr(X_train, 'shape'):
        print(f"X_train shape: {X_train.shape}")
    
    if isinstance(X_train, pd.DataFrame):
        print("Using existing DataFrame for feature importance")
        X_train_df = X_train
    else:
        print("Warning: X_train is not a DataFrame. Feature names might be missing.")
        # If we got here, something went wrong in our pipeline
        # Try to recover by creating a generic DataFrame
        X_train_df = pd.DataFrame(X_train)
        X_train_df.columns = [f"Feature_{i}" for i in range(X_train_df.shape[1])]
    
    # Plot feature importance with the DataFrame
    plot_feature_importance(model, X_train_df, output_dir)
    
    # Compile results into a dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    return results

def save_model_and_scaler(model, scaler, output_dir="outputs"):
    """
    Save the trained model and scaler as .pkl files.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "xgboost_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    print(f"Model and scaler saved to '{output_dir}/'")

def main(filepath: str, tune_hyperparams=False):
    """
    Main function to orchestrate the entire workflow.
    """
    print("Loading data...")
    df = load_data_local(filepath)
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Store feature names before splitting and scaling
    feature_names = X.columns.tolist()
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Convert numpy arrays back to DataFrames with original feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print("Training model...")
    model = train_model(X_train_scaled_df, y_train, tune_hyperparams=tune_hyperparams)
    
    print("Evaluating model...")
    # Pass DataFrame objects to preserve feature names
    results = evaluate_model(model, X_test_scaled_df, y_test, X_train_scaled_df, y_train)
    
    print("Saving model and scaler...")
    save_model_and_scaler(model, scaler)
    
    return model, scaler, results

if __name__ == "__main__":
    # File path to your dataset
    FILE_PATH = r"D:\eme\GradProject\Data\sampled_NF-CSE-CIC-IDS2018-v2.csv"
    
    # Set to True if you want to perform hyperparameter tuning (takes longer)
    TUNE_HYPERPARAMS = False
    
    # Run the main function
    model, scaler, results = main(FILE_PATH, tune_hyperparams=TUNE_HYPERPARAMS)
    
    print("\n===== Final Results =====")
    for metric, value in results.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")