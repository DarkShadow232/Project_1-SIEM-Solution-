"""
SIEM Integration - Executable Python Script
Converts the notebook to a runnable Python script
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("SIEM Security Solution - Complete Integration")
print("One-Class SVM Anomaly Detection (Recommended by Eng Mariam)")
print("=" * 80)
print(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# Step 1: Load and Explore Data
# ============================================================================
print("=" * 80)
print("STEP 1: Loading Datasets")
print("=" * 80)

# Load all three CSV files
try:
    df1 = pd.read_csv('Output1.csv')
    df2 = pd.read_csv('output2.csv')
    df3 = pd.read_csv('output3.csv')
    
    print(f"\nDataset 1 (Output1.csv): {df1.shape[0]} rows, {df1.shape[1]} columns")
    print(f"Dataset 2 (output2.csv): {df2.shape[0]} rows, {df2.shape[1]} columns")
    print(f"Dataset 3 (output3.csv): {df3.shape[0]} rows, {df3.shape[1]} columns")
    
    # Combine datasets
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"\n[OK] Combined Dataset: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns")
    
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("Please ensure Output1.csv, output2.csv, and output3.csv are in the current directory")
    exit(1)

# Display first few rows
print("\n" + "=" * 80)
print("First 5 Rows")
print("=" * 80)
print(df_combined.head())

# Data Information
print("\n" + "=" * 80)
print("Dataset Information")
print("=" * 80)
print(f"\nColumn Names ({len(df_combined.columns)} total):")
print(df_combined.columns.tolist()[:10], "..." if len(df_combined.columns) > 10 else "")

print(f"\nData Types:")
print(df_combined.dtypes.value_counts())

print(f"\nMissing Values:")
missing = df_combined.isnull().sum()
missing_pct = (missing / len(df_combined)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df.head(10))
else:
    print("[OK] No missing values found!")

# ============================================================================
# Step 2: Data Preprocessing
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Data Preprocessing")
print("=" * 80)

# Store filename column for later reference
filenames = df_combined['Filename'].copy()

# Remove filename column (non-numeric)
df_features = df_combined.drop(columns=['Filename'])

# Check for any remaining non-numeric columns
non_numeric_cols = df_features.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"[WARNING] Found non-numeric columns: {non_numeric_cols}")
    df_features = df_features.drop(columns=non_numeric_cols)

print(f"\nFeatures shape: {df_features.shape}")
print(f"Feature columns: {len(df_features.columns)}")

# Handle missing values (fill with median)
if df_features.isnull().sum().sum() > 0:
    print(f"\nFilling {df_features.isnull().sum().sum()} missing values with median...")
    df_features = df_features.fillna(df_features.median())
else:
    print("\n[OK] No missing values found!")

# Check for infinite values
inf_count = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
if inf_count > 0:
    print(f"\n[WARNING] Found {inf_count} infinite values. Replacing with NaN then median...")
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.median())

print(f"\n[OK] Preprocessing complete! Final shape: {df_features.shape}")

# ============================================================================
# Step 3: Train One-Class SVM Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Training One-Class SVM Model")
print("=" * 80)
print("(Recommended by Eng Mariam for anomaly detection)")

# Prepare data
X = df_features.values
print(f"\nData shape: {X.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[OK] Features scaled successfully!")
print(f"Scaled data shape: {X_scaled.shape}")
print(f"Mean: {X_scaled.mean():.6f}, Std: {X_scaled.std():.6f}")

# Model parameters
nu = 0.1  # Expected fraction of outliers (10%)
gamma = 'scale'  # Kernel coefficient
kernel = 'rbf'  # Radial Basis Function kernel

print(f"\nModel Parameters:")
print(f"  nu (outlier fraction): {nu}")
print(f"  gamma: {gamma}")
print(f"  kernel: {kernel}")

# Initialize and train model
ocsvm_model = OneClassSVM(
    nu=nu,
    gamma=gamma,
    kernel=kernel
)

print("\nTraining model... (this may take a few minutes)")
ocsvm_model.fit(X_scaled)

print("[OK] Model training completed!")

# Get predictions on training data
train_predictions = ocsvm_model.predict(X_scaled)
train_scores = ocsvm_model.decision_function(X_scaled)

# Statistics
n_normal = (train_predictions == 1).sum()
n_anomaly = (train_predictions == -1).sum()

print(f"\nModel Statistics:")
print(f"  Support vectors: {ocsvm_model.n_support_[0]}")
print(f"  Samples predicted as normal: {n_normal} ({n_normal/len(train_predictions)*100:.2f}%)")
print(f"  Samples predicted as anomaly: {n_anomaly} ({n_anomaly/len(train_predictions)*100:.2f}%)")
print(f"  Decision score range: [{train_scores.min():.2f}, {train_scores.max():.2f}]")

# Save model and scaler
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'one_class_svm_model.pkl')
scaler_path = os.path.join(model_dir, 'standard_scaler.pkl')
feature_names_path = os.path.join(model_dir, 'feature_names.json')

joblib.dump(ocsvm_model, model_path)
joblib.dump(scaler, scaler_path)

# Save feature names
feature_names = df_features.columns.tolist()
with open(feature_names_path, 'w') as f:
    json.dump(feature_names, f)

print(f"\n[OK] Model and scaler saved successfully!")
print(f"  Model: {model_path}")
print(f"  Scaler: {scaler_path}")
print(f"  Feature names: {feature_names_path}")

# ============================================================================
# Step 4: Anomaly Detection
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Anomaly Detection")
print("=" * 80)

# Get predictions and scores
predictions = ocsvm_model.predict(X_scaled)
decision_scores = ocsvm_model.decision_function(X_scaled)

# Create results dataframe
results_df = pd.DataFrame({
    'Filename': filenames,
    'Prediction': predictions,
    'Decision_Score': decision_scores,
    'Is_Anomaly': (predictions == -1).astype(int)
})

# Add original features for analysis
results_df = pd.concat([results_df, df_features.reset_index(drop=True)], axis=1)

print(f"\nDetection Results:")
print(f"  Total samples: {len(results_df)}")
print(f"  Normal samples: {(results_df['Is_Anomaly'] == 0).sum()} ({(results_df['Is_Anomaly'] == 0).sum()/len(results_df)*100:.2f}%)")
print(f"  Anomaly samples: {(results_df['Is_Anomaly'] == 1).sum()} ({(results_df['Is_Anomaly'] == 1).sum()/len(results_df)*100:.2f}%)")

# Display top anomalies (lowest decision scores)
print("\n" + "=" * 80)
print("Top 10 Anomalies (Lowest Decision Scores)")
print("=" * 80)
top_anomalies = results_df.nsmallest(10, 'Decision_Score')[['Filename', 'Decision_Score', 'Is_Anomaly', 
                                                             'malfind.ninjections', 'pslist.nproc', 'handles.nhandles']]
print(top_anomalies.to_string(index=False))

# ============================================================================
# Visualizations: Anomaly Detection Results
# ============================================================================
print("\n" + "=" * 80)
print("Generating Visualizations...")
print("=" * 80)

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# 1. Visualize anomalies
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Decision scores vs Malware injections
scatter = axes[0, 0].scatter(results_df['malfind.ninjections'], results_df['Decision_Score'], 
                             c=results_df['Is_Anomaly'], cmap='RdYlGn', alpha=0.6, edgecolors='black')
axes[0, 0].set_xlabel('Malware Injections', fontsize=12)
axes[0, 0].set_ylabel('Decision Score', fontsize=12)
axes[0, 0].set_title('Decision Score vs Malware Injections', fontsize=14, fontweight='bold')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 0], label='Anomaly (1) / Normal (0)')

# Decision scores vs Number of processes
scatter = axes[0, 1].scatter(results_df['pslist.nproc'], results_df['Decision_Score'], 
                             c=results_df['Is_Anomaly'], cmap='RdYlGn', alpha=0.6, edgecolors='black')
axes[0, 1].set_xlabel('Number of Processes', fontsize=12)
axes[0, 1].set_ylabel('Decision Score', fontsize=12)
axes[0, 1].set_title('Decision Score vs Number of Processes', fontsize=14, fontweight='bold')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 1], label='Anomaly (1) / Normal (0)')

# Decision scores distribution by anomaly status
normal_scores = results_df[results_df['Is_Anomaly'] == 0]['Decision_Score']
anomaly_scores = results_df[results_df['Is_Anomaly'] == 1]['Decision_Score']

axes[1, 0].hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='green', edgecolor='black')
axes[1, 0].hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
axes[1, 0].axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Decision Boundary')
axes[1, 0].set_xlabel('Decision Score', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Decision Score Distribution', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Feature importance (correlation with decision scores)
feature_corr = df_features.corrwith(results_df['Decision_Score']).abs().sort_values(ascending=False).head(10)
axes[1, 1].barh(range(len(feature_corr)), feature_corr.values, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].set_yticks(range(len(feature_corr)))
axes[1, 1].set_yticklabels(feature_corr.index, fontsize=9)
axes[1, 1].set_xlabel('Absolute Correlation with Decision Score', fontsize=12)
axes[1, 1].set_title('Top 10 Features Correlated with Anomaly Score', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('plots/anomaly_detection_results.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plots/anomaly_detection_results.png")
plt.close()

# 2. Normal vs Anomaly Comparison (Box Plots)
compare_features = ['pslist.nproc', 'handles.nhandles', 'dlllist.ndlls', 'malfind.ninjections']
compare_features = [f for f in compare_features if f in results_df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(compare_features):
    normal_data = results_df[results_df['Is_Anomaly'] == 0][feature].dropna()
    anomaly_data = results_df[results_df['Is_Anomaly'] == 1][feature].dropna()
    
    bp = axes[idx].boxplot([normal_data, anomaly_data], 
                           labels=['Normal', 'Anomaly'],
                           patch_artist=True, showmeans=True, meanline=True)
    
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[idx].set_title(f'{feature}: Normal vs Anomaly', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    normal_mean = normal_data.mean()
    anomaly_mean = anomaly_data.mean()
    textstr = f'Normal Mean: {normal_mean:.2f}\nAnomaly Mean: {anomaly_mean:.2f}\nDifference: {abs(anomaly_mean - normal_mean):.2f}'
    axes[idx].text(0.02, 0.98, textstr, transform=axes[idx].transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/normal_vs_anomaly_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plots/normal_vs_anomaly_comparison.png")
plt.close()

print("[OK] Visualizations completed!")

# ============================================================================
# Step 5: SIEM Alert Generation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SIEM Alert Generation")
print("=" * 80)

def calculate_severity(decision_score):
    """Calculate alert severity based on decision score"""
    score = abs(decision_score)
    if score > 10:
        return 'critical'
    elif score > 5:
        return 'high'
    elif score > 2:
        return 'medium'
    else:
        return 'low'

def generate_recommendations(row):
    """Generate security recommendations based on detected anomalies"""
    recommendations = []
    
    if row['malfind.ninjections'] > 0:
        recommendations.append("Investigate malware injection - check for code injection attacks")
    
    if row.get('psxview.not_in_pslist', 0) > 0:
        recommendations.append("Hidden process detected - investigate rootkit activity")
    
    if row['handles.nhandles'] > results_df['handles.nhandles'].quantile(0.95):
        recommendations.append("Unusually high number of handles - possible resource exhaustion attack")
    
    if row['dlllist.ndlls'] > results_df['dlllist.ndlls'].quantile(0.95):
        recommendations.append("High DLL count - check for DLL hijacking or injection")
    
    if not recommendations:
        recommendations.append("Investigate anomalous system behavior")
    
    return recommendations

# Generate alerts for anomalies only
anomalies = results_df[results_df['Is_Anomaly'] == 1].copy()

alerts = []
for idx, row in anomalies.iterrows():
    alert = {
        'alert_id': f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{idx}",
        'timestamp': datetime.now().isoformat(),
        'severity': calculate_severity(row['Decision_Score']),
        'status': 'new',
        'source': 'One-Class SVM Anomaly Detection',
        'anomaly_score': abs(row['Decision_Score']),
        'decision_score': float(row['Decision_Score']),
        'filename': row['Filename'],
        'key_indicators': {
            'malware_injections': int(row['malfind.ninjections']),
            'num_processes': int(row['pslist.nproc']),
            'num_handles': int(row['handles.nhandles']),
            'num_dlls': int(row['dlllist.ndlls']),
            'hidden_processes': int(row.get('psxview.not_in_pslist', 0))
        },
        'recommendations': generate_recommendations(row)
    }
    alerts.append(alert)

print(f"\n[OK] Generated {len(alerts)} SIEM alerts")

# Display alert summary
alert_severity = {}
for alert in alerts:
    severity = alert['severity']
    alert_severity[severity] = alert_severity.get(severity, 0) + 1

print("\nAlert Summary by Severity:")
severity_order = ['critical', 'high', 'medium', 'low']
for severity in severity_order:
    count = alert_severity.get(severity, 0)
    if count > 0:
        print(f"  {severity.upper()}: {count} alerts")

# Display sample alerts
print("\n" + "=" * 80)
print("Sample SIEM Alerts (Top 5 Critical/High)")
print("=" * 80)

sorted_alerts = sorted(alerts, key=lambda x: (severity_order.index(x['severity']), -x['anomaly_score']))[:5]

for i, alert in enumerate(sorted_alerts, 1):
    print(f"\n{'='*80}")
    print(f"Alert #{i}")
    print(f"{'='*80}")
    print(f"Alert ID: {alert['alert_id']}")
    print(f"Timestamp: {alert['timestamp']}")
    print(f"Severity: {alert['severity'].upper()}")
    print(f"Filename: {alert['filename']}")
    print(f"Anomaly Score: {alert['anomaly_score']:.4f}")
    print(f"Decision Score: {alert['decision_score']:.4f}")
    print(f"\nKey Indicators:")
    for key, value in alert['key_indicators'].items():
        print(f"  - {key}: {value}")
    print(f"\nRecommendations:")
    for rec in alert['recommendations']:
        print(f"  - {rec}")

# Save alerts to JSON file
alerts_file = 'siem_alerts.json'
with open(alerts_file, 'w') as f:
    json.dump(alerts, f, indent=2, default=str)

# Save results to CSV
results_file = 'detection_results.csv'
results_df.to_csv(results_file, index=False)

print(f"\n[OK] Alerts saved to: {alerts_file}")
print(f"[OK] Detection results saved to: {results_file}")
print(f"\nTotal alerts generated: {len(alerts)}")

# ============================================================================
# Visualizations: SIEM Alert Dashboard
# ============================================================================
print("\n" + "=" * 80)
print("Generating SIEM Alert Visualizations...")
print("=" * 80)

alerts_df = pd.DataFrame(alerts)
key_indicators_df = pd.json_normalize(alerts_df['key_indicators'])
alerts_extended = pd.concat([alerts_df[['severity', 'anomaly_score', 'decision_score']], 
                            key_indicators_df], axis=1)

# Alert Severity Distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Severity distribution
severity_counts = alerts_df['severity'].value_counts().reindex(['critical', 'high', 'medium', 'low'], fill_value=0)
colors = {'critical': 'darkred', 'high': 'red', 'medium': 'orange', 'low': 'yellow'}
bars = axes[0, 0].bar(severity_counts.index, severity_counts.values, 
               color=[colors[s] for s in severity_counts.index], alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Alert Distribution by Severity', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Severity Level')
axes[0, 0].set_ylabel('Number of Alerts')
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(severity_counts.values):
    axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# 2. Anomaly score distribution
axes[0, 1].hist(alerts_df['anomaly_score'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Anomaly Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 3. Key indicators comparison
if len(key_indicators_df.columns) > 0:
    indicator_totals = key_indicators_df.sum().sort_values(ascending=True).tail(8)
    axes[1, 0].barh(range(len(indicator_totals)), indicator_totals.values, 
            color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_yticks(range(len(indicator_totals)))
    axes[1, 0].set_yticklabels([col.replace('_', ' ').title() for col in indicator_totals.index], fontsize=10)
    axes[1, 0].set_xlabel('Total Count', fontsize=11)
    axes[1, 0].set_title('Top Key Indicators in Alerts', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Processes vs Handles colored by severity
if 'num_processes' in alerts_extended.columns and 'num_handles' in alerts_extended.columns:
    scatter = axes[1, 1].scatter(alerts_extended['num_processes'], alerts_extended['num_handles'],
                                c=[severity_order.index(s) if s in severity_order else 3 for s in alerts_df['severity']],
                                cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
    axes[1, 1].set_xlabel('Number of Processes', fontsize=11)
    axes[1, 1].set_ylabel('Number of Handles', fontsize=11)
    axes[1, 1].set_title('Processes vs Handles (by Severity)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Severity')

plt.tight_layout()
plt.savefig('plots/siem_alert_dashboard.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plots/siem_alert_dashboard.png")
plt.close()

print("[OK] SIEM alert visualizations completed!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("[OK] PIPELINE EXECUTION COMPLETED!")
print("=" * 80)
print("\nSummary:")
print(f"  - Total samples processed: {len(results_df)}")
print(f"  - Anomalies detected: {(results_df['Is_Anomaly'] == 1).sum()}")
print(f"  - SIEM alerts generated: {len(alerts)}")
print(f"  - Model saved: {model_path}")
print(f"  - Alerts saved: {alerts_file}")
print(f"  - Results saved: {results_file}")
print(f"  - Visualizations saved: plots/ directory")
print("\nNext Steps:")
print("  1. Review generated alerts in 'siem_alerts.json'")
print("  2. Configure SIEM API endpoint to send alerts to dashboard")
print("  3. Fine-tune model parameters (nu, gamma) based on your requirements")
print("  4. Integrate with real-time data streams for continuous monitoring")
print("=" * 80)

