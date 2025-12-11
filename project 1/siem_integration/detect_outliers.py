"""
Outlier Detection Script
Detects outliers in Output1.csv, output2.csv, and output3.csv using multiple methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Outlier Detection Analysis")
print("=" * 80)

# Load datasets
print("\nLoading datasets...")
df1 = pd.read_csv('Output1.csv')
df2 = pd.read_csv('output2.csv')
df3 = pd.read_csv('output3.csv')

# Combine datasets
df_combined = pd.concat([df1, df2, df3], ignore_index=True)
print(f"Total samples: {len(df_combined)}")

# Store filenames
filenames = df_combined['Filename'].copy()

# Remove filename column
df_features = df_combined.drop(columns=['Filename'])

# Handle missing values
df_features = df_features.fillna(df_features.median())
df_features = df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.fillna(df_features.median())

print(f"Features: {len(df_features.columns)}")

# ============================================================================
# Method 1: IQR (Interquartile Range) Method
# ============================================================================
print("\n" + "=" * 80)
print("Method 1: IQR (Interquartile Range) Outlier Detection")
print("=" * 80)

def detect_outliers_iqr(df, threshold=1.5):
    """Detect outliers using IQR method"""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
    return outliers, lower_bound, upper_bound

iqr_outliers, lower_bound, upper_bound = detect_outliers_iqr(df_features)
n_iqr_outliers = iqr_outliers.sum()

print(f"\nIQR Outliers detected: {n_iqr_outliers} ({n_iqr_outliers/len(df_features)*100:.2f}%)")
print(f"Normal samples: {(~iqr_outliers).sum()} ({(~iqr_outliers).sum()/len(df_features)*100:.2f}%)")

# ============================================================================
# Method 2: Z-Score Method
# ============================================================================
print("\n" + "=" * 80)
print("Method 2: Z-Score Outlier Detection")
print("=" * 80)

def detect_outliers_zscore(df, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(df))
    outliers = (z_scores > threshold).any(axis=1)
    return outliers, z_scores

zscore_outliers, z_scores = detect_outliers_zscore(df_features, threshold=3)
n_zscore_outliers = zscore_outliers.sum()

print(f"\nZ-Score Outliers detected: {n_zscore_outliers} ({n_zscore_outliers/len(df_features)*100:.2f}%)")
print(f"Normal samples: {(~zscore_outliers).sum()} ({(~zscore_outliers).sum()/len(df_features)*100:.2f}%)")

# ============================================================================
# Method 3: Isolation Forest (if available)
# ============================================================================
print("\n" + "=" * 80)
print("Method 3: Isolation Forest Outlier Detection")
print("=" * 80)

try:
    from sklearn.ensemble import IsolationForest
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_outliers = iso_forest.fit_predict(X_scaled)
    iso_outliers = (iso_outliers == -1)
    n_iso_outliers = iso_outliers.sum()
    
    print(f"\nIsolation Forest Outliers detected: {n_iso_outliers} ({n_iso_outliers/len(df_features)*100:.2f}%)")
    print(f"Normal samples: {(~iso_outliers).sum()} ({(~iso_outliers).sum()/len(df_features)*100:.2f}%)")
except ImportError:
    print("Isolation Forest not available. Skipping...")
    iso_outliers = None

# ============================================================================
# Combine Results
# ============================================================================
print("\n" + "=" * 80)
print("Combined Outlier Detection Results")
print("=" * 80)

# Create results dataframe
results = pd.DataFrame({
    'Filename': filenames,
    'IQR_Outlier': iqr_outliers.astype(int),
    'ZScore_Outlier': zscore_outliers.astype(int),
})

if iso_outliers is not None:
    results['IsolationForest_Outlier'] = iso_outliers.astype(int)
    results['Any_Method'] = (iqr_outliers | zscore_outliers | iso_outliers).astype(int)
else:
    results['Any_Method'] = (iqr_outliers | zscore_outliers).astype(int)

# Add max z-score for reference (handle NaN values)
max_z_scores = z_scores.max(axis=1)
max_z_scores = np.nan_to_num(max_z_scores, nan=0.0)
results['Max_ZScore'] = max_z_scores

# Add original features for analysis
results = pd.concat([results, df_features.reset_index(drop=True)], axis=1)

# Summary
print(f"\nOutlier Summary:")
print(f"  IQR Method: {results['IQR_Outlier'].sum()} outliers")
print(f"  Z-Score Method: {results['ZScore_Outlier'].sum()} outliers")
if iso_outliers is not None:
    print(f"  Isolation Forest: {results['IsolationForest_Outlier'].sum()} outliers")
print(f"  Detected by ANY method: {results['Any_Method'].sum()} outliers ({results['Any_Method'].sum()/len(results)*100:.2f}%)")

# ============================================================================
# Top Outliers
# ============================================================================
print("\n" + "=" * 80)
print("Top 20 Outliers (Highest Z-Scores)")
print("=" * 80)

top_outliers = results.nlargest(20, 'Max_ZScore')[
    ['Filename', 'Max_ZScore', 'IQR_Outlier', 'ZScore_Outlier', 
     'malfind.ninjections', 'pslist.nproc', 'handles.nhandles', 'dlllist.ndlls']
]

if iso_outliers is not None:
    top_outliers = results.nlargest(20, 'Max_ZScore')[
        ['Filename', 'Max_ZScore', 'IQR_Outlier', 'ZScore_Outlier', 'IsolationForest_Outlier',
         'malfind.ninjections', 'pslist.nproc', 'handles.nhandles', 'dlllist.ndlls']
    ]

print(top_outliers.to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("Generating Outlier Visualizations...")
print("=" * 80)

os.makedirs('plots', exist_ok=True)

# 1. Outlier comparison by method
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Outlier counts by method
methods = ['IQR', 'Z-Score']
counts = [results['IQR_Outlier'].sum(), results['ZScore_Outlier'].sum()]
if iso_outliers is not None:
    methods.append('Isolation Forest')
    counts.append(results['IsolationForest_Outlier'].sum())

axes[0, 0].bar(methods, counts, color=['steelblue', 'coral', 'lightgreen'][:len(methods)], 
               alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Outlier Count by Detection Method', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Outliers')
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(counts):
    axes[0, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# Z-score distribution
axes[0, 1].hist(results['Max_ZScore'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=3, color='red', linestyle='--', linewidth=2, label='Z-Score Threshold (3)')
axes[0, 1].set_title('Distribution of Maximum Z-Scores', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Max Z-Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Outliers vs Key Features
scatter = axes[1, 0].scatter(results['malfind.ninjections'], results['handles.nhandles'],
                            c=results['Any_Method'], cmap='RdYlGn', alpha=0.6, 
                            s=50, edgecolors='black', linewidths=0.5)
axes[1, 0].set_xlabel('Malware Injections', fontsize=12)
axes[1, 0].set_ylabel('Number of Handles', fontsize=12)
axes[1, 0].set_title('Outliers: Malware Injections vs Handles', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Outlier (1) / Normal (0)')

# Outlier overlap between methods
if iso_outliers is not None:
    # Venn-like comparison
    iqr_only = results[(results['IQR_Outlier'] == 1) & (results['ZScore_Outlier'] == 0) & (results['IsolationForest_Outlier'] == 0)].shape[0]
    zscore_only = results[(results['IQR_Outlier'] == 0) & (results['ZScore_Outlier'] == 1) & (results['IsolationForest_Outlier'] == 0)].shape[0]
    iso_only = results[(results['IQR_Outlier'] == 0) & (results['ZScore_Outlier'] == 0) & (results['IsolationForest_Outlier'] == 1)].shape[0]
    iqr_zscore = results[(results['IQR_Outlier'] == 1) & (results['ZScore_Outlier'] == 1) & (results['IsolationForest_Outlier'] == 0)].shape[0]
    iqr_iso = results[(results['IQR_Outlier'] == 1) & (results['ZScore_Outlier'] == 0) & (results['IsolationForest_Outlier'] == 1)].shape[0]
    zscore_iso = results[(results['IQR_Outlier'] == 0) & (results['ZScore_Outlier'] == 1) & (results['IsolationForest_Outlier'] == 1)].shape[0]
    all_three = results[(results['IQR_Outlier'] == 1) & (results['ZScore_Outlier'] == 1) & (results['IsolationForest_Outlier'] == 1)].shape[0]
    
    categories = ['IQR Only', 'Z-Score Only', 'Isolation Forest Only', 
                  'IQR+ZScore', 'IQR+ISO', 'ZScore+ISO', 'All Three']
    values = [iqr_only, zscore_only, iso_only, iqr_zscore, iqr_iso, zscore_iso, all_three]
    
    axes[1, 1].barh(range(len(categories)), values, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(categories)))
    axes[1, 1].set_yticklabels(categories, fontsize=9)
    axes[1, 1].set_xlabel('Number of Outliers')
    axes[1, 1].set_title('Outlier Overlap Between Methods', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
else:
    # Simple comparison for two methods
    iqr_only = results[(results['IQR_Outlier'] == 1) & (results['ZScore_Outlier'] == 0)].shape[0]
    zscore_only = results[(results['IQR_Outlier'] == 0) & (results['ZScore_Outlier'] == 1)].shape[0]
    both = results[(results['IQR_Outlier'] == 1) & (results['ZScore_Outlier'] == 1)].shape[0]
    
    categories = ['IQR Only', 'Z-Score Only', 'Both Methods']
    values = [iqr_only, zscore_only, both]
    
    axes[1, 1].barh(range(len(categories)), values, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(categories)))
    axes[1, 1].set_yticklabels(categories)
    axes[1, 1].set_xlabel('Number of Outliers')
    axes[1, 1].set_title('Outlier Overlap Between Methods', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('plots/outlier_detection_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plots/outlier_detection_analysis.png")
plt.close()

# 2. Feature-wise outlier analysis
key_features = ['pslist.nproc', 'handles.nhandles', 'dlllist.ndlls', 'malfind.ninjections']
key_features = [f for f in key_features if f in df_features.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    # Box plot with outliers highlighted
    bp = axes[idx].boxplot([df_features[feature]], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Mark outliers
    Q1 = df_features[feature].quantile(0.25)
    Q3 = df_features[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df_features[feature] < lower_bound) | (df_features[feature] > upper_bound)
    outlier_values = df_features[feature][outlier_mask]
    
    if len(outlier_values) > 0:
        axes[idx].scatter([1] * len(outlier_values), outlier_values, 
                         color='red', s=50, alpha=0.6, label=f'Outliers ({len(outlier_values)})',
                         edgecolors='black', linewidths=0.5)
    
    axes[idx].set_title(f'{feature}: Outlier Detection', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].set_xticks([1])
    axes[idx].set_xticklabels([feature.split('.')[-1]])
    axes[idx].grid(True, alpha=0.3, axis='y')
    if len(outlier_values) > 0:
        axes[idx].legend()

plt.tight_layout()
plt.savefig('plots/feature_wise_outliers.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plots/feature_wise_outliers.png")
plt.close()

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "=" * 80)
print("Saving Results...")
print("=" * 80)

# Save all outliers
outliers_df = results[results['Any_Method'] == 1].copy()
outliers_df.to_csv('outliers_detected.csv', index=False)
print(f"[OK] Saved: outliers_detected.csv ({len(outliers_df)} outliers)")

# Save full results
results.to_csv('outlier_detection_results.csv', index=False)
print(f"[OK] Saved: outlier_detection_results.csv ({len(results)} total samples)")

# Save summary
summary = {
    'Total_Samples': len(results),
    'IQR_Outliers': int(results['IQR_Outlier'].sum()),
    'ZScore_Outliers': int(results['ZScore_Outlier'].sum()),
    'Any_Method_Outliers': int(results['Any_Method'].sum()),
    'Outlier_Percentage': float(results['Any_Method'].sum() / len(results) * 100)
}

if iso_outliers is not None:
    summary['IsolationForest_Outliers'] = int(results['IsolationForest_Outlier'].sum())

with open('outlier_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"[OK] Saved: outlier_summary.json")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("OUTLIER DETECTION COMPLETED!")
print("=" * 80)
print(f"\nSummary:")
print(f"  Total samples analyzed: {len(results)}")
print(f"  Outliers detected (IQR): {results['IQR_Outlier'].sum()}")
print(f"  Outliers detected (Z-Score): {results['ZScore_Outlier'].sum()}")
if iso_outliers is not None:
    print(f"  Outliers detected (Isolation Forest): {results['IsolationForest_Outlier'].sum()}")
print(f"  Total unique outliers: {results['Any_Method'].sum()} ({results['Any_Method'].sum()/len(results)*100:.2f}%)")
print(f"\nFiles saved:")
print(f"  - outliers_detected.csv: List of all detected outliers")
print(f"  - outlier_detection_results.csv: Full results with outlier flags")
print(f"  - outlier_summary.json: Summary statistics")
print(f"  - plots/outlier_detection_analysis.png: Visualization")
print(f"  - plots/feature_wise_outliers.png: Feature-wise outlier analysis")
print("=" * 80)

