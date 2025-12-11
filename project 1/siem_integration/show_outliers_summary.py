"""Quick summary of detected outliers"""
import pandas as pd

df = pd.read_csv('outlier_detection_results.csv')

print("=" * 80)
print("OUTLIER DETECTION SUMMARY")
print("=" * 80)
print(f"\nTotal Samples: {len(df)}")
print(f"\nOutliers by Method:")
print(f"  IQR Method: {df['IQR_Outlier'].sum()} ({df['IQR_Outlier'].sum()/len(df)*100:.2f}%)")
print(f"  Z-Score Method: {df['ZScore_Outlier'].sum()} ({df['ZScore_Outlier'].sum()/len(df)*100:.2f}%)")
print(f"  Isolation Forest: {df['IsolationForest_Outlier'].sum()} ({df['IsolationForest_Outlier'].sum()/len(df)*100:.2f}%)")
print(f"  ANY Method: {df['Any_Method'].sum()} ({df['Any_Method'].sum()/len(df)*100:.2f}%)")

# Most significant outliers (detected by multiple methods)
df['Method_Count'] = df['IQR_Outlier'] + df['ZScore_Outlier'] + df['IsolationForest_Outlier']
multi = df[df['Method_Count'] >= 2].sort_values('Method_Count', ascending=False)

print(f"\nMost Significant Outliers (Detected by 2+ Methods): {len(multi)}")
if len(multi) > 0:
    print("\nTop 15 Multi-Method Outliers:")
    print(multi[['Filename', 'IQR_Outlier', 'ZScore_Outlier', 'IsolationForest_Outlier', 
                'Method_Count', 'malfind.ninjections', 'pslist.nproc', 'handles.nhandles', 'dlllist.ndlls']].head(15).to_string(index=False))

# Z-Score outliers (most extreme)
zscore_outliers = df[df['ZScore_Outlier'] == 1].sort_values('Max_ZScore', ascending=False)
print(f"\n\nTop 15 Z-Score Outliers (Most Extreme):")
print(zscore_outliers[['Filename', 'Max_ZScore', 'malfind.ninjections', 'pslist.nproc', 
                       'handles.nhandles', 'dlllist.ndlls']].head(15).to_string(index=False))

# Isolation Forest outliers (most anomalous)
iso_outliers = df[df['IsolationForest_Outlier'] == 1]
print(f"\n\nIsolation Forest Outliers ({len(iso_outliers)} total):")
print(iso_outliers[['Filename', 'malfind.ninjections', 'pslist.nproc', 
                    'handles.nhandles', 'dlllist.ndlls']].head(15).to_string(index=False))

print("\n" + "=" * 80)
print("Files saved:")
print("  - outliers_detected.csv: All 395 detected outliers")
print("  - outlier_detection_results.csv: Full dataset with outlier flags")
print("  - outlier_summary.json: Summary statistics")
print("=" * 80)

