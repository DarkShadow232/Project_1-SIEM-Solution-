"""
Debug script to check if data is loading correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DashboardDataLoader
import json

print("=" * 80)
print("DEBUGGING DATA LOADER")
print("=" * 80)

loader = DashboardDataLoader()

# Test statistics
print("\n1. Testing statistics...")
try:
    stats = loader.get_statistics()
    print(f"   [OK] Total alerts: {stats['total_alerts']}")
    print(f"   [OK] Total outliers: {stats['total_outliers']}")
    print(f"   [OK] Alert timeline data points: {len(stats['alert_timeline']['hours'])}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test alerts
print("\n2. Testing alerts...")
try:
    alerts = loader.load_alerts()
    print(f"   [OK] Loaded {len(alerts)} alerts")
    if alerts:
        print(f"   [OK] First alert ID: {alerts[0].get('alert_id', 'N/A')}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test outliers
print("\n3. Testing outliers...")
try:
    outliers = loader.load_outliers()
    print(f"   [OK] Loaded {len(outliers)} outliers")
    if not outliers.empty:
        print(f"   [OK] Columns: {list(outliers.columns)[:5]}...")
        print(f"   [OK] First filename: {outliers.iloc[0]['Filename']}")
        
        # Test JSON conversion
        outliers_json = json.dumps(outliers.to_dict('records'))
        print(f"   [OK] JSON conversion successful ({len(outliers_json)} characters)")
    else:
        print("   [WARNING] Outliers DataFrame is empty")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Check file existence
print("\n4. Checking data files...")
import os
siem_path = "../siem_integration"
files = {
    'siem_alerts.json': os.path.join(siem_path, 'siem_alerts.json'),
    'outliers_detected.csv': os.path.join(siem_path, 'outliers_detected.csv'),
    'outlier_summary.json': os.path.join(siem_path, 'outlier_summary.json')
}

for name, path in files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   [OK] {name}: {size:,} bytes")
    else:
        print(f"   [ERROR] {name}: NOT FOUND at {path}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

