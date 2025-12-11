"""
Data Loader for SIEM Dashboard
Loads and processes SIEM alerts and outlier detection results
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import re


class DashboardDataLoader:
    """Load and process data for the SIEM dashboard"""
    
    def __init__(self, siem_data_path="../siem_integration"):
        self.siem_data_path = siem_data_path
        self.alerts_file = os.path.join(siem_data_path, "siem_alerts.json")
        self.outliers_file = os.path.join(siem_data_path, "outliers_detected.csv")
        self.outlier_summary_file = os.path.join(siem_data_path, "outlier_summary.json")
        
        # Cache for data
        self._alerts_cache = None
        self._outliers_cache = None
        self._last_load_time = None
    
    def load_alerts(self, filters=None) -> List[Dict]:
        """
        Load SIEM alerts from JSON file
        
        Parameters:
        -----------
        filters : dict, optional
            Filters to apply: 'severity', 'status', 'time_range', 'search'
            
        Returns:
        --------
        list: List of alert dictionaries
        """
        try:
            with open(self.alerts_file, 'r') as f:
                alerts = json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []
        
        if not filters:
            return alerts
        
        filtered = alerts
        
        # Filter by severity
        if 'severity' in filters and filters['severity']:
            filtered = [a for a in filtered if a.get('severity') == filters['severity']]
        
        # Filter by status
        if 'status' in filters and filters['status']:
            filtered = [a for a in filtered if a.get('status') == filters['status']]
        
        # Filter by time range
        if 'time_range' in filters and filters['time_range']:
            hours = filters['time_range']
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered = [a for a in filtered 
                       if datetime.fromisoformat(a['timestamp']) >= cutoff_time]
        
        # Search filter
        if 'search' in filters and filters['search']:
            search_term = filters['search'].lower()
            filtered = [a for a in filtered 
                       if search_term in str(a).lower()]
        
        return filtered
    
    def load_outliers(self, filters=None) -> pd.DataFrame:
        """
        Load outlier detection results from CSV file
        
        Parameters:
        -----------
        filters : dict, optional
            Filters to apply: 'detection_method', 'search'
            
        Returns:
        --------
        pd.DataFrame: Outliers dataframe
        """
        try:
            df = pd.read_csv(self.outliers_file)
        except FileNotFoundError:
            return pd.DataFrame()
        
        if not filters:
            return df
        
        filtered = df.copy()
        
        # Filter by detection method
        if 'detection_method' in filters and filters['detection_method']:
            method = filters['detection_method']
            if method == 'IQR':
                filtered = filtered[filtered['IQR_Outlier'] == 1]
            elif method == 'ZScore':
                filtered = filtered[filtered['ZScore_Outlier'] == 1]
            elif method == 'IsolationForest':
                filtered = filtered[filtered['IsolationForest_Outlier'] == 1]
        
        # Search filter
        if 'search' in filters and filters['search']:
            search_term = filters['search'].lower()
            mask = filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            filtered = filtered[mask]
        
        return filtered
    
    def get_statistics(self) -> Dict:
        """
        Calculate KPIs and summary statistics
        
        Returns:
        --------
        dict: Statistics dictionary
        """
        alerts = self.load_alerts()
        outliers = self.load_outliers()
        
        # Alert statistics
        total_alerts = len(alerts)
        high_severity = sum(1 for a in alerts if a.get('severity') == 'high')
        medium_severity = sum(1 for a in alerts if a.get('severity') == 'medium')
        low_severity = sum(1 for a in alerts if a.get('severity') == 'low')
        
        # Status counts
        new_alerts = sum(1 for a in alerts if a.get('status') == 'new')
        investigating = sum(1 for a in alerts if a.get('status') == 'investigating')
        resolved = sum(1 for a in alerts if a.get('status') == 'resolved')
        
        # Outlier statistics
        total_outliers = len(outliers)
        iqr_outliers = outliers['IQR_Outlier'].sum() if not outliers.empty else 0
        zscore_outliers = outliers['ZScore_Outlier'].sum() if not outliers.empty else 0
        iso_outliers = outliers['IsolationForest_Outlier'].sum() if not outliers.empty else 0
        
        # Top anomalous files
        top_files = []
        if not outliers.empty:
            # Calculate a combined score for ranking
            outliers_copy = outliers.copy()
            outliers_copy['combined_score'] = (
                outliers_copy['IQR_Outlier'] + 
                outliers_copy['ZScore_Outlier'] + 
                outliers_copy['IsolationForest_Outlier']
            )
            top_files = outliers_copy.nlargest(10, 'combined_score')[
                ['Filename', 'combined_score', 'malfind.ninjections', 
                 'pslist.nproc', 'handles.nhandles']
            ].to_dict('records')
        
        # Time series data (alerts over time)
        alert_timeline = []
        if alerts:
            # Group alerts by hour
            for alert in alerts:
                try:
                    timestamp = datetime.fromisoformat(alert['timestamp'])
                    alert_timeline.append(timestamp)
                except:
                    continue
            
            if alert_timeline:
                # Create hourly bins for the last 24 hours
                now = datetime.now()
                hours = []
                counts = []
                for i in range(24, 0, -1):
                    hour_start = now - timedelta(hours=i)
                    hour_end = now - timedelta(hours=i-1)
                    count = sum(1 for t in alert_timeline if hour_start <= t < hour_end)
                    hours.append(hour_start.strftime('%H:00'))
                    counts.append(count)
                
                alert_timeline = {'hours': hours, 'counts': counts}
            else:
                alert_timeline = {'hours': [], 'counts': []}
        else:
            alert_timeline = {'hours': [], 'counts': []}
        
        return {
            'total_alerts': total_alerts,
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'low_severity': low_severity,
            'new_alerts': new_alerts,
            'investigating': investigating,
            'resolved': resolved,
            'total_outliers': total_outliers,
            'iqr_outliers': int(iqr_outliers),
            'zscore_outliers': int(zscore_outliers),
            'iso_outliers': int(iso_outliers),
            'top_files': top_files,
            'alert_timeline': alert_timeline
        }
    
    def search(self, query: str, search_type: str = 'all') -> Dict:
        """
        Search across alerts and outliers
        
        Parameters:
        -----------
        query : str
            Search query (supports regex if enabled)
        search_type : str
            Type of search: 'all', 'alerts', 'outliers'
            
        Returns:
        --------
        dict: Search results
        """
        results = {
            'query': query,
            'alerts': [],
            'outliers': []
        }
        
        if not query:
            return results
        
        # Search alerts
        if search_type in ['all', 'alerts']:
            alerts = self.load_alerts({'search': query})
            results['alerts'] = alerts
        
        # Search outliers
        if search_type in ['all', 'outliers']:
            outliers = self.load_outliers({'search': query})
            results['outliers'] = outliers.to_dict('records') if not outliers.empty else []
        
        return results
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Dict]:
        """Get a specific alert by ID"""
        alerts = self.load_alerts()
        for alert in alerts:
            if alert.get('alert_id') == alert_id:
                return alert
        return None
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get most recent alerts"""
        alerts = self.load_alerts()
        # Sort by timestamp descending
        sorted_alerts = sorted(alerts, 
                             key=lambda x: x.get('timestamp', ''), 
                             reverse=True)
        return sorted_alerts[:limit]
    
    def get_severity_distribution(self) -> Dict:
        """Get alert severity distribution for pie chart"""
        alerts = self.load_alerts()
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for alert in alerts:
            severity = alert.get('severity', 'low')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return severity_counts
    
    def get_detection_method_comparison(self) -> Dict:
        """Get outlier detection method comparison"""
        outliers = self.load_outliers()
        
        if outliers.empty:
            return {'IQR': 0, 'Z-Score': 0, 'Isolation Forest': 0}
        
        return {
            'IQR': int(outliers['IQR_Outlier'].sum()),
            'Z-Score': int(outliers['ZScore_Outlier'].sum()),
            'Isolation Forest': int(outliers['IsolationForest_Outlier'].sum())
        }

