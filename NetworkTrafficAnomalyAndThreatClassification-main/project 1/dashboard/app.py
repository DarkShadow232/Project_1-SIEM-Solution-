"""
SIEM Dashboard - Flask Application
Real-time visualization of security alerts and network anomalies
"""

from flask import Flask, render_template, request, jsonify, Response, send_file
from data_loader import DashboardDataLoader
import json
import io
import csv
from datetime import datetime
import webbrowser
from threading import Timer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'siem-dashboard-secret-key-change-in-production'

# Initialize data loader
data_loader = DashboardDataLoader()


@app.route('/')
def index():
    """Overview dashboard page"""
    try:
        stats = data_loader.get_statistics()
        # Ensure all values are JSON serializable (convert numpy types)
        for key in stats:
            if isinstance(stats[key], (int, float)):
                stats[key] = float(stats[key]) if '.' in str(stats[key]) else int(stats[key])
        return render_template('index.html', stats=stats)
    except Exception as e:
        print(f"Error loading index: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', stats={
            'total_alerts': 0,
            'high_severity': 0,
            'medium_severity': 0,
            'low_severity': 0,
            'total_outliers': 0,
            'iqr_outliers': 0,
            'zscore_outliers': 0,
            'iso_outliers': 0,
            'top_files': [],
            'alert_timeline': {'hours': [], 'counts': []},
            'new_alerts': 0,
            'investigating': 0,
            'resolved': 0
        })


@app.route('/alerts')
def alerts():
    """Alerts page with filtering"""
    try:
        # Get filter parameters from query string
        filters = {}
        if request.args.get('severity'):
            filters['severity'] = request.args.get('severity')
        if request.args.get('status'):
            filters['status'] = request.args.get('status')
        if request.args.get('time_range'):
            filters['time_range'] = int(request.args.get('time_range'))
        if request.args.get('search'):
            filters['search'] = request.args.get('search')
        
        alerts_data = data_loader.load_alerts(filters if filters else None)
        return render_template('alerts.html', alerts=alerts_data)
    except Exception as e:
        print(f"Error loading alerts: {e}")
        return render_template('alerts.html', alerts=[])


@app.route('/outliers')
def outliers():
    """Outliers analysis page"""
    try:
        outliers_df = data_loader.load_outliers()
        
        # Convert to JSON for JavaScript
        if not outliers_df.empty:
            # Replace NaN with 0 for numeric columns
            outliers_clean = outliers_df.fillna(0)
            outliers_json = json.dumps(outliers_clean.to_dict('records'))
        else:
            outliers_json = json.dumps([])
        
        return render_template('outliers.html', 
                             outliers=outliers_df,
                             outliers_json=outliers_json)
    except Exception as e:
        print(f"Error loading outliers: {e}")
        import traceback
        traceback.print_exc()
        return render_template('outliers.html', 
                             outliers=None,
                             outliers_json='[]')


@app.route('/monitor')
def monitor():
    """Real-time monitoring page"""
    return render_template('monitor.html')


@app.route('/search')
def search():
    """Search page"""
    query = request.args.get('q', '')
    search_type = request.args.get('type', 'all')
    
    results = None
    if query:
        try:
            results = data_loader.search(query, search_type)
        except Exception as e:
            print(f"Error in search: {e}")
            results = {'query': query, 'alerts': [], 'outliers': []}
    
    return render_template('search.html', 
                         query=query, 
                         search_type=search_type,
                         results=results)


@app.route('/test-charts')
def test_charts():
    """Test page for chart rendering"""
    return render_template('test_charts.html')


# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/api/statistics')
def api_statistics():
    """Get current statistics"""
    try:
        stats = data_loader.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts')
def api_alerts():
    """Get alerts with optional filtering"""
    try:
        filters = {}
        if request.args.get('severity'):
            filters['severity'] = request.args.get('severity')
        if request.args.get('status'):
            filters['status'] = request.args.get('status')
        if request.args.get('time_range'):
            filters['time_range'] = int(request.args.get('time_range'))
        if request.args.get('search'):
            filters['search'] = request.args.get('search')
        
        limit = request.args.get('limit', type=int)
        
        alerts = data_loader.load_alerts(filters if filters else None)
        
        if limit:
            alerts = alerts[:limit]
        
        return jsonify(alerts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alert/<alert_id>')
def api_alert_detail(alert_id):
    """Get specific alert details"""
    try:
        alert = data_loader.get_alert_by_id(alert_id)
        if alert:
            return jsonify(alert)
        else:
            return jsonify({'error': 'Alert not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/outliers')
def api_outliers():
    """Get outliers data"""
    try:
        filters = {}
        if request.args.get('detection_method'):
            filters['detection_method'] = request.args.get('detection_method')
        if request.args.get('search'):
            filters['search'] = request.args.get('search')
        
        outliers = data_loader.load_outliers(filters if filters else None)
        
        if not outliers.empty:
            return jsonify(outliers.to_dict('records'))
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/alerts')
def export_alerts():
    """Export alerts to CSV"""
    try:
        filters = {}
        if request.args.get('severity'):
            filters['severity'] = request.args.get('severity')
        if request.args.get('status'):
            filters['status'] = request.args.get('status')
        if request.args.get('time_range'):
            filters['time_range'] = int(request.args.get('time_range'))
        
        alerts = data_loader.load_alerts(filters if filters else None)
        
        # Create CSV in memory
        output = io.StringIO()
        if alerts:
            # Get all keys from first alert
            fieldnames = alerts[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(alerts)
        
        # Convert to bytes
        output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        output_bytes.seek(0)
        
        return send_file(
            output_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'siem_alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/outliers')
def export_outliers():
    """Export outliers to CSV"""
    try:
        outliers = data_loader.load_outliers()
        
        if not outliers.empty:
            # Create CSV in memory
            output = io.StringIO()
            outliers.to_csv(output, index=False)
            output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
            output_bytes.seek(0)
            
            return send_file(
                output_bytes,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'outliers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        else:
            return jsonify({'error': 'No outliers data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream')
def stream():
    """Server-Sent Events endpoint for real-time updates"""
    def generate():
        # Send initial connection message
        yield f"data: {json.dumps({'status': 'connected'})}\n\n"
        
        # In a real implementation, you would:
        # 1. Monitor the alerts file for changes
        # 2. Stream new alerts as they appear
        # For now, we'll send periodic updates
        import time
        last_count = 0
        
        while True:
            try:
                alerts = data_loader.load_alerts()
                current_count = len(alerts)
                
                if current_count > last_count:
                    # New alerts detected
                    new_alerts = alerts[:current_count - last_count]
                    for alert in new_alerts:
                        yield f"data: {json.dumps(alert)}\n\n"
                    last_count = current_count
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('base.html'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('base.html'), 500


# =============================================================================
# Main Entry Point
# =============================================================================

def open_browser():
    """Open browser after server starts"""
    webbrowser.open('http://127.0.0.1:5000')


if __name__ == '__main__':
    print("=" * 80)
    print("SIEM Dashboard Starting...")
    print("=" * 80)
    print(f"\nDashboard will be available at: http://127.0.0.1:5000")
    print("\nPages:")
    print("  - Overview:   http://127.0.0.1:5000/")
    print("  - Alerts:     http://127.0.0.1:5000/alerts")
    print("  - Outliers:   http://127.0.0.1:5000/outliers")
    print("  - Monitor:    http://127.0.0.1:5000/monitor")
    print("  - Search:     http://127.0.0.1:5000/search")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    # Open browser after 1.5 seconds
    Timer(1.5, open_browser).start()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

