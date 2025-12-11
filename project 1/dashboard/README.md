# SIEM Dashboard

Real-time visualization and monitoring dashboard for network traffic anomaly detection and security alerts.

## Overview

This dashboard provides an ELK Stack-like interface for visualizing and analyzing security alerts and network anomalies detected by the One-Class SVM model. It features real-time monitoring, interactive visualizations, and comprehensive search capabilities.

## Features

### 📊 Overview Dashboard
- Real-time KPI cards (Total Alerts, High Severity, Medium Severity, Outliers)
- 24-hour alerts timeline chart
- Severity distribution pie chart
- Alert status breakdown
- Detection method comparison
- Top 10 anomalous files table

### 🚨 Alerts Page
- Comprehensive alerts table with DataTables
- Advanced filtering (severity, status, time range, search)
- Click to view detailed alert information in modal
- Export alerts to CSV
- Auto-refresh every 30 seconds

### 📈 Outliers Analysis
- 3D scatter plot (Processes vs Handles vs DLLs)
- Detection method distribution
- Multi-method detection breakdown
- Feature distribution box plots
- Feature correlation heatmap
- Interactive data table with search
- Download outliers data

### 💻 Real-Time Monitor
- Live alert feed with SSE (Server-Sent Events)
- System status indicators
- Session uptime tracker
- Alert rate chart (per minute)
- Severity gauge
- Auto-refresh every 5 seconds

### 🔍 Advanced Search
- Search across all alerts and outliers
- Filter by data source (all, alerts, outliers)
- Regex pattern support
- Highlighted search results
- Keyboard shortcut (Ctrl+K) for quick access

## Installation

### Prerequisites
- Python 3.8 or higher
- Flask, pandas, plotly (see requirements.txt)

### Setup

1. **Install dependencies:**
```bash
cd dashboard
pip install -r requirements.txt
```

2. **Verify SIEM data is available:**
Ensure the following files exist in `../siem_integration/`:
- `siem_alerts.json`
- `outliers_detected.csv`

## Running the Dashboard

### Option 1: Using the Batch File (Windows, Recommended)
Simply double-click or run:
```bash
run_dashboard.bat
```

The batch file will:
- Automatically find the correct Python interpreter
- Check and install dependencies if needed
- Start the Flask server
- Open your browser automatically

### Option 2: Using Python Directly
```bash
cd dashboard
python app.py
```

Then open your browser to: `http://127.0.0.1:5000`

## Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| **Overview** | `/` | Main dashboard with KPIs and summary charts |
| **Alerts** | `/alerts` | Detailed alerts table with filtering and search |
| **Outliers** | `/outliers` | Outlier detection visualizations and analysis |
| **Monitor** | `/monitor` | Real-time monitoring with live alert feed |
| **Search** | `/search` | Advanced search across all data |

## API Endpoints

The dashboard exposes several API endpoints for programmatic access:

- `GET /api/statistics` - Get current statistics and KPIs
- `GET /api/alerts` - Get alerts (supports filtering)
- `GET /api/alert/<alert_id>` - Get specific alert details
- `GET /api/outliers` - Get outliers data
- `GET /api/export/alerts` - Export alerts to CSV
- `GET /api/export/outliers` - Export outliers to CSV
- `GET /api/stream` - SSE stream for real-time updates

### Example API Usage

```bash
# Get statistics
curl http://localhost:5000/api/statistics

# Get high-severity alerts
curl "http://localhost:5000/api/alerts?severity=high"

# Get recent alerts (last 10)
curl "http://localhost:5000/api/alerts?limit=10"

# Export alerts to CSV
curl "http://localhost:5000/api/export/alerts" -o alerts.csv
```

## Architecture

### Backend
- **Flask**: Web framework for routing and API
- **Pandas**: Data processing and filtering
- **Plotly**: Chart generation and visualization

### Frontend
- **Bootstrap 5**: Responsive UI framework with dark theme
- **Plotly.js**: Interactive charts and graphs
- **DataTables**: Advanced table features (sorting, searching, pagination)
- **Server-Sent Events**: Real-time data streaming

### Data Flow
```
SIEM Integration (One-Class SVM)
    ↓
    ├── siem_alerts.json
    └── outliers_detected.csv
    ↓
Data Loader (data_loader.py)
    ↓
Flask Routes (app.py)
    ↓
Templates (HTML + Jinja2)
    ↓
JavaScript (Plotly + DataTables)
    ↓
Dashboard UI (Browser)
```

## Configuration

### Port Configuration
To change the default port (5000), edit `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=YOUR_PORT)
```

### Data Source Configuration
To change the data source path, edit `data_loader.py`:
```python
def __init__(self, siem_data_path="../siem_integration"):
```

### Auto-Refresh Intervals
- **Overview**: 10 seconds (in index.html)
- **Alerts**: 30 seconds (in alerts.html)
- **Monitor**: 5 seconds (in monitor.html)

To change, edit the respective template files.

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, kill the process or change the port in `app.py`.

### Module Not Found Errors
Install missing dependencies:
```bash
pip install -r requirements.txt
```

### No Data Displayed
Ensure SIEM integration has been run and generated:
- `../siem_integration/siem_alerts.json`
- `../siem_integration/outliers_detected.csv`

### Browser Not Opening Automatically
Manually navigate to: `http://127.0.0.1:5000`

## Performance Tips

1. **Large Datasets**: The dashboard loads all data into memory. For very large datasets (>100K alerts), consider implementing pagination in the data loader.

2. **Real-Time Updates**: SSE connections can consume resources. The dashboard automatically pauses updates when the tab is not visible.

3. **Chart Performance**: For better performance with large datasets, reduce the number of data points in time-series charts.

## Security Considerations

⚠️ **Important**: This dashboard is designed for local/internal use only:

- No authentication is implemented
- Debug mode is enabled by default
- All API endpoints are publicly accessible
- Secret key should be changed for production

For production deployment:
1. Disable debug mode in `app.py`
2. Implement authentication (Flask-Login, OAuth, etc.)
3. Change the secret key
4. Use HTTPS
5. Implement rate limiting
6. Add input validation and sanitization

## Browser Compatibility

Tested and supported on:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

Note: Internet Explorer is not supported.

## File Structure

```
dashboard/
├── app.py                      # Flask application
├── data_loader.py              # Data processing module
├── requirements.txt            # Python dependencies
├── run_dashboard.bat           # Windows launcher
├── README.md                   # This file
├── static/
│   ├── css/
│   │   └── dashboard.css       # Custom styles
│   └── js/
│       ├── dashboard.js        # Main JavaScript
│       ├── charts.js           # Chart configurations
│       └── realtime.js         # Real-time updates
└── templates/
    ├── base.html               # Base template
    ├── index.html              # Overview dashboard
    ├── alerts.html             # Alerts page
    ├── outliers.html           # Outliers page
    ├── monitor.html            # Real-time monitor
    └── search.html             # Search page
```

## Contributing

This dashboard is part of the Network Traffic Anomaly Detection project. For issues or improvements, please refer to the main project documentation.

## License

Part of the Network Traffic Anomaly Detection System project.

## Acknowledgments

- Bootstrap 5 for the UI framework
- Plotly.js for interactive visualizations
- DataTables for advanced table features
- ELK Stack for inspiration

