# SIEM Dashboard - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Ensure SIEM Data is Generated

Before running the dashboard, make sure you have run the SIEM integration to generate the necessary data files:

```bash
cd "project 1\siem_integration"
python run_siem_notebook.py
# OR
run_siem_notebook.bat
```

This creates:
- `siem_alerts.json` - Security alerts from anomaly detection
- `outliers_detected.csv` - Outlier detection results

### Step 2: Launch the Dashboard

**Windows (Recommended):**
```bash
cd "project 1\dashboard"
run_dashboard.bat
```

**Or use Python directly:**
```bash
cd "project 1\dashboard"
python app.py
```

### Step 3: Open Your Browser

The dashboard will automatically open at: **http://127.0.0.1:5000**

If it doesn't open automatically, manually navigate to the URL.

## 📊 Dashboard Overview

### Main Pages

1. **Overview (/)** - Main dashboard
   - Total alerts and severity breakdown
   - 24-hour timeline
   - Top anomalous files
   - Real-time KPIs

2. **Alerts (/alerts)** - Detailed alerts
   - Searchable table with filtering
   - Export to CSV
   - View detailed alert information
   - Auto-refresh every 30 seconds

3. **Outliers (/outliers)** - Data analysis
   - 3D scatter plots
   - Detection method comparison
   - Feature correlation heatmap
   - Download outlier data

4. **Monitor (/monitor)** - Real-time monitoring
   - Live alert feed
   - System status
   - Alert rate charts
   - Auto-refresh every 5 seconds

5. **Search (/search)** - Advanced search
   - Search across all data
   - Regex support
   - Filter by data source
   - Keyboard shortcut: Ctrl+K

## 🎯 Key Features

✅ **Real-time Updates** - Auto-refresh with SSE streaming  
✅ **Dark Theme** - ELK Stack-inspired UI  
✅ **Interactive Charts** - Powered by Plotly.js  
✅ **Advanced Filtering** - Multi-criteria search  
✅ **Data Export** - CSV downloads  
✅ **Responsive Design** - Mobile-friendly  

## 🔧 Troubleshooting

### Dashboard Won't Start

**Issue**: Port 5000 already in use  
**Solution**: Kill the process or change the port in `app.py`

**Issue**: Module not found errors  
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### No Data Displayed

**Issue**: Empty dashboard or "No data" messages  
**Solution**: Run SIEM integration first
```bash
cd "..\siem_integration"
python run_siem_notebook.py
```

### Browser Not Opening

**Issue**: Dashboard starts but browser doesn't open  
**Solution**: Manually open: http://127.0.0.1:5000

## 📁 Required Files

The dashboard expects these files in `../siem_integration/`:

```
project 1/
├── siem_integration/
│   ├── siem_alerts.json          ← Required
│   ├── outliers_detected.csv     ← Required
│   ├── outlier_summary.json      ← Optional
│   └── Output1.csv               ← Source data
└── dashboard/
    └── app.py                    ← Dashboard app
```

## 💡 Tips

1. **Keyboard Shortcuts**
   - `Ctrl+K` - Quick search

2. **Best Practices**
   - Keep the dashboard open for real-time monitoring
   - Use filters to narrow down specific alerts
   - Export data for offline analysis

3. **Performance**
   - Dashboard loads all data into memory
   - For large datasets (>100K alerts), expect slower load times
   - Charts auto-resize on window resize

## 🔒 Security Note

⚠️ This dashboard is for **local/internal use only**:
- No authentication implemented
- Debug mode enabled
- Not suitable for public-facing deployment

## 📚 More Information

See `README.md` for complete documentation including:
- API endpoints
- Configuration options
- Architecture details
- Advanced features

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. Review error messages in the terminal
3. Ensure all dependencies are installed
4. Verify SIEM data files exist

## 🎉 You're All Set!

Your SIEM Dashboard is ready. Navigate through the pages to explore your security data!

**Homepage**: http://127.0.0.1:5000

Press `Ctrl+C` in the terminal to stop the dashboard.

