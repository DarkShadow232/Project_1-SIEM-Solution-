# Dashboard Troubleshooting Guide

## Issue: Charts Not Displaying (Black Boxes)

If you see empty black boxes where charts should be, follow these steps:

### Step 1: Test if Plotly.js is Working

1. **Restart the dashboard** (stop with Ctrl+C and restart)
   ```bash
   run_dashboard.bat
   ```

2. **Open the test page** in your browser:
   ```
   http://127.0.0.1:5000/test-charts
   ```

3. **Check the results:**
   - ✅ **If you see 3 colorful charts** → Plotly is working! Issue is with data
   - ❌ **If you still see black boxes** → Plotly.js not loading properly

### Step 2: Check Browser Console for Errors

1. **Open Developer Tools**:
   - Press `F12` or `Ctrl+Shift+I`
   - Click on the **"Console"** tab

2. **Look for errors** (red text):
   - `Plotly is not defined` → CDN connection issue
   - `NaN` or `undefined` errors → Data formatting issue
   - `JSON.parse` error → JSON serialization issue

3. **Take a screenshot** of any errors you see

### Step 3: Check Terminal Output

Look at the terminal where the dashboard is running for error messages:
```
Error loading outliers: ...
Error loading index: ...
```

### Step 4: Verify Data Files

Run the diagnostic script:
```bash
cd "project 1\dashboard"
python debug_data.py
```

You should see:
```
[OK] Total alerts: XX
[OK] Total outliers: XXX
[OK] JSON conversion successful
[OK] siem_alerts.json: XX,XXX bytes
[OK] outliers_detected.csv: XXX,XXX bytes
```

## Common Fixes

### Fix 1: Restart Dashboard with Fixes

I've updated the code to handle NaN values. **Restart the dashboard**:

1. Stop the current server (Ctrl+C)
2. Run again:
   ```bash
   run_dashboard.bat
   ```
3. **Hard refresh** your browser (Ctrl+F5 or Ctrl+Shift+R)

### Fix 2: Check Internet Connection

The dashboard uses CDN links for Plotly.js. If you're offline:
1. Charts won't work
2. Make sure you have an internet connection

### Fix 3: Try a Different Browser

- Chrome/Edge usually works best
- Avoid Internet Explorer (not supported)

### Fix 4: Clear Browser Cache

1. Press `Ctrl+Shift+Delete`
2. Clear cached images and files
3. Reload the dashboard

### Fix 5: Check for JavaScript Errors in Templates

Open browser console (F12) and look for specific errors:

**If you see "Plotly is not defined":**
- CDN link is blocked or not loading
- Check your firewall/antivirus

**If you see "Cannot read property 'x' of undefined":**
- Data structure issue
- Run the debug_data.py script

## Debugging Steps

### 1. Test Individual Chart
Open browser console and try:
```javascript
Plotly.newPlot('test', [{y:[1,2,3], type:'bar'}], {})
```

If this works, the issue is with your data, not Plotly.

### 2. Check JSON Data
In browser console, type:
```javascript
console.log(statsData)  // On Overview page
console.log(outliersData)  // On Outliers page
```

Look for:
- `null` values
- `NaN` values
- Empty arrays `[]`
- Malformed objects

### 3. Inspect Network Tab
1. Open Developer Tools (F12)
2. Go to **Network** tab
3. Refresh the page
4. Look for failed requests (red)
5. Check if `plotly-2.26.0.min.js` loaded successfully

## Still Not Working?

### Collect This Information:

1. **Browser**: Chrome/Firefox/Edge? Version?
2. **Console Errors**: Screenshot of red errors in console
3. **Test Page Result**: Does http://127.0.0.1:5000/test-charts work?
4. **Debug Output**: Output from `python debug_data.py`
5. **Terminal Errors**: Any errors in the terminal running Flask?

### Quick Diagnostic Checklist

- [ ] Dashboard is running (no errors in terminal)
- [ ] Browser is Chrome/Firefox/Edge (not IE)
- [ ] Internet connection is working
- [ ] Data files exist (run debug_data.py)
- [ ] Test page shows charts (http://127.0.0.1:5000/test-charts)
- [ ] Browser console shows no red errors
- [ ] Hard refresh performed (Ctrl+F5)

## Expected Behavior

### Overview Page Should Show:
- 4 KPI cards with numbers (not charts, just numbers)
- Line chart with timeline
- Pie chart with colors
- 2 bar charts
- Table with top files

### Outliers Page Should Show:
- 4 number cards at top
- 3D rotating scatter plot
- 2 bar charts side by side
- 2 box plots side by side
- Correlation heatmap
- Data table at bottom

## Manual Fix: Verify plotly.min.js Loads

1. Open: http://127.0.0.1:5000/
2. Press F12
3. Click "Network" tab
4. Look for `plotly-2.26.0.min.js`
5. Status should be `200 OK`
6. If it's `404` or `Failed`, you have a CDN/network issue

## Contact Info

If none of these fixes work, provide:
1. Screenshot of browser console (F12 → Console tab)
2. Screenshot of /test-charts page
3. Output of `python debug_data.py`
4. Which browser and version you're using

## Quick Reference

| Problem | Solution |
|---------|----------|
| Black boxes instead of charts | Restart dashboard, hard refresh browser (Ctrl+F5) |
| "Plotly is not defined" error | Check internet connection, try different browser |
| Charts load but show wrong data | Run debug_data.py, check for errors |
| Some charts work, others don't | Check browser console for specific errors |
| Dashboard won't start | Install dependencies: `pip install -r requirements.txt` |

---

**Most Common Solution**: Restart the dashboard and do a hard refresh (Ctrl+F5) in your browser!

