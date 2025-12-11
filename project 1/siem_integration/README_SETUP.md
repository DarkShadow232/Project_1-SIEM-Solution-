# SIEM Integration - Setup Guide

## Python Environment Issue

If you encounter `ModuleNotFoundError: No module named 'pandas'`, you have multiple Python installations and your IDE is using the wrong one.

## Quick Fix Options

### Option 1: Use the Batch File (Recommended)
Run the script using the provided batch file which automatically finds the correct Python:

```bash
run_siem_notebook.bat
```

### Option 2: Configure Your IDE

**For VS Code:**
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the Anaconda Python (usually `I:\Programs\anaconda3\python.exe` or similar)

**For PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Select the Anaconda Python interpreter

### Option 3: Install Packages for Current Python

If you must use the 32-bit Python, you'll need to install packages manually. However, **pandas doesn't have pre-built wheels for 32-bit Python 3.13**, so you'll need to:

1. Use 64-bit Python instead, OR
2. Use Anaconda Python (recommended)

### Option 4: Use Anaconda Environment

```bash
# Activate Anaconda environment
conda activate base

# Navigate to script directory
cd "project 1/siem_integration"

# Run the script
python run_siem_notebook.py
```

## Verify Installation

Check if packages are installed:
```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages installed!')"
```

## Recommended Solution

**Use Anaconda Python** - It comes with all required packages pre-installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

The batch file (`run_siem_notebook.bat`) will automatically find and use Anaconda Python.

