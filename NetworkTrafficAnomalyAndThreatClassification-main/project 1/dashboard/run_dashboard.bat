@echo off
REM SIEM Dashboard Launcher
REM Starts the Flask dashboard application

echo ================================================================================
echo                         SIEM Dashboard Launcher
echo ================================================================================
echo.

REM Try to find Anaconda Python
set PYTHON_EXE=
if exist "I:\Programs\anaconda3\python.exe" (
    set PYTHON_EXE=I:\Programs\anaconda3\python.exe
) else if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
) else if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe
) else (
    REM Try to use python from PATH
    where python >nul 2>&1
    if %ERRORLEVEL% == 0 (
        set PYTHON_EXE=python
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Could not find Python with required packages installed.
    echo Please install Anaconda Python or install packages manually.
    echo.
    pause
    exit /b 1
)

echo [INFO] Using Python interpreter: %PYTHON_EXE%
echo.

REM Change to dashboard directory
cd /d "%~dp0"

REM Check if requirements are installed
echo [INFO] Checking dependencies...
"%PYTHON_EXE%" -c "import flask, pandas, plotly" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some dependencies are missing. Installing requirements...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies.
        echo Please install manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [OK] Dependencies check passed
echo.
echo ================================================================================
echo                    Starting SIEM Dashboard Server...
echo ================================================================================
echo.
echo The dashboard will open automatically in your browser.
echo.
echo Available at: http://127.0.0.1:5000
echo.
echo Pages:
echo   - Overview:   http://127.0.0.1:5000/
echo   - Alerts:     http://127.0.0.1:5000/alerts
echo   - Outliers:   http://127.0.0.1:5000/outliers
echo   - Monitor:    http://127.0.0.1:5000/monitor
echo   - Search:     http://127.0.0.1:5000/search
echo.
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

REM Run the dashboard
"%PYTHON_EXE%" app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Dashboard server failed to start.
    echo Check the error messages above for details.
    pause
)

