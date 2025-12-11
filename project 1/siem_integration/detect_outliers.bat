@echo off
REM Outlier Detection Script Launcher
REM Uses Anaconda Python which has all required packages

REM Try to find Anaconda Python
set ANACONDA_PYTHON=
if exist "I:\Programs\anaconda3\python.exe" (
    set ANACONDA_PYTHON=I:\Programs\anaconda3\python.exe
) else if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set ANACONDA_PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
) else if exist "%USERPROFILE%\anaconda3\python.exe" (
    set ANACONDA_PYTHON=%USERPROFILE%\anaconda3\python.exe
) else (
    REM Try to use python from PATH (should be Anaconda)
    where python >nul 2>&1
    if %ERRORLEVEL% == 0 (
        set ANACONDA_PYTHON=python
    )
)

if "%ANACONDA_PYTHON%"=="" (
    echo ERROR: Could not find Python with required packages installed.
    echo Please install Anaconda Python or install packages manually.
    pause
    exit /b 1
)

echo Using Python: %ANACONDA_PYTHON%
echo.

REM Change to script directory
cd /d "%~dp0"

REM Run the script
"%ANACONDA_PYTHON%" detect_outliers.py

pause

