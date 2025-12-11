@echo off
REM =============================================================================
REM Network Traffic Anomaly & Threat Classification
REM Docker Run Script for Windows
REM =============================================================================

echo.
echo ============================================================
echo   SIEM Dashboard - Docker Run Script
echo ============================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running! Please start Docker Desktop.
    pause
    exit /b 1
)

REM Check if image exists, if not build it
docker images network-threat-classifier --format "{{.Repository}}" | findstr /C:"network-threat-classifier" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Image not found. Building...
    docker-compose build
)

echo [INFO] Starting SIEM Dashboard...
echo.
echo Dashboard URL: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

docker-compose up

echo.
echo [INFO] Dashboard stopped.
pause

