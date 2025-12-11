@echo off
REM =============================================================================
REM Network Traffic Anomaly & Threat Classification
REM Docker Build and Run Script for Windows
REM =============================================================================

echo.
echo ============================================================
echo   SIEM Dashboard - Docker Build Script
echo ============================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [INFO] Docker found!
echo.

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [INFO] Docker daemon is running!
echo.

REM Build the Docker image
echo [INFO] Building Docker image...
echo.
docker-compose build --no-cache

if errorlevel 1 (
    echo.
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Build completed successfully!
echo ============================================================
echo.
echo To start the dashboard, run:
echo   docker-compose up
echo.
echo Or run in background:
echo   docker-compose up -d
echo.
echo Dashboard will be available at: http://localhost:5000
echo.

pause

