@echo off
REM =============================================================================
REM Network Traffic Anomaly & Threat Classification
REM Docker Stop Script for Windows
REM =============================================================================

echo.
echo ============================================================
echo   SIEM Dashboard - Docker Stop Script
echo ============================================================
echo.

echo [INFO] Stopping containers...
docker-compose down

echo.
echo [INFO] Containers stopped successfully!
echo.

pause

