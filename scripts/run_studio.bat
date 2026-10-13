@echo off
setlocal
cd /d "%~dp0\.."
echo ========================================================
echo Launching Autonomous CFX Studio v1.0.0
echo ========================================================

set PYTHONPATH=%CD%\src;%PYTHONPATH%

python -m autonomous_cfx.ui.app %*
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Application exited with error code %ERRORLEVEL%
    pause
)
endlocal
