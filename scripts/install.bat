@echo off
setlocal
cd /d "%~dp0\.."
echo ========================================================
echo Installing Autonomous CFX Studio Dependencies
echo ========================================================

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================================
echo Running Test Suite
echo ========================================================
python -m pytest tests -v

echo.
echo ========================================================
echo Installation and Verification Complete!
echo You can now run Autonomous CFX Studio via scripts\run_studio.bat
echo ========================================================
endlocal
