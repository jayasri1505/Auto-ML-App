@echo off
REM Auto-ML Application Startup Script for Windows
REM This script automates the setup and startup of the Streamlit Auto-ML application

echo ==========================================
echo   Auto-ML Application Startup Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed. Please install Python 3 first.
    pause
    exit /b 1
)

echo [OK] Python found

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo Starting Streamlit application...
echo The app will open in your browser automatically.
echo.

REM Start the Streamlit app
streamlit run app.py

pause
