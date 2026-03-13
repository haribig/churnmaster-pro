@echo off
title ChurnMaster Pro
color 0B
echo.
echo  ████████████████████████████████████████████████████████
echo    CHURNMASTER PRO — Starting...
echo  ████████████████████████████████████████████████████████
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Please install Python 3.8+ from python.org
    pause & exit /b 1
)

:: Install requirements
echo  Installing dependencies...
pip install -r requirements.txt -q

:: Check model files
if not exist "churn_model.pkl" (
    echo.
    echo  WARNING: churn_model.pkl not found!
    echo  Copy churn_model.pkl, scaler.pkl, feature_names.pkl into this folder.
    echo  Or copy them from your original 'customer churn' project.
    echo.
)

echo.
echo  Starting Flask server...
echo  Open http://localhost:5000 in your browser
echo.
python app.py
pause
