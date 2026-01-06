@echo off
REM Activate virtual environment for YOLOv8 Sublot Segmentation

echo ========================================
echo Activating Virtual Environment
echo ========================================

call venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo.
echo Next steps:
echo   1. Install dependencies: pip install -r requirements.txt
echo   2. Split dataset: python dataset_split.py
echo   3. Train model: python train.py
echo.
