# Activate virtual environment for YOLOv8 Sublot Segmentation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Activating Virtual Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Install dependencies: pip install -r requirements.txt"
Write-Host "  2. Split dataset: python dataset_split.py"
Write-Host "  3. Train model: python train.py"
Write-Host ""
