@echo off
echo ==========================================
echo  NeuroGAN — Starting Backend Server
echo ==========================================
cd /d "%~dp0"
pip install -r requirements_app.txt --quiet
python -m uvicorn app.backend.main:app --reload --port 8000
