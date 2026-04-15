@echo off
echo ==========================================
echo  NeuroGAN — Starting Frontend Dashboard
echo ==========================================
cd /d "%~dp0\app\frontend"
call npm install
call npm run dev
