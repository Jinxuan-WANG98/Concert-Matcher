@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  python -m venv .venv
)

echo Installing required packages if needed...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Stopping any previous Concert Matcher instance on port 5050...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "127.0.0.1:5050" ^| findstr LISTENING') do (
  taskkill /F /PID %%p >nul 2>&1
)

echo Opening Concert Matcher at http://127.0.0.1:5050/
start "" "http://127.0.0.1:5050/"
".venv\Scripts\python.exe" -m flask --app app run --host 127.0.0.1 --port 5050
