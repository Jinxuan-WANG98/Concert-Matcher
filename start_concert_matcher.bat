@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  python -m venv .venv
)

echo Installing required packages if needed...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Opening Concert Matcher at http://127.0.0.1:5050/
start "" "http://127.0.0.1:5050/"
".venv\Scripts\python.exe" -m flask --app app run --host 127.0.0.1 --port 5050
