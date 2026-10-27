@echo off
REM Launches FinPilot. Run this AFTER installing the dependencies (see README.md).
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment .venv was not found. Using the system Python.
    echo           See README.md, section "Installation", to create it.
)

if not exist ".env" (
    echo [INFO] No .env file found. Gemini features will be disabled until you create one.
    echo        Copy .env.example to .env and add your API key.
)

python -m streamlit run app.py
pause
