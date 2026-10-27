@echo off
REM Runs the automated test suite (no Gemini API key or internet needed).
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment .venv was not found. Using the system Python.
)

python -m pytest -v
pause
