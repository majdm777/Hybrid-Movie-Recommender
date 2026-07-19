
@echo off

REM Start the backend
start "Backend" cmd /k "cd /d "%~dp0backend" && .\.venv\Scripts\python.exe -m uvicorn app.api:app --reload"

REM Start the frontend
start "Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

REM Wait for the servers to start
timeout /t 3 >nul

REM Open the browser
start http://localhost:5173