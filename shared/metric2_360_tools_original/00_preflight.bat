@echo off
REM Run this from your activated Anaconda Prompt; it does NOT activate a new env.
cd /d "%~dp0.."
python "%~dp0run_metric2_360.py" preflight
if errorlevel 1 (
  echo.
  echo STOP - Review the error above. Do not launch the next stage.
  exit /b 2
)
