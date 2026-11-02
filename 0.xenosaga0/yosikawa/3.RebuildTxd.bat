@echo off
setlocal
chcp 65001 >nul
python -B "%~dp0txd_tool.py" r
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [FAILED] Rebuild stopped. Do not use output from this failed run.
pause
exit /b %RESULT%
