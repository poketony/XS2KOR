@echo off
setlocal
chcp 65001 >nul
python -B "%~dp0title_tool.py" rebuild
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [FAILED] Do not use outputs from this failed run.
pause
exit /b %RESULT%
