@echo off
setlocal
chcp 65001 >nul
python -B "%~dp0title_tool.py" extract
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [FAILED] Title extraction stopped. Existing edits were preserved.
pause
exit /b %RESULT%
