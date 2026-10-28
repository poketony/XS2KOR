@echo off
setlocal
pushd "%~dp0"
python -B "..\texture_graphics.py" menu extract
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [FAILED] Check the error above.
popd
pause
exit /b %RESULT%
