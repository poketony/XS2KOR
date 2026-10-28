@echo off
setlocal
pushd "%~dp0"
python -B "..\..\texture_graphics.py" map rebuild
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [FAILED] Do not use outputs from this failed run.
popd
pause
exit /b %RESULT%
