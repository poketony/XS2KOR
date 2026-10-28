@echo off
setlocal
pushd "%~dp0" || exit /b 1
python -B fontsheetgenerator.py
set "result=%errorlevel%"
popd
pause
exit /b %result%
