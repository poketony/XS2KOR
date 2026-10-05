@echo off
setlocal
chcp 65001 >nul
python -B "%~dp0..\disc1metadata\elf_text_tool.py" extract --disc 2
set "result=%errorlevel%"
if not "%result%"=="0" echo [FAILED] Existing translations and originals were not overwritten.
pause
exit /b %result%
