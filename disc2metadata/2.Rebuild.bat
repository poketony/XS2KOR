@echo off
setlocal
chcp 65001 >nul
python -B "%~dp0..\disc1metadata\elf_text_tool.py" rebuild --disc 2
set "result=%errorlevel%"
if not "%result%"=="0" echo [FAILED] Do not use incomplete output. Check the error above.
pause
exit /b %result%
