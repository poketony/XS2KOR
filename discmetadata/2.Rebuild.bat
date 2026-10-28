@echo off
setlocal
chcp 65001 >nul
python -B "%~dp0elf_text_tool.py" rebuild
set "result=%errorlevel%"
if not "%result%"=="0" echo [FAILED] Do not use incomplete output. Check the error above.
if not defined ELF_NO_PAUSE pause
exit /b %result%
