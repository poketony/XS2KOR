@echo off
@chcp 65001 >nul
setlocal DisableDelayedExpansion
python -B "%~dp00.sb2tool.py" rebuild %*
set "rc=%errorlevel%"
if not "%rc%"=="0" echo [FAILED] Do not use .new files from this run.
if not defined SB2TOOL_NO_PAUSE pause
exit /b %rc%
