@echo off
@chcp 65001 >nul
setlocal DisableDelayedExpansion
python -B "%~dp0..\tools\subtitles\subtitles.py" encode --disc 2
set "rc=%errorlevel%"
if not "%rc%"=="0" echo [FAILED] Do not rebuild ISO. Fix the error and rerun.
if not defined SUBTITLES_NO_PAUSE pause
exit /b %rc%
