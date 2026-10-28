@echo off
@chcp 65001 >nul
setlocal DisableDelayedExpansion
python -B "%~dp0..\tools\subtitles\subtitles.py" prepare --disc 1
set "rc=%errorlevel%"
if not "%rc%"=="0" echo [FAILED] Existing translations were not overwritten.
if not defined SUBTITLES_NO_PAUSE pause
exit /b %rc%
