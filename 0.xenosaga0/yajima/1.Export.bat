@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if "%~1"=="" goto all

:next
python -B "%~dp0image_tool.py" export "%~1"
if errorlevel 1 goto failed
shift
if not "%~1"=="" goto next
goto done

:all
python -B "%~dp0image_tool.py" export-all
if errorlevel 1 goto failed

:done
echo Synopsis export completed. Existing originals were not modified.
pause
exit /b 0

:failed
echo Export stopped. Existing edits were not overwritten.
echo This tool accepts synopsis.bin only.
pause
exit /b 1
