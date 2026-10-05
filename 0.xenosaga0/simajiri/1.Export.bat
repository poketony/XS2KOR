@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if "%~1"=="" goto selected

:next
python -B "%~dp0esd_esp_tool.py" export "%~1"
if errorlevel 1 goto failed
shift
if not "%~1"=="" goto next
echo Export completed. See the graphics_extract folder beside this batch file.
pause
exit /b 0

:selected
python -B "%~dp0esd_esp_tool.py" export-text
if errorlevel 1 goto failed
echo Reviewed text graphics exported. See graphics_extract.
pause
exit /b 0

:failed
echo Export stopped. Existing edits and source files were not overwritten.
pause
exit /b 1
