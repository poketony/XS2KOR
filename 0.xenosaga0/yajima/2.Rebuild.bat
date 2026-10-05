@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if "%~1"=="" goto usage

:next
python -B "%~dp0image_tool.py" rebuild "%~1"
if errorlevel 1 goto failed
shift
if not "%~1"=="" goto next
echo Synopsis rebuild completed. See graphics_rebuilt beside this batch file.
pause
exit /b 0

:usage
echo Save loading_KOR.png or start_KOR.png, then drop the synopsis work folder here.
pause
exit /b 1

:failed
echo Rebuild stopped. Original files and existing results were not overwritten.
pause
exit /b 1
