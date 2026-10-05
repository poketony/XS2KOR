@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if "%~1"=="" goto usage

:next
python -B "%~dp0esd_esp_tool.py" rebuild "%~1"
if errorlevel 1 goto failed
shift
if not "%~1"=="" goto next
echo Rebuild completed. See the graphics_rebuilt folder beside this batch file.
pause
exit /b 0

:usage
echo Drop an exported folder or its manifest.json onto this batch file.
echo Save edited PNGs beside originals as name_KOR.png. KOR copies take priority.
echo Sources under original automatically synchronize identical embedded ESD copies.
pause
exit /b 1

:failed
echo Rebuild stopped. Existing results and source files were not overwritten.
pause
exit /b 1
