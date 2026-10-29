@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if "%~1"=="" goto prompt

:next
python -B "%~dp0esd_esp_tool.py" rebuild "%~1"
if errorlevel 1 goto failed
shift
if not "%~1"=="" goto next
:done
echo Rebuild completed. See the graphics_rebuilt folder beside this batch file.
pause
exit /b 0

:prompt
echo Enter an exported folder name, for example ce001_999.
echo You can still drop an exported folder or manifest.json onto this batch file.
python -B "%~dp0esd_esp_tool.py" rebuild
if errorlevel 1 goto failed
goto done

:failed
echo Rebuild stopped. Existing results and source files were not overwritten.
pause
exit /b 1
