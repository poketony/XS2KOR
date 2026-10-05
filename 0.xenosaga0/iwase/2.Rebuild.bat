@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if "%~1"=="" goto all

:next
python -B "%~dp0field_text_tool.py" rebuild "%~1"
if errorlevel 1 goto failed
shift
if not "%~1"=="" goto next
goto done

:all
python -B "%~dp0field_text_tool.py" rebuild-all
if errorlevel 1 goto failed

:done
echo Rebuild completed. See rebuilt beside this batch file.
pause
exit /b 0

:failed
echo Rebuild stopped. Original files and existing results were not overwritten.
pause
exit /b 1
