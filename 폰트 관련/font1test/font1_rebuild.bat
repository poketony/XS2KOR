@echo off
setlocal
pushd "%~dp0" || exit /b 1
python -B xeno1font_rb.py -s font1_sheet1.png font1_sheet2.png font1_sheet3.png font1_sheet4.png -r font1.tex -o font1.tex.new
set "result=%errorlevel%"
popd
pause
exit /b %result%
