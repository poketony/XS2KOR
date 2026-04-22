@echo off
@chcp 65001
setlocal

set "OUT=merged_dialogue_final.txt"
set "PS=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

if exist "%OUT%" del "%OUT%"

echo 가독성을 개선하여 병합 중...

%PS% -NoProfile -ExecutionPolicy Bypass ^
  "$outFile = Join-Path (Get-Location) '%OUT%';" ^
  "$files = Get-ChildItem -Filter *.txt | Where-Object { $_.Name -ne '%OUT%' } | Sort-Object Name;" ^
  "$result = New-Object System.Collections.Generic.List[string];" ^
  "foreach ($file in $files) {" ^
  "    $result.Add('[FILE:' + $file.BaseName + ']' + \"`r`n\");" ^
  "    $content = [System.IO.File]::ReadAllText($file.FullName, [System.Text.Encoding]::UTF8);" ^
  "    $content = $content -replace '(?s)\*{8,}\r?\n.*BGM.*\r?\n.*?\*{8,}\r?\n?', '';" ^
  "    $content = $content -replace '(?m)^[a-zA-Z_]+\s*\(.*?\)\r?\n?', '';" ^
  "    $result.Add($content);" ^
  "    $result.Add(\"`r`n`r`n\");" ^
  "}" ^
  "[System.IO.File]::WriteAllText($outFile, ([string]::Join(\"\", $result)), [System.Text.Encoding]::GetEncoding('utf-8'));"

echo.
echo 가독성 개선 및 병합 완료: %OUT%
pause