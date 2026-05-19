@echo off
@chcp 65001
setlocal enabledelayedexpansion

echo [.txt 드롭 방식] 리팩 작업을 시작합니다...

:: 전달된 인자가 없는 경우 안내
if "%~1" == "" (
    echo.
    echo [오류] 처리할 .txt 파일을 이 배치 파일 위로 드래그해 주세요.
    pause
    exit /b
)

:: 드롭된 모든 파일들에 대해 루프
for %%f in (%*) do (
    :: 확장자가 .txt인 경우만 처리
    if /i "%%~xf" == ".txt" (
        set "FILENAME=%%~nf"
        echo.
        echo 대상 파일명: !FILENAME!
        
        :: .sb 파일이 존재하는지 확인 후 명령 실행
        if exist "!FILENAME!.sb" (
            echo 명령 실행: python 0.sb2tool.py import "!FILENAME!.sb" "%%~nxf"
            python 0.sb2tool.py import "!FILENAME!.sb" "%%~nxf"
        ) else (
            echo [경고] "!FILENAME!.sb" 파일을 찾을 수 없어 건너뜁니다.
        )
    ) else (
        echo [건너뜀] .txt 파일이 아닙니다: %%~nxf
    )
)

echo.
echo 모든 작업이 완료되었습니다!
pause