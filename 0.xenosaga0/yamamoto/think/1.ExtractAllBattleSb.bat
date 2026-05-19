@echo off
@chcp 65001
echo .sb 파일 추출을 시작합니다...

:: 현재 폴더의 모든 .sb 파일에 대해 루프 실행
for %%f in (*.sb) do (
    echo 처리 중: %%f
    python 0.sb2tool.py extract "%%f"
)

echo.
echo 모든 작업이 완료되었습니다!
pause