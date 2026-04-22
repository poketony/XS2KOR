@echo off
@chcp 65001
echo .sb 파일 리팩을 시작합니다...

:: 현재 폴더의 모든 .sb 파일에 대해 루프 실행
for %%f in (*.sb) do (
    echo 처리 중: %%f
    :: %%~nf는 파일명에서 확장자(.sb)를 제거한 이름만 가져옵니다.
    python 0.sb2tool.py import "%%f" "%%~nf.txt"
)

echo.
echo 모든 작업이 완료되었습니다!
pause