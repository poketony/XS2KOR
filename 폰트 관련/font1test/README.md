# font1 편집

확정 글꼴: **Pretendard SemiBold 20px**, 공통 기준선 Y=19. 글자별 확대·축소는 하지 않는다.

- **시트 수정:** 이 폴더의 `font1_sheet2.png`~`font1_sheet4.png`를 수정하고 `font1_rebuild.bat`을 실행하면 `font1.tex.new`가 생성된다. 게임 적용 위치에는 `.new`를 뺀 이름으로 직접 옮긴다.
- **한글 시트 재생성:** `korfont/MakeFontSheet.bat`을 실행한 뒤 위 리빌드 BAT를 실행한다. **재생성은 한글 셀의 수동 수정을 덮어쓴다.** 재생성할 때만 사용한다.

`korfont/Pretendard-SemiBold.otf`와 green/blue/alpha 문자 목록으로 한글 2,350자를 생성한다. 설치된 글꼴이나 외부 ROM 작업 폴더는 참조하지 않는다. sheet 1 및 다른 문자 셀은 유지한다. 원본 `font1.tex`는 리빌드 기준이므로 교체하지 않는다.

검증·백업 기록: `../../codex-lab/font1-spacing/final/`. font0, ELF, 번역문은 이 작업에서 변경하지 않는다.
