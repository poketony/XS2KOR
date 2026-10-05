# 전투 설명 SB 번역

**기존 `think*.txt`를 수정하고 `4.DragBattleTxt.bat`에 드롭하면 됩니다.**
전체 처리는 `3.RepackAllBattleSb.bat`를 더블클릭합니다. TXT와 원본 SB/BIN은 덮어쓰지 않습니다.

- 결과: TXT 옆의 `thinkNNN.sb.new`. 대응 BIN이 있으면 `../mapFromModel/이름.bin.new`도 함께 생성합니다.
- 예: `think002.txt` → `think002.sb.new`와 `../mapFromModel/E2_KML03.bin.new`.
- 게임에는 두 결과를 모두 적용하세요. SB는 `yamamoto/think`, BIN은 **`model/map`** 경로입니다.
- `.new`만 뺀 이름으로 아카이브에 넣습니다. 작업 폴더의 기준 원본에 덮어쓰지 마세요.
- 기존 번역을 재추출할 필요가 없습니다. 성공 시 `.new`만 최신 결과로 갱신합니다.
- 실패 표시가 나오면 해당 실행의 결과를 사용하지 마세요. 원인을 해결하고 다시 실행합니다.
- 원본과 대응 관계를 확인하지 못한 `E2_MID06.bin`은 변경하지 않습니다.

편집 형식·검증 명령·길이 증가 처리: [공통 사용법](../../cf/README_SB.md).
