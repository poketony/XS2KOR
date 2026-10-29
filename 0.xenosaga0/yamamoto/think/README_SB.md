# 전투 설명 SB 번역

**`think*.txt`의 `[KOR]` 아래에 번역하고 `4.DragBattleTxt.bat`에 드롭하면 됩니다.**
전체 처리는 `3.RepackAllBattleSb.bat`를 더블클릭합니다. TXT와 원본 SB/BIN은 덮어쓰지 않습니다.

- `[JPN]`과 주소·첫 줄 헤더는 그대로 두세요. KOR가 비어 있거나 공백뿐이면 원본을 유지합니다.
- 기존 번역은 KOR에 보존했습니다. 새로 추출하는 TXT도 JPN/KOR 형식입니다. 재추출은 필요 없습니다.
- 결과: TXT 옆의 `thinkNNN.sb.new`. 대응 BIN이 있으면 `../mapFromModel/이름.bin.new`도 함께 생성합니다.
- 예: `think002.txt` → `think002.sb.new`와 `../mapFromModel/E2_KML03.bin.new`.
- 게임에는 두 결과를 모두 적용하세요. SB는 `yamamoto/think`, BIN은 **`model/map`** 경로입니다.
- `.new`만 뺀 이름으로 아카이브에 넣습니다. 작업 폴더의 기준 원본에 덮어쓰지 마세요.
- 기존 번역을 재추출할 필요가 없습니다. 성공 시 `.new`만 최신 결과로 갱신합니다.
- 실패 표시가 나오면 해당 실행의 결과를 사용하지 마세요. 원인을 해결하고 다시 실행합니다.
- 원본과 대응 관계를 확인하지 못한 `E2_MID06.bin`은 변경하지 않습니다.

편집 형식·검증 명령·길이 증가 처리: [공통 사용법](../../cf/README_SB.md).

전환 전 TXT 115개와 도구·README 백업:
`codex-lab/sb-bilingual/think-20261029/backup`.
기존 번역 1,779항목을 보존했고 전환 전후 SB/BIN 빌드 결과가 같습니다. 백업은 삭제하지 마세요.
