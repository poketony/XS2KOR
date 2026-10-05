# SB 번역 사용법

**기존 `sc*.txt`를 수정한 뒤 `4.DragStoryTxt.bat`에 드롭하면 TXT 옆에 `sc*.sb.new`가 만들어집니다.**
여러 TXT를 함께 드롭해도 됩니다. 이 폴더 전체는 `3.RepackAllStorySb.bat`를 더블클릭합니다.
`cf*.txt`는 장면 설정 파일이며 이 번역 도구의 편집 대상이 아닙니다.

전투 설명은 `../yamamoto/think`의 기존 `think*.txt`를 수정하고 그 폴더의
`4.DragBattleTxt.bat`에 드롭하거나 `3.RepackAllBattleSb.bat`를 실행합니다.

## 기존 번역 보존

- 기존 TXT는 재추출하거나 자동 수정하지 않습니다. 형식은 `[0x주소]` 다음 줄부터 번역문입니다.
- UTF-8로 저장하고 주소·`[n]`·`{XX}` 등의 기존 표기를 유지하세요. 공백은 자동 변환하지 않습니다.
- 원본 `.sb`·`.bin`은 읽기만 합니다. **작업 폴더의 원본을 `.new`로 덮어쓰지 마세요.**
- 입력 원본과 치환표의 해시를 검사합니다. 다른 버전이나 이미 번역된 SB는 거부합니다.
- 결과 `.new`는 성공적으로 만든 최신 내용으로 갱신합니다. 무변경 항목은 새 사본을 만들지 않습니다.
  다만 기존 `.new`가 있는 항목은 원문으로 되돌린 경우에도 갱신해 오래된 번역이 남지 않게 합니다.
- 하나라도 TXT 누락·중복 주소·인코딩 오류·불명확한 EXT·원본 불일치가 있으면 요청 전체를 저장 전에 중단합니다.
- 저장 중 디스크 오류가 발생하면 일부 `.new`만 갱신되었을 수 있습니다. 실패 표시가 나온 실행의
  결과를 적용하지 말고 오류를 해결한 뒤 같은 요청을 다시 실행하세요. 원본·TXT는 그대로입니다.
- `extract`도 기존 TXT가 있으면 중단합니다. 기존 번역 작업에 재추출은 필요 없습니다.

## BIN 자동 동기화와 게임 적용

원본 전체가 같은 것으로 확인한 BIN이 있으면 `.bin.new`도 함께 만듭니다.
예: `think002.txt` 리빌드 시 `think/think002.sb.new`와
`mapFromModel/E2_KML03.bin.new`가 같은 내용으로 생성됩니다. EXT로 파일이 커져도 처리합니다.

아카이브에 넣을 때 `.new`를 뺀 이름으로 **생성된 SB와 BIN을 모두** 교체하세요.
프로젝트 `yamamoto/mapFromModel`의 BIN은 게임 아카이브의 **`model/map`**에 들어갑니다.
파일이 커졌다면 아카이브의 크기·위치 정보도 갱신하는 기존 리팩 절차를 거쳐야 합니다.
이 도구가 아카이브나 ISO까지 자동 수정하지는 않습니다.

대응 SB가 없는 `think/E2_MID06.bin`은 자동 동기화하지 않습니다.
기존 `MatchThinkSb2bin.py patch think002.sb`도 이제 TXT 기반 리빌드로 연결되며 원본을 덮어쓰지 않습니다.
`scan`·`info`는 확인용이며 기존 `think_map.json`을 수정하지 않습니다.

## 검증만 실행

XS2KOR 루트에서 다음 명령은 결과 파일을 쓰지 않습니다.

```powershell
python -B 0.xenosaga0/sb_tool.py verify 0.xenosaga0/cf
python -B 0.xenosaga0/sb_tool.py verify 0.xenosaga0/yamamoto/think
```

Python 3.10 이상이 필요합니다. 기존 BAT/`0.sb2tool.py` 이름은 유지하며 내부적으로
`0.xenosaga0/sb_tool.py`를 공유합니다. `sb_sources.json`은 검증된 원본 471개와 BIN 120개의
해시·대응 관계를 담은 **도구 내부 자료**입니다. 번역 파일이 아니므로 편집하지 마세요.
치환표는 해당 원본 SB 폴더의 `XENOSAGA KOR-JPN.json`을 사용합니다.

검증·분석 기록: `codex-lab/font-rendering/sb-tool/README.md`.
