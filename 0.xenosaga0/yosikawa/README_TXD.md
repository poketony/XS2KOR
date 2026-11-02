# TXD 텍스트 리빌드

**`Location.txd.txt` 또는 `Tutorial.txd.txt`를 수정한 뒤 `3.RebuildTxd.bat`을 실행하고, 파일명을 입력해 Enter를 누르면 됩니다.**

## 실행 방법

1. 번역 TXT를 UTF-8로 저장합니다.
2. 같은 폴더의 `3.RebuildTxd.bat`을 더블클릭합니다.
3. `Location` 또는 `Tutorial`을 입력하고 Enter를 누릅니다. 대소문자는 구분하지 않습니다.

`Location.txd`, `Location.txd.txt`처럼 확장자를 붙여 입력해도 됩니다. 전체 경로는 입력하지 않습니다.
잘못된 이름이나 원본/TXT 누락은 결과를 생성하기 전에 검사합니다.

| 입력 | 생성되는 파일 | ROM에 적용할 이름 |
|---|---|---|
| `Location` | `Location_repacked.txd` | `Location.txd` |
| `Tutorial` | `Tutorial_repacked.txd` | `Tutorial.txd` |

생성된 파일을 ROM 작업 폴더의 같은 위치에 수동으로 옮기고, 위 이름으로 바꿔 교체합니다.
XS2KOR 안의 원본 `.txd`는 덮어쓰지 마세요. BAT는 원본이나 번역 TXT를 수정하지 않습니다.

## 명령어로 실행할 경우

이 README가 있는 `C:\Users\JO\XS2KOR\0.xenosaga0\yosikawa` 폴더에서 실행합니다.

```powershell
python -B txd_tool.py r Location.txd Location.txd.txt
python -B txd_tool.py r Tutorial.txd Tutorial.txd.txt
```

## 주의 사항

- TXT는 UTF-8로 저장하고 `[번호]`와 항목 순서를 유지하세요.
- 한글은 같은 폴더의 `XENOSAGA KOR-JPN.json` 대응표로 자동 변환됩니다. 오류가 뜨면 결과를 적용하지 말고 원인을 확인하세요.
- 같은 이름의 `_repacked.txd`가 이미 있으면 다시 생성하면서 덮어씁니다.
- 번역 중에는 추출 명령(`e`)을 실행하지 마세요. 기존 TXT를 원문으로 덮어씁니다.
- `MainMenu.txd.txt`는 별도 통합 패치와 짝을 이루므로 `discmetadata/2.Rebuild.bat`으로 처리합니다. 이 BAT는 Location과 Tutorial을 리빌드하지 않습니다.
