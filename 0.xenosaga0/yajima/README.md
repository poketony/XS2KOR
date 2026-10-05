# Synopsis BIN 작업

## 핵심 사용법

아래 작업 경로는 이 README가 있는 `0.xenosaga0/yajima` 폴더 기준입니다.

**`graphics_extract/synopsis/images/loading.png`, `start.png`를 포토샵으로 수정해서 옆에 `loading_KOR.png`, `start_KOR.png`로 저장하고, `graphics_extract/synopsis` 폴더를 `2.Rebuild.bat`에 드롭하면 됩니다.** 결과 `graphics_rebuilt/synopsis/synopsis.bin`을 게임 아카이브의 `yajima/synopsis.bin`에 교체·리팩합니다.

이 도구는 **`synopsis.bin`만** 지원합니다. `diskchg`, `end`, `gameov1`, `mnt_set`은 작업 대상에서 제외했으며, 기존 JPG는 그대로 보존합니다.

프로젝트 파일을 수정하거나 리빌드하는 것만으로 ISO가 자동 갱신되지는 않습니다. 외부 `Xenosaga2` 원본 디렉터리는 계속 읽기 전용으로 유지합니다.

## 실제 수정 대상

| 내용 | 편집 파일 | 게임에 넣는 경로 |
| --- | --- | --- |
| 줄거리 본문 22장 | **기존 `synopsis/01.jpg`~`22.jpg`** | `yajima/synopsis/01.jpg`~`22.jpg` |
| 로딩·동그라미 버튼 시작 안내 | `graphics_extract/synopsis/images/loading.png`, `start.png` | `yajima/synopsis.bin` |

줄거리 본문은 기존 JPG를 포토샵으로 수정하고 같은 이름·게임 경로로 교체·리팩합니다. **본문은 `synopsis.bin`에 넣지 않으며 `2.Rebuild.bat`도 사용하지 않습니다.**

**`graphics_extract/synopsis/images/01.png`~`22.png`는 BIN 안에 있던 별도 본문 사본입니다. 확인한 줄거리 표시 경로에서는 이 사본을 사용하지 않으므로 번역하지 않습니다.** 기존 `synopsis` 폴더의 JPG와 문구·배치도 다릅니다. BIN 쪽 PNG를 기존 JPG 대신 번역하거나, JPG와 자동으로 동기화되는 것으로 생각하면 안 됩니다.

현재 도구는 BIN 내부 24장을 모두 읽으므로 **BIN 본문 PNG 22장과 `manifest.json`은 수정·삭제하지 말고 그대로 둡니다.** 안내를 바꿀 때는 `loading_KOR.png`, `start_KOR.png`만 추가하면 됩니다. 추출된 PNG 24장이 모두 번역 대상이라는 뜻은 아닙니다.

## 이미지 형식

- 기존 줄거리 JPG 22장: 원본과 같은 **512×448, RGB, baseline JPEG, 4:4:4**로 저장합니다. 프로그레시브 JPEG나 CMYK로 바꾸지 않습니다. 현재 `image_tool.py`는 이 개별 JPG들의 변환·검사를 지원하지 않습니다.
- `loading.png`, `start.png`: **512×64, 투명도를 유지한 RGBA PNG**로 저장합니다. 원본은 직접 RGBA XTX이며 알파는 PS2의 표현 단계로 반올림됩니다.
- 포토샵의 8비트 RGBA PNG를 사용하고 크기는 바꾸지 않습니다.

리빌드는 `_KOR.png`를 우선 사용하고, 없으면 원래 PNG를 사용합니다. 원래 PNG를 직접 수정하는 방식도 지원합니다. 안내 이미지는 팔레트 양자화 대상이 아니며, 무편집 데이터는 그대로 보존합니다.

## 폴더와 재실행

- `synopsis.bin`: 프로젝트에 원래 있던 기준 파일을 직접 읽습니다. 별도 `original` 사본은 만들지 않습니다.
- `graphics_extract/synopsis`: 이미 추출된 PNG 편집 공간.
- `graphics_rebuilt/synopsis`: 변경된 BIN과 `build-report.json`. 무편집이면 보고서만 나옵니다.
- `1.Export.bat`: 새로 추출할 때 더블클릭하거나 `synopsis.bin`을 드롭합니다. 기존 추출본을 덮어쓰지 않습니다.

기준 `synopsis.bin`이 추출 이후 바뀌면 해시 검사에서 중단합니다. 게임에 적용할 때는 `graphics_rebuilt`의 결과를 사용하고, 기준 파일을 바꿨다면 새 작업 폴더로 다시 추출합니다.

리빌드는 원본 파일 크기, 헤더와 편집하지 않은 데이터를 유지합니다. 결과 폴더가 이미 있으면 덮어쓰지 않고 중단합니다. 이전 결과와 다른 폴더로 만들 때만 이 폴더의 PowerShell에서 실행합니다.

```powershell
python -B image_tool.py rebuild graphics_extract/synopsis graphics_rebuilt/synopsis_v2
```

보고서에는 적용 PNG와 변경 내용이 기록됩니다. `--quality`는 BIN 내부 JPEG 편집용 옵션이므로 로딩·시작 안내 작업에는 필요하지 않습니다.

## 원본 범위와 검증

2026-10-04 Disc 1 실행 파일 정적 분석에서 `0x1e8e40` 함수가 BIN의 첫 두 XTX를 안내용으로 가져온 뒤, `yajima\synopsis\00.jpg`의 번호를 바꿔 개별 JPG를 로드하고 JPEG 디코더로 넘기는 흐름을 확인했습니다. 외부 원본 ELF와 프로젝트 ELF의 해당 함수는 동일합니다. BIN에 본문 사본이 남은 이유는 미확정이며, 현지화용·개발 잔재라는 해석은 가설입니다.

BIN 내부 24장 무편집 왕복과 XTX 수정 시 다른 데이터 보존을 검사했습니다. BIN 내부 JPEG 22장도 포맷 보존 검사는 유지하지만, 게임 내 사용을 보증하는 것은 아닙니다. 기존 `synopsis` 폴더의 JPG 22장은 이 도구의 왕복 검사 대상이 아닙니다. 제외한 JPG의 입력 거절, 기존 루트 BIN 참조와 기준 파일 변경 거절을 포함해 전체 회귀 테스트 80개가 통과했습니다. 실제 게임 표시는 별도 확인해야 하며 외부 원본 자동 적용이나 ISO 제작은 하지 않습니다.

Python 3.10 이상, NumPy, Pillow가 필요합니다. 상세 검증 기록은 `codex-lab/asset-pipeline`에 있습니다.
