# XS2 영상 자막

**`1.xenosaga1/movie/kr` 또는 `2.xenosaga2/movie/kr`의 TXT를 번역하고,
해당 디스크 폴더의 `2.ApplySubtitles.bat`를 실행하면 게임 인코딩으로 변환하여
외부 Xenosaga2의 `xenosaga1_UNPACKED/movie` 또는 `xenosaga2_UNPACKED/movie`에 자동 적용합니다.**
그 뒤 외부 작업 폴더의 디스크별 ROM 리빌드 BAT를 실행하세요. 자막 BAT는 ISO를 만들지 않습니다.

## 편집과 준비

- `movie/jp`: 일본판 원본에서 복원한 UTF-8 일본어. 편집하지 않습니다.
- `movie/kr`: UTF-8 번역 작업본. 미번역 부분이 일본어여도 됩니다. 기존 번역 45개는 바이트 그대로 이곳으로 옮겼습니다.
- **게임에는 jp/kr 폴더 구분이 없습니다.** 자동 적용 경로는 `movie/파일명.txt`이며 `movie/jp`가 아닙니다.
- `1.PrepareSubtitles.bat`는 보존 원본을 검증하고 없는 jp/kr만 준비합니다. 기존 kr는 덮어쓰지 않습니다. 이미 준비돼 있으므로 평소에는 2번 BAT만 사용하면 됩니다.
- 일본어 원본은 `tools/subtitles/original/1`·`2`의 게임 인코딩 파일입니다. 해시를 검사하며 ISO나 외부 적용본에서 다시 가져오지 않습니다.
- 원본 ISO 등록 목록 기준 Disc 1은 20개, Disc 2는 25개입니다. Disc 1에 잘못 섞였던 `s040110`·`s050210`·`s050400`은 백업 및 Disc 2본과 동일함을 확인한 뒤 사용자 승인으로 삭제했습니다.
- 최초 기존 TXT 48개는 `codex-lab/subtitles/backup`에 그대로 백업했습니다. 이 백업은 삭제하지 마세요.

## 보존과 검증

한글만 `0.xenosaga0/yosikawa/XENOSAGA KOR-JPN.json`으로 치환한 뒤 **EUC-JIS-2004 strict**로 변환합니다.
대응 한글 폰트도 게임에 적용돼 있어야 합니다. 일본어 원문을 번역본의 역치환으로 만들지 않았습니다.
탭·공백·CRLF/LF·문장부호·리터럴 `\n`·주석을 자동 정리하거나 번역하지 않습니다. UTF-8 BOM은 출력에 넣지 않습니다.

타이밍·구분 공백/탭·`$ay...;` 등의 서식 명령은 원본과 동일하게 유지하세요. 줄바꿈은 기존처럼 리터럴 `\n`을 사용합니다.
시간만 있고 문구가 없는 줄도 정상적인 지우기 레코드이므로 유지합니다. `end`도 필요합니다.
없는 한글·인코딩 불가 문장부호·NUL·잘못된 형식·원본/대응표 불일치·타 디스크 TXT·누락은 자동 수정하지 않고 오류로 중단합니다.

모든 자막과 외부 디스크 등록 목록, 기존 대상 파일을 먼저 검사한 뒤 변경 파일만 복사합니다.
같은 결과는 다시 쓰지 않습니다. 디스크가 다른 자막, 추가 파일, 누락된 적용 대상을 자동 생성하지 않습니다.
외부 폴더는 기본값 `C:\Users\JO\Desktop\XENO_ARC\Xenosaga\Xenosaga2`입니다.
CLI에서 위치를 지정할 수도 있지만 올바른 디스크 LBA 해시·등록 파일이 있는 별도 작업 폴더여야 합니다.

파일마다 `.pending`에 완전히 저장·검사한 뒤 교체합니다. 사전 검증 오류는 복사 시작 전에 차단하지만,
실제 저장 중 디스크 오류가 나면 일부 파일만 갱신됐을 수 있습니다. 실패한 상태로 ISO를 만들지 말고
오류와 표시된 `.pending`을 확인한 뒤 다시 실행하세요. 중단된 pending은 자동 삭제하지 않습니다.

## 검사만 실행

XS2KOR 루트에서:

```powershell
python -B tools/subtitles/subtitles.py encode --disc 1 --check
python -B tools/subtitles/subtitles.py encode --disc 2 --check
```

Python 3.10 이상, 추가 패키지 불필요. ROM 빌더는 변경하지 않았으며 XS2KOR·lab·XS3를 읽지 않습니다.
이번 자동 복사는 사용자 승인에 따른 **자막 전용 예외**이며 다른 파이프라인의 자동 연동을 허용하지 않습니다.
원본·기존 번역 보존과 시험 기록은 `codex-lab/subtitles/README.md`를 참고하세요. 실제 게임 자막 표시는 별도 확인이 필요합니다.
