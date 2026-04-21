#!/usr/bin/env python3
"""
sb2tool.py - 제노사가 에피소드 2 .sb 스크립트 파일 텍스트 추출/삽입 툴

인터페이스는 제노사가 3 sbtool과 동일하게 맞춤.
인코딩: 항상 EUC-JP. 임포트 시 JSON 치환표로 한글→한자 변환 후 EUC-JP 인코딩.

사용법:
  python sb2tool.py extract <input.sb>
  python sb2tool.py import  <original.sb> <translated.txt>

텍스트 파일 형식 (extract 출력):
  [0x절대오프셋]
  원문
  (빈 줄)

JSON 치환표: 스크립트와 같은 폴더의 "XENOSAGA KOR-JPN.json"
  {"replace-table": {"가": "あ", "나": "に", ...}}
"""

import sys, os, re, struct, json

ENCODING   = 'euc-jp'
JSON_FILE  = 'XENOSAGA KOR-JPN.json'


# ── 커스텀 바이트 에스케이프 ────────────────────────────────
# EUC-JP로 표현 불가한 게임 전용 바이트를 {XX}/{XXYY} 플레이스홀더로 보존.
# 예: 0xADA1 -> {ADA1}, 0xAD -> {AD}

def raw_to_text(raw_bytes):
    """raw EUC-JP 바이트열 -> 유니코드 문자열. 불가 바이트는 {XXYY} 에스케이프."""
    result = []
    i = 0
    while i < len(raw_bytes):
        b = raw_bytes[i]
        # EUC-JP 2바이트 문자 시도
        if 0xA1 <= b <= 0xFE and i + 1 < len(raw_bytes):
            pair = raw_bytes[i:i+2]
            try:
                result.append(pair.decode(ENCODING))
                i += 2
                continue
            except (UnicodeDecodeError, LookupError):
                result.append('{%02X%02X}' % (b, raw_bytes[i+1]))
                i += 2
                continue
        # ASCII (줄바꿈 포함)
        if b < 0x80:
            result.append(chr(b))
            i += 1
            continue
        # 단독 고바이트
        result.append('{%02X}' % b)
        i += 1
    return ''.join(result)


def text_to_raw(text):
    """유니코드 문자열 -> raw EUC-JP 바이트열. {XX}/{XXYY} 플레이스홀더 복원."""
    result = bytearray()
    i = 0
    while i < len(text):
        m = re.match(r'\{([0-9A-Fa-f]{2,4})\}', text[i:])
        if m:
            result.extend(bytes.fromhex(m.group(1)))
            i += len(m.group(0))
            continue
        ch = text[i]
        try:
            result.extend(ch.encode(ENCODING))
        except UnicodeEncodeError:
            # 치환표에서 변환 못 한 글자 -> 그대로 두면 오류, 호출자에서 처리
            raise
        i += 1
    return bytes(result)



# ── 헤더 파싱 ──────────────────────────────────────────────

def parse_header(data):
    assert data[:4] == b'SB  ', f"매직 불일치: {data[:4]!r}"
    o = [struct.unpack_from('<I', data, 0x10 + i*4)[0] for i in range(7)]
    return {
        'unk10':          o[0],
        'code_start':     o[1],
        'secmeta_start':  o[2],
        'sp_start':       o[3],   # string pool 시작
        'sp_end':         o[4],   # string pool 끝
        'ptrtable_start': o[5],
        'end_section':    o[6],
    }


# ── string pool 파싱 ────────────────────────────────────────

def read_pool(data, sp_start, sp_end):
    """
    반환: [(abs_offset, raw_bytes), ...]
    raw_bytes = NULL 미포함 원본 바이트열
    """
    pool = data[sp_start:sp_end]
    result = []
    off = 0
    while off < len(pool):
        null = pool.find(b'\x00', off)
        if null == -1:
            break
        result.append((sp_start + off, pool[off:null]))
        off = null + 1
    return result


def is_jp(raw):
    """일본어 포함 여부 (비-ASCII 바이트 존재)"""
    return any(b > 0x7F for b in raw)


# ── 포인터 테이블 (lo값 업데이트) ──────────────────────────

def find_ptr_binary_end(data, ptr_start, ptr_limit):
    """
    ptr_start 부터 u32 배열을 읽으면서 hi16 > 0x100 이면 중단.
    반환: 포인터 binary 부분의 끝 오프셋 (exclusive)
    """
    end = ptr_start
    pos = ptr_start
    while pos + 4 <= ptr_limit:
        v = struct.unpack_from('<I', data, pos)[0]
        if (v >> 16) > 0x100:
            break
        end = pos + 4
        pos += 4
    return end


def update_ptr_table(data, hdr, old_sp_start, offset_map):
    """
    포인터 테이블의 lo16 값을 offset_map 에 따라 갱신.
    offset_map: {old_pool_rel_off -> new_pool_rel_off}
    (pool_rel_off = 절대오프셋 - sp_start)
    """
    ptr_start = hdr['ptrtable_start']
    ptr_end   = hdr['end_section']
    ptr_bin_end = find_ptr_binary_end(data, ptr_start, ptr_end)

    # string pool 경계 목록 (pool 내 상대 오프셋 기준)
    pool = data[old_sp_start:hdr['sp_end']]
    boundaries = []
    off = 0
    while off < len(pool):
        null = pool.find(b'\x00', off)
        if null == -1: break
        boundaries.append((off, null))
        off = null + 1

    result = bytearray(data)
    for pos in range(ptr_start, ptr_bin_end, 4):
        v  = struct.unpack_from('<I', data, pos)[0]
        hi = v >> 16
        lo = v & 0xFFFF
        # lo가 속한 문자열 찾기
        for (s, e) in boundaries:
            if s <= lo <= e:
                intra  = lo - s
                new_lo = offset_map.get(s, s) + intra
                struct.pack_into('<I', result, pos, (hi << 16) | (new_lo & 0xFFFF))
                break
    return bytes(result)


# ── 텍스트 파일 파싱 ────────────────────────────────────────

def parse_txt(content):
    """[0x절대오프셋] 태그 기반 파싱. {abs_off(int): text(str)} 반환."""
    result = {}
    current_off  = None
    current_lines = []
    for line in content.splitlines():
        m = re.match(r'^\[(0x[0-9a-fA-F]+)\]$', line)
        if m:
            if current_off is not None:
                result[current_off] = '\n'.join(current_lines).rstrip('\n')
            current_off   = int(m.group(1), 16)
            current_lines = []
        else:
            if current_off is not None:
                current_lines.append(line)
    if current_off is not None:
        result[current_off] = '\n'.join(current_lines).rstrip('\n')
    return result


# ── JSON 치환표 ─────────────────────────────────────────────

def load_table(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        return json.load(f).get('replace-table', {})


def convert(text, table):
    return ''.join(table.get(c, c) for c in text)


# ── extract ─────────────────────────────────────────────────

def cmd_extract(sb_path):
    data = open(sb_path, 'rb').read()
    hdr  = parse_header(data)
    pool = read_pool(data, hdr['sp_start'], hdr['sp_end'])

    base_name = os.path.splitext(os.path.basename(sb_path))[0] + '.txt'
    txt_path  = os.path.join(os.getcwd(), base_name)
    count = 0
    with open(txt_path, 'w', encoding='utf-8-sig') as f:
        for abs_off, raw in pool:
            if not is_jp(raw):
                continue
            text = raw_to_text(raw)
            f.write(f'[{hex(abs_off)}]\n{text.replace(chr(10), "[n]")}\n\n')
            count += 1

    print(f'[*] {txt_path} 추출 완료 ({count}개)')


# ── import ──────────────────────────────────────────────────

def cmd_import(sb_path, txt_path):
    table = load_table(JSON_FILE)
    if not table:
        print(f'[!] 치환표 없음: {JSON_FILE}')
        print('    치환표 없이는 임포트 불가. JSON 파일을 같은 폴더에 두세요.')
        return

    data   = open(sb_path, 'rb').read()
    hdr    = parse_header(data)
    pool   = read_pool(data, hdr['sp_start'], hdr['sp_end'])

    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        translations = parse_txt(f.read())

    print(f'[*] 원본: {len(pool)}개 문자열 / 번역: {len(translations)}개')

    sp_start = hdr['sp_start']
    sp_end   = hdr['sp_end']

    # ── 1단계: 인플레이스 패치 + 초과분 수집 ────────────────
    sb_data    = bytearray(data)
    overflow   = bytearray()   # sp_end 뒤에 붙을 초과 텍스트
    offset_map = {}            # old_pool_rel -> new_pool_rel

    for abs_off, raw in pool:
        pool_rel = abs_off - sp_start

        if abs_off not in translations:
            # 미번역: 오프셋 변화 없음
            offset_map[pool_rel] = pool_rel
            continue

        text_raw = translations[abs_off]
        converted = convert(text_raw.replace('[n]', '\n'), table)

        try:
            new_bytes = text_to_raw(converted)
        except UnicodeEncodeError:
            print(f'  [!] 인코딩 에러 {hex(abs_off)}: 치환표에 없는 문자 포함, 원문 유지')
            offset_map[pool_rel] = pool_rel
            continue

        orig_len = len(raw)

        if len(new_bytes) <= orig_len:
            # 인플레이스: 덮어쓰고 나머지 0x00
            sb_data[abs_off : abs_off + len(new_bytes)] = new_bytes
            for j in range(abs_off + len(new_bytes), abs_off + orig_len):
                sb_data[j] = 0x00
            offset_map[pool_rel] = pool_rel
            print(f'  [OK] {hex(abs_off)} 인플레이스 ({orig_len}->{len(new_bytes)} bytes)')
        else:
            # 초과: sp_end 뒤에 추가
            new_abs  = sp_end + len(overflow)
            new_rel  = new_abs - sp_start
            overflow.extend(new_bytes + b'\x00')
            offset_map[pool_rel] = new_rel
            print(f'  [EXT] {hex(abs_off)} 확장 ({orig_len}->{len(new_bytes)} bytes) -> {hex(new_abs)}')

    # ── 2단계: 초과 텍스트가 있으면 헤더 오프셋 업데이트 ────
    if overflow:
        ext_size = len(overflow)
        print(f'[*] 초과 텍스트 {ext_size} bytes를 sp_end에 추가')

        # sp_end 이후 오프셋에 ext_size 더하기
        def shift(v):
            return v + ext_size if v >= sp_end else v

        new_offsets = [
            hdr['unk10'],
            hdr['code_start'],
            hdr['secmeta_start'],
            hdr['sp_start'],
            hdr['sp_end'] + ext_size,
            shift(hdr['ptrtable_start']),
            shift(hdr['end_section']),
        ]
        for i, v in enumerate(new_offsets):
            struct.pack_into('<I', sb_data, 0x10 + i*4, v)

        # 포인터 테이블 lo 업데이트 (초과 항목)
        ptr_bin_end = find_ptr_binary_end(
            bytes(sb_data), hdr['ptrtable_start'], hdr['end_section']
        )
        pool_data = bytes(sb_data)[sp_start:sp_end]
        boundaries = []
        off = 0
        while off < len(pool_data):
            null = pool_data.find(b'\x00', off)
            if null == -1: break
            boundaries.append((off, null))
            off = null + 1

        for pos in range(hdr['ptrtable_start'], ptr_bin_end, 4):
            v  = struct.unpack_from('<I', sb_data, pos)[0]
            hi = v >> 16
            lo = v & 0xFFFF
            for (s, e) in boundaries:
                if s <= lo <= e:
                    intra  = lo - s
                    new_lo = offset_map.get(s, s) + intra
                    struct.pack_into('<I', sb_data, pos, (hi << 16) | (new_lo & 0xFFFF))
                    break

        # 초과 텍스트 삽입 (sp_end 위치에)
        sb_data = sb_data[:sp_end] + overflow + sb_data[sp_end:]

    # ── 3단계: 저장 ─────────────────────────────────────────
    out_name = os.path.basename(sb_path) + '.new'
    out_path  = os.path.join(os.getcwd(), out_name)
    open(out_path, 'wb').write(bytes(sb_data))
    print(f'[+] 저장: {out_path} ({len(sb_data):,} bytes)')


# ── 진입점 ──────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'extract':
        cmd_extract(sys.argv[2])
    elif cmd == 'import':
        if len(sys.argv) < 4:
            print('사용법: sb2tool.py import <orig.sb> <trans.txt>')
            sys.exit(1)
        cmd_import(sys.argv[2], sys.argv[3])
    else:
        print(f'알 수 없는 명령: {cmd}')
        sys.exit(1)


if __name__ == '__main__':
    main()
