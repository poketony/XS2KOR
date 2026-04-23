#!/usr/bin/env python3
"""
sb2tool.py - 제노사가 에피소드 2 .sb 스크립트 파일 텍스트 추출/삽입 툴

사용법:
  python sb2tool.py extract <input.sb>
  python sb2tool.py import  <original.sb> <translated.txt>

JSON 치환표: 스크립트와 같은 폴더의 "XENOSAGA KOR-JPN.json"
  {"replace-table": {"가": "あ", "나": "に", ...}}
"""

import sys, os, re, struct, json

ENCODING  = 'euc-jp'
JSON_FILE = 'XENOSAGA KOR-JPN.json'


# ── 커스텀 바이트 에스케이프 ────────────────────────────────

def raw_to_text(raw_bytes):
    """raw EUC-JP -> 유니코드. 불가 바이트는 {XXYY} 플레이스홀더."""
    result = []
    i = 0
    while i < len(raw_bytes):
        b = raw_bytes[i]
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
        if b < 0x80:
            result.append(chr(b))
            i += 1
            continue
        result.append('{%02X}' % b)
        i += 1
    return ''.join(result)


def text_to_raw(text):
    """유니코드 -> raw EUC-JP. {XX}/{XXYY} 플레이스홀더 복원."""
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
            raise
        i += 1
    return bytes(result)


# ── 헤더 파싱 ──────────────────────────────────────────────

def parse_header(data):
    assert data[:4] == b'SB  ', f"매직 불일치: {data[:4]!r}"
    o = [struct.unpack_from('<I', data, 0x10 + i*4)[0] for i in range(7)]
    return {
        'secmeta_start': o[2],   # 바이트코드 오프셋 기준점 (base)
        'sp_start':      o[3],   # string pool 시작
        'sp_end':        o[4],   # string pool 끝
    }


# ── string pool 파싱 ────────────────────────────────────────

def read_pool(data, sp_start, sp_end):
    """[(abs_offset, raw_bytes), ...] 반환. raw_bytes = NULL 미포함."""
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
    return any(b > 0x7F for b in raw)


# ── 텍스트 파일 파싱 ────────────────────────────────────────

def parse_txt(content):
    """{abs_off(int): text(str)} 반환."""
    result = {}
    current_off   = None
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

def patch_bytecode_refs(sb_data, code_start, code_end, old_rel, new_rel):
    """
    바이트코드(code_start~code_end)에서 텍스트 참조 opcode(0x0028 + subtype 0x0006/0x0004)
    인수 위치의 old_rel(u16)만 찾아 new_rel로 패치.
    반환: 패치된 위치 수
    """
    import struct as _struct
    old_bytes = _struct.pack('<H', old_rel)
    new_bytes = _struct.pack('<H', new_rel)
    count = 0
    pos = code_start
    while pos < code_end - 1:
        if sb_data[pos:pos+2] == old_bytes:
            # 앞 4바이트 확인: opcode=0x0028, subtype=0x0006 or 0x0004
            if pos >= 4:
                op      = _struct.unpack_from('<H', sb_data, pos - 4)[0]
                subtype = _struct.unpack_from('<H', sb_data, pos - 2)[0]
                if op == 0x0028 and subtype == 0x0006:
                    sb_data[pos:pos+2] = new_bytes
                    count += 1
        pos += 2
    return count


def cmd_import(sb_path, txt_path):
    table = load_table(JSON_FILE)
    if not table:
        print(f'[!] 치환표 없음: {JSON_FILE}')
        return

    data   = open(sb_path, 'rb').read()
    hdr    = parse_header(data)
    pool   = read_pool(data, hdr['sp_start'], hdr['sp_end'])

    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        translations = parse_txt(f.read())

    print(f'[*] 원본: {len(pool)}개 문자열 / 번역: {len(translations)}개')

    secmeta    = hdr['secmeta_start']
    sp_start   = hdr['sp_start']
    sp_end     = hdr['sp_end']
    code_start = 0x40
    code_end   = secmeta  # 바이트코드 영역

    sb_data = bytearray(data)

    for abs_off, raw in pool:
        if abs_off not in translations:
            continue

        text_raw  = translations[abs_off]
        converted = convert(text_raw.replace('[n]', '\n'), table)

        try:
            new_bytes = text_to_raw(converted)
        except UnicodeEncodeError:
            print(f'  [!] 인코딩 에러 {hex(abs_off)}: 치환표에 없는 문자, 원문 유지')
            continue

        orig_len = len(raw)
        old_rel  = abs_off - secmeta

        if len(new_bytes) <= orig_len:
            # 인플레이스: 덮어쓰고 남는 자리 0x00
            sb_data[abs_off : abs_off + len(new_bytes)] = new_bytes
            for j in range(abs_off + len(new_bytes), abs_off + orig_len):
                sb_data[j] = 0x00
            print(f'  [OK] {hex(abs_off)} ({orig_len}->{len(new_bytes)} bytes)')

        else:
            # 초과: 파일 끝에 추가 + 바이트코드 내 오프셋 패치
            new_abs = len(sb_data)
            new_rel = new_abs - secmeta
            sb_data.extend(new_bytes + b'\x00')

            n = patch_bytecode_refs(sb_data, code_start, code_end, old_rel, new_rel)
            print(f'  [EXT] {hex(abs_off)} ({orig_len}->{len(new_bytes)} bytes)'
                  f' -> {hex(new_abs)} (바이트코드 {n}곳 패치)')

    out_name = os.path.basename(sb_path) + '.new'
    out_path = os.path.join(os.getcwd(), out_name)
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
