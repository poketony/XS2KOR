#!/usr/bin/env python3
"""
sb2tool.py - 제노사가 에피소드 2 .sb 스크립트 파일 텍스트 추출/삽입 툴

사용법:
  python sb2tool.py extract <input.sb>
  python sb2tool.py import  <original.sb> <translated.txt>

추가 명령: rebuild <TXT...>, batch <folder>, verify <folder|TXT...>
JSON 치환표: 원본 SB와 같은 폴더의 "XENOSAGA KOR-JPN.json"
  {"replace-table": {"가": "あ", "나": "に", ...}}
"""

import sys, os, re, struct, json, tempfile, hashlib
from pathlib import Path

ENCODING  = 'euc_jis_2004'
JSON_FILE = 'XENOSAGA KOR-JPN.json'
BILINGUAL_HEADER = '# SB-TEXT: JPN-KOR v1'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(SCRIPT_DIR)

# Original ov01: VA 0xAB5650, lengths in u16 words (helper 0xA9D4C0).
OPCODE_WORDS = bytes.fromhex(
    '01030303030303030303010101010303030103010103010101010101010101010101010101000101010101010101'
)

# Exact original SBs only. See FORMAT.md for the consuming game APIs.
# End offsets are exclusive; only matching aligned u32 values are patched.
VERIFIED_DATA_RANGES = {
    # sc1610: two 40-entry talk.menuDeptQuest name arrays.
    '7998307f5c75d890fa35bf3cf761ab931cd531f5c5adbdd081c875a22a0e188d':
        ((0x6a8c, 0x6b2c), (0x6bd4, 0x6c74)),
    # think002 and think003: bat.msgPut tutorial arrays.
    '872e97d302ce9737ca4cb2e97396b64bc54aea271bc2a7220e45cab7d91b6c18':
        ((0x61c4, 0x62d4),),
    '56d75e8970d0d56715c3be3497606e69eb0c3cf4e77dd04045058ef173b7f6c7':
        ((0xd6b4, 0xd720),),
}


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
    if len(data) < 64 or data[:4] != b'SB  ':
        raise ValueError('완전한 SB 헤더가 아닙니다')
    o = [struct.unpack_from('<I', data, 0x10 + i*4)[0] for i in range(7)]
    if not 64 <= o[1] <= o[2] <= o[3] <= o[4] <= o[5] <= (o[6] or len(data)) <= len(data):
        raise ValueError('SB 영역 경계가 잘못되었습니다')
    return {
        'code_start':    o[1],
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
            raise ValueError('종단 NUL이 없는 문자열 풀')
        result.append((sp_start + off, pool[off:null]))
        off = null + 1
    return result


def is_jp(raw):
    return any(b > 0x7F for b in raw)


# ── 텍스트 파일 파싱 ────────────────────────────────────────

def parse_txt(content):
    """{abs_off(int): text(str)} 반환."""
    if is_bilingual(content):
        originals, translations = parse_bilingual(content)
        return {off: translations.get(off, text) for off, text in originals.items()}
    result = {}
    current_off   = None
    current_lines = []
    seen = {}
    for line_number, line in enumerate(content.splitlines(), 1):
        m = re.match(r'^\[(0x[0-9a-fA-F]+)\]$', line)
        if m:
            next_off = int(m.group(1), 16)
            if next_off in seen:
                raise ValueError(f'중복 offset {hex(next_off)}: '
                                 f'{seen[next_off]}행 / {line_number}행')
            seen[next_off] = line_number
            if current_off is not None:
                result[current_off] = '\n'.join(current_lines).rstrip('\n')
            current_off   = next_off
            current_lines = []
        else:
            if current_off is not None:
                current_lines.append(line)
    if current_off is not None:
        result[current_off] = '\n'.join(current_lines).rstrip('\n')
    return result


def is_bilingual(content):
    return any(line in (BILINGUAL_HEADER, '[JPN]', '[KOR]') or line.startswith('# SB-TEXT:')
               for line in content.splitlines())


def parse_bilingual(content):
    lines = content.splitlines()
    if not lines or lines[0] != BILINGUAL_HEADER:
        raise ValueError('JPN/KOR TXT 형식 헤더가 없거나 지원하지 않는 버전입니다')
    originals, translations = {}, {}
    current, section, fields = None, None, {}

    def finish():
        if current is None:
            return
        if 'JPN' not in fields:
            raise ValueError(f'{hex(current)}: [JPN]이 없습니다')
        originals[current] = '\n'.join(fields['JPN']).rstrip('\n')
        value = '\n'.join(fields.get('KOR', [])).rstrip('\n')
        if value.strip():
            translations[current] = value

    for line_number, line in enumerate(lines[1:], 2):
        address = re.fullmatch(r'\[(0x[0-9a-fA-F]+)\]', line)
        if address:
            finish()
            current = int(address[1], 16)
            if current in originals:
                raise ValueError(f'중복 offset {hex(current)}: {line_number}행')
            section, fields = None, {}
        elif line in ('[JPN]', '[KOR]'):
            name = line[1:-1]
            if current is None or name in fields or (name == 'KOR' and 'JPN' not in fields):
                raise ValueError(f'잘못된 JPN/KOR 구획: {line_number}행')
            if name == 'JPN' and section is not None:
                raise ValueError(f'JPN/KOR 순서 오류: {line_number}행')
            section = name
            fields[name] = []
        elif section is not None:
            fields[section].append(line)
        elif line:
            raise ValueError(f'구획 밖 내용: {line_number}행')
    finish()
    return originals, translations


# ── JSON 치환표 ─────────────────────────────────────────────

def convert(text, table):
    return ''.join(table.get(c, c) for c in text)


# ── extract ─────────────────────────────────────────────────

def export_text(data, bilingual=False):
    hdr  = parse_header(data)
    pool = read_pool(data, hdr['sp_start'], hdr['sp_end'])
    entries = [(off, raw_to_text(raw).replace(chr(10), '[n]'))
               for off, raw in pool if is_jp(raw)]
    if bilingual:
        return BILINGUAL_HEADER + '\n\n' + ''.join(
            f'[{hex(off)}]\n[JPN]\n{text}\n[KOR]\n\n' for off, text in entries)
    return ''.join(f'[{hex(off)}]\n{text}\n\n' for off, text in entries)


# ── import ──────────────────────────────────────────────────

def patch_code_type6_refs(sb_data, code_start, code_end, old_rel, new_rel):
    """
    바이트코드(code_start~code_end)의 type=0x0006 문자열 참조를 패치.
    값은 16비트가 아니라 32비트 little-endian으로 저장된다.
    """
    new_bytes = struct.pack('<I', new_rel)
    count = 0
    pos = code_start
    while pos < code_end:
        if pos + 2 > code_end:
            raise ValueError(f'잘린 명령어: {hex(pos)}')
        opcode = struct.unpack_from('<H', sb_data, pos)[0]
        if opcode >= len(OPCODE_WORDS):
            raise ValueError(f'알 수 없는 SB 명령어 {hex(opcode)}: {hex(pos)}')
        words = OPCODE_WORDS[opcode]
        if opcode == 0x25:
            if pos + 6 > code_end:
                raise ValueError(f'잘린 switch 명령어: {hex(pos)}')
            words = 4 * struct.unpack_from('<I', sb_data, pos + 2)[0] + 5
        size = words * 2
        if not size or pos + size > code_end:
            raise ValueError(f'명령어가 코드 영역을 벗어남: {hex(pos)}')
        if opcode == 0x06 and struct.unpack_from('<I', sb_data, pos + 2)[0] == old_rel:
            sb_data[pos + 2:pos + 6] = new_bytes
            count += 1
        pos += size
    return count


def patch_secmeta_u32_refs(sb_data, secmeta_start, sp_start, old_rel, new_rel, verified_sites=()):
    """검증된 데이터 참조만 패치. 주소와 같은 일반 숫자는 자동 판별하지 않는다."""
    old_bytes = struct.pack('<I', old_rel)
    new_bytes = struct.pack('<I', new_rel)
    count = 0
    pos = secmeta_start
    while pos <= sp_start - 4:
        found = sb_data.find(old_bytes, pos, sp_start)
        if found == -1:
            break

        if (found - secmeta_start) % 4 == 0:
            if found not in verified_sites:
                raise ValueError(f'검증되지 않은 데이터 값 {hex(found)} = {hex(old_rel)}: '
                                 '문자열 참조인지 숫자인지 확정할 수 없어 EXT 저장을 취소합니다')
            sb_data[found:found + 4] = new_bytes
            count += 1
            pos = found + 4
        else:
            pos = found + 1
    return count


def patch_string_refs(sb_data, secmeta_start, sp_start, code_start, code_end, old_rel, new_rel,
                      verified_sites=()):
    code_count = patch_code_type6_refs(sb_data, code_start, code_end, old_rel, new_rel)
    secmeta_count = patch_secmeta_u32_refs(sb_data, secmeta_start, sp_start, old_rel, new_rel, verified_sites)
    return code_count, secmeta_count


def build(data, content, table):
    """Build in memory using the reviewed VM boundaries and data references."""
    if not table:
        raise ValueError('치환표가 비어 있습니다')
    hdr    = parse_header(data)
    pool   = read_pool(data, hdr['sp_start'], hdr['sp_end'])
    originals = None
    if is_bilingual(content):
        originals, translations = parse_bilingual(content)
        source_text = {off: raw_to_text(raw).replace(chr(10), '[n]') for off, raw in pool}
        for off, text in originals.items():
            if off not in source_text or text != source_text[off]:
                raise ValueError(f'{hex(off)}: JPN이 원본 SB와 다릅니다')
    else:
        translations = parse_txt(content)

    pool_offsets = {abs_off for abs_off, _raw in pool}
    translatable = [(abs_off, raw) for abs_off, raw in pool if raw and is_jp(raw)]
    translatable_offsets = {abs_off for abs_off, _raw in translatable}
    listed = set(originals if originals is not None else translations)
    missing = sorted(translatable_offsets - listed)
    extra = sorted(listed - pool_offsets)

    if missing:
        raise ValueError('번역 누락 offset: ' + ', '.join(map(hex, missing)))
    if extra:
        raise ValueError('원본 string pool에 없는 offset: ' + ', '.join(map(hex, extra)))

    encoded = {}
    encoding_errors = []
    for abs_off, text in translations.items():
        converted = convert(text.replace('[n]', '\n'), table)
        try:
            encoded[abs_off] = text_to_raw(converted)
        except (UnicodeEncodeError, ValueError) as exc:
            encoding_errors.append((abs_off, str(exc)))

    if encoding_errors:
        raise ValueError('인코딩/바이트 표기 오류: ' + '; '.join(
            f'{hex(off)}: {message}' for off, message in encoding_errors))

    secmeta    = hdr['secmeta_start']
    sp_start   = hdr['sp_start']
    code_start = hdr['code_start']
    code_end   = secmeta  # 바이트코드 영역
    verified_sites = {pos for start, end in VERIFIED_DATA_RANGES.get(hashlib.sha256(data).hexdigest(), ())
                      for pos in range(start, end, 4)}

    sb_data = bytearray(data)
    failed_ext = []
    changed, extended = 0, 0

    for abs_off, raw in pool:
        if abs_off not in translations:
            continue

        new_bytes = encoded[abs_off]
        if new_bytes == raw:
            continue
        changed += 1

        orig_len = len(raw)
        old_rel  = abs_off - secmeta

        if len(new_bytes) <= orig_len:
            # 인플레이스: 덮어쓰고 남는 자리 0x00
            sb_data[abs_off : abs_off + len(new_bytes)] = new_bytes
            for j in range(abs_off + len(new_bytes), abs_off + orig_len):
                sb_data[j] = 0x00

        else:
            # 초과: 파일 끝에 추가 + 바이트코드 내 오프셋 패치
            new_abs = len(sb_data)
            new_rel = new_abs - secmeta
            sb_data.extend(new_bytes + b'\x00')

            code_n, secmeta_n = patch_string_refs(
                sb_data, secmeta, sp_start, code_start, code_end, old_rel, new_rel, verified_sites
            )
            n = code_n + secmeta_n
            extended += 1
            if n == 0:
                failed_ext.append(abs_off)

    if failed_ext:
        raise ValueError('참조 없는 EXT: ' + ', '.join(map(hex, failed_ext)))
    return bytes(sb_data), {'changed': changed, 'extended': extended}


def write_result(out_path, data):
    """Atomic replacement of one generated result; never an input file."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=out_path.parent,
                                         prefix=out_path.name + '.', suffix='.tmp',
                                         delete=False) as f:
            temp_path = f.name
            f.write(data)
        os.replace(temp_path, out_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    print(f'[SAVED] {out_path} ({len(data):,} bytes)')


# ── 진입점 ──────────────────────────────────────────────────

def load_catalog():
    return json.loads((ROOT / 'sb_sources.json').read_text(encoding='utf-8'))


def asset_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('프로젝트 밖 경로는 사용할 수 없습니다')
    return path


def load_source(path, catalog):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('등록된 프로젝트 원본 SB만 사용할 수 있습니다')
    name = path.relative_to(ROOT.resolve()).as_posix()
    if name not in catalog['sources']:
        raise ValueError(f'등록되지 않은 원본: {path}')
    record = catalog['sources'][name]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError(f'원본 SB가 변경되었습니다. 번역/EXT 결과를 원본으로 쓰지 마세요: {path}')
    return path, data, record


def prepare_import(sb_path, txt_path, catalog):
    path, data, record = load_source(sb_path, catalog)
    table_bytes = (path.parent / JSON_FILE).read_bytes()
    if hashlib.sha256(table_bytes).hexdigest() != catalog['table_sha256']:
        raise ValueError(f'치환표가 변경되었습니다. 재검증이 필요합니다: {path.parent / JSON_FILE}')
    table = json.loads(table_bytes.decode('utf-8-sig'))['replace-table']
    built, stats = build(data, Path(txt_path).read_text(encoding='utf-8-sig'), table)
    targets = [path]
    for relative in record['copies']:
        copy = asset_path(relative)
        if copy.read_bytes() != data:
            raise ValueError(f'대응 BIN이 기준 SB와 다릅니다. 원본/번역을 확인하세요: {copy}')
        targets.append(copy)
    outputs = [p.with_name(p.name + '.new') for p in targets]
    if any(p.is_symlink() or (p.exists() and not p.is_file()) for p in outputs):
        raise ValueError('출력 경로는 일반 .new 파일이어야 합니다')
    # Refresh old outputs even if a translation was reverted to the original.
    needed = built != data or any(p.exists() for p in outputs)
    return ({p: built for p in outputs} if needed else {}), stats


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) < 2:
            raise ValueError(__doc__)
        cmd, args = argv[0], argv[1:]
        catalog = load_catalog()
        if cmd == 'extract' and len(args) == 1:
            source, data, _record = load_source(args[0], catalog)
            target = source.with_suffix('.txt')
            with target.open('x', encoding='utf-8-sig', newline='\n') as stream:
                stream.write(export_text(data, bilingual=source.parent == ROOT / 'cf'))
            print(f'[SAVED] {target}')
            return 0
        if cmd == 'import' and len(args) == 2:
            pairs = [(Path(args[0]), Path(args[1]))]
        elif cmd in ('batch', 'verify') and len(args) == 1 and Path(args[0]).is_dir():
            directory = Path(args[0]).resolve()
            sources = sorted(directory.glob('*.sb'))
            if not sources:
                raise ValueError(f'SB 파일이 없습니다: {directory}')
            pairs = [(p, p.with_suffix('.txt')) for p in sources]
        elif cmd in ('rebuild', 'verify') and all(Path(p).suffix.lower() == '.txt' for p in args):
            pairs = [(Path(p).with_suffix('.sb'), Path(p)) for p in args]
        else:
            raise ValueError('명령/인자를 확인하세요: extract, import, rebuild, batch, verify')
        outputs, failures = {}, []
        changed = 0
        for source, text in pairs:
            try:
                prepared, stats = prepare_import(source, text, catalog)
                for path, data in prepared.items():
                    if path in outputs and outputs[path] != data:
                        raise ValueError(f'서로 다른 번역이 같은 출력에 지정되었습니다: {path}')
                    outputs[path] = data
                if stats['changed']:
                    changed += 1
                    print(f'[CHECK] {source.name}: {stats["changed"]} changed, {stats["extended"]} EXT')
            except (OSError, ValueError, KeyError, struct.error) as exc:
                failures.append(f'{text}: {exc}')
        if failures:
            raise ValueError('\n'.join(failures) + '\n전체 사전 검증 실패: 아무 결과도 저장하지 않았습니다.')
        print(f'[CHECK] {len(pairs)} SB checked; {changed} changed; {len(outputs)} SB/BIN outputs')
        if cmd == 'verify':
            print('[NO OUTPUT] 검증만 완료했습니다.')
            return 0
        for path, data in outputs.items():
            write_result(path, data)
        print('[OK] 모든 요청이 완료되었습니다. 원본과 번역 TXT는 수정하지 않았습니다.')
        return 0
    except (OSError, ValueError, KeyError, struct.error) as exc:
        print(f'[!] 작업 실패: {exc}')
        print('[실패] .new를 이번 성공 결과로 사용하지 마세요. 저장 중 오류라면 일부만 갱신되었을 수 있습니다.')
        return 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sys.exit(main())
