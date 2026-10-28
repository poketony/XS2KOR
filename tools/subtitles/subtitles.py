"""XS2 UTF-8 subtitle preparation and explicitly requested ROM delivery."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
ENCODING = 'euc_jis_2004'
ROM = Path('C:/Users/JO/Desktop/XENO_ARC/Xenosaga/Xenosaga2')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inside(root, path):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes workspace: {path}')
    return path


def hangul(char):
    code = ord(char)
    return (0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F or
            0xA960 <= code <= 0xA97F or 0xAC00 <= code <= 0xD7FF)


def editor_text(raw):
    text = raw.decode('utf-8-sig')
    if '\0' in text:
        raise ValueError('NUL in editor; save as UTF-8, not UTF-16')
    return text


def decode_game(raw):
    text = raw.decode(ENCODING)
    if '\0' in text or text.encode(ENCODING) != raw:
        raise ValueError('Original must roundtrip exactly without NUL')
    return text.encode('utf-8')


def encode_editor(raw, table):
    text = editor_text(raw)
    converted = ''.join(table.get(c, c) for c in text)
    for index, char in enumerate(converted):
        if hangul(char):
            line = converted.count('\n', 0, index)+1
            raise ValueError(f'Line {line}: Hangul missing from font table: {char!r}')
    try:
        result = converted.encode(ENCODING)
    except UnicodeEncodeError as exc:
        line = converted.count('\n', 0, exc.start)+1
        raise ValueError(f'Line {line}: cannot encode {converted[exc.start:exc.end]!r}') from exc
    if result.decode(ENCODING) != converted:
        raise ValueError('Encoded subtitle does not roundtrip exactly')
    return result


def structure(text):
    """Validate active timing records, preserving their separators and commands."""
    result, ended = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith(';'):
            continue
        if line == 'end' and not ended:
            ended = True
            continue
        match = re.fullmatch(r'(\+?(?:[0-9]+:){0,2}[0-9]+)(?:([\t ]+)(.*))?', line)
        if ended or match is None:
            raise ValueError(f'Invalid subtitle record at line {number}')
        result.append((match[1], match[2] or '', tuple(re.findall(r'\$[^;\r\n]*;', match[3] or ''))))
    if not ended:
        raise ValueError('Missing end record')
    return result


def publish(changes, snapshots, check=False):
    # Finish all validation, including race/pending checks, before any writes.
    for path, before in snapshots.items():
        current = path.read_bytes() if path.exists() else None
        if current != before:
            raise ValueError(f'Input/destination changed during validation: {path}')
    for path, _raw in changes:
        if path.with_name(path.name+'.pending').exists():
            raise FileExistsError(f'Inspect interrupted output: {path}.pending')
    if check:
        return
    for path, raw in changes:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name+'.pending')
        with pending.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if pending.read_bytes() != raw:
            raise OSError(f'Pending verification failed: {pending}')
        os.replace(pending, path)
        if path.read_bytes() != raw:
            raise OSError(f'Output verification failed: {path}')


def inputs(root, home, disc):
    snapshots = {}

    def read(path):
        snapshots[path] = path.read_bytes()
        return snapshots[path]

    manifest = json.loads(read(inside(root, home / 'manifest.json')))
    if manifest['encoding'] != ENCODING or disc not in (1, 2):
        raise ValueError('Unsupported subtitle profile')
    table_path = inside(root, root / manifest['table'])
    table_raw = read(table_path)
    if sha(table_raw) != manifest['table_sha256']:
        raise ValueError('Font table hash mismatch')
    table = json.loads(table_raw.decode('utf-8-sig'))['replace-table']
    if not table or not all(len(k) == 1 and hangul(k) and isinstance(v, str) and
                            len(v) == 1 and not hangul(v) for k,v in table.items()):
        raise ValueError('Expected single Hangul -> game glyph mappings')
    for value in table.values():
        if value.encode(ENCODING).decode(ENCODING) != value:
            raise ValueError('Non-roundtripping font mapping')
    profile = manifest['discs'][str(disc)]
    folder = inside(root, root / f'{disc}.xenosaga{disc}' / 'movie')
    names = set(profile['files'])
    for name in names:
        if Path(name).name != name or not re.fullmatch(r'(?:s[0-9]{6}|EP07|EPSTAFF)\.txt', name):
            raise ValueError('Unsafe subtitle filename')
    for kind in ('jp', 'kr'):
        extra = {p.name for p in (folder / kind).glob('*.txt')}-names
        if extra:
            raise ValueError(f'Unregistered Disc {disc} {kind} subtitles: {sorted(extra)}')
    originals = {}
    for name, digest in profile['files'].items():
        path = inside(root, home / 'original' / str(disc) / name)
        raw = read(path)
        if sha(raw) != digest:
            raise ValueError(f'Original hash mismatch: {path}')
        originals[name] = decode_game(raw)
        structure(originals[name].decode('utf-8'))
    return profile, folder, originals, table, snapshots, read


def prepare(root, home, disc, check=False):
    _profile, folder, originals, _table, snapshots, read = inputs(root, home, disc)
    changes, kept = [], 0
    for name, japanese in originals.items():
        for kind in ('jp', 'kr'):
            path = inside(root, folder / kind / name)
            before = read(path) if path.exists() else None
            snapshots[path] = before
            if before is not None:
                editor_text(before)
                if kind == 'jp' and before != japanese:
                    raise ValueError(f'Japanese reference differs from original: {path}')
                if kind == 'kr':
                    kept += 1
            else:
                changes.append((path, japanese))
    publish(changes, snapshots, check)
    return {'total': len(originals), 'created': len(changes), 'kr_preserved': kept}


def encode(root, home, disc, rom, check=False):
    if rom.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(rom.resolve()):
        raise ValueError('ROM and translation workspaces must be separate')
    profile, folder, originals, table, snapshots, read = inputs(root, home, disc)
    index = inside(rom, rom / f'LBA{disc}.txt')
    raw_index = read(index)
    if sha(raw_index) != profile['lba_sha256']:
        raise ValueError(f'Disc {disc} LBA hash mismatch')
    registered = {line.split('|')[-1].replace('\\', '/').lstrip('/')
                  for line in raw_index.decode('cp949').rstrip('\0\r\n').splitlines()}
    unpacked = inside(rom, rom / f'xenosaga{disc}_UNPACKED')
    changes, targets = [], []
    for name, japanese in originals.items():
        if 'movie/'+name not in registered:
            raise ValueError(f'Not registered on Disc {disc}: {name}')
        jp = inside(root, folder / 'jp' / name)
        if read(jp) != japanese:
            raise ValueError(f'Japanese reference changed: {jp}')
        editor = inside(root, folder / 'kr' / name)
        raw_editor = read(editor)
        try:
            if structure(editor_text(raw_editor)) != structure(japanese.decode('utf-8')):
                raise ValueError('Timing, separator or formatting commands changed')
            after = encode_editor(raw_editor, table)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f'{editor}: {exc}') from exc
        # XS2 has no jp/kr directory in the game archive.
        target = inside(unpacked, unpacked / 'movie' / name)
        if not target.is_file():
            raise FileNotFoundError(f'Expected registered ROM subtitle: {target}')
        targets.append(target)
        before = read(target)
        if before != after:
            changes.append((target, after))
    protected = [p for p in snapshots if p.is_relative_to(root)]
    for target in targets:
        if any(target.samefile(p) for p in protected):
            raise ValueError(f'ROM target aliases repository input: {target}')
    publish(changes, snapshots, check)
    return {'total': len(originals), 'changed': len(changes), 'unchanged': len(originals)-len(changes)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'encode'))
    parser.add_argument('--disc', required=True, type=int, choices=(1, 2))
    parser.add_argument('--rom-root', type=Path, default=ROM)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.action == 'prepare':
            stats = prepare(ROOT, HOME, args.disc, args.check)
        else:
            stats = encode(ROOT, HOME, args.disc, args.rom_root, args.check)
            print('Destination:', args.rom_root / f'xenosaga{args.disc}_UNPACKED/movie')
        print('[CHECK]' if args.check else '[OK]', stats)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print('[FAILED]', exc, file=sys.stderr)
        print('Do not build an ISO after failure. Inspect any .pending file; rerun after fixing the error.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
