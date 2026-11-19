"""Edit XS2 title/title2 RGBA images against hash-checked Japanese originals."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct

from PIL import Image

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 512, 448
RAW_SIZE = WIDTH * HEIGHT * 4
SOURCES = {
    'title': '1d1d748d37ed46aca0c7f51baae952e54411f0e01425f753456187b86cd8c2d3',
    'title2': '90655d97f7bc93e88717d855deb036e482fb556f0ec8659262ad881cf4814362',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decompress_arx(data):
    if len(data) < 136 or data[:4] != b'ARX\0' or len(data) % 4:
        raise ValueError('Invalid/truncated ARX header')
    size, packed = struct.unpack_from('<II', data, 4)
    if size % 4 or size > RAW_SIZE or packed not in (len(data), len(data) - 136):
        raise ValueError('Invalid ARX size fields')
    table = struct.unpack_from('<30I', data, 16)
    position, control, remaining = 136, 0, 0

    def word():
        nonlocal position
        if position + 4 > len(data):
            raise ValueError('Truncated ARX control/literal')
        value = struct.unpack_from('<I', data, position)[0]
        position += 4
        return value

    def bits(count):
        nonlocal control, remaining
        value = 0
        for _ in range(count):
            if not remaining:
                control, remaining = word(), 32
            remaining -= 1
            value = (value << 1) | ((control >> remaining) & 1)
        return value

    output = bytearray()
    for _ in range(size // 4):
        if not bits(1):
            value = word()
        else:
            if not bits(1):
                index = bits(1)
            elif not bits(1):
                index = 2 + bits(2)
            elif not bits(1):
                index = 6 + bits(3)
            elif not bits(1):
                index = 14 + bits(4)
            else:
                raise ValueError('Invalid ARX LUT code')
            if index >= len(table):
                raise ValueError('ARX LUT index outside its table')
            value = table[index]
        output.extend(struct.pack('<I', value))
    if position != len(data) or control & ((1 << remaining) - 1):
        raise ValueError('Unexpected ARX trailing data')
    return bytes(output)


def compress_arx(raw, original):
    if len(raw) % 4 or len(raw) > RAW_SIZE:
        raise ValueError('Invalid raw ARX size')
    words = [value for value, in struct.iter_unpack('<I', raw)]
    table = [value for value, _ in Counter(words).most_common(30)]
    table += [0] * (30 - len(table))
    lookup = {}
    for i, value in enumerate(table):
        lookup.setdefault(value, i)
    chunks, literals = [], []
    control, used = 0, 0

    def flush():
        nonlocal control, used
        chunks.append(struct.pack('<I', control << (32 - used)))
        chunks.extend(struct.pack('<I', value) for value in literals)
        literals.clear()
        control, used = 0, 0

    for value in words:
        index = lookup.get(value)
        if index is None:
            code, count = 0, 1
            # A literal belongs to the control word containing its zero bit.
            literals.append(value)
        elif index < 2:
            code, count = 0b100 | index, 3
        elif index < 6:
            code, count = 0b11000 | (index - 2), 5
        elif index < 14:
            code, count = 0b1110000 | (index - 6), 7
        else:
            code, count = 0b111100000 | (index - 14), 9
        for shift in range(count - 1, -1, -1):
            control = (control << 1) | ((code >> shift) & 1)
            used += 1
            if used == 32:
                flush()
    if used:
        flush()
    body = b''.join(chunks)
    # Preserve whether +8 counts the whole file or only the compressed body.
    packed = len(body) + (136 if struct.unpack_from('<I', original, 8)[0] == len(original) else 0)
    header = b'ARX\0' + struct.pack('<II', len(raw), packed) + original[12:16]
    return header + struct.pack('<30I', *table) + body


def png_bytes(raw):
    if len(raw) != RAW_SIZE or max(raw[3::4]) > 128:
        raise ValueError('Expected 512x448 PS2 RGBA pixels')
    pixels = bytearray(raw)
    pixels[3::4] = bytes(min(a * 2, 255) for a in raw[3::4])
    output = io.BytesIO()
    Image.frombytes('RGBA', (WIDTH, HEIGHT), bytes(pixels)).save(output, format='PNG')
    return output.getvalue()


def raw_pixels(png):
    with Image.open(io.BytesIO(png)) as image:
        if image.format != 'PNG' or png[24] != 8 or image.mode != 'RGBA' or image.size != (WIDTH, HEIGHT):
            raise ValueError('Use a 512x448, 8-bit RGBA PNG; no automatic resizing/conversion')
        pixels = bytearray(image.tobytes())
    if min(pixels[3::4]) == 255:
        raise ValueError('PNG lost its transparent background; do not use an opaque preview')
    pixels[3::4] = bytes((a + 1) // 2 for a in pixels[3::4])
    return bytes(pixels)


def build(original, png):
    baseline = decompress_arx(original)
    if len(baseline) != RAW_SIZE:
        raise ValueError('Unexpected title dimensions')
    raw = raw_pixels(png)
    result = original if raw == baseline else compress_arx(raw, original)
    if decompress_arx(result) != raw:
        raise ValueError('ARX verification failed; no output saved')
    return result


def checked(path):
    if not path.is_relative_to(ROOT):
        raise ValueError('Path outside this tool folder')
    for node in (path, *path.parents):
        if node.exists() or node.is_symlink():
            if getattr(node.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or node.is_symlink():
                raise ValueError(f'Linked path is not allowed: {node}')
        if node == ROOT:
            break
    return path


def sources():
    result = {}
    for name, digest in SOURCES.items():
        path = checked(ROOT / (name + '.bin.original'))
        data = path.read_bytes()
        if sha(data) != digest:
            raise ValueError(f'{path.name}: Japanese original hash mismatch')
        if len(decompress_arx(data)) != RAW_SIZE:
            raise ValueError(f'{path.name}: unexpected raw size')
        result[name] = data
    return result


def extract(originals):
    folder = checked(ROOT / 'title_graphics')
    images = [(checked(folder / (name + '.png')), png_bytes(raw))
              for name, data in originals.items() for raw in (decompress_arx(data),)]
    folder.mkdir(exist_ok=True)
    for path, raw in images:
        if path.exists():
            print(f'[KEEP] {path.name}')
        else:
            with path.open('xb') as stream:
                stream.write(raw)
            print(f'[PNG] {path}')
    print('[OK] Japanese editing PNGs; existing PNG/KOR/PSD files preserved')


def prepare(originals):
    folder = checked(ROOT / 'title_graphics')
    allowed = {name + suffix + '.png' for name in SOURCES for suffix in ('', '_KOR')}
    for path in folder.iterdir():
        checked(path)
        if path.is_file() and path.suffix.lower() == '.png' and path.name not in allowed:
            raise ValueError(f'Unknown PNG: {path.name}')
    outputs, inputs = {}, {}
    for name, data in originals.items():
        path = checked(folder / (name + '_KOR.png'))
        if not path.exists():
            path = checked(folder / (name + '.png'))
        png = path.read_bytes()
        outputs[name + '.bin'] = build(data, png)
        inputs[path.name] = sha(png)
    return outputs, inputs


def rebuild(originals, outputs, inputs):
    folder = checked(ROOT / 'title_rebuilt')
    report_path = checked(folder / 'build.json')
    previous = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    owned = previous.get('outputs', {})
    names = [name for name, data in outputs.items()
             if data != originals[Path(name).stem] or checked(folder / name).exists()]
    if not names:
        print('[OK] No changed pixels; no BINs generated')
        return
    pending = []
    for name in names:
        path = checked(folder / name)
        data = outputs[name]
        if path.exists() and path.read_bytes() != data and sha(path.read_bytes()) != owned.get(name):
            raise ValueError(f'{name}: result was edited outside this tool; refusing to overwrite')
        pending.append((path, data))
    report = dict(format=1, originals=SOURCES, inputs=inputs,
                  outputs={name:sha(outputs[name]) for name in names}, verified_raw_bytes=RAW_SIZE)
    pending.append((report_path, (json.dumps(report, indent=2) + '\n').encode('utf-8')))
    for path, _ in pending:
        stage = checked(path.with_name(path.name + '.pending'))
        if stage.exists():
            raise ValueError(f'Incomplete earlier write: {stage.name}')
    folder.mkdir(exist_ok=True)
    for path, data in pending:
        stage = path.with_name(path.name + '.pending')
        with stage.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if stage.read_bytes() != data:
            raise OSError(f'Staged output verification failed: {stage}')
    for path, _ in pending:
        path.with_name(path.name + '.pending').replace(path)
        print(f'[SAVED] {path}')
    print('[OK] Verified ARX/RGBA; manually transfer BINs to yajima in the ROM workspace')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('extract', 'verify', 'rebuild'))
    args = parser.parse_args()
    try:
        original = sources()
        if args.command == 'extract':
            extract(original)
        else:
            outputs, inputs = prepare(original)
            if args.command == 'verify':
                print('[OK] Both titles verified; no output written')
            else:
                rebuild(original, outputs, inputs)
    except (OSError, ValueError, struct.error, IndexError) as error:
        print(f'[ERROR] {error}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
