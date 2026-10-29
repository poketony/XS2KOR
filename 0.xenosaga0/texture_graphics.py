"""Selective menu XTX / map-sign TXY editing; XS2KOR-local originals only."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'menu': {
        'ctrl01.xtx': '69b3bd4b50d1afc8ad03ceff5ff8fdd0b9a16ffa8cacc5f77e457d64cc388e43',
        'font.xtx': 'ce83cc04e00c81de4182d45c6465b3a5d9c26f640a11e591babf18a1bcb75d00',
        'obj01.xtx': '15f7792e6510fb3f40f19af5d6bf27d7df160ba3a9d2ef60714496bc79050209',
    },
    'map': {
        'CFEVE15.txy': '9924e36a447a1cd84dfa9eaf3a18b81ea7f21d35e73b2bbe8132c3eaa79f09af',
        'E2_UZU01.txy': '7538608a3857308b7fd64dc6bf92919e1655680b527d3be6b9f0000b1b5fbfdf',
        'E2_UZU06.txy': '32358de49b3ad074f5d094e25bc1bfbb1ce80bf67563500b2f2c1029b9571c11',
        'E2_KUK11.txy': 'b3ed03a487c6534cabe3ea3e9fbc1fc633549765f065b44196f3ebe5a6414c33',
    },
}

# Coordinates are in the original logical atlas, not the CT32 upload image.
MENU_RECTS = {
    'ctrl01.xtx': [
        ('ctrl_l2_r2', 44, 23, 76, 17),
        ('ctrl_l1_r1', 44, 65, 76, 17),
        ('ctrl_dpad', 49, 111, 47, 16),
        ('ctrl_select', 42, 157, 74, 17),
        ('ctrl_left_stick', 47, 203, 61, 16),
        ('ctrl_right_stick', 401, 203, 61, 16),
        ('ctrl_start', 222, 223, 69, 17),
        *[('ctrl_button', 426, y, 33, 18) for y in (22, 62, 102, 142)],
        ('ctrl_menu_category', 21, 331, 121, 17),
        ('ctrl_menu_open_close', 383, 331, 103, 16),
        ('ctrl_character_switch', 30, 375, 102, 16),
        ('ctrl_confirm', 419, 371, 25, 16),
        ('ctrl_cursor_1', 39, 421, 66, 16),
        ('ctrl_cancel', 394, 411, 82, 16),
        ('ctrl_skill_reset', 369, 452, 123, 30),
        ('ctrl_unused', 57, 467, 36, 16),
        ('ctrl_cursor_2', 44, 513, 66, 16),
        ('ctrl_navigator_angle', 374, 512, 110, 30),
        ('ctrl_leader_switch', 215, 532, 98, 30),
        ('ctrl_status', 68, 868, 55, 16),
        ('ctrl_skill', 77, 928, 37, 16),
        ('ctrl_party', 6, 948, 54, 16),
    ],
    'font.xtx': [('font_character_select', 8, 384, 96, 18)],
    'obj01.xtx': [
        ('obj_item', 8, 288, 50, 18),
        ('obj_special_item', 74, 288, 75, 18),
        ('obj_normal_attack', 339, 288, 55, 18),
        ('obj_special_attack', 410, 288, 42, 18),
        ('obj_accessory', 8, 324, 97, 18),
        ('obj_ether', 8, 360, 52, 18),
        ('obj_skill', 76, 360, 41, 18),
        ('obj_solo', 132, 360, 42, 18),
        ('obj_duo', 190, 360, 42, 18),
        ('obj_class', 248, 360, 48, 18),
        ('obj_battle_formation', 312, 360, 63, 18),
        ('obj_learned_skill', 391, 358, 66, 18),
        ('obj_tutorial', 303, 394, 81, 18),
        ('obj_equipped_skill', 8, 428, 66, 18),
        ('obj_ether_skill', 90, 428, 88, 18),
        ('obj_leader_change', 194, 428, 94, 18),
    ],
}
MAP_MATERIALS = {
    'CFEVE15.txy': ('uzu06_kanban', 86),
    'E2_UZU01.txy': ('uzu06_kanban', 56),
    'E2_UZU06.txy': ('uzu06_kanban', 81),
    'E2_KUK11.txy': ('kanban01_new', 184),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def codec():
    path = ROOT / 'xtx \uac1c\ubc1c\uc18c' / 'xtx_psmt4_codec.py'
    spec = importlib.util.spec_from_file_location('xs2_graphics_codec', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ct32(x, y, width):
    c = codec()
    block = np.array(c.BLOCK32)[((x & 63) >> 3) + ((y & 31) >> 3)*8]
    word = np.array(c.COLUMN_WORD32)[(x & 7) + (y & 1)*8]
    return ((y >> 5)*(width >> 6) + (x >> 6))*2048 + block*64 + ((y & 7) >> 1)*16 + word


def upload8(x, y, width):
    return ((y & ~15)*width + (x & ~15)*2
            + ((((y & ~3) >> 1)+(y & 1)) & 7)*width*2
            + ((x+(((y+2) >> 2 & 1)*4)) & 7)*4
            + ((y >> 1) & 1) + ((x >> 2) & 2))


def psmt4(x, y, width):
    c = codec()
    block = np.array(c.BLOCK4)[((x & 127) >> 5) + ((y & 127) >> 4)*4]
    col, ix = (y & 15) >> 2, (x & 31) + (y & 3)*32
    word = np.array(c.COLUMN_WORD4)[col & 1, ix]
    nib = np.array(c.COLUMN_BYTE4)[ix]
    return (((y >> 7)*(width//128) + (x >> 7))*2048 + block*64 + col*16 + word)*8 + nib


@dataclass
class Region:
    key: str
    source: str
    offsets: np.ndarray
    shifts: np.ndarray
    mask: int
    indices: np.ndarray
    palette: np.ndarray | None = None

    def pixels(self):
        return self.indices*17 if self.palette is None else self.palette[self.indices]


def read_region(data, key, source, offsets, shifts, mask, palette=None):
    if offsets.min() < 0 or offsets.max() >= len(data):
        raise ValueError(f'{key}: pixel points outside loaded source')
    # PSMT4 siblings share a byte but never the same nibble.
    if len(np.unique(offsets*8+shifts)) != offsets.size:
        raise ValueError(f'{key}: aliased pixel addresses')
    indices = ((np.frombuffer(data, np.uint8)[offsets] >> shifts) & mask).astype(np.uint8)
    return Region(key, source, offsets, shifts, mask, indices, palette)


def menu_regions(name, data):
    inverse = None
    if name == 'ctrl01.xtx':
        yy, xx = np.indices((256, 256))
        positions = ct32(xx, yy, 256)[..., None]*4 + np.arange(4)
        inverse = np.full(262144, -1, np.int64)
        inverse[positions] = np.arange(80, len(data)).reshape(256, 256, 4)
    for key, x, y, w, h in MENU_RECTS[name]:
        yy, xx = np.indices((h, w))
        xx, yy = xx+x, yy+y
        if inverse is not None:
            nib = psmt4(xx, yy, 512)
            offsets, shifts, mask = inverse[nib//2], (nib & 1)*4, 15
        else:
            offsets, shifts, mask = 80+upload8(xx, yy, 512), np.zeros_like(xx), 255
        region = read_region(data, key, name, offsets, shifts, mask)
        if region.indices.max() > 15:
            raise ValueError(f'{key}: region contains non-text indices')
        yield region


def map_region(name, data):
    key, material = MAP_MATERIALS[name]
    if data[:4] != b'txy\0' or struct.unpack_from('<I', data, 8)[0] != len(data):
        raise ValueError(f'{name}: TXY header mismatch')
    off = struct.unpack_from('<I', data, 16)[0]
    header = struct.unpack_from('<8I', data, off)
    if header[:5] != (64, 16, 512, 128, 0) or not 0 < header[5] <= 8 or material >= header[6]:
        raise ValueError(f'{name}: unexpected bank header')
    inverse = np.full(512*1024, -1, np.int64)
    for j in range(header[5]):
        addr, base, w, h, pad, qwc, blocks, flags = struct.unpack_from('<8I', data, off+32+j*32)
        if pad or not 0 < w <= 512 or h not in (8, 16, 24, 32) or qwc != w*h//4+2 or addr+32+w*h*4 > len(data):
            raise ValueError(f'{name}: unexpected upload')
        yy, xx = np.indices((h, w))
        positions = (ct32(xx, yy, 512)*4+base*256)[..., None] + np.arange(4)
        if positions.max() >= len(inverse) or np.any(inverse[positions] >= 0):
            raise ValueError(f'{name}: overlapping/out-of-range GS upload')
        inverse[positions] = np.arange(addr+32, addr+32+w*h*4).reshape(h, w, 4)
    pos = off+288+material*96
    v = struct.unpack_from('<16I', data, pos)
    actual_name = data[pos+64:pos+96].split(b'\0', 1)[0].decode('ascii')
    tbp, psm, w, h, x, y, cbp, flag, px, py = v[:10]
    if actual_name != key or psm != 19 or v[12] != 0 or not (0 < w <= 1024 and 0 < h <= 1024 and x+w <= 1024 and y+h <= 1024):
        raise ValueError(f'{name}: material mismatch')
    yy, xx = np.indices((h, w))
    src = upload8(xx+x, yy+y, 1024)
    addresses = ct32((src//4) % 512, src//2048, 512)*4 + src % 4
    cy, cx = np.indices((16, 16))
    palpos = ct32(cx+px, cy+py, 512)[..., None]*4 + np.arange(4)
    if addresses[0, 0] != tbp*256 or palpos[0, 0, 0] != cbp*256 or np.any(inverse[palpos] < 0):
        raise ValueError(f'{name}: material/CLUT address mismatch')
    palette = np.frombuffer(data, np.uint8)[inverse[palpos]].reshape(256, 4).copy()
    order = np.arange(256)
    order = (order & ~24) | ((order & 8) << 1) | ((order & 16) >> 1)
    palette = palette[order]
    alpha = palette[:, 3].astype(np.uint16)
    if alpha.max() <= 128 and alpha.any():
        palette[:, 3] = alpha*255//128
    if np.intersect1d(inverse[addresses], inverse[palpos]).size:
        raise ValueError(f'{name}: texture overlaps CLUT')
    return read_region(data, key, name, inverse[addresses], np.zeros_like(addresses), 255, palette)


def folder(group):
    return ROOT / '0.xenosaga0' / ('yosikawa' if group == 'menu' else 'model/map')


def load(group):
    originals, regions = {}, []
    for name, digest in SOURCES[group].items():
        data = (folder(group) / name).read_bytes()
        if sha(data) != digest:
            raise ValueError(f'{name}: original SHA-256 mismatch; nothing written')
        originals[name] = data
        regions.extend(menu_regions(name, data) if group == 'menu' else [map_region(name, data)])
    examples = {}
    for r in regions:
        pixels = r.pixels()
        if r.key in examples and not np.array_equal(pixels, examples[r.key]):
            raise ValueError(f'{r.key}: shared copies no longer agree')
        examples[r.key] = pixels
    return originals, regions, examples


def png_bytes(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return stream.getvalue()


def decode_edit(raw, example, key):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.size != (example.shape[1], example.shape[0]) or getattr(image, 'n_frames', 1) != 1:
            raise ValueError(f'{key}: expected a single PNG of {example.shape[1]}x{example.shape[0]} pixels')
        if image.mode not in ('1', 'L', 'LA', 'P', 'RGB', 'RGBA'):
            raise ValueError(f'{key}: save as an 8-bit grayscale/RGB/RGBA PNG')
        if example.ndim == 3 and np.any(example[:, :, 3] != 255):
            if image.mode not in ('LA', 'RGBA') and 'transparency' not in image.info:
                raise ValueError(f'{key}: original has transparency; retain alpha when exporting PNG')
        rgba = np.array(image.convert('RGBA'))
    if example.ndim == 2:
        if not np.all(rgba[:, :, 3] == 255) or not np.all(rgba[:, :, :3] == rgba[:, :, :1]):
            raise ValueError(f'{key}: menu masks must be opaque grayscale (black background, white letters)')
        return rgba[:, :, 0]
    return rgba


def quantize(r, pixels):
    if r.palette is None:
        return ((pixels.astype(np.uint16)+8)//17).astype(np.uint8)
    out = r.indices.copy()
    changed = np.any(pixels != r.pixels(), axis=2)
    if not changed.any():
        return out
    colors, lookup = np.unique(pixels[changed], axis=0, return_inverse=True)
    nearest = np.empty(len(colors), np.uint8)
    # Exact unchanged pixels keep their original indices, including CLUT duplicates.
    for start in range(0, len(colors), 512):
        delta = colors[start:start+512, None, :].astype(np.int32)-r.palette[None, :, :].astype(np.int32)
        nearest[start:start+512] = np.argmin((delta*delta).sum(axis=2), axis=1)
    out[changed] = nearest[lookup]
    return out


def build(originals, regions, edits):
    output = {name: np.frombuffer(raw, np.uint8).copy() for name, raw in originals.items()}
    masks = {name: np.zeros(len(raw), np.uint8) for name, raw in originals.items()}
    for r in regions:
        if r.key not in edits:
            continue
        indices = quantize(r, edits[r.key])
        target, allowed = output[r.source], masks[r.source]
        # Split low/high nibble writes so advanced-index assignment cannot lose a sibling.
        for shift in np.unique(r.shifts):
            select = r.shifts == shift
            offsets = r.offsets[select]
            mask = r.mask << int(shift)
            if np.any(allowed[offsets] & mask):
                raise ValueError(f'{r.key}: edit regions overlap')
            allowed[offsets] |= mask
            target[offsets] = (target[offsets] & (255 ^ mask)) | (indices[select] << shift).astype(np.uint8)
        actual = ((target[r.offsets] >> r.shifts) & r.mask).astype(np.uint8)
        if not np.array_equal(actual, indices):
            raise ValueError(f'{r.key}: encoded pixel verification failed')
    for name, target in output.items():
        delta = target ^ np.frombuffer(originals[name], np.uint8)
        if np.any(delta & ~masks[name]):
            raise ValueError(f'{name}: write outside selected pixels')
    return {name: pixels.tobytes() for name, pixels in output.items()}


def extract(group, examples):
    dest = folder(group) / 'graphics'
    dest.mkdir(exist_ok=True)
    for key, pixels in examples.items():
        path = dest / f'{key}.png'
        if path.exists():
            print(f'[KEEP] {path.name}')
        else:
            with path.open('xb') as stream:
                stream.write(png_bytes(pixels))
            print(f'[PNG] {path.name}')
    print(f'[PASS] {len(examples)} editing PNGs; existing PNG / _KOR / PSD files preserved')


def rebuild(group, originals, regions, examples):
    edit_dir = folder(group) / 'graphics'
    expected = {f'{key}{suffix}.png' for key in examples for suffix in ('', '_KOR')}
    unknown = [p.name for p in edit_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png' and p.name not in expected]
    if unknown:
        raise ValueError('Unknown PNG filenames (not silently ignored): '+', '.join(sorted(unknown)))
    edits, inputs = {}, {}
    for key, example in examples.items():
        path = edit_dir / f'{key}_KOR.png'
        if not path.exists():
            path = edit_dir / f'{key}.png'
        if not path.exists():
            raise ValueError(f'Missing {path.name}; run ExtractGraphics first')
        raw = path.read_bytes()
        pixels = decode_edit(raw, example, key)
        inputs[path.name] = sha(raw)
        if not np.array_equal(pixels, example):
            edits[key] = pixels
    results = build(originals, regions, edits)
    dest = folder(group) / 'graphics_rebuilt'
    report_path = dest / 'build.json'
    previous = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    owned = previous.get('outputs', {})
    changed = {name for name in results if results[name] != originals[name]}
    # Restore a previously generated file to baseline if its translation is removed.
    names = sorted(changed | {name for name in results if (dest / name).exists()})
    pending = []
    for name in names:
        path = dest / name
        raw = results[name]
        if path.exists() and path.read_bytes() != raw and sha(path.read_bytes()) != owned.get(name):
            raise ValueError(f'{path}: not the last generated result; refusing to overwrite')
        if (dest / (name+'.pending')).exists():
            raise ValueError(f'{name}.pending exists from an interrupted build; inspect it first')
        pending.append((path, raw))
    if report_path.with_suffix('.pending').exists():
        raise ValueError('build.pending exists from an interrupted build; inspect it first')
    if not names:
        print('[PASS] No changed pixels; no game files generated')
        return
    dest.mkdir(exist_ok=True)
    report = {'format': 1, 'group': group, 'originals': SOURCES[group], 'inputs': inputs,
              'edited_images': sorted(edits), 'outputs': {name: sha(results[name]) for name in names}}
    # Preflight/encode every input before staging any output; originals are never writable targets.
    for path, raw in pending:
        with path.with_suffix(path.suffix+'.pending').open('xb') as stream:
            stream.write(raw)
        if sha(path.with_suffix(path.suffix+'.pending').read_bytes()) != sha(raw):
            raise ValueError(f'{path.name}: staged output verification failed')
    report_path.with_suffix('.pending').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for path, raw in pending:
        path.with_suffix(path.suffix+'.pending').replace(path)
        print(f'[WRITE] {path.name}: {len(raw):,} bytes'+(' (baseline restored)' if path.name not in changed else ''))
    report_path.with_suffix('.pending').replace(report_path)
    print(f'[PASS] {len(edits)} edited PNGs; {len(changed)} changed game files; transfer outputs manually')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=('menu', 'map'))
    parser.add_argument('action', choices=('extract', 'rebuild', 'check'))
    args = parser.parse_args()
    try:
        originals, regions, examples = load(args.group)
        if args.action == 'extract':
            extract(args.group, examples)
        elif args.action == 'rebuild':
            rebuild(args.group, originals, regions, examples)
        else:
            if build(originals, regions, examples) != originals:
                raise ValueError('Unedited round-trip mismatch')
            print(f'[PASS] {len(originals)} source hashes and {len(examples)} PNG round-trips')
    except (ValueError, OSError, KeyError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        print('[FAILED] Do not transfer outputs from this failed run.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
