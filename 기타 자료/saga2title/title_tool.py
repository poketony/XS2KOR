#!/usr/bin/env python3
"""
title_tool.py  –  Xenosaga Episode I  title.bin  이미지 추출 / 리빌드 툴

포맷:
  ARX 압축 컨테이너 → 압축 해제 → 512×448 RGBA (PS2 half-alpha)
  픽셀 배열: little-endian u32, 순서 [R][G][B][A]
  PS2 alpha: 0x00=완전투명, 0x80=완전불투명 (×2 → 0~255)

Usage:
  python title_tool.py extract <title.bin>  [--out <output.png>]
  python title_tool.py rebuild <title.bin>  <new_image.png>  [--out <title_new.bin>]
"""

import sys
import os
import struct
from collections import Counter

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("필요 패키지: pip install pillow numpy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# ARX 압축 해제 / 재압축
# ─────────────────────────────────────────────────────────────────────────────

def decompress_arx(data: bytes) -> bytes:
    if data[:4] != b'ARX\x00':
        raise ValueError(f"ARX magic 불일치: {data[:4]!r}")
    size_orig = struct.unpack_from('<I', data, 4)[0]
    lut = list(struct.unpack_from('<30I', data, 16))
    out = bytearray(size_orig); out_pos=0; fp_pos=136
    buf=0; buf_len=0; top=1<<63; s=0; lut_val=lut_idx=lut_len=0
    def read_u32():
        nonlocal fp_pos
        if fp_pos+4>len(data): return None
        v=struct.unpack_from('<I',data,fp_pos)[0]; fp_pos+=4; return v
    while True:
        val=read_u32()
        if val is None: break
        buf|=(val<<(32-buf_len))&0xFFFFFFFFFFFFFFFF; buf_len+=32
        while buf_len:
            bit=(buf&top)>>63
            if s==0:
                if bit: s=1
                else:
                    v=read_u32()
                    if v is None: buf_len=0; break
                    if out_pos+4<=len(out): struct.pack_into('<I',out,out_pos,v)
                    out_pos+=4; buf=(buf<<1)&0xFFFFFFFFFFFFFFFF; buf_len-=1
            elif s==1:
                lut_val=lut_idx=lut_len=0; s=2
                buf=(buf<<1)&0xFFFFFFFFFFFFFFFF; buf_len-=1
            elif s==2:
                lut_val=((lut_val<<1)|bit)&0xFF
                if lut_idx==0: lut_len=4 if bit else 2
                if lut_idx==1 and lut_len==4 and bit: lut_len=6
                if lut_idx==2 and lut_len==6 and bit: lut_len=8
                lut_idx+=1
                if lut_idx==lut_len:
                    s=0
                    if lut_len==2: idx=lut_val
                    elif lut_len==4: idx=2+(lut_val&0x7)
                    elif lut_len==6: idx=6+(lut_val&0xF)
                    else: idx=14+(lut_val&0x1F)
                    if out_pos+4<=len(out): struct.pack_into('<I',out,out_pos,lut[idx] if idx<len(lut) else 0)
                    out_pos+=4
                buf=(buf<<1)&0xFFFFFFFFFFFFFFFF; buf_len-=1
    return bytes(out)


def _lut_idx_to_bits(idx: int) -> list:
    if idx <= 1:   return [1, 0, idx]
    elif idx <= 5: val=idx-2;  return [1,1,0,(val>>1)&1,val&1]
    elif idx <= 13:val=idx-6;  return [1,1,1,0,(val>>2)&1,(val>>1)&1,val&1]
    else:          val=idx-14; return [1,1,1,1,0,(val>>3)&1,(val>>2)&1,(val>>1)&1,val&1]


def compress_arx(data: bytes) -> bytes:
    if len(data) % 4 != 0:
        data += b'\x00' * (4 - len(data) % 4)
    words = [struct.unpack_from('<I',data,i)[0] for i in range(0,len(data),4)]
    cnt = Counter(words)
    lut = [v for v,_ in cnt.most_common(30)]
    while len(lut) < 30: lut.append(0)
    lut_map = {v:i for i,v in enumerate(lut)}
    result=[]; ctrl_bits=[]; ctrl_raws=[]
    def flush():
        bits=ctrl_bits[:32]
        while len(bits)<32: bits.append(0)
        w=0
        for b in bits: w=(w<<1)|b
        result.append(struct.pack('<I',w))
        for r in ctrl_raws: result.append(struct.pack('<I',r))
        ctrl_bits.clear(); ctrl_raws.clear()
    for word in words:
        if word in lut_map: bits=_lut_idx_to_bits(lut_map[word]); raw=None
        else: bits=[0]; raw=word
        if raw is not None: ctrl_raws.append(raw)
        for bit in bits:
            ctrl_bits.append(bit)
            if len(ctrl_bits)==32: flush()
    if ctrl_bits or ctrl_raws: flush()
    body=b''.join(result)
    hdr=(b'ARX\x00'
         +struct.pack('<I',len(data))
         +struct.pack('<I',len(body))
         +struct.pack('<I',0)
         +struct.pack('<30I',*lut))
    return hdr+body


# ─────────────────────────────────────────────────────────────────────────────
# 픽셀 변환
# ─────────────────────────────────────────────────────────────────────────────

W, H = 512, 448  # title.bin 고정 해상도


def raw_to_png(raw_pixels: bytes) -> Image.Image:
    """
    압축 해제된 raw 픽셀(RGBA, PS2 half-alpha) → PIL RGBA Image
    PS2 alpha: ×2 clamp 255
    """
    arr = np.frombuffer(raw_pixels, dtype=np.uint8).reshape(H, W, 4).copy()
    arr[:,:,3] = np.clip(arr[:,:,3].astype(np.uint16) * 2, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')


def png_to_raw(img: Image.Image) -> bytes:
    """
    PIL RGBA Image → PS2 half-alpha raw 픽셀 bytes
    alpha: ÷2  (255→127≈0x7F, 완전불투명은 0x80으로 clamp)
    """
    if img.size != (W, H):
        print(f"  경고: 이미지 크기 {img.size} → {W}×{H} 리사이즈")
        img = img.resize((W, H), Image.LANCZOS)
    img = img.convert('RGBA')
    arr = np.array(img, dtype=np.uint16).copy()
    # PS2 half-alpha: 255→0x80, 0→0x00
    arr[:,:,3] = np.clip((arr[:,:,3] + 1) // 2, 0, 0x80).astype(np.uint16)
    return arr.astype(np.uint8).tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# 커맨드
# ─────────────────────────────────────────────────────────────────────────────

def cmd_extract(src: str, dst_png: str):
    print(f"[extract] {src}")
    data = open(src, 'rb').read()
    raw  = decompress_arx(data)
    print(f"  ARX 압축 해제: {len(data):,} → {len(raw):,} bytes")

    img  = raw_to_png(raw)

    # 검정 배경 합성본도 함께 저장
    base = os.path.splitext(dst_png)[0]
    img.save(dst_png)
    print(f"  저장(RGBA): {dst_png}")

    bg   = Image.new('RGBA', img.size, (0,0,0,255))
    comp = Image.alpha_composite(bg, img)
    preview = base + '_preview.png'
    comp.save(preview)
    print(f"  저장(미리보기/검정BG): {preview}")


def cmd_rebuild(src_bin: str, src_png: str, dst_bin: str):
    print(f"[rebuild] {src_bin} + {src_png} → {dst_bin}")

    # 원본 압축 해제 (라운드트립 기준점)
    orig_data = open(src_bin, 'rb').read()
    orig_raw  = decompress_arx(orig_data)
    print(f"  원본 ARX 압축 해제: {len(orig_data):,} → {len(orig_raw):,} bytes")

    # 새 이미지 로드 → PS2 raw 픽셀
    new_img = Image.open(src_png)
    new_raw = png_to_raw(new_img)
    print(f"  새 이미지 변환: {new_img.size} → {len(new_raw):,} bytes raw")

    if len(new_raw) != len(orig_raw):
        raise ValueError(f"raw 크기 불일치: {len(new_raw)} != {len(orig_raw)}")

    # 재압축
    comp = compress_arx(new_raw)
    open(dst_bin, 'wb').write(comp)
    print(f"  재압축: {len(new_raw):,} → {len(comp):,} bytes")
    print(f"  저장: {dst_bin}")

    # 검증
    verify = decompress_arx(comp)
    ok = verify == new_raw
    print(f"  라운드트립 검증: {'✓ 일치' if ok else '✗ 불일치'}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == 'extract':
        src = sys.argv[2]
        dst = (sys.argv[4] if len(sys.argv) >= 5 and sys.argv[3] == '--out'
               else os.path.splitext(src)[0] + '.png')
        cmd_extract(src, dst)

    elif cmd == 'rebuild':
        if len(sys.argv) < 4:
            print("rebuild: title.bin 과 새 이미지 PNG 둘 다 필요")
            sys.exit(1)
        src_bin = sys.argv[2]
        src_png = sys.argv[3]
        dst = (sys.argv[5] if len(sys.argv) >= 6 and sys.argv[4] == '--out'
               else os.path.splitext(src_bin)[0] + '_rebuilt.bin')
        cmd_rebuild(src_bin, src_png, dst)

    else:
        print(f"알 수 없는 명령: {cmd}")
        print("사용법: extract / rebuild")
        sys.exit(1)


if __name__ == '__main__':
    main()
