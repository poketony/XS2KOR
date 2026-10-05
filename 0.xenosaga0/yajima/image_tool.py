"""XS2 synopsis.bin: PNG export and fixed-size JPEG/XTX rebuilds."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "synopsis.bin"
EXTRACT = ROOT / "graphics_extract"
REBUILT = ROOT / "graphics_rebuilt"
PROFILES = {"synopsis.bin": 24}
SOURCES = tuple(PROFILES)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def output_path(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root) and path != root for root in (EXTRACT, REBUILT)):
        raise ValueError("Work/output must stay inside graphics_extract or graphics_rebuilt; originals are read-only")
    return path


def write_new(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def write_json(path, value):
    write_new(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def png_bytes(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def check_jpeg(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        if im.format != "JPEG" or im.size != (512, 448) or im.mode != "RGB" \
                or im.info.get("progressive") or im.layer != [(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)]:
            raise ValueError("Unverified JPEG profile (expected baseline 512x448 RGB 4:4:4)")


def parse(data, filename):
    count = PROFILES.get(filename)
    if count is None or len(data) < 4 or struct.unpack_from("<I", data)[0] != count*4:
        raise ValueError("Unknown bundle profile/header")
    base = (4 + count*4 + 63) // 64 * 64
    if len(data) < base:
        raise ValueError("Truncated offset table")
    ends = struct.unpack_from(f"<{count}I", data, 4)
    start, entries = base, []
    for index, relative_end in enumerate(ends):
        end = base + relative_end
        if end <= start or end > len(data):
            raise ValueError("Invalid member extent")
        raw = data[start:end]
        if raw[:4] == b"XTX\0" and filename == "synopsis.bin" and index < 2:
            if len(raw) != 131200 or struct.unpack_from("<3I", raw, 4) != (131152, 1, 16) \
                    or struct.unpack_from("<4H3I", raw, 16) != (512, 8, 64, 0, 0, 8194, 48):
                raise ValueError("Unknown synopsis direct-RGBA XTX profile")
            entry = {"index": index, "type": "XTX", "offset": start, "length": end-start,
                     "pixel_offset": start+80, "pixel_length": 512*64*4,
                     "png": f"images/{'loading' if index == 0 else 'start'}.png", "width": 512, "height": 64}
        else:
            if index < 2:
                raise ValueError("Missing synopsis XTX member")
            check_jpeg(raw)
            entry = {"index": index, "type": "JPEG", "offset": start, "length": end-start,
                     "png": f"images/{index-1:02d}.png", "width": 512, "height": 448}
        entries.append(entry)
        start = (end + 63) // 64 * 64
    if start != len(data):
        raise ValueError("Unexpected bundle trailer")
    return entries


def pixels(data, entry):
    if entry["type"] == "XTX":
        rgba = np.frombuffer(data, np.uint8, count=entry["pixel_length"], offset=entry["pixel_offset"]).reshape(64, 512, 4).copy()
        rgba[:, :, 3] = np.minimum(rgba[:, :, 3].astype(np.uint16)*2, 255)
        return rgba
    start = entry["offset"]
    with Image.open(io.BytesIO(data[start:start+entry["length"]])) as im:
        im.load()
        return np.array(im.convert("RGB"))


def rebuild_bytes(data, filename, edits, quality=None):
    if quality is not None and not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be 1..100")
    entries = parse(data, filename)
    editable = {e["index"]: e for e in entries}
    if edits.keys() - editable.keys():
        raise ValueError("Unknown image member")
    result = bytearray(data)
    reports = []
    for index, edited in edits.items():
        entry = editable[index]
        original = pixels(data, entry)
        if edited.shape != original.shape or edited.dtype != np.uint8:
            raise ValueError(f"Member {index}: expected uint8 image of shape {original.shape}")
        if np.array_equal(edited, original):
            continue
        if entry["type"] == "XTX":
            start = entry["pixel_offset"]
            raw = np.frombuffer(data, np.uint8, count=entry["pixel_length"], offset=start).reshape(original.shape).copy()
            encoded = edited.copy()
            encoded[:, :, 3] = (edited[:, :, 3].astype(np.uint16)+1)//2
            changed = edited != original
            raw[changed] = encoded[changed]
            result[start:start+entry["pixel_length"]] = raw.tobytes()
            reports.append({"index": index, "type": "XTX", "edited_pixels": int(np.any(changed, axis=-1).sum()),
                            "changed_bytes": int(np.count_nonzero(np.frombuffer(data[start:start+entry['pixel_length']], np.uint8) != raw.reshape(-1)))})
            continue
        stream = io.BytesIO()
        if quality is None:
            start = entry["offset"]
            with Image.open(io.BytesIO(data[start:start+entry["length"]])) as source_image:
                compression = {"qtables": source_image.quantization}
        else:
            compression = {"quality": quality}
        # Pillow can underestimate optimized 4:4:4 output; restore its shared limit.
        old_block = ImageFile.MAXBLOCK
        try:
            ImageFile.MAXBLOCK = max(old_block, edited.nbytes*2 + 65536)
            Image.fromarray(edited).save(stream, format="JPEG", subsampling=0,
                                         progressive=False, optimize=True, **compression)
        finally:
            ImageFile.MAXBLOCK = old_block
        raw = stream.getvalue()
        if len(raw) > entry["length"]:
            raise ValueError(f"Member {index}: {len(raw)} JPEG bytes exceeds slot {entry['length']}; "
                             "lower --quality or simplify the image")
        start = entry["offset"]
        result[start:start+entry["length"]] = raw.ljust(entry["length"], b"\0")
        decoded = np.array(Image.open(io.BytesIO(raw)))
        reports.append({"index": index, "type": "JPEG", "quality": quality if quality is not None else "source", "jpeg_bytes": len(raw),
                        "slot_bytes": entry["length"],
                        "mean_absolute_error": float(np.abs(decoded.astype(np.int16)-edited).mean())})
    if parse(bytes(result), filename) != entries:
        raise AssertionError("Bundle structure changed")
    allowed = np.zeros(len(data), dtype=bool)
    for report in reports:
        entry = editable[report["index"]]
        start = entry.get("pixel_offset", entry["offset"])
        length = entry.get("pixel_length", entry["length"])
        allowed[start:start+length] = True
    if np.any((np.frombuffer(data, np.uint8) != np.frombuffer(result, np.uint8)) & ~allowed):
        raise AssertionError("Bytes outside edited image slots changed")
    return bytes(result), reports


def export(source, destination):
    destination = output_path(destination)
    if destination.exists():
        raise ValueError("Export directory exists")
    data = source.read_bytes()
    entries = parse(data, source.name)
    for entry in entries:
        write_new(destination / entry["png"], png_bytes(pixels(data, entry)))
    write_json(destination / "manifest.json", {"format": "xs2-yajima-images-v1", "source": str(source.resolve()),
               "source_sha256": sha(data), "entries": entries})
    return {"images": len(entries), "directory": str(destination)}


def rebuild(work, destination, quality=None):
    work, destination = output_path(work), output_path(destination)
    if destination.exists():
        raise ValueError("Output directory exists")
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "xs2-yajima-images-v1":
        raise ValueError("Unknown manifest format")
    source = Path(manifest["source"])
    data = source.read_bytes()
    entries = parse(data, source.name)
    if sha(data) != manifest["source_sha256"] or entries != manifest["entries"]:
        raise ValueError("Source or manifest recipe changed")
    expected_kor = {Path(e["png"]).stem.casefold()+"_kor.png" for e in entries}
    for path in (work / "images").glob("*.png"):
        if path.stem.lower().endswith("_kor") and path.name.casefold() not in expected_kor:
            raise ValueError(f"Unrecognized _KOR filename: {path.name}")
    edits, inputs = {}, {}
    kor_count = 0
    for entry in entries:
        base = work / entry["png"]
        kor = output_path(base.with_name(base.stem+"_KOR.png"))
        is_kor = kor.exists()
        path = kor if is_kor else output_path(base)
        with Image.open(path) as im:
            if im.format != "PNG" or im.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                raise ValueError(f"Use an 8-bit PNG: {path.name}")
            if im.size != (entry["width"], entry["height"]):
                raise ValueError("PNG dimensions changed")
            rgba = np.array(im.convert("RGBA"))
            if entry["type"] == "JPEG" and not np.all(rgba[:, :, 3] == 255):
                raise ValueError("JPEG cannot preserve transparency; provide an opaque image")
            edits[entry["index"]] = rgba if entry["type"] == "XTX" else rgba[:, :, :3]
        inputs[entry["index"]] = {"input_png": path.relative_to(work).as_posix(), "kor_override": is_kor}
        kor_count += is_kor
    built, reports = rebuild_bytes(data, source.name, edits, quality)
    for report in reports:
        report.update(inputs[report["index"]])
    if built != data:
        write_new(destination / source.name, built)
    write_json(destination / "build-report.json", {"source": str(source), "before": sha(data), "after": sha(built),
               "policy": "Fixed slots; preserve tables, XTX headers, padding and unedited members", "changes": reports})
    return {"changed_images": len(reports), "kor_images": kor_count, "directory": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("source", type=Path)
    exp.add_argument("destination", type=Path, nargs="?")
    sub.add_parser("export-all", help="Export the existing synopsis.bin beside this tool")
    build = sub.add_parser("rebuild")
    build.add_argument("work", type=Path)
    build.add_argument("destination", type=Path, nargs="?")
    build.add_argument("--quality", type=int, help="1..100; omitted: preserve each source JPEG quantization table")
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export(args.source, args.destination or EXTRACT / args.source.stem)
        elif args.command == "export-all":
            result = export(SOURCE, EXTRACT / SOURCE.stem)
        else:
            work = args.work.parent if args.work.name.lower() == "manifest.json" else args.work
            result = rebuild(work, args.destination or REBUILT / work.name, args.quality)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
