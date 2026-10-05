"""XS2 ESD/ESP indexed-image export and size-preserving rebuild.

Inputs are read-only. Outputs stay inside this simajiri project directory.
"""

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "original"
EXTRACT = ROOT / "graphics_extract"
REBUILT = ROOT / "graphics_rebuilt"
SELECTION = ROOT / "text_graphics.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def effect_blocks(data):
    if len(data) < 64 or struct.unpack_from("<I", data)[0] != len(data):
        raise ValueError("Root size mismatch")
    blocks = []
    pos = 64
    while pos < len(data):
        if pos + 64 > len(data):
            raise ValueError(f"Truncated block {pos:#x}")
        size, field04 = struct.unpack_from("<II", data, pos)
        if size == 0 and not any(data[pos:]):
            break
        if size < 64 or pos + size > len(data):
            raise ValueError(f"Block extent {pos:#x}: {size:#x}/{field04:#x}")
        name = data[pos+20:pos+64].split(b"\0", 1)[0].decode("ascii")
        fields = struct.unpack_from("<6H", data, pos+8)
        blocks.append({"offset": pos, "size": size, "payload_size": size-64, "field04": field04,
                       "fields": fields, "name": name,
                       "payload_sha256": sha(data[pos+64:pos+size])})
        pos += size
    return blocks


def source_files(root):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    folders = [root / name for name in ("esd", "esp") if (root / name).is_dir()]
    for folder in folders or [root]:
        for path in sorted(folder.rglob("*")):
            if any(path.is_relative_to(folder) for folder in (EXTRACT, REBUILT, ROOT / "work", ROOT / "out")):
                continue
            if path.suffix.lower() in (".esd", ".esp") and path.is_file():
                yield path



def output_path(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT) or result == ROOT:
        raise ValueError(f"Output must be inside {ROOT}")
    if result.is_relative_to(ORIGINAL):
        raise ValueError("The original directory is read-only for pipeline outputs")
    return result


def extraction_directory(source):
    source = source.resolve()
    relative = source.relative_to(ORIGINAL) if source.is_relative_to(ORIGINAL) \
               else Path(source.suffix[1:].lower()) / source.name
    return EXTRACT / relative.with_suffix("")


def rebuild_directory(work):
    work = work.resolve()
    if work.is_relative_to(EXTRACT):
        relative = work.relative_to(EXTRACT)
        return REBUILT / (relative.name + "_" + relative.parent.name)
    return REBUILT / work.name


def payload(data, block):
    start = block["offset"] + 64
    return data[start:start + block["payload_size"]]


def containers(data, base=0, route=()):
    blocks = effect_blocks(data)
    yield data, base, route, blocks
    if len(route) >= 8:
        raise ValueError("Unexpected ESD nesting depth")
    for block in blocks:
        if block["fields"][0] == 10:
            yield from containers(payload(data, block), base + block["offset"] + 64,
                                  route + (block["offset"],))


def palette_rgba(raw, bpp):
    if bpp not in (4, 8) or len(raw) != 4 * (1 << bpp):
        raise ValueError("Unexpected CLUT extent")
    palette = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).copy()
    if bpp == 8:
        i = np.arange(256)
        # CSM1 CLUT uploaded as 16x16 PSMCT32: swap index bits 3 and 4.
        palette = palette[(i & ~24) | ((i & 8) << 1) | ((i & 16) >> 1)]
    palette[:, 3] = np.minimum(palette[:, 3].astype(np.uint16) * 2, 255)
    return palette


def image_records(data):
    records, unsupported = [], []
    for local, base, route, blocks in containers(data):
        palettes = defaultdict(list)
        for block in blocks:
            if block["fields"][0] == 3:
                palettes[block["name"]].append(block)
        for block in blocks:
            kind, resource_id, width, height, bpp, flags = block["fields"]
            if kind not in (4, 7):
                continue
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(block["name"]).stem)
            key = f"{base + block['offset']:08x}_{stem}"
            candidates = palettes[block["name"][:-4] + ".clt"]
            reason = None
            if not block["name"].endswith(".bmp") or bpp not in (4, 8) or flags not in (0, 1):
                reason = "Unknown image profile"
            elif width < 1 or height < 1 or width > 4096 or height > 4096 or width % 2:
                reason = "Unsupported dimensions"
            elif len(candidates) != 1:
                reason = "Missing or ambiguous same-container palette"
            else:
                palette = candidates[0]
                expected = width * height * bpp // 8
                if block["payload_size"] != (expected + 63) // 64 * 64:
                    reason = "Unexpected image extent"
                elif palette["fields"][4] != bpp or palette["payload_size"] != 4 * (1 << bpp):
                    reason = "Palette depth/extent mismatch"
            if reason:
                unsupported.append({"key": key, "reason": reason})
                continue
            pal_bytes = payload(local, palette)
            pal = palette_rgba(pal_bytes, bpp)
            mode = "alpha" if np.all(pal[:, :3] == 0) else "rgba"
            records.append({"key": key, "name": block["name"], "route": list(route),
                            "offset": base + block["offset"] + 64, "length": expected,
                            "palette_offset": base + palette["offset"] + 64,
                            "palette_length": len(pal_bytes), "palette_sha256": sha(pal_bytes),
                            "width": width, "height": height, "bpp": bpp, "mode": mode,
                            "png": f"images/{key}.{mode}.png"})
    return records, unsupported


def indices(data, record):
    raw = np.frombuffer(data, dtype=np.uint8, count=record["length"], offset=record["offset"])
    if record["bpp"] == 4:
        raw = np.column_stack((raw & 15, raw >> 4)).reshape(-1)
    return raw.reshape(record["height"], record["width"])


def record_palette(data, record):
    off = record["palette_offset"]
    return palette_rgba(data[off:off + record["palette_length"]], record["bpp"])


def rendered(data, record):
    rgba = record_palette(data, record)[indices(data, record)]
    return rgba[:, :, 3] if record["mode"] == "alpha" else rgba


def features(rgba):
    rgba = np.asarray(rgba, dtype=np.float32)
    alpha = rgba[..., 3:4]
    return np.concatenate((rgba[..., :3] * alpha / 255, alpha * 2), axis=-1)


def quantize(edited, palette, original, mode):
    expected = original.shape if mode == "alpha" else original.shape + (4,)
    if edited.shape != expected or edited.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image of shape {expected}, got {edited.shape}")
    before = palette[original]
    before = before[..., 3] if mode == "alpha" else before
    mask = edited != before if mode == "alpha" else np.any(edited != before, axis=-1)
    result = original.copy()
    if not np.any(mask):
        return result, {"edited_pixels": 0, "changed_indices": 0, "max_channel_error": 0}
    values = edited[mask]
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    pf = palette[:, 3:4].astype(np.float32) if mode == "alpha" else features(palette)
    uf = unique.reshape(-1, 1).astype(np.float32) if mode == "alpha" else features(unique)
    nearest = np.empty(len(unique), dtype=np.uint8)
    for start in range(0, len(unique), 1024):
        distances = np.sum((uf[start:start+1024, None] - pf[None]) ** 2, axis=-1)
        nearest[start:start+1024] = np.argmin(distances, axis=1)
    result[mask] = nearest[inverse]
    after = palette[result]
    after = after[..., 3] if mode == "alpha" else after
    return result, {"edited_pixels": int(mask.sum()),
                    "changed_indices": int(np.count_nonzero(result != original)),
                    "max_channel_error": int(np.max(np.abs(after[mask].astype(np.int16) - values)))}


def rebuild_bytes(data, records, edits):
    result = bytearray(data)
    reports = []
    by_key = {r["key"]: r for r in records}
    if edits.keys() - by_key.keys():
        raise ValueError("Unknown edit key")
    for key, pixels in edits.items():
        record = by_key[key]
        new, report = quantize(pixels, record_palette(data, record), indices(data, record), record["mode"])
        flat = new.reshape(-1)
        if record["bpp"] == 4:
            flat = flat[::2] | (flat[1::2] << 4)
        raw = flat.tobytes()
        if len(raw) != record["length"]:
            raise ValueError("Pixel data resized unexpectedly")
        off = record["offset"]
        result[off:off+len(raw)] = raw
        reports.append({"key": key, **report})
    allowed = np.zeros(len(data), dtype=bool)
    for key in edits:
        r = by_key[key]
        allowed[r["offset"]:r["offset"]+r["length"]] = True
    changed = np.frombuffer(data, np.uint8) != np.frombuffer(result, np.uint8)
    if np.any(changed & ~allowed):
        raise AssertionError("Rebuild changed non-pixel bytes")
    for r in records:
        off = r["palette_offset"]
        if result[off:off+r["palette_length"]] != data[off:off+r["palette_length"]]:
            raise ValueError("Image/palette overlap")
    return bytes(result), reports


def png_bytes(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def write_new(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def write_json(path, value):
    write_new(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def text_selection():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("format") != "xs2-text-selection-v1":
        raise ValueError("Unknown text selection format")
    return selection["files"]


def selected_records(source, data, records):
    relative = source.suffix[1:].lower() + "/" + source.name
    entry = text_selection().get(relative)
    if entry is None:
        raise ValueError(f"Not a reviewed text asset: {relative}")
    if sha(data) != entry["sha256"]:
        raise ValueError(f"Text selection source hash mismatch: {relative}")
    keys = [r["key"] for r in entry["images"]]
    chosen = [r for r in records if r["key"] in keys]
    if not keys or len(keys) != len(set(keys)) or len(chosen) != len(keys):
        raise ValueError("Text selection does not match source images")
    return chosen


def export(source, destination, text_only=False):
    source = source.resolve()
    destination = output_path(destination)
    if destination.exists():
        raise ValueError("Export directory already exists; refusing to overwrite edits")
    data = source.read_bytes()
    records, unsupported = image_records(data)
    total_images = len(records)
    if text_only:
        records = selected_records(source, data, records)
    if not records:
        raise ValueError("No supported images in this container")
    for r in records:
        write_new(destination / r["png"], png_bytes(rendered(data, r)))
    write_json(destination / "manifest.json", {"format": "xs2-effects-text-v1" if text_only else "xs2-effects-v1", "source": str(source),
               "source_sha256": sha(data), "source_size": len(data),
               "images": records, "unsupported": unsupported})
    return {"images": len(records), "preserved_images": total_images - len(records),
            "unsupported": len(unsupported), "directory": str(destination)}


def export_text():
    jobs = []
    for relative in text_selection():
        source = (ORIGINAL / relative).resolve()
        if not source.is_relative_to(ORIGINAL):
            raise ValueError("Text selection source must stay under original")
        work = extraction_directory(source)
        if work.exists():
            raise ValueError(f"Export directory already exists; refusing to overwrite edits: {work}")
        data = source.read_bytes()
        records, _ = image_records(data)
        selected_records(source, data, records)
        jobs.append((source, work))
    if (EXTRACT / "catalog.json").exists():
        raise ValueError("Export catalog already exists")
    files = []
    for source, work in jobs:
        result = export(source, work, text_only=True)
        files.append({"path": source.relative_to(ORIGINAL).as_posix(),
                      "extract": work.relative_to(ROOT).as_posix(), "images": result["images"]})
    write_json(EXTRACT / "catalog.json", {"format": "xs2-text-catalog-v1", "selection": SELECTION.name,
               "files": files})
    return {"files": len(files), "images": sum(f["images"] for f in files)}


def exact_replacements(original, rebuilt, suffix):
    replacements = {}
    if suffix.lower() == ".esd" and original != rebuilt:
        replacements[sha(original)] = (original, rebuilt)
    for local, base, route, blocks in containers(original):
        if route and local != rebuilt[base:base+len(local)]:
            replacements[sha(local)] = (local, rebuilt[base:base+len(local)])
    return replacements


def sync_exact(data, replacements):
    replacement = replacements.get(sha(data))
    if replacement and replacement[0] == data:
        return replacement[1]
    result = bytearray(data)
    for local, base, route, blocks in containers(data):
        if not route:
            continue
        replacement = replacements.get(sha(local))
        if replacement and replacement[0] == local:
            if len(local) != len(replacement[1]):
                raise ValueError("Exact-copy replacement changed size")
            result[base:base+len(local)] = replacement[1]
    return bytes(result)


def read_edit(path, record):
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
            raise ValueError(f"Use an 8-bit PNG: {path.name}")
        if im.size != (record["width"], record["height"]):
            raise ValueError(f"Image dimensions changed: {path.name}")
        rgba = im.convert("RGBA")
        if record["mode"] != "alpha":
            return np.array(rgba), "rgba"
        # Opaque masks encode coverage in brightness; transparent artwork in alpha.
        alpha = rgba.getchannel("A")
        if alpha.getextrema()[0] < 255:
            return np.array(alpha), "alpha"
        return np.array(im.convert("L")), "luminance"


def rebuild(work, destination, sync_root=None):
    work = output_path(work)
    destination = output_path(destination)
    if destination.exists():
        raise ValueError("Output directory already exists")
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in ("xs2-effects-v1", "xs2-effects-text-v1"):
        raise ValueError("Unknown manifest format")
    source = Path(manifest["source"]).resolve()
    data = source.read_bytes()
    if sha(data) != manifest["source_sha256"] or len(data) != manifest["source_size"]:
        raise ValueError("Source changed after export")
    records, unsupported = image_records(data)
    editable = selected_records(source, data, records) if manifest["format"] == "xs2-effects-text-v1" else records
    if manifest["images"] != editable or manifest["unsupported"] != unsupported:
        raise ValueError("Manifest recipe changed; export again from the original")
    kor_names = {Path(r["png"]).stem.casefold() + "_kor.png" for r in editable}
    for p in (work / "images").glob("*.png"):
        if p.suffix.lower() == ".png" and p.stem.lower().endswith("_kor") and p.name.casefold() not in kor_names:
            raise ValueError(f"Unrecognized _KOR image; check the original filename: {p.name}")
    edits, inputs = {}, {}
    kor_count = 0
    for r in editable:
        base = work / r["png"]
        kor = output_path(base.with_name(base.stem + "_KOR.png"))
        is_kor = kor.exists()
        path = kor if is_kor else output_path(base)
        edits[r["key"]], conversion = read_edit(path, r)
        inputs[r["key"]] = {"input_png": path.relative_to(work).as_posix(), "conversion": conversion,
                             "kor_override": is_kor}
        kor_count += is_kor
    new, reports = rebuild_bytes(data, records, edits)
    for image_report in reports:
        image_report.update(inputs[image_report["key"]])
    root = Path(sync_root).resolve() if sync_root else (ORIGINAL if source.is_relative_to(ORIGINAL) else None)
    if root and not source.is_relative_to(root):
        raise ValueError("Source must be under --sync-root")
    pending = {}
    if new != data:
        pending[source.relative_to(root) if root else Path(source.name)] = (data, new)
    copies = exact_replacements(data, new, source.suffix)
    if root and copies:
        for p in source_files(root):
            if p.resolve() == source:
                continue
            raw = p.read_bytes()
            synced = sync_exact(raw, copies)
            if synced != raw:
                pending[p.relative_to(root)] = (raw, synced)
    report = {"source": str(source), "source_sha256": sha(data), "images": reports,
              "sync_policy": "Exact full-ESD copies only; no same-name or partial matching",
              "sync_root": str(root) if root else None, "unsupported_preserved": unsupported,
              "unselected_images_preserved": len(records) - len(editable),
              "files": [{"path": p.as_posix(), "size": len(old), "before": sha(old), "after": sha(new),
                         "changed_bytes": sum(a != b for a, b in zip(old, new))}
                        for p, (old, new) in pending.items()]}
    for path in pending:
        output_path(destination / path)
    for path, (_, built) in pending.items():
        write_new(destination / path, built)
    write_json(destination / "build-report.json", report)
    return {"changed_files": len(pending), "kor_images": kor_count, "directory": str(destination)}


def audit(root):
    files, unsupported, copy_locations = [], [], defaultdict(list)
    root = root.resolve()
    for p in source_files(root):
        raw = p.read_bytes()
        records, skipped = image_records(raw)
        relative = p.relative_to(root).as_posix()
        for r in skipped:
            unsupported.append({"file": relative, **r})
        edits = {r["key"]: rendered(raw, r) for r in records}
        built, _ = rebuild_bytes(raw, records, edits)
        if built != raw:
            raise AssertionError(f"No-edit roundtrip failed: {p}")
        files.append({"path": relative, "sha256": sha(raw), "size": len(raw),
                      "images": len(records), "unsupported": len(skipped)})
        for local, base, route, _ in containers(raw):
            if route or p.suffix.lower() == ".esd":
                copy_locations[sha(local)].append({"file": relative, "offset": base})
    if not files:
        raise ValueError("No ESD/ESP source files found; specify --root with an unpacked simajiri directory")
    return {"root": str(root.resolve()), "files": files, "unsupported": unsupported,
            "exact_esd_copy_groups": [v for v in copy_locations.values() if len(v) > 1],
            "roundtrip": "All supported images identical without edits"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("source", type=Path)
    exp.add_argument("destination", type=Path, nargs="?")
    exp.add_argument("--all-images", action="store_true", help="Explicit opt-in for non-text effects")
    sub.add_parser("export-text", help="Export only the reviewed text selection from original")
    build = sub.add_parser("rebuild")
    build.add_argument("work", type=Path)
    build.add_argument("destination", type=Path, nargs="?")
    build.add_argument("--sync-root", type=Path)
    check = sub.add_parser("audit")
    check.add_argument("--root", type=Path, default=ORIGINAL)
    check.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "export":
            if args.source.suffix.lower() not in (".esd", ".esp"):
                raise ValueError("Select an .esd or .esp source file")
            destination = args.destination or extraction_directory(args.source)
            result = export(args.source, destination, text_only=not args.all_images)
        elif args.command == "export-text":
            result = export_text()
        elif args.command == "rebuild":
            work = args.work.parent if args.work.name.lower() == "manifest.json" else args.work
            destination = args.destination or rebuild_directory(work)
            result = rebuild(work, destination, args.sync_root)
        else:
            result = audit(args.root)
            if args.report:
                write_json(args.report, result)
            result = {"files": len(result["files"]), "images": sum(f["images"] for f in result["files"]),
                      "unsupported": len(result["unsupported"]),
                      "exact_esd_copy_groups": len(result["exact_esd_copy_groups"])}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
