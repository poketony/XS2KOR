#!/usr/bin/env python3
"""
Xenosaga II batdat.bin text extractor/patcher.

Usage:
  python batdat_tool.py extract batdat.bin batdat.json
  python batdat_tool.py patch   batdat.bin batdat.json [batdat_patched.bin] [char_table.json]

The patch command keeps the original file size and section layout. When a
section is changed, it rebuilds only that section's string pool, pads remaining
space with NUL bytes, and updates only documented text pointer fields.
Korean text can be converted through the same
char_table.json "replace-table" style used by the Xenosaga III workflow.
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ENCODING = "euc-jis-2004"
MENU_SITA_TAG = "[menu_sita]"


@dataclass(frozen=True)
class Section:
    label: str
    index: int
    base: int
    end: int
    stride: int
    hdr_off: int
    sec_type: str


SECTIONS = [
    Section("00_party_names", 0, 0x00054, 0x00650, 92, 0, "name_only"),
    Section("01_enemy_names", 1, 0x00650, 0x03AD4, 92, 0, "name_only"),
    Section("06_ether", 6, 0x0A854, 0x0C5B8, 12, 0, "name_desc"),
    Section("07_skills", 7, 0x0C5B8, 0x0D65C, 12, 0, "name_desc"),
    Section("08_skill_nodes", 8, 0x0D65C, 0x0DE44, 16, 0, "name_desc"),
    Section("09_items", 9, 0x0DE44, 0x0E6E8, 16, 0, "name_desc"),
    Section("10_secret_keys", 10, 0x0E6E8, 0x0EF48, 4, 0, "name_desc"),
    Section("11_segment_decoders", 11, 0x0EF48, 0x10E28, 8, 0, "name_desc"),
    Section("12_double_tech", 12, 0x10E28, 0x112A4, 16, 0, "name_desc"),
    Section("13_character_tech", 13, 0x112A4, 0x11888, 16, 0, "name_desc"),
    Section("14_es_tech", 14, 0x11888, 0x11A78, 16, 0, "name_desc"),
    Section("18_battle_messages", 18, 0x125B4, 0x128D8, 2, 0, "string_table"),
]


def read_header(data: bytes) -> list[int]:
    if len(data) < 8:
        raise ValueError("truncated header")
    count = struct.unpack_from("<I", data, 0)[0]
    if count != 19 or len(data) < 8 + count * 4:
        raise ValueError("expected the Xenosaga II 19-section header")
    file_size = struct.unpack_from("<I", data, 4)[0]
    if file_size != len(data):
        raise ValueError(f"header size mismatch: header={file_size}, actual={len(data)}")
    offsets = [struct.unpack_from("<I", data, 8 + i * 4)[0] for i in range(count)]
    if offsets[0] != 84 or offsets != sorted(set(offsets)) or offsets[-1] >= len(data):
        raise ValueError("invalid section offsets")
    return offsets


def validate_sections(data: bytes) -> None:
    offsets = read_header(data)
    for sec in SECTIONS:
        if sec.index >= len(offsets):
            raise ValueError(f"{sec.label}: header index {sec.index} is missing")
        if offsets[sec.index] != sec.base:
            raise ValueError(
                f"{sec.label}: base mismatch: expected 0x{sec.base:X}, "
                f"header has 0x{offsets[sec.index]:X}"
            )
        actual_end = offsets[sec.index + 1] if sec.index + 1 < len(offsets) else len(data)
        if actual_end != sec.end:
            raise ValueError(
                f"{sec.label}: end mismatch: expected 0x{sec.end:X}, "
                f"header has 0x{actual_end:X}"
            )


def read_raw_str(data: bytes, abs_addr: int, end: int) -> bytes:
    if not 0 <= abs_addr < end <= len(data):
        raise ValueError(f"invalid string address: 0x{abs_addr:X}")
    nul = data.find(b"\x00", abs_addr, end)
    if nul < 0:
        raise ValueError(f"unterminated string: 0x{abs_addr:X}")
    return bytes(data[abs_addr:nul])


def read_str(data: bytes, abs_addr: int, end: int) -> str:
    return read_raw_str(data, abs_addr, end).decode(ENCODING)


def encode_str(text: str) -> bytes:
    if not isinstance(text, str):
        raise ValueError("translation must be a string")
    if "\x00" in text:
        raise ValueError("translation contains NUL")
    return text.encode(ENCODING)


def load_char_table(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, encoding="utf-8-sig") as f:
        obj = json.load(f)
    table = obj.get("replace-table") if isinstance(obj, dict) else None
    if not isinstance(table, dict):
        raise ValueError("char_table must contain a replace-table object")
    for key, value in table.items():
        if not isinstance(key, str) or len(key) != 1 or not isinstance(value, str) or len(value) != 1:
            raise ValueError("char_table mappings must be one character to one character")
        encode_str(value)
    return table


def apply_char_table(text: str, table: dict[str, str]) -> str:
    if not isinstance(text, str):
        raise ValueError("translation must be a string")
    if not table:
        return text
    return "".join(table.get(ch, ch) for ch in text)


def apply_menu_sita(text: str) -> str:
    if text.endswith(MENU_SITA_TAG):
        text = text[: -len(MENU_SITA_TAG)]
        text = text.replace(" ", "\u3000")
        text = text.replace("@", " ")
    return text


def count_entries(data: bytes, sec: Section) -> int:
    first_ptr = struct.unpack_from("<H", data, sec.base + sec.hdr_off)[0]
    table_size = first_ptr - sec.hdr_off
    expected = {0: 15, 1: 125, 6: 109, 7: 104, 8: 40, 9: 40,
                10: 31, 11: 107, 12: 16, 13: 21, 14: 10, 18: 54}
    if table_size != expected[sec.index] * sec.stride:
        raise ValueError(
            f"{sec.label}: invalid pointer table size {table_size} for stride {sec.stride}"
        )
    return table_size // sec.stride


def is_pool_ptr(sec: Section, rel: int, pool_start_rel: int) -> bool:
    return pool_start_rel <= rel < sec.end - sec.base


def entry_fields(data: bytes, sec: Section, idx: int, pool_start_rel: int) -> list[tuple[str, int, int]]:
    ptr_pos = sec.base + sec.hdr_off + idx * sec.stride
    name_ptr = struct.unpack_from("<H", data, ptr_pos)[0]
    if not is_pool_ptr(sec, name_ptr, pool_start_rel):
        raise ValueError(f"{sec.label}[{idx}]: name pointer outside string pool")
    if sec.sec_type == "name_only":
        return [("name", ptr_pos, name_ptr)]
    if sec.sec_type == "string_table":
        return [("text", ptr_pos, name_ptr)]

    desc_ptr = struct.unpack_from("<H", data, ptr_pos + 2)[0]
    fields = [("name", ptr_pos, name_ptr)]
    if not is_pool_ptr(sec, desc_ptr, pool_start_rel):
        raise ValueError(f"{sec.label}[{idx}]: description pointer outside string pool")
    fields.append(("desc", ptr_pos + 2, desc_ptr))
    return fields


def make_entry(data: bytes, sec: Section, idx: int, pool_start_rel: int) -> dict:
    fields = entry_fields(data, sec, idx, pool_start_rel)
    entry = {"idx": idx, "ptr_pos": sec.base + sec.hdr_off + idx * sec.stride}

    if sec.sec_type == "name_only":
        _, ptr_pos, name_ptr = fields[0]
        entry.update(
            {
                "name_ptr": name_ptr,
                "jp": read_str(data, sec.base + name_ptr, sec.end),
                "ko": read_str(data, sec.base + name_ptr, sec.end),
            }
        )
    elif sec.sec_type == "string_table":
        _, ptr_pos, text_ptr = fields[0]
        entry.update(
            {
                "text_ptr": text_ptr,
                "jp": read_str(data, sec.base + text_ptr, sec.end),
                "ko": read_str(data, sec.base + text_ptr, sec.end),
            }
        )
    else:
        name_ptr = struct.unpack_from("<H", data, entry["ptr_pos"])[0]
        desc_ptr = struct.unpack_from("<H", data, entry["ptr_pos"] + 2)[0]
        entry.update(
            {
                "name_ptr": name_ptr,
                "desc_ptr": desc_ptr,
                "jp_name": read_str(data, sec.base + name_ptr, sec.end)
                if is_pool_ptr(sec, name_ptr, pool_start_rel)
                else "",
                "jp_desc": read_str(data, sec.base + desc_ptr, sec.end)
                if is_pool_ptr(sec, desc_ptr, pool_start_rel)
                else "",
                "ko_name": read_str(data, sec.base + name_ptr, sec.end)
                if is_pool_ptr(sec, name_ptr, pool_start_rel)
                else "",
                "ko_desc": read_str(data, sec.base + desc_ptr, sec.end)
                if is_pool_ptr(sec, desc_ptr, pool_start_rel)
                else "",
            }
        )
    return entry


def cmd_extract(bin_path: str, out_json: str) -> None:
    if Path(bin_path).resolve() == Path(out_json).resolve() or (
        Path(out_json).exists() and Path(bin_path).samefile(out_json)
    ):
        raise ValueError("output must not overwrite the source binary")
    data = Path(bin_path).read_bytes()
    validate_sections(data)
    result = {}

    print(f"[extract] {bin_path} ({len(data):,} bytes)")
    for sec in SECTIONS:
        n = count_entries(data, sec)
        pool_start_rel = struct.unpack_from("<H", data, sec.base + sec.hdr_off)[0]
        entries = [make_entry(data, sec, i, pool_start_rel) for i in range(n)]
        result[sec.label] = {
            "index": sec.index,
            "base": sec.base,
            "end_sec": sec.end,
            "stride": sec.stride,
            "hdr_off": sec.hdr_off,
            "sec_type": sec.sec_type,
            "entries": entries,
        }
        print(f"  [{sec.label:<20}] {n:4d} entries")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(sec["entries"]) for sec in result.values())
    print(f"[extract] wrote {out_json} ({total:,} entries)")


def patch_section(buf: bytearray, sec: Section, sec_json: dict, char_table: dict[str, str]) -> tuple[bool, str]:
    n = count_entries(buf, sec)
    pool_start_rel = struct.unpack_from("<H", buf, sec.base + sec.hdr_off)[0]
    pool_start = sec.base + pool_start_rel
    pool_size = sec.end - pool_start
    if not isinstance(sec_json, dict) or not isinstance(sec_json.get("entries"), list):
        raise ValueError(f"{sec.label}: expected an entries list")
    for key, value in {"index": sec.index, "base": sec.base, "end_sec": sec.end,
                       "stride": sec.stride, "hdr_off": sec.hdr_off, "sec_type": sec.sec_type}.items():
        if key in sec_json and sec_json[key] != value:
            raise ValueError(f"{sec.label}: mismatched {key}")

    # Addresses always come from the binary; JSON addresses are consistency checks.
    originals = [make_entry(buf, sec, i, pool_start_rel) for i in range(n)]
    submitted = {}
    for entry in sec_json["entries"]:
        if not isinstance(entry, dict):
            raise ValueError(f"{sec.label}: entry must be an object")
        idx = entry.get("idx")
        if type(idx) is not int or not 0 <= idx < n or idx in submitted:
            raise ValueError(f"{sec.label}: invalid or duplicate idx {idx!r}")
        for key, value in originals[idx].items():
            if not key.startswith("ko") and entry.get(key) != value:
                raise ValueError(f"{sec.label}[{idx}]: source mismatch in {key}; extract from this binary again")
        submitted[idx] = entry

    pointers = []
    translations = {}
    covered = set()
    patched_fields = 0
    for idx in range(n):
        entry = submitted.get(idx, originals[idx])
        for field, pos, rel in entry_fields(buf, sec, idx, pool_start_rel):
            suffix = "_" + field if sec.sec_type == "name_desc" else ""
            jp_key, ko_key = "jp" + suffix, "ko" + suffix
            old_raw = read_raw_str(buf, sec.base + rel, sec.end)
            text = entry.get(ko_key, originals[idx][jp_key])
            raw = old_raw
            if text != originals[idx][jp_key]:
                try:
                    raw = encode_str(apply_menu_sita(apply_char_table(text, char_table)))
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"{sec.label}[{idx}].{ko_key}: {exc}") from exc
            if rel in translations and translations[rel] != raw:
                raise ValueError(f"{sec.label}[{idx}]: conflicting translations for shared pointer 0x{rel:X}")
            translations[rel] = raw
            pointers.append((pos, rel))
            covered.update(range(rel, rel + len(old_raw) + 1))
            patched_fields += raw != old_raw
    if not patched_fields:
        return False, "unchanged"
    if any(buf[sec.base + rel] and rel not in covered for rel in range(pool_start_rel, sec.end - sec.base)):
        raise ValueError(f"{sec.label}: unreferenced nonzero pool data; cannot rebuild safely")

    new_pool = bytearray()
    rel_map = {}
    for rel, raw in sorted(translations.items()):
        rel_map[rel] = pool_start_rel + len(new_pool)
        if rel_map[rel] > 0xFFFF:
            raise ValueError(f"{sec.label}: 16-bit pointer overflow")
        new_pool += raw + b"\x00"

    if len(new_pool) > pool_size:
        over = len(new_pool) - pool_size
        return False, f"pool overflow by {over} bytes ({len(new_pool)} > {pool_size})"

    buf[pool_start : sec.end] = new_pool + b"\x00" * (pool_size - len(new_pool))

    for pos, rel in pointers:
        struct.pack_into("<H", buf, pos, rel_map[rel])

    return True, f"{patched_fields} fields, pool {len(new_pool)}/{pool_size} bytes"


def cmd_patch(bin_path: str, json_path: str, out_path: str, char_table_path: str | None = None) -> None:
    protected = [Path(bin_path), Path(json_path), Path(json_path).parent / "char_table.json"]
    if char_table_path:
        protected.append(Path(char_table_path))
    if any(Path(out_path).resolve() == path.resolve() or
           (Path(out_path).exists() and path.exists() and Path(out_path).samefile(path))
           for path in protected):
        raise ValueError("output must not overwrite an input file")
    buf = bytearray(Path(bin_path).read_bytes())
    validate_sections(buf)

    with open(json_path, encoding="utf-8-sig") as f:
        trans = json.load(f)
    if not isinstance(trans, dict) or not trans:
        raise ValueError("translation JSON must be a nonempty section object")
    unknown = set(trans) - {sec.label for sec in SECTIONS}
    if unknown:
        raise ValueError(f"unknown JSON sections: {sorted(unknown)}")

    if char_table_path is None:
        auto = Path(json_path).parent / "char_table.json"
        char_table_path = str(auto) if auto.exists() else None
    char_table = load_char_table(char_table_path)

    print(f"[patch] {bin_path} ({len(buf):,} bytes)")
    if char_table_path:
        print(f"[patch] char_table: {char_table_path} ({len(char_table):,} mappings)")

    changed_sections = 0
    errors = []
    for sec in SECTIONS:
        if sec.label not in trans:
            print(f"  [{sec.label:<20}] missing in json, skipped")
            continue
        sec_json = trans[sec.label]
        try:
            changed, message = patch_section(buf, sec, sec_json, char_table)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if changed:
            changed_sections += 1
            print(f"  [{sec.label:<20}] patched: {message}")
        elif message != "unchanged":
            errors.append(f"{sec.label}: {message}")
            print(f"  [{sec.label:<20}] error: {message}")
        else:
            print(f"  [{sec.label:<20}] unchanged")

    if errors:
        print("[patch] failed; output was not written")
        for error in errors:
            print(f"  - {error}")
        sys.exit(2)

    if len(buf) != struct.unpack_from("<I", buf, 4)[0]:
        raise AssertionError("file size changed unexpectedly")

    Path(out_path).write_bytes(buf)
    print(f"[patch] wrote {out_path} ({changed_sections} changed sections)")


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "extract":
        cmd_extract(sys.argv[2], sys.argv[3])
    elif cmd in ("patch", "insert"):
        out_path = sys.argv[4] if len(sys.argv) > 4 else "batdat_patched.bin"
        char_table_path = sys.argv[5] if len(sys.argv) > 5 else None
        cmd_patch(sys.argv[2], sys.argv[3], out_path, char_table_path)
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
