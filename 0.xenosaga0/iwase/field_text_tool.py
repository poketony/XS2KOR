"""Edit original-layout UTF-8 TXT, then rebuild checked game-encoded TXT."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "original"
REBUILT = ROOT / "rebuilt"
TABLE = ROOT.parent / "yosikawa/XENOSAGA KOR-JPN.json"
PROFILES = {
    "Citizen.txt": (50, {"Name": 16, "Sex": 4, "Age": 16, "Hobby": 16, "Profile": 150}),
    "QuestMes.txt": (36, {"Subject": 24, "Locate": 16, "Client": 16, "MesIPG": 256,
                          "MesSCD": 256, "MesFLD": 256, "UkunSCD": 1024, "UkunFLD": 1024}),
    "Treasure.txt": (18, {"Item": 16, "Locate": 16, "Expl": 120}),
}
SOURCE_HASHES = {
    "Citizen.txt": "1a4f1b06e2dc40398b7227d11db50396e3d76c7538c45e89ffbf8d61fc189b86",
    "QuestMes.txt": "b4b37a4c986a91a1eb9d45c9a034a9aa7fea0f26d5a89c2d63e18f4461fa8cff",
    "Treasure.txt": "c82c3f678a9ab4f727d4cfa337e1932628f9383bc1c4524e95d0fbbf18485107",
}
TABLE_HASH = "1f461ef9eee4b58a856ef214e80f38ffa84c24865bddd7a494662685f00ab007"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def output_path(path):
    path = Path(path).resolve()
    editable = {ROOT / name for name in PROFILES}
    if path not in editable and (path.parent != REBUILT or path.suffix.lower() != ".txt"):
        raise ValueError("Only editable TXT and TXT directly inside rebuilt may be written; originals are read-only")
    return path


def write_new(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def sjis_pair(lead, trail):
    """Exact byte arithmetic of SLPS-25368 function 0x23a728."""
    doubled = lead * 2
    first = doubled + (0x9f if (doubled & 0xfe) < 0x3f else 0x1f)
    if trail < 0x9f:
        second = trail + (0x61 if trail < 0x7f else 0x60)
    else:
        first += 1
        second = trail + 2
    return bytes((first & 255, second & 255))


INVERSE = {sjis_pair(a, b): bytes((a, b))
           for a in (*range(0x81, 0xa0), *range(0xe0, 0xf0))
           for b in (*range(0x40, 0x7f), *range(0x80, 0xfd))}


def runtime_text(raw):
    """Model quoted-field processing at 0x22a718, without unbounded memory reads."""
    result = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if raw[i:i+2] == b"\r\n":
            i += 2
        elif c in (32, 9):
            i += 1
        elif c >= 128:
            if i + 1 == len(raw):
                raise ValueError("Incomplete Shift-JIS pair")
            result += sjis_pair(c, raw[i+1])
            i += 2
        elif c == 92:
            if raw[i:i+2] != b"\\n":
                raise ValueError("Unsupported escape in source field")
            result.append(10)
            i += 2
        else:
            if c in (0, 13, 34):
                raise ValueError("Unexpected control/quote in field")
            result.append(c)
            i += 1
    return bytes(result)


def parse(data, filename):
    if filename not in PROFILES:
        raise ValueError("Only verified Citizen.txt, QuestMes.txt, Treasure.txt profiles are supported")
    count, limits = PROFILES[filename]
    text = data.decode("cp932")
    if text.encode("cp932") != data:
        raise ValueError("Non-roundtrippable CP932 source")
    records = list(re.finditer(r"(?m)^No\.(\d+)\s*\{(.*?)^\}", text, re.DOTALL))
    if len(records) != count:
        raise ValueError(f"Expected {count} records, found {len(records)}")
    fields = []
    for index, record in enumerate(records, 1):
        if int(record[1]) != index:
            raise ValueError("Record numbering differs from runtime order")
        matches = list(re.finditer(r'(?m)^[ \t]*([A-Za-z]+):"([^"]*)"', record[2]))
        if [m[1] for m in matches] != list(limits):
            raise ValueError(f"Unexpected fields/order at record {index}")
        for match in matches:
            start = record.start(2) + match.start(2)
            end = record.start(2) + match.end(2)
            offset = len(text[:start].encode("cp932"))
            raw = text[start:end].encode("cp932")
            runtime = runtime_text(raw)
            fields.append({"id": f"{index:02d}.{match[1]}", "offset": offset, "length": len(raw),
                           "max_runtime_bytes": limits[match[1]], "runtime_bytes": len(runtime),
                           "original": runtime.decode("euc_jis_2004")})
    return fields


def encode_translation(text, table, spaces="fullwidth"):
    if not isinstance(text, str) or any(c in text for c in ('"', "\\", "\r", "\t", "\0")):
        raise ValueError("Use actual line breaks; quotes, backslashes, tabs and CR/NUL are unsupported")
    if " " in text:
        if spaces == "fullwidth":
            text = text.replace(" ", "\u3000")
        else:
            raise ValueError("Game discards ASCII spaces; use fullwidth spaces or --spaces fullwidth")
    converted = "".join(table.get(c, c) for c in text)
    desired = converted.encode("euc_jis_2004")
    result = bytearray()
    i = 0
    while i < len(desired):
        c = desired[i]
        if c == 10:
            result += b"\\n"
            i += 1
        elif 0x21 <= c <= 0x7e and c not in (34, 92):
            result.append(c)
            i += 1
        else:
            pair = desired[i:i+2]
            if len(pair) != 2 or pair not in INVERSE:
                raise ValueError(f"Not representable by the game's two-byte converter: {pair.hex()}")
            result += INVERSE[pair]
            i += 2
    if runtime_text(result) != desired:
        raise AssertionError("Runtime conversion did not reproduce intended font codes")
    return bytes(result), len(desired)


def rebuild_bytes(data, filename, translations, table, spaces="fullwidth"):
    fields = parse(data, filename)
    expected = {f["id"] for f in fields}
    if not isinstance(translations, dict) or set(translations) != expected:
        raise ValueError("Missing or extra translation IDs")
    changes = []
    for field in fields:
        new = translations[field["id"]]
        if new == field["original"]:
            continue
        try:
            raw, size = encode_translation(new, table, spaces)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"{field['id']}: {exc}") from exc
        if size > field["max_runtime_bytes"]:
            raise ValueError(f"{field['id']}: {size} bytes exceeds {field['max_runtime_bytes']}")
        changes.append((field, raw, size))
    out = bytearray(data)
    for field, raw, _ in reversed(changes):
        start = field["offset"]
        out[start:start+field["length"]] = raw
    # Verify record structure and runtime bytes after all replacements.
    after = parse(bytes(out), filename)
    for field, _, size in changes:
        changed = next(f for f in after if f["id"] == field["id"])
        if changed["runtime_bytes"] != size:
            raise AssertionError("Rebuilt field differs from planned runtime length")
    return bytes(out), [{"id": f["id"], "runtime_bytes": n, "limit": f["max_runtime_bytes"]}
                        for f, _, n in changes]


def load_table(path):
    raw = path.read_bytes()
    table = json.loads(raw.decode("utf-8-sig"))["replace-table"]
    if not isinstance(table, dict) or any(len(k) != 1 or not isinstance(v, str) or not v
                                          for k, v in table.items()):
        raise ValueError("Invalid translation table")
    return table, sha(raw)


def checked_source(source):
    data = source.read_bytes()
    if source.name not in SOURCE_HASHES or sha(data) != SOURCE_HASHES[source.name]:
        raise ValueError(f"Source changed or unsupported: {source.name}")
    return data


def text_fields(text, filename):
    count, limits = PROFILES[filename]
    records = list(re.finditer(r"(?m)^No\.(\d+)\s*\{(.*?)^\}", text, re.DOTALL))
    if len(records) != count or [int(r[1]) for r in records] != list(range(1, count+1)):
        raise ValueError("Record count or No. order changed")
    fields = []
    for record in records:
        matches = list(re.finditer(r'(?m)^[ \t]*([A-Za-z]+):"([^"]*)"', record[2]))
        if [m[1] for m in matches] != list(limits):
            raise ValueError(f"Field names/order or quotes changed at No.{record[1]}")
        for match in matches:
            fields.append({"id": f"{int(record[1]):02d}.{match[1]}", "value": match[2],
                           "start": record.start(2)+match.start(2), "end": record.start(2)+match.end(2)})
    return fields


def text_layout(text, fields):
    pieces, end = [], 0
    for field in fields:
        pieces.append(text[end:field["start"]])
        pieces.append("\0")
        end = field["end"]
    pieces.append(text[end:])
    return re.sub(r"\s+", "", "".join(pieces))


def rebuild_text(data, filename, text, table, spaces="fullwidth"):
    original = data.decode("cp932").replace("\r\n", "\n")
    text = text.replace("\r\n", "\n")
    before, after = text_fields(original, filename), text_fields(text, filename)
    if text_layout(original, before) != text_layout(text, after):
        raise ValueError("Edit only quoted text; record labels, comments and other structure must stay unchanged")
    edits = {f["id"]: f["original"] for f in parse(data, filename)}
    for old, new in zip(before, after):
        if old["value"] == new["value"]:
            continue
        # Physical lines/indentation are source formatting; literal \\n is a game line break.
        value = re.sub(r"\n[ \t]*", "", new["value"])
        edits[new["id"]] = value.replace("\\n", "\n")
    return rebuild_bytes(data, filename, edits, table, spaces)


def export(source):
    data = checked_source(source)
    destination = output_path(ROOT / source.name)
    if destination.exists():
        raise ValueError("Editable TXT exists; existing translations are never overwritten")
    parse(data, source.name)
    write_new(destination, data.decode("cp932").encode("utf-8-sig"))
    return {"translation": str(destination), "encoding": "UTF-8"}


def plan_rebuild(translation, destination, spaces="fullwidth"):
    translation = Path(translation).resolve()
    if translation.parent != ROOT or translation.name not in PROFILES:
        raise ValueError("Use Citizen.txt, QuestMes.txt or Treasure.txt beside this tool")
    destination = output_path(destination)
    if destination.parent != REBUILT or destination.suffix.lower() != ".txt":
        raise ValueError("Rebuilt TXT must be directly inside rebuilt")
    data = checked_source(ORIGINAL / translation.name)
    table, table_hash = load_table(TABLE)
    if table_hash != TABLE_HASH:
        raise ValueError("Font mapping changed from the verified table")
    text = translation.read_bytes().decode("utf-8-sig")
    built, report = rebuild_text(data, translation.name, text, table, spaces)
    if report and destination.exists():
        raise ValueError("Output TXT exists; existing results are never overwritten")
    return destination, built, {"file": translation.name, "changed_fields": len(report),
                                "output": str(destination) if report else None}


def rebuild(translation, destination, spaces="fullwidth"):
    destination, built, result = plan_rebuild(translation, destination, spaces)
    if result["changed_fields"]:
        write_new(destination, built)
    return result


def rebuild_all(spaces="fullwidth"):
    plans = [plan_rebuild(ROOT / name, REBUILT / name, spaces) for name in PROFILES]
    for destination, built, result in plans:
        if result["changed_fields"]:
            write_new(destination, built)
    return [result for _, _, result in plans]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("source", type=Path)
    sub.add_parser("export-all", help="Export the three verified original TXT files")
    build = sub.add_parser("rebuild")
    build.add_argument("translation", type=Path)
    build.add_argument("destination", type=Path, nargs="?")
    build.add_argument("--spaces", choices=("reject", "fullwidth"), default="fullwidth",
                       help="ASCII space policy (default: fullwidth)")
    all_build = sub.add_parser("rebuild-all", help="Rebuild changed editable TXT files after checking all three")
    all_build.add_argument("--spaces", choices=("reject", "fullwidth"), default="fullwidth")
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export(args.source)
        elif args.command == "export-all":
            for name in PROFILES:
                if (ROOT / name).exists():
                    raise ValueError("Editable TXT exists; existing translations are never overwritten")
                parse(checked_source(ORIGINAL / name), name)
            result = [export(ORIGINAL / name) for name in PROFILES]
        elif args.command == "rebuild-all":
            result = rebuild_all(args.spaces)
        else:
            result = rebuild(args.translation, args.destination or REBUILT / args.translation.with_suffix(".txt").name, args.spaces)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, UnicodeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
