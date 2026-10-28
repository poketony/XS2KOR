"""Original-based XS2 ELF text export/rebuild; never patches input files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "discmetadata"
ORIGINALS = {
    1: METADATA / "slps_253.68.original",
    2: METADATA / "slps_253.68.original",
}
ELF_NAMES = {1: "slps_253.68", 2: "SLPS_253.69"}
ELF_HASH = "4e6b7fa2d0fb21af561519d914cf5b84598b6ad8622412a0daa6963825942c75"
OV02_HASH = "1d61c26c667bb9e4b6187cede22b386d6062d805b141711b921682162ae4fdaf"
TABLE = ROOT / "0.xenosaga0/yosikawa/XENOSAGA KOR-JPN.json"
TABLE_HASH = "1f461ef9eee4b58a856ef214e80f38ffa84c24865bddd7a494662685f00ab007"
ENCODING = "euc_jis_2004"
OV02_MAIN_START, OV02_MAIN_END = 0x57A000, 0x57DE08
OV02_BIAS = 0x579000
MENU_TEXT = ROOT / "0.xenosaga0/yosikawa/MainMenu.txd.txt"
MENU_FONT = ROOT / "폰트 관련/font1test/font1.tex"

# Byte-identical to the final 2026-10-04 spacing patch, including both icon
# spacer boundaries. File offsets, not virtual addresses; no assembler needed.
SPACING_PATCHES = (
    (0x64, "9819190098191900", "001a1900001a1900"),
    (0x15E560,
     "20ad85272c00838c0000a0a4000062900000a2a0000060a0010062800300401001006490"
     "0100a4a0010060a00800e003000000000000000020ad82272c00858c000043900000a3a0"
     "0100448002008010010042900100a2a00800e00300000000",
     "2d58e00366c60a0c20ad85270000a0a4000062900000a2a0000060a00100628003004010"
     "010064900100a4a0010060a0080060012df860012d58e00366c60a0c20ad822700004690"
     "0100449002008010000066a0010064a0080060012df86001"),
    (0x192998, "00" * 104,
     "2c00838c2800888c2b400301120000112d406000ffff0691200007240400c750ffff0825"
     "3b0007240b00c714000000002b00093cf819293508002a2500000691000027910400c714"
     "01002925fbff2a15010008252d1800010800e0030000000020247878312c343b"),
)
# Prior fixed coordinates and six-byte slot prefixes require matching menu data.
MENU_PATCHES = (
    (0x233B68, 0x90430008, 0x90430005),
    (0x233B70, 0xA2030008, 0xA2030005),
    (0x233B74, 0x90440009, 0),
    (0x233B78, 0xA2040009, 0),
    (0x2370AC, 0x240400B8, 0x240400D3),
    (0x237174, 0x240400B8, 0x240400D3),
    (0x2370DC, 0x24050133, 0x24050123),
    (0x2371A4, 0x24050133, 0x24050123),
    (0x23710C, 0x24050132, 0x24050122),
    (0x2371D4, 0x24050132, 0x24050122),
    (0x23712C, 0x24040070, 0x24040053),
    (0x2371F4, 0x240400B0, 0x2404009E),
)
SLOT_IDS = {218, 220, 221, 222, 223, 514, 515, 539, 541, 543, 544, 545, 546, 547, 548}
DYNAMIC_SLOT_IDS = {218, 539, 541, 543, 544, 545, 546, 547, 548}
# Reviewed text-only calls: confirmation body and save/load status labels.
# Never include character names, play times, numeric fields or image widgets.
STATUS_SPACE_CALLS = (0x233AB4, 0x2347B0, 0x234B80, 0x234BC8, 0x234BEC, 0x234C54)
MENU_ITEM_NAME_SPACE_CALLS = (0x21B858,)
ITEM_COUNT_WRITES = {
    0x5651C0: ((0x0CFF24, 0xA2C30002), (0x0CFF2C, 0xA2C60003)),
    0x578D98: ((0x0DBF54, 0xA0430002), (0x0DBF5C, 0xA0470003)),
}
CAMPAIGN_NUMBER = 0x2C9348
CAMPAIGN_NUMBER_WRITES = ((0x10B570, 0xA0C30012), (0x10B57C, 0xA0C20013))
SHORT_ITEM = 0x578D98
SHORT_ITEM_STORAGE = 0x578D96

# Reviewed NUL-terminated movie labels: each has one independent LUI/ADDIU
# address construction, used by both the width and draw calls at 0x1D71F0.
# Other strings may have interior references or fixed-size copies; fail closed.
MAIN_FILE_BIAS = 0x11F000
RELOCATION_GROUPS = ((0x1945E0, 0x194610),)
RELOCATION_REFERENCES = {
    0x1945E0: (0x0B8200, 0x3C10002B, 0x261035E0),
    0x194610: (0x0B8238, 0x3C10002B, 0x26103610),
}
# File-backed alignment gap, not zero-initialized variables or a code cave.
# Leave the four bytes after .lit4 untouched; align string starts to eight.
RELOCATION_POOLS = ((0x5773B0, 0x577400),)

# Reviewed display strings, not a blind scan of code, pointers or font tables.
GROUPS = {
    "UI / movie pause": (0x1945E0, 0x194610),
    "UI / disc change": (0x199CD0, 0x199CF8, 0x199D30, 0x199D70),
    "UI / movie error and save": (0x564DA0, 0x564E20, 0x564E98, 0x564F40, 0x564F90),
    "UI / item message fragments": (0x5651C0, 0x578D98, 0x579670, 0x579678),
    "UI / segment address": (0x5666F0, 0x566718, 0x566748),
    "UI / quest and citizen help": (0x5669F0, 0x566B48, 0x566B58, 0x566BB8,
                                    0x566C88, 0x566E68, 0x5791C8, 0x5791D8,
                                    0x5791E0, 0x5791E8),
    "UI / shop": (0x568F48, 0x568F78, 0x569010, 0x579640, 0x579648),
    "UI / HDD read error": (0x569CE8,),
    "UI / campaign number": (CAMPAIGN_NUMBER,),
    "Optional / English UI": (
        0x566438, 0x566448, 0x566458, 0x566468, 0x566478, 0x566498,
        0x5664B0, 0x5664C0, 0x5664D0, 0x5664E0, 0x566500, 0x566510,
        0x566530, 0x566540, 0x566550, 0x566560, 0x566570, 0x566580, 0x566590,
        0x566AB0, 0x566DD0, 0x566DE0, 0x566DF0, 0x566EE0, 0x566F00,
        0x5670B8, 0x5670C8, 0x567398, 0x578EA8, 0x578EC0,
        0x578F80, 0x578F88, 0x578F90, 0x578F98, 0x578FA0, 0x578FA8,
        0x578FB0, 0x578FB8, 0x578FC0, 0x578FC8, 0x578FD0, 0x578FD8,
        0x578FE8, 0x578FF0, 0x578FF8, 0x579000, 0x579010, 0x579018,
        0x579020, 0x579028, 0x579030, 0x579038, 0x579058, 0x579068,
        0x579070, 0x579078, 0x579080, 0x579088, 0x579090, 0x579098,
        0x5790A0, 0x5790A8, 0x5790B8, 0x5791A0, 0x5791A8,
        0x5791B0, 0x5791B8, 0x5791C0, 0x579238, 0x579240,
        0x579248, 0x5792B0, 0x5792B8, 0x5792C8, 0x5792D0,
        0x5792D8, 0x5792E0, 0x5792E8, 0x579310, 0x579318,
        0x579320, 0x579328, 0x579330, 0x579338, 0x579340,
        0x579348, 0x579350, 0x579398, 0x5793A0, 0x5793B0,
    ),
    "Debug / Japanese test and diagnostics": (
        0x2C98D0, 0x2C98E8, 0x2C9900, 0x2C9A80, 0x2C9A90, 0x2C9AA8,
        0x2C9AC0, 0x2C9AD8, 0x2C9AF0, 0x2C9B08, 0x5620B8, 0x5620C8,
        0x564CE0, 0x564D08, 0x565490, 0x5654A0, 0x5654B0, 0x5654C0,
        0x5654F0, 0x565850, 0x565868, 0x565878, 0x565888, 0x565898,
        0x565920, 0x565970, 0x5659C0, 0x5674D0, 0x5674E0, 0x5674F0,
        0x567508, 0x567518, 0x567528, 0x567538, 0x567568, 0x567588,
        0x5682E0, 0x56B440, 0x578DB0, 0x578DB8, 0x578DC0, 0x578DE0,
        0x5793C0, 0x579400,
    ),
    "OV02 / memory card": (0x57D180, 0x57D1E8, 0x57D238, 0x57D280,
                              0x57D2C8, 0x57D320, 0x57D380),
    "OV02 / HDD": (0x57D518, 0x57D608, 0x57D628, 0x57D7A8, 0x57D818,
                      0x57D860, 0x57D8D0, 0x57D948, 0x57D998, 0x57DA28,
                      0x57DB18, 0x57DB50, 0x57DB80, 0x57DBC8),
    "OV02 / English installation status": (
        0x57D578, 0x57D588, 0x57D5A0, 0x57D5B0, 0x57D5D0,
        0x57D5E8, 0x57D5F8, 0x57D690, 0x57D6A8, 0x57D6B8,
        0x57D6C8, 0x57D6E0, 0x57D6F0, 0x57D718,
    ),
}
PRINTF = re.compile(rb"%(?:%|[-+ #0]*\d*(?:\.\d+)?[diuoxXfFeEgGcs])")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def control_size(data, offset, limit):
    code = data[offset]
    sizes = {0x09: 1, 0x0A: 1, 0x0B: 1, 0x0C: 4, 0x0D: 2,
             0x0E: 6, 0x0F: 4, 0x19: 2, 0x1B: 2}
    if code in (8, 0x15):
        if offset + 2 > limit:
            raise ValueError("Truncated control")
        operand = data[offset + 1]
        if code == 8:
            size = 2 + sum(n for bit, n in ((1, 2), (2, 2), (4, 3), (8, 2), (16, 2))
                           if operand & bit)
        else:
            size = 2 + operand
    elif code in sizes:
        size = sizes[code]
    else:
        raise ValueError(f"Unsupported control {code:02X}")
    if offset + size > limit:
        raise ValueError("Truncated control")
    return size


def read_string(data, offset, limit):
    """Decode one whole control-aware string, including zero-valued operands."""
    start, display, controls = offset, [], []
    while offset < limit:
        byte = data[offset]
        if byte == 0:
            return data[start:offset], "".join(display), tuple(controls)
        if byte < 32:
            size = control_size(data, offset, limit)
            packet = data[offset:offset + size]
            if byte == 10:
                display.append("\\n")
            else:
                controls.append(packet)
                display.append("{" + packet.hex().upper() + "}")
        elif 32 <= byte < 127:
            size = 1
            display.append("\\\\" if byte == 92 else "{" + f"{byte:02X}" + "}"
                           if byte in (123, 125) else chr(byte))
        else:
            size = 3 if byte == 0x8F else 2
            if offset + size > limit:
                raise ValueError("Truncated character")
            raw = data[offset:offset + size]
            text = raw.decode(ENCODING)
            if text.encode(ENCODING) != raw:
                raise ValueError("Non-roundtripping character")
            display.append(text)
        offset += size
    raise ValueError("Missing string terminator")


def encode_text(text, table):
    out, literal = bytearray(), []

    def flush():
        converted = "".join(table.get(c, c) for c in literal)
        if re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]", converted):
            raise ValueError("Korean character missing from font mapping")
        out.extend(converted.encode(ENCODING))
        literal.clear()

    index = 0
    while index < len(text):
        c = text[index]
        if c == "{":
            end = text.find("}", index + 1)
            token = text[index + 1:end] if end >= 0 else ""
            if not token or len(token) % 2 or not re.fullmatch(r"[0-9a-fA-F]+", token):
                raise ValueError("Malformed {HEX} token")
            flush()
            out.extend(bytes.fromhex(token))
            index = end + 1
        elif c == "\\":
            if index + 1 >= len(text) or text[index + 1] not in ("n", "\\"):
                raise ValueError("Use \\n for 0A or \\\\ for a literal backslash")
            flush()
            out.append(10 if text[index + 1] == "n" else 92)
            index += 2
        elif c == "}" or ord(c) < 32:
            raise ValueError("Unexpected brace or literal control character")
        else:
            literal.append(c)
            index += 1
    flush()
    return bytes(out)


@dataclass(frozen=True)
class Record:
    offset: int
    group: str
    raw: bytes
    text: str
    controls: tuple
    capacity: int
    write_offset: int


def records(data):
    if sha(data) != ELF_HASH:
        raise ValueError("Not the approved original ELF; patched/translated inputs are rejected")
    result = []
    seen = set()
    for group, offsets in GROUPS.items():
        for offset in offsets:
            if offset in seen:
                raise ValueError("Duplicate catalogue offset")
            seen.add(offset)
            raw, text, controls = read_string(data, offset, offset + 2048)
            if encode_text(text, {}) != raw:
                raise ValueError(f"Original decode mismatch at {offset:08X}")
            # Default to the current 8-byte-aligned literal slot. The reviewed
            # short-item exception below also uses two preceding padding bytes.
            stop = offset + len(raw) + 1
            boundary = (stop + 7) & ~7
            while stop < boundary and data[stop] == 0:
                stop += 1
            capacity = stop - offset - 1
            write_offset = offset
            if offset == SHORT_ITEM:
                # Keep Dummy's NUL at ...D95 and the referenced empty string at
                # ...DA0. The translated fragment shares that empty string's NUL.
                if data[0x578D90:0x578DA1] != b"Dummy\0\0\0" + raw + b"\0\0\0":
                    raise ValueError("Short item message storage layout changed")
                write_offset, capacity = SHORT_ITEM_STORAGE, 10
            result.append(Record(offset, group, raw, text, controls, capacity, write_offset))
    ordered = sorted(result, key=lambda r: r.write_offset)
    for left, right in zip(ordered, ordered[1:]):
        if left.write_offset + left.capacity + 1 > right.write_offset:
            raise ValueError("Overlapping literal slots")
    return result


def load_table():
    raw = TABLE.read_bytes()
    # Fail closed when the shared font layout changes, rather than corrupt text.
    if sha(raw) != TABLE_HASH:
        raise ValueError("Font mapping changed; revalidate before rebuilding")
    table = json.loads(raw.decode("utf-8-sig"))["replace-table"]
    if not isinstance(table, dict) or any(len(k) != 1 or not isinstance(v, str) for k, v in table.items()):
        raise ValueError("Invalid font mapping")
    return table


def export_text(data):
    lines = ["# XS2 ELF text v1", f"# origin-sha256: {ELF_HASH}",
             "# offset|original bytes/in-place bytes|text",
             "# Edit only text after the second |. Missing rows remain original.",
             "# \\n = line break (0A); \\\\n = literal backslash+n; {HEX} = control packet.",
             "# Keep {HEX}, % formats and &03/&04. Item counts require one ASCII 00.",
             "# English/debug groups are optional. Rebuild includes runtime fixes, not the font asset.",
             "# Reviewed movie labels can relocate on overflow; other rows retain their limits."]
    group = None
    for rec in records(data):
        if group != rec.group:
            group = rec.group
            lines.extend(["", "# " + group])
        if rec.offset in ITEM_COUNT_WRITES:
            lines.append("# COUNT: Keep exactly one ASCII 00. Rebuild adjusts the numeric write position.")
        if rec.offset == CAMPAIGN_NUMBER:
            lines.append("# CAMPAIGN: Keep exactly one ASCII 01. Rebuild adjusts the numeric write position.")
        lines.append(f"{rec.offset:08X}|{len(rec.raw)}/{rec.capacity}|{rec.text}")
    return "\n".join(lines) + "\n"


def parse_edits(text, catalog):
    edits = {}
    headers = []
    for number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        if line.startswith("# origin-sha256: "):
            headers.append(line.split(": ", 1)[1])
        if not line or line.startswith("#"):
            continue
        fields = line.split("|", 2)
        if len(fields) != 3 or not re.fullmatch(r"[0-9A-Fa-f]{8}", fields[0]):
            raise ValueError(f"Line {number}: malformed record")
        offset = int(fields[0], 16)
        if offset in edits or offset not in catalog:
            raise ValueError(f"Line {number}: duplicate or unknown offset {offset:08X}")
        rec = catalog[offset]
        if fields[1] != f"{len(rec.raw)}/{rec.capacity}":
            raise ValueError(f"Line {number}: original length/capacity was changed")
        edits[offset] = fields[2]
    if headers != [ELF_HASH]:
        raise ValueError("Missing, duplicated or wrong original hash header")
    return edits


def printf_tokens(raw):
    tokens = PRINTF.findall(raw)
    if b"%" in PRINTF.sub(b"", raw):
        raise ValueError("Unrecognized printf format; keep original % placeholders")
    return tokens


def item_count_position(raw):
    matches = list(re.finditer(rb"(?<![0-9])00(?![0-9])", raw))
    if len(matches) != 1 or raw.count(b"00") != 1:
        raise ValueError("Item count requires exactly one standalone ASCII 00")
    return matches[0].start()


def campaign_number_position(raw):
    matches = list(re.finditer(rb"(?<![0-9])01(?![0-9])", raw))
    if len(matches) != 1 or raw.count(b"01") != 1:
        raise ValueError("Campaign number requires exactly one standalone ASCII 01")
    return matches[0].start()


def validate_relocation_layout(data, catalog):
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    programs = [struct.unpack_from("<8I", data, header[5] + i * header[9])
                for i in range(header[10])]
    sections = [struct.unpack_from("<10I", data, header[6] + i * header[11])
                for i in range(header[12])]
    for start, end in RELOCATION_POOLS:
        if start % 8 or not start < end <= len(data) or any(data[start:end]):
            raise ValueError("Relocation pool layout changed")
        va, limit = start + MAIN_FILE_BIAS, end + MAIN_FILE_BIAS
        if not any(p[0] == 1 and p[1] <= start and end <= p[1] + p[4]
                   and p[2] - p[1] == MAIN_FILE_BIAS for p in programs):
            raise ValueError("Relocation pool is not file-backed loadable memory")
        if any(s[2] & 2 and s[3] < limit and va < s[3] + s[5] for s in sections):
            raise ValueError("Relocation pool overlaps an allocated section")
        if any(r.write_offset < end and start < r.write_offset + r.capacity + 1
               for r in catalog.values()):
            raise ValueError("Relocation pool overlaps a text slot")
        if any(off < end and start < off + len(after) for off, _before, after in runtime_regions()):
            raise ValueError("Relocation pool overlaps a runtime patch")
    for group in RELOCATION_GROUPS:
        for left, right in zip(group, group[1:]):
            if catalog[left].write_offset + catalog[left].capacity + 1 != right:
                raise ValueError("Relocation group is not contiguous")
        for offset in group:
            address, hi, lo = RELOCATION_REFERENCES[offset]
            if struct.unpack_from("<II", data, address) != (hi, lo):
                raise ValueError(f"Relocation reference changed at {address:08X}")
            low = lo & 0xFFFF
            va = ((hi & 0xFFFF) << 16) + (low - 0x10000 if low & 0x8000 else low)
            if va != offset + MAIN_FILE_BIAS:
                raise ValueError("Relocation reference does not match its string")


def plan_text_layout(data, catalog, changes):
    positions = {r.offset: r.write_offset for r, _raw in changes}
    regions = {r.offset: (r.write_offset, r.write_offset + r.capacity + 1) for r, _raw in changes}
    payloads = {r.offset: r.raw for r in catalog.values()}
    payloads.update((r.offset, raw) for r, raw in changes)
    overflow = {r.offset for r, raw in changes if len(raw) > r.capacity}
    for offset in sorted(overflow):
        if offset not in RELOCATION_REFERENCES:
            raise ValueError(f"{offset:08X}: {len(payloads[offset])} bytes exceeds {catalog[offset].capacity}; "
                             "relocation references are not reviewed for this row")
    if not overflow:
        return positions, list(regions.values()), payloads
    validate_relocation_layout(data, catalog)
    pools = [list(bounds) for bounds in RELOCATION_POOLS]
    extra_regions = []
    for group in RELOCATION_GROUPS:
        if not overflow.intersection(group):
            continue
        start = catalog[group[0]].write_offset
        end = catalog[group[-1]].write_offset + catalog[group[-1]].capacity + 1
        cursor, packed = start, {}
        for offset in group:
            cursor = (cursor + 7) & ~7
            packed[offset] = cursor
            cursor += len(payloads[offset]) + 1
        if cursor <= end:
            # Borrow adjacent savings only inside this reviewed literal block.
            positions.update(packed)
            for offset in group:
                regions.pop(offset, None)
            extra_regions.append((start, end))
            continue
        for offset in group:
            if offset not in overflow:
                continue
            needed = len(payloads[offset]) + 1
            for pool in pools:
                destination = (pool[0] + 7) & ~7
                if destination + needed <= pool[1]:
                    positions[offset] = destination
                    regions.pop(offset, None)
                    extra_regions.append((destination, destination + needed))
                    pool[0] = destination + needed
                    break
            else:
                raise ValueError(f"{offset:08X}: {len(payloads[offset])} bytes exceeds local storage; "
                                 "reviewed relocation pools have no contiguous space (including NUL/alignment)")
    return positions, list(regions.values()) + extra_regions, payloads


def relocation_moves(built):
    moves = []
    for offset, (address, hi, lo) in RELOCATION_REFERENCES.items():
        new_hi, new_lo = struct.unpack_from("<II", built, address)
        if new_hi & 0xFFFF0000 != hi & 0xFFFF0000 or new_lo & 0xFFFF0000 != lo & 0xFFFF0000:
            raise ValueError("Unexpected relocation instruction")
        low = new_lo & 0xFFFF
        destination = ((new_hi & 0xFFFF) << 16) + (low - 0x10000 if low & 0x8000 else low) - MAIN_FILE_BIAS
        if destination != offset:
            moves.append((offset, destination))
    return moves


def build(data, text, table):
    catalog = {r.offset: r for r in records(data)}
    edits = parse_edits(text, catalog)
    changes = []
    for offset, translated in edits.items():
        rec = catalog[offset]
        if translated == rec.text:
            continue
        try:
            raw = encode_text(translated, table)
            parsed, _display, controls = read_string(raw + b"\0", 0, len(raw) + 1)
        except ValueError as error:
            raise ValueError(f"{offset:08X}: {error}") from error
        if parsed != raw:
            raise ValueError(f"{offset:08X}: embedded terminator or malformed control")
        if controls != rec.controls:
            raise ValueError(f"{offset:08X}: control packets changed, missing or reordered")
        if printf_tokens(raw) != printf_tokens(rec.raw):
            raise ValueError(f"{offset:08X}: printf placeholders changed")
        if re.findall(rb"&\d{2}", raw) != re.findall(rb"&\d{2}", rec.raw):
            raise ValueError(f"{offset:08X}: button placeholders changed")
        if re.findall(rb"\$[^;]*;", raw) != re.findall(rb"\$[^;]*;", rec.raw):
            raise ValueError(f"{offset:08X}: inline font commands changed")
        if offset in ITEM_COUNT_WRITES:
            try:
                item_count_position(raw)
            except ValueError as error:
                raise ValueError(f"{offset:08X}: {error}") from error
        if offset == CAMPAIGN_NUMBER:
            try:
                campaign_number_position(raw)
            except ValueError as error:
                raise ValueError(f"{offset:08X}: {error}") from error
        if raw != rec.raw:
            changes.append((rec, raw))
    positions, regions, payloads = plan_text_layout(data, catalog, changes)
    result = bytearray(data)
    for start, end in regions:
        result[start:end] = bytes(end - start)
    for offset, destination in positions.items():
        raw = payloads[offset]
        result[destination:destination + len(raw) + 1] = raw + b"\0"
        if offset in RELOCATION_REFERENCES and destination != offset:
            address, hi, lo = RELOCATION_REFERENCES[offset]
            va = destination + MAIN_FILE_BIAS
            # ADDIU sign-extends its low half, so the high half needs carry.
            struct.pack_into("<II", result, address,
                             (hi & 0xFFFF0000) | ((va + 0x8000) >> 16),
                             (lo & 0xFFFF0000) | (va & 0xFFFF))
    if len(result) != len(data):
        raise ValueError("ELF length changed")
    return bytes(result), [(r.offset, len(r.raw), len(raw)) for r, raw in changes]


def build_overlay(original, source_elf, built_elf):
    if sha(original) != OV02_HASH:
        raise ValueError("Not the approved original OV02")
    if original[0x1000:0x4E08] != source_elf[OV02_MAIN_START:OV02_MAIN_END]:
        raise ValueError("Original ELF and OV02 payloads differ")
    result = bytearray(original)
    result[0x1000:0x4E08] = built_elf[OV02_MAIN_START:OV02_MAIN_END]
    return bytes(result)


def save_title_regions():
    """Patch only the icon.sys prefix/line boundary, not the location composer."""
    # File-backed padding before .sbss, separate from the text relocation pool.
    # The helper uses 96 bytes at VA 0x698b10 and does not consume BSS memory.
    prefix = b"".join(
        b"\x81\x7c" if c == "-" else chr(ord(c) + 0xFEE0).encode("shift_jis")
        for c in "XenosagaEPII-00"
    ) + b"\0"
    words = (
        # Preserve the unused NULL-location path, including its old digit positions.
        0x16200003, 0x2402000A, 0x08067705, 0x2508FFFA,
        0x3C0A0069, 0x354A8B50,             # t2 = prefix
        0x250900C0,                         # t1 = icon.sys title
        0x91430000, 0x254A0001, 0xA1230000, # copy through NUL
        0x1460FFFC, 0x25290001,
        0x2402001E, 0xA5020006,             # newline after 15 SJIS characters
        0x08067705, 0x2402000A,             # resume 0x19dc14; displaced li v0,10
    )
    payload = (struct.pack("<16I", *words) + prefix).ljust(0x60, b"\0")
    return [
        (0x84, struct.pack("<I", 0x3E710C), struct.pack("<I", 0x3E7170)),
        (0x8C, struct.pack("<I", 6), struct.pack("<I", 7)),
        (0x579B10, bytes(0x60), payload),
        (0x7EC0C, struct.pack("<I", 0x2402000A), struct.pack("<I", 0x081A62C4)),
        (0x7EC34, struct.pack("<I", 0xA10300D7), struct.pack("<I", 0xA10300DD)),
        (0x7EC44, struct.pack("<I", 0xA10400D5), struct.pack("<I", 0xA10400DB)),
        (0x7EC4C, struct.pack("<I", 0x24F00118), struct.pack("<I", 0x24F0011E)),
    ]


def popup_font_regions():
    # File header padding is loaded after BSS by an independent RX segment.
    # Reserve it from BOTH kernel heap and libc sbrk; never map onto the stack.
    bias = 0x9D0000
    marker, select, metric = bias + 0x2F4, bias + 0x340, bias + 0x500
    table = bias + 0x700
    scope_callback, scope_queue, literals = bias + 0x780, bias + 0x7C0, bias + 0x980
    before = b"$init;$bg2;$shadow+1,+3,000000;\0"
    after = (b"\x0b\x0d\x02$shadow+1,+3,000000;\x17"
             + marker.to_bytes(3, "big") + b"\x01\x01\x01\x01\0")
    assert len(after) == len(before) and b"\0" not in after[:-1]

    def imm(op, rt, rs, value):
        return (op << 26) | (rs << 21) | (rt << 16) | (value & 0xFFFF)

    def reg(fn, rd, rs, rt=0, shift=0):
        return (rs << 21) | (rt << 16) | (rd << 11) | (shift << 6) | fn

    def jump(target):
        return 0x08000000 | (target >> 2)

    def assemble(start, rows):
        labels, words, fixups = {}, [], []
        for row in rows:
            if isinstance(row, str):
                labels[row] = start + 4 * len(words)
            elif isinstance(row, tuple):
                if len(row) == 2:
                    fixups.append((len(words), None, 0, 0, row[1]))
                    words.append(0)
                    continue
                op, rs, rt, label = row
                fixups.append((len(words), op, rs, rt, label))
                words.append(0)
            else:
                words.append(row)
        for index, op, rs, rt, label in fixups:
            if op is None:
                words[index] = jump(labels[label])
                continue
            delta = (labels[label] - (start + 4 * index + 4)) // 4
            assert -32768 <= delta <= 32767
            words[index] = imm(op, rt, rs, delta)
        return struct.pack('<' + 'I' * len(words), *words), labels

    # State bit 7 is private to this prefix. INIT and the normal flush clear it.
    # Low bits remain the original ruby/proportional/ASCII-large/small settings.
    mark, _ = assemble(marker, [
        imm(0x24, 2, 4, 0x18),           # lbu v0,mode(a0)
        imm(0x0C, 2, 2, 0xF0),         # same font reset + han1 as original prefix
        imm(0x0D, 2, 2, 0x84),         # popup marker + han1
        reg(8, 0, 31),                  # jr ra
        imm(0x28, 2, 4, 0x18),         # sb v0,mode(a0)
    ])
    # Draw the original quad/UV unchanged, then override its advance only.
    # Keep at least ten source pixels and retain the original left bearing;
    # Native right bounds + 4 cover bitmap differences and the shadow. Backtick
    # retains full advance: its normal/proportional glyphs have different bounds.
    choose, _ = assemble(select, [
        imm(9, 29, 29, -0x40),
        imm(0x3F, 21, 29, 0), imm(0x3F, 31, 29, 8),
        imm(0x3F, 4, 29, 0x10), imm(0x3F, 5, 29, 0x18),
        imm(0x3F, 6, 29, 0x20), imm(0x3F, 7, 29, 0x28),
        imm(0x2B, 0, 29, 0x30),
        imm(0x24, 2, 19, 0x18),         # lbu v0,mode(s3)
        imm(0x0C, 3, 2, 0x4F),         # bit 6: spaces only, not punctuation
        imm(9, 1, 0, 0x44),
        (5, 3, 1, 'punctuation'), 0,
        imm(9, 3, 0, 0x20),
        (5, 20, 3, 'punctuation'), 0,
        (4, 16, 0, 'space'), 0,        # literal ASCII space maps to glyph zero
        'punctuation',
        imm(0x0C, 3, 2, 0x8F),
        imm(9, 1, 0, 0x84),
        (5, 3, 1, 'normal'), 0,
        imm(9, 3, 20, -0x20),          # source byte in s4, not mapped glyph
        imm(0x0B, 1, 3, 0x5F),
        (4, 1, 0, 'normal'), 0,
        imm(15, 1, 0, table >> 16),
        imm(0x0D, 1, 1, table & 0xFFFF),
        reg(0x21, 1, 1, 20),
        imm(0x24, 3, 1, 0),
        (4, 3, 0, 'normal'), 0,
        imm(9, 3, 3, -1),
        (5, 3, 16, 'normal'), 0,        # do not treat #HH escapes as literal #
        imm(9, 1, 0, 0x20),
        (4, 20, 1, 'space'), 0,
        imm(9, 1, 0, 0x2C),
        (4, 20, 1, 'space'), 0,
        imm(9, 1, 0, 0x2E),
        (4, 20, 1, 'space'), 0,
        imm(15, 1, 0, 0x8D),
        imm(0x24, 2, 1, 0x470E),
        (5, 2, 0, 'normal'), 0,        # retain explicit xx3 fallback
        0x0C06626A, reg(0x21, 4, 16),  # jal 0x1989a8; move a0,s0
        reg(2, 2, 0, 2, 8),
        imm(0x0C, 2, 2, 0xFF),
        imm(9, 2, 2, 4),
        imm(0x0B, 1, 2, 10),
        (5, 1, 0, 'space'), 0,
        imm(0x0B, 1, 2, 20),
        (5, 1, 0, 'advance'), 0,
        ('j', 'advance'), imm(9, 2, 0, 19),
        'space',
        imm(9, 2, 0, 10),
        'advance',
        imm(0x21, 3, 19, 0x2C),
        reg(0x18, 0, 2, 3), reg(0x12, 2, 0),
        reg(3, 2, 0, 2, 4),
        imm(0x21, 21, 19, 8),
        reg(0x21, 21, 21, 2),
        imm(9, 2, 0, 1), imm(0x2B, 2, 29, 0x30),
        'normal',
        imm(0x37, 4, 29, 0x10), imm(0x37, 5, 29, 0x18),
        imm(0x37, 6, 29, 0x20), imm(0x37, 7, 29, 0x28),
        0x0C065C20, 0,                 # original glyph draw
        imm(0x23, 2, 29, 0x30),
        (4, 2, 0, 'return'), 0,
        imm(0x29, 21, 19, 8),
        'return',
        imm(0x37, 21, 29, 0), imm(0x37, 31, 29, 8),
        reg(8, 0, 31), imm(9, 29, 29, 0x40),
    ])
    # Popup width walker is leaf code. Only v0/v1/at are scratch at this point.
    # Use current font right bounds, just as the draw wrapper does. Scale the
    # actual 19-pixel large glyph, not the previous nominal 20-pixel estimate.
    measure, measure_labels = assemble(metric, [
        imm(15, 1, 0, table >> 16),
        imm(0x0D, 1, 1, table & 0xFFFF),
        reg(0x21, 1, 1, 3),
        imm(0x24, 2, 1, 0),
        (4, 2, 0, 'wide'), 0,
        imm(9, 1, 0, 0x20),
        (4, 3, 1, 'ten'), 0,
        imm(9, 1, 0, 0x2C),
        (4, 3, 1, 'ten'), 0,
        imm(9, 1, 0, 0x2E),
        (4, 3, 1, 'ten'), 0,
        imm(15, 1, 0, 0x8D),
        imm(0x24, 3, 1, 0x470E),
        (5, 3, 0, 'wide'), 0,
        imm(0x23, 3, 1, 0x4700),        # live font pointer
        imm(9, 2, 2, -1),
        reg(0, 2, 0, 2, 1),
        reg(0x21, 3, 3, 2),
        imm(15, 1, 0, 8),
        reg(0x21, 3, 3, 1),
        imm(0x25, 2, 3, -0x7FC0),
        imm(9, 2, 2, 0x1FF),
        imm(0x0C, 2, 2, 0xFFFF),
        imm(0x0C, 3, 2, 0xFF),
        imm(9, 1, 0, 0xFF),
        (4, 3, 1, 'fallback'), 0,
        reg(2, 2, 0, 2, 8),
        'right',
        imm(9, 2, 2, 4),
        imm(0x0B, 1, 2, 10),
        (5, 1, 0, 'ten'), 0,
        imm(0x0B, 1, 2, 20),
        (5, 1, 0, 'scale'), 0,
        ('j', 'wide'), 0,
        'fallback',
        imm(15, 1, 0, 0x8D),
        imm(0x24, 2, 1, 0x470C),
        ('j', 'right'), 0,
        'wide',
        ('j', 'scale'), imm(9, 2, 0, 19),
        'ten',
        ('j', 'scale'), imm(9, 2, 0, 10),
        'scale',
        imm(0x23, 3, 28, -0x5600),
        reg(0x18, 0, 2, 3), reg(0x12, 2, 0),
        jump(0x2310F8), 0,
    ])
    # Fixed slots keep the code's absolute entry points easy to audit.
    assert len(mark) <= select - marker and len(choose) <= metric - select
    assert len(measure) <= table - metric
    # ASCII remap -> glyph indices, checked against the original renderer in
    # CPU tests. Zero means retain the full-width path. All indices < 192.
    indices = {32: 1, 33: 10, 34: 41, 35: 84, 36: 80, 37: 83, 38: 85, 39: 39,
               40: 42, 41: 43, 42: 86, 43: 60, 44: 4, 45: 61, 46: 5, 47: 31,
               58: 7, 59: 8, 60: 67, 61: 65, 62: 68, 63: 9, 64: 87, 91: 46,
               92: 79, 93: 47, 94: 16, 95: 18, 123: 48, 124: 35,
               125: 49, 126: 33}
    glyphs = bytes(indices.get(c, 0) for c in range(128))
    # Scope only one queued name/help/status string. No parser changes to it,
    # no permanent global font mode and no marker on quantity/HP columns.
    callback, _ = assemble(scope_callback, [
        imm(0x24, 2, 4, 0x18), imm(0x0C, 3, 5, 0xC0), imm(0x0C, 1, 5, 1),
        (4, 1, 0, 'clear'), 0,
        reg(0x25, 2, 2, 3), ('j', 'store'), 0,
        'clear', reg(0x27, 3, 3), reg(0x24, 2, 2, 3),
        'store', reg(8, 0, 31), imm(0x28, 2, 4, 0x18),
    ])
    controls = b''.join(b'\x17' + scope_callback.to_bytes(3, 'big')
                        + bytes([mode, mode, mode, mode | enabled]) + b'\0'
                        for mode, enabled in ((0x40, 1), (0x40, 0), (0x80, 1), (0x80, 0)))

    def address(rt, value):
        return [imm(15, rt, 0, value >> 16), imm(0x0D, rt, rt, value & 0xFFFF)]

    queue, queue_labels = assemble(scope_queue, [
        # s0 is the reviewed caller's list descriptor. Only the battle overlay's
        # named rows with a separate quantity table qualify; other lists bypass.
        imm(0x23, 2, 16, 0x14), (4, 2, 0, 'original'), 0,
        imm(0x23, 2, 16, 0x10), *address(3, 0xAC3258),
        (5, 2, 3, 'original'), 0,
        'space',
        *address(8, literals), *address(9, literals + 9),
        ('j', 'common'), 0,
        'help', *address(8, literals + 18), *address(9, literals + 27),
        'common', imm(9, 29, 29, -0x40),
        imm(0x3F, 16, 29, 0), imm(0x3F, 17, 29, 8),
        imm(0x3F, 18, 29, 0x10), imm(0x3F, 19, 29, 0x18),
        imm(0x3F, 20, 29, 0x20), imm(0x3F, 31, 29, 0x28),
        reg(0x21, 16, 4), reg(0x21, 17, 5), reg(0x21, 18, 6),
        reg(0x21, 19, 7), reg(0x21, 20, 9),
        reg(0x21, 4, 0), reg(0x21, 5, 0), reg(0x21, 6, 8),
        0x0C08585C, reg(0x21, 7, 0),   # queue begin control
        reg(0x21, 4, 16), reg(0x21, 5, 17), reg(0x21, 6, 18),
        0x0C08585C, reg(0x21, 7, 19),  # original text/position/flags
        reg(0x21, 4, 0), reg(0x21, 5, 0), reg(0x21, 6, 20),
        0x0C08585C, reg(0x21, 7, 0),   # queue end control
        imm(0x37, 16, 29, 0), imm(0x37, 17, 29, 8),
        imm(0x37, 18, 29, 0x10), imm(0x37, 19, 29, 0x18),
        imm(0x37, 20, 29, 0x20), imm(0x37, 31, 29, 0x28),
        reg(8, 0, 31), imm(9, 29, 29, 0x40),
        'original', jump(0x216170), 0,
    ])
    assert len(callback) <= scope_queue - scope_callback
    assert len(queue) <= literals - scope_queue
    assert literals + len(controls) <= bias + 0xA00
    payload = bytearray(0x800)
    for location, value in ((marker, mark), (select, choose), (metric, measure), (table, glyphs),
                            (scope_callback, callback), (scope_queue, queue), (literals, controls)):
        offset = location - bias - 0x200
        payload[offset:offset + len(value)] = value
    return [
        (0x2C, struct.pack('<H', 4), struct.pack('<H', 5)),
        (0xB4, bytes(32), struct.pack('<8I', 1, 0x200, bias + 0x200,
                                    bias + 0x200, len(payload), len(payload), 5, 0x1000)),
        (0x1074, struct.pack('<I', 0x2484F770), struct.pack('<I', 0x24841000)),
        (0x524934, struct.pack('<I', 0x9CF770), struct.pack('<I', bias + 0x1000)),
        (0x200, bytes(len(payload)), bytes(payload)),
        (0x566F88, before, after),
        (0x78CA8, struct.pack('<I', 0x308400F8), struct.pack('<I', 0x30840038)),
        (0x773C0, struct.pack('<I', 0x304200F8), struct.pack('<I', 0x30420038)),
        (0x79748, struct.pack('<I', 0x0C065C20), struct.pack('<I', 0x0C000000 | (select >> 2))),
        (0x113B08, struct.pack('<I', 0x0C08585C), struct.pack('<I', 0x0C000000 | (scope_queue >> 2))),
        (0x1049DC, struct.pack('<I', 0x0C08585C), struct.pack('<I', 0x0C000000 | (queue_labels['help'] >> 2))),
        *((address - 0x11F000, struct.pack('<I', 0x0C08585C),
           struct.pack('<I', 0x0C000000 | (queue_labels['space'] >> 2)))
          for address in STATUS_SPACE_CALLS + MENU_ITEM_NAME_SPACE_CALLS),
        (0x112000, struct.pack('<I', 0x8F83AA00), struct.pack('<I', jump(measure_labels['wide']))),
        (0x1120F0, struct.pack('<II', 0x8F82AA00, 0x000210C0), struct.pack('<II', jump(metric), 0)),
    ]


def runtime_regions():
    return ([(offset, bytes.fromhex(before), bytes.fromhex(after))
             for offset, before, after in SPACING_PATCHES]
            + [(address - 0x11F000, struct.pack("<I", before), struct.pack("<I", after))
               for address, before, after in MENU_PATCHES]
            + save_title_regions() + popup_font_regions())


def item_patch_regions(source, translated):
    catalog = {r.offset: r for r in records(source)}
    regions = []
    for offset, words in ITEM_COUNT_WRITES.items():
        row = catalog[offset]
        # Only an edited short fragment occupies the former zero padding. Every
        # valid fragment contains 00 and cannot begin with a NUL terminator.
        location = row.write_offset if translated[row.write_offset] else row.offset
        raw, _text, _controls = read_string(translated, location, row.write_offset + row.capacity + 1)
        position = item_count_position(raw)
        for digit, (address, expected) in enumerate(words):
            replacement = (expected & 0xFFFF0000) | (position + digit)
            regions.append((address, struct.pack("<I", expected), struct.pack("<I", replacement)))
        if offset == SHORT_ITEM:
            if translated[0x578D90:0x578D96] != b"Dummy\0" or translated[0x578DA0] != 0:
                raise ValueError("Short item message overwrote a neighboring string")
            delta = location - offset
            for address, expected in ((0x0DBF40, 0x24427D98), (0x0DBF60, 0x2785A4A8)):
                replacement = (expected & 0xFFFF0000) | ((expected + delta) & 0xFFFF)
                regions.append((address, struct.pack("<I", expected), struct.pack("<I", replacement)))
    return regions


def apply_runtime_patches(source, translated):
    catalog = records(source)
    if len(translated) != len(source):
        raise ValueError("Runtime patch input size changed")
    result = bytearray(translated)
    for offset, before, after in (runtime_regions() + item_patch_regions(source, translated)
                                  + campaign_patch_regions(source, translated)):
        end = offset + len(before)
        if len(before) != len(after) or source[offset:end] != before:
            raise ValueError(f"Runtime patch source mismatch at {offset:08X}")
        if any(r.write_offset < end and offset < r.write_offset + r.capacity + 1 for r in catalog):
            raise ValueError(f"Runtime patch overlaps a text slot at {offset:08X}")
        if translated[offset:end] != before:
            raise ValueError(f"Runtime patch conflict/already applied at {offset:08X}")
        result[offset:end] = after
    return bytes(result)


def campaign_patch_regions(source, translated):
    row = next(r for r in records(source) if r.offset == CAMPAIGN_NUMBER)
    raw, _text, _controls = read_string(translated, row.write_offset,
                                       row.write_offset + row.capacity + 1)
    position = campaign_number_position(raw)
    return [(address, struct.pack("<I", expected),
             struct.pack("<I", (expected & 0xFFFF0000) | (position + digit)))
            for digit, (address, expected) in enumerate(CAMPAIGN_NUMBER_WRITES)]


def encode_menu_text(text, table):
    # Match the existing TXD tool's notation; ELF text uses a different notation.
    if "[menu_sita]" in text:
        text = text.replace("[menu_sita]", "").replace(" ", "、")
    output = bytearray()
    for token in re.split(r"(\[0x[0-9A-Fa-f]+\])", text):
        match = re.fullmatch(r"\[0x([0-9A-Fa-f]+)\]", token)
        if match:
            value = match[1]
            output.extend(bytes.fromhex(value.zfill(len(value) + len(value) % 2)))
        else:
            token = token.translate(str.maketrans(table))
            if re.search(r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]", token):
                raise ValueError("Unmapped Korean text")
            output.extend(token.replace("\n", "\r\n").encode(ENCODING))
    if 0 in output:
        raise ValueError("Embedded NUL in menu text")
    return bytes(output)


def build_menu_txd(text, table, font):
    text = text.lstrip("\ufeff").replace("\r\n", "\n")
    markers = list(re.finditer(r"^\[(\d+)\]", text, re.M))
    if [int(m[1]) for m in markers] != list(range(569)) or text[:markers[0].start()].strip():
        raise ValueError("MainMenu TXT must contain ordered entries [0] through [568]")
    entries = {}
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        entries[i] = text[marker.end():end].rstrip("\n")
    expected = {211: "특전 획득", 212: "지기와 $prop1;KOS-MOS$prop0;의 수영복 스킬을 획득했다",
                213: "노인용 수영복을 획득했다"}
    if any(entries[i] != value for i, value in expected.items()) or "$han1;" not in entries[34]:
        raise ValueError("MainMenu reward text/font setup changed; review fixed coordinates")
    if not entries[224].startswith("클리어 데이터"):
        raise ValueError("MainMenu [224] must retain the corrected leading-space layout")
    indices = (148, 152, 156, 60, 150, 152, 156)
    if len(font) < 0x78040 + 2 * (max(indices) + 1):
        raise ValueError("Truncated menu font metrics")
    metrics = [(struct.unpack_from("<H", font, 0x78040 + 2 * i)[0] + 0x1FF) & 0xFFFF
               for i in indices]
    if [((m >> 8) - (m & 255) + 1) * 13 // 16 for m in metrics] != [10, 11, 9, 12, 13, 11, 9]:
        raise ValueError("Menu font metrics changed; review fixed coordinates")
    pointers, pool = [], bytearray()
    for index, value in entries.items():
        try:
            raw = encode_menu_text(value, table)
        except ValueError as error:
            raise ValueError(f"MainMenu [{index}]: {error}") from error
        if index in SLOT_IDS:
            number = 0 if index in DYNAMIC_SLOT_IDS else (1 if index in (220, 221, 514) else 2)
            prefix = encode_menu_text(f"슬롯 {number}", table)
            if len(prefix) != 6 or not raw.startswith(prefix) or (index in (514, 515) and len(raw) != 6):
                raise ValueError(f"MainMenu [{index}]: expected six-byte slot prefix")
        pointers.append(569 * 4 + len(pool) if raw else 0)
        if raw:
            pool.extend(raw + b"\0")
    result = struct.pack("<569I", *pointers) + pool
    if len(result) > 0x3000:
        raise ValueError("MainMenu.txd exceeds the 12288-byte area before Tutorial.txd")
    return result


def load_menu(table):
    return build_menu_txd(MENU_TEXT.read_text(encoding="utf-8-sig"), table, MENU_FONT.read_bytes())


def folder(disc=None):
    return METADATA


def load_sources(disc=None):
    disc = 1 if disc is None else disc
    data = ORIGINALS[disc].read_bytes()
    if sha(data) != ELF_HASH:
        raise ValueError(f"Original ELF changed: {ORIGINALS[disc]}")
    overlay = (folder(disc) / "OV02.OVL").read_bytes()
    build_overlay(overlay, data, data)
    return data, overlay


def write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def write_result(path, data):
    if path.exists() and path.read_bytes() == data:
        return False
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if pending.read_bytes() != data:
        raise OSError(f"Output verification failed; old result preserved, inspect: {pending}")
    try:
        pending.replace(path)
    except OSError as error:
        raise OSError(f"Cannot replace {path}; old result preserved, pending file: {pending}") from error
    return True


def run(command, disc=None, text_path=None):
    source, ov02 = load_sources(disc)
    target = folder(disc) / "SLPS_strings.txt" if text_path is None else Path(text_path)
    if command == "extract":
        if target.exists():
            raise ValueError(f"TXT already exists; translations were preserved: {target}")
        if target.resolve().parent != folder(disc).resolve() or target.suffix.lower() != ".txt":
            raise ValueError("Export TXT must be directly inside the chosen metadata folder")
        write_new(target, export_text(source).encode("utf-8-sig"))
        print(f"[OK] {len(records(source))} records -> {target}")
        return
    table = load_table()
    built, changes = build(source, target.read_text(encoding="utf-8-sig"), table)
    built = apply_runtime_patches(source, built)
    menu = load_menu(table)
    overlay = build_overlay(ov02, source, built)
    for off, before, after in changes:
        print(f"[OK] {off:08X}: {before} -> {after} bytes")
    for old, new in relocation_moves(built):
        mode = "POOL" if any(a <= new < b for a, b in RELOCATION_POOLS) else "NEAR"
        print(f"[MOVE/{mode}] {old:08X} -> {new:08X}; width/draw reference updated")
    print(f"[CHECK] {len(changes)} changed; ELF size unchanged; OV02 copies agree")
    print("[PATCH] Spacing/icon boundaries, menu alignment and slot numbering included")
    print("[PATCH] Item count positions and short-fragment references match the TXT")
    print("[PATCH] Campaign number write positions match the TXT")
    print("[PATCH] Memory-card first line: XenosagaEPII-##; location and save ID unchanged")
    print(f"[MENU] {len(menu)} bytes; paired MainMenu.txd must be applied with the ELF")
    if command == "verify":
        print("[NO OUTPUT] Verification only")
        return
    discs = (1, 2) if disc is None else (disc,)
    outputs = [(folder() / (ELF_NAMES[number] + ".new"), built) for number in discs]
    outputs.append((folder() / "MainMenu.txd.new", menu))
    overlay_output = folder(disc) / "OV02.OVL.new"
    if overlay != ov02 or overlay_output.exists() or overlay_output.is_symlink():
        # A reverted translation must not leave an older translated OV02 result.
        outputs.append((overlay_output, overlay))
    # Validate every destination before writing any output. Original paths are
    # fixed inputs and are never accepted as output destinations.
    for path, _data in outputs:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Output is not a regular file; nothing written: {path}")
        pending = path.with_name(path.name + ".pending")
        if pending.exists() or pending.is_symlink():
            raise ValueError(f"Previous incomplete write; nothing written. Inspect: {pending}")
    for path, data in outputs:
        changed = write_result(path, data)
        print(f"[{'SAVED' if changed else 'UNCHANGED'}] {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "verify", "rebuild"))
    parser.add_argument("--disc", type=int, choices=(1, 2),
                        help="Optional single-disc output; default writes both ELF names")
    parser.add_argument("--text", type=Path)
    args = parser.parse_args()
    try:
        run(args.command, args.disc, args.text)
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
