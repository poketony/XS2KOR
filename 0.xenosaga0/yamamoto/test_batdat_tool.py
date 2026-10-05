"""Regression tests against the repository's original batdat.bin (no writes)."""
import copy
import io
import json
import struct
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import batdat_tool as tool

ROOT = Path(__file__).resolve().parent


class BatdatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "batdat.bin").read_bytes()
        cls.table = tool.load_char_table(str(ROOT / "char_table.json"))

    def document(self, sec, source=None):
        source = self.source if source is None else source
        pool = struct.unpack_from("<H", source, sec.base)[0]
        return {"entries": [tool.make_entry(source, sec, i, pool)
                            for i in range(tool.count_entries(source, sec))]}

    def test_noop_and_translation_preserve_every_other_field(self):
        for sec in tool.SECTIONS:
            with self.subTest(section=sec.label):
                doc = self.document(sec)
                out = bytearray(self.source)
                self.assertEqual(tool.patch_section(out, sec, doc, self.table), (False, "unchanged"))
                self.assertEqual(out, self.source)
                key = "ko_name" if sec.sec_type == "name_desc" else "ko"
                doc["entries"][0][key] = "\uac00"
                changed, _ = tool.patch_section(out, sec, doc, self.table)
                self.assertTrue(changed)
                self.assertEqual(len(out), len(self.source))
                pool = struct.unpack_from("<H", self.source, sec.base)[0]
                allowed = set(range(sec.base + pool, sec.end))
                for i in range(tool.count_entries(self.source, sec)):
                    for field, pos, rel in tool.entry_fields(self.source, sec, i, pool):
                        allowed.update((pos, pos + 1))
                        newrel = struct.unpack_from("<H", out, pos)[0]
                        old = tool.read_raw_str(self.source, sec.base + rel, sec.end)
                        expected = b"\xb0\xa1" if i == 0 and field in ("name", "text") else old
                        self.assertEqual(tool.read_raw_str(out, sec.base + newrel, sec.end), expected)
                self.assertTrue(all(a == b or i in allowed for i, (a, b) in enumerate(zip(self.source, out))))
                reextracted = self.document(sec, out)
                snapshot = bytes(out)
                self.assertEqual(tool.patch_section(out, sec, reextracted, self.table), (False, "unchanged"))
                self.assertEqual(out, snapshot)

    def test_empty_description_can_be_translated(self):
        sec = tool.SECTIONS[2]
        doc = self.document(sec)
        doc["entries"][0]["ko_desc"] = "A"
        blank = next(e for e in doc["entries"] if e["jp_desc"] == "")
        blank["ko_desc"] = "B"
        out = bytearray(self.source)
        self.assertTrue(tool.patch_section(out, sec, doc, self.table)[0])
        ptr = struct.unpack_from("<H", out, blank["ptr_pos"] + 2)[0]
        self.assertEqual(tool.read_raw_str(out, sec.base + ptr, sec.end), b"B")

    def test_invalid_translation_and_metadata_leave_buffer_unchanged(self):
        sec = tool.SECTIONS[0]
        for key, value in (("ko", "\ud7a3"), ("ko", "A\x00B"), ("ko", None),
                           ("name_ptr", 1400), ("jp", "wrong"), ("idx", -1)):
            with self.subTest(key=key, value=value):
                doc = self.document(sec)
                doc["entries"][0][key] = value
                out = bytearray(self.source)
                with self.assertRaises(ValueError):
                    tool.patch_section(out, sec, doc, self.table)
                self.assertEqual(out, self.source)

    def test_overflow_and_duplicate_ids(self):
        sec = tool.SECTIONS[0]
        doc = self.document(sec)
        doc["entries"][0]["ko"] = "A" * 2000
        out = bytearray(self.source)
        changed, reason = tool.patch_section(out, sec, doc, self.table)
        self.assertFalse(changed)
        self.assertIn("overflow", reason)
        self.assertEqual(out, self.source)
        doc = self.document(sec)
        doc["entries"].append(copy.deepcopy(doc["entries"][0]))
        with self.assertRaises(ValueError):
            tool.patch_section(out, sec, doc, self.table)

    def test_partial_reordered_json(self):
        sec = tool.SECTIONS[0]
        doc = self.document(sec)
        doc["entries"] = [doc["entries"][2], doc["entries"][0]]
        doc["entries"][1]["ko"] = "A"
        out = bytearray(self.source)
        self.assertTrue(tool.patch_section(out, sec, doc, self.table)[0])

    def test_command_never_writes_on_validation_failure(self):
        sec = tool.SECTIONS[0]
        doc = {sec.label: self.document(sec)}
        doc[sec.label]["entries"][0]["ko"] = "\ud7a3"
        with patch.object(tool.json, "load", return_value=doc), \
                patch.object(tool, "load_char_table", return_value=self.table), \
                patch.object(Path, "write_bytes") as write:
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                tool.cmd_patch(str(ROOT / "batdat.bin"), str(ROOT / "batdat.json"),
                               str(ROOT / "test_output.bin"), None)
            write.assert_not_called()

    def test_source_overwrite_and_missing_table_rejected(self):
        with self.assertRaises(ValueError):
            tool.cmd_patch(str(ROOT / "batdat.bin"), str(ROOT / "batdat.json"), str(ROOT / "batdat.bin"))
        with self.assertRaises(FileNotFoundError):
            tool.load_char_table(str(ROOT / "not_a_char_table.json"))
        self.assertEqual(tool.apply_menu_sita("a b@c[menu_sita]"), "a\u3000b c")

    def test_current_translation_command_output(self):
        # Exercise the real JSON/table loaders and command, intercept only disk output.
        with patch.object(Path, "write_bytes") as write, redirect_stdout(io.StringIO()):
            tool.cmd_patch(str(ROOT / "batdat.bin"), str(ROOT / "batdat.json"),
                           str(ROOT / "test_output.bin"))
        write.assert_called_once()
        output = bytes(write.call_args.args[0])
        self.assertEqual(len(output), len(self.source))
        translation = json.loads((ROOT / "batdat.json").read_text(encoding="utf-8-sig"))
        allowed = set()
        for sec in tool.SECTIONS:
            pool = struct.unpack_from("<H", self.source, sec.base)[0]
            allowed.update(range(sec.base + pool, sec.end))
            entries = {e["idx"]: e for e in translation[sec.label]["entries"]}
            for i in range(tool.count_entries(self.source, sec)):
                for field, pos, rel in tool.entry_fields(self.source, sec, i, pool):
                    allowed.update((pos, pos + 1))
                    suffix = "_" + field if sec.sec_type == "name_desc" else ""
                    entry = entries[i]
                    jp = entry["jp" + suffix]
                    ko = entry.get("ko" + suffix, jp)
                    expected = (tool.read_raw_str(self.source, sec.base + rel, sec.end) if ko == jp else
                                tool.apply_menu_sita(tool.apply_char_table(ko, self.table)).encode(tool.ENCODING))
                    target = struct.unpack_from("<H", output, pos)[0]
                    self.assertEqual(tool.read_raw_str(output, sec.base + target, sec.end), expected)
        self.assertTrue(all(a == b or i in allowed for i, (a, b) in enumerate(zip(self.source, output))))


if __name__ == "__main__":
    unittest.main()
