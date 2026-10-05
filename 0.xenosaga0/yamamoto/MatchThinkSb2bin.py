#!/usr/bin/env python3
"""Legacy entry point: rebuild TXT to paired .sb.new/.bin.new, never in-place."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sb_tool


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        catalog = sb_tool.load_catalog()
        if args == ["scan"]:
            count = 0
            for name, record in catalog["sources"].items():
                source, data, _ = sb_tool.load_source(sb_tool.asset_path(name), catalog)
                for relative in record["copies"]:
                    target = sb_tool.asset_path(relative)
                    if target.read_bytes() != data:
                        raise ValueError(f"Original BIN mismatch: {target}")
                    print(f"{source.name} -> {relative}")
                    count += 1
            print(f"[OK] {count} whole-file matches. No files changed.")
            return 0
        if len(args) != 2 or args[0] not in ("info", "patch"):
            raise ValueError("Usage: scan | info <thinkNNN.sb> | patch <thinkNNN.sb>")
        source = Path(args[1])
        if not source.is_absolute():
            source = Path(__file__).resolve().parent / "think" / source
        source, _data, record = sb_tool.load_source(source, catalog)
        if args[0] == "patch":
            return sb_tool.main(["import", str(source), str(source.with_suffix(".txt"))])
        print(source)
        for relative in record["copies"]:
            print(f"  -> {sb_tool.asset_path(relative)}.new")
        if not record["copies"]:
            print("No verified BIN copy.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAILED] {exc}")
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
