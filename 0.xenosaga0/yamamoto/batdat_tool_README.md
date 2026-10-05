# Xenosaga II batdat text tool

Run from this directory:

```powershell
python batdat_tool.py extract batdat.bin fresh_batdat.json
python batdat_tool.py patch batdat.bin batdat.json batdat_patched.bin char_table.json
```

The existing `patch.bat` remains supported. `insert` is an alias for `patch`.
Keep the original binary as the patch input for an existing translation JSON.
Extracting again to an existing JSON path replaces that JSON, including translations.

Edit only `ko`, `ko_name`, and `ko_desc`. Missing translation fields preserve
the original bytes; an empty translation string deliberately clears a field.
Missing sections or entries preserve their original text. Entries may be reordered.
Keep indices, pointer values, original Japanese text and section metadata intact:
the patcher checks them against the input binary before rebuilding each section.
An old JSON cannot be applied to a binary whose strings/pointers have changed.

The optional character table uses a `replace-table` object containing single
character mappings. If omitted, `char_table.json` beside the translation JSON
is loaded automatically when present. Changed text passes through that table,
then optional `[menu_sita]` handling, then strict EUC-JIS-2004 encoding.
The trailing tag turns spaces into fullwidth spaces and `@` into ASCII spaces.
Unchanged original text is copied byte for byte without character substitution.
For game newline commands, preserve the original literal backslash plus `n`
(represented as `\\n` in JSON); no automatic newline conversion is performed.

Only known name/description/message pointers are rewritten. Empty strings are
included in relocation. File size, section offsets and all other record fields
are preserved. Rebuilding must fit the original section pool. Overflow, invalid
encoding, NUL characters, source mismatches and conflicting shared-pointer
translations abort the command without writing output (exit status 2).
Existing output files are left untouched on validation failure; do not mistake
an older output for a successful new patch. Output cannot replace an input file.

This tool targets the supplied 75,992-byte, 19-section XS2 layout and its known
record counts. It is not a general parser for other regional/game versions.
Extracting a patched file shows encoded substitute characters, not reverse-mapped
Korean. Keep the translation JSON as the editable Korean source.

Regression checks from the repository root:

```powershell
python -B -m unittest discover -s 0.xenosaga0/yamamoto -p test_batdat_tool.py -v
```

Tests use the original binary and current translation JSON, and intercept command
output writes. They verify translated strings, untouched fields, empty strings,
re-extraction, invalid input and overflow. Font rendering, layout and runtime
behavior still require testing in the game with the matching patched font.
