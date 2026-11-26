# Workspace Boundary

- XS2KOR is the translation/editing workspace. The separate Xenosaga2 directory is the ROM staging/rebuild workspace.
- Keep both workspaces independent. No pipeline may automatically read inputs, settings, originals, or executable code from the other workspace, or write outputs into it.
- The user manually transfers finished files. Never replace this step with synchronization, fallback paths, or automatic discovery of the other workspace.
- Translation tools must keep immutable, hash-checked originals inside XS2KOR. Files in the ROM staging workspace are replaceable build inputs, never translation-tool originals.
- Historical reports may retain provenance paths. They must not become live input paths.

## Approved Subtitle Delivery Exception (2026-10-24)

- The dedicated subtitle apply BAT/explicit encode command may copy validated subtitle outputs to the user's external Xenosaga2 workspace.
- Use repository-local hash-checked originals and font mapping. External LBA indexes and existing movie TXT targets are read only to validate disc membership and replace the requested subtitle outputs.
- XS2 game targets are `xenosaga1_UNPACKED/movie/*.txt` and `xenosaga2_UNPACKED/movie/*.txt`, without a jp/kr subdirectory.
- Validate the complete subtitle batch before copying; preserve UTF-8 editors and skip identical output.
- The ROM builder remains independent and may not read XS2KOR, XS3KOR, or codex-lab code/settings/inputs.

## Approved ISO Metadata Delivery Exception (2026-11-25)

- `discmetadata/2.Rebuild.bat` may deliver only the validated per-disc SLPS and OV02.OVL, and the shared `xenosaga0_UNPACKED/yosikawa/MainMenu.txd`, to the explicitly specified Xenosaga2 staging root.
- Require both disc metadata folders and the shared menu folder before delivery. If delivery is unavailable, keep outputs in XS2KOR as `.new`. Existing destination files may be inspected only to skip unchanged writes and restore them if delivery fails; they are not build inputs.
- No other automatic cross-workspace copying is authorized. The ISO builder is unchanged and independent.
