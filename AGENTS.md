# Workspace Boundary

- XS2KOR is the translation/editing workspace. The separate Xenosaga2 directory is the ROM staging/rebuild workspace.
- Keep both workspaces independent. No pipeline may automatically read inputs, settings, originals, or executable code from the other workspace, or write outputs into it.
- The user manually transfers finished files. Never replace this step with synchronization, fallback paths, or automatic discovery of the other workspace.
- Translation tools must keep immutable, hash-checked originals inside XS2KOR. Files in the ROM staging workspace are replaceable build inputs, never translation-tool originals.
- Historical reports may retain provenance paths. They must not become live input paths.
