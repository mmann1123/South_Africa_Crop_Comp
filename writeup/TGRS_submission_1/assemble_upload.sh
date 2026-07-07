#!/usr/bin/env bash
# Stage the IEEE DataPort upload bundle in one folder without duplicating the
# ~7.6 GB of data: the (large) data files are symlinked from the repo's data/
# directory and the documentation files are copied. Then either upload the
# folder's contents directly, or build a self-contained archive that resolves
# the symlinks:
#     tar -czhf dataport_upload.tar.gz -C dataport_upload .   (-h follows symlinks)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"     # writeup/TGRS_submission_1
REPO="$(cd "$HERE/../.." && pwd)"         # repo root
DATADIR="$REPO/data"
DOCDIR="$HERE/data"
OUT="$HERE/dataport_upload"

DATA=(final_data.parquet merged_dl_train.parquet patch_level_data.parquet
      merged_dl_test.parquet test_patch_data.parquet combined_test_features.parquet
      combined_training_fields.geojson test_fields.geojson patch_level.geojson
      test_patches.geojson)
DOCS=(README.txt FILE_MANIFEST.txt DATA_DICTIONARY.txt PROVENANCE_AND_LICENSE.txt
      REPRODUCIBILITY.txt CITATION.txt SHA256SUMS.txt LICENSE.txt)

mkdir -p "$OUT"
miss=0
for f in "${DATA[@]}"; do
  if [ -f "$DATADIR/$f" ]; then ln -sf "$DATADIR/$f" "$OUT/$f"; else echo "MISSING data: $f"; miss=1; fi
done
for f in "${DOCS[@]}"; do
  if [ -f "$DOCDIR/$f" ]; then cp -f "$DOCDIR/$f" "$OUT/$f"; else echo "MISSING doc: $f"; miss=1; fi
done

echo "Staged $((${#DATA[@]}+${#DOCS[@]})) files in: $OUT"
echo "  (data files are symlinks -> $DATADIR ; docs are copies)"
echo "Bundle size (resolving symlinks):"; du -shL "$OUT" 2>/dev/null || true
echo "To make a self-contained archive:"
echo "  tar -czhf \"$HERE/dataport_upload.tar.gz\" -C \"$OUT\" ."
[ "$miss" -eq 0 ] && echo "OK: all 18 files present." || echo "WARNING: some files missing (see above)."
