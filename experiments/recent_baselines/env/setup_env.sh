#!/usr/bin/env bash
# Create the isolated `ts_baselines` env for the R2-Q3a baselines (MiniRocket, TimesNet).
#
# Why a separate env: `pip install aeon einops` into deep_field would downgrade
# numpy 2.4.2 -> 2.3.5 and numba 0.65 -> 0.63 (pip dry-run, 2026-10-09), which could
# change the published models. deep_field is cloned with --copy (no shared hardlinks)
# and only the clone is modified. The script fails if deep_field changes.
#
# Run from the repo root:  bash experiments/recent_baselines/env/setup_env.sh
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
SRC_PY="$CONDA_BASE/envs/deep_field/bin/python"
NEW_ENV=ts_baselines
NEW_PY="$CONDA_BASE/envs/$NEW_ENV/bin/python"

AEON_VERSION=1.6.0
EINOPS_VERSION=0.8.2

"$SRC_PY" -m pip freeze | sort > "$ENV_DIR/deep_field_freeze_before.txt"

if [ ! -x "$NEW_PY" ]; then
    conda create --clone deep_field --copy -n "$NEW_ENV" -y
fi
"$NEW_PY" -m pip install "aeon==$AEON_VERSION" "einops==$EINOPS_VERSION"

# deep_field carries a stale numpy-2.0.2.dist-info next to the real numpy (pre-existing,
# harmless there); in the clone it makes `pip freeze` misreport numpy. Remove any numpy
# dist-info in the clone whose version differs from the numpy that actually imports.
"$NEW_PY" - <<'EOF'
import glob, os, shutil, numpy
site = os.path.dirname(os.path.dirname(numpy.__file__))
for d in glob.glob(os.path.join(site, "numpy-*.dist-info")):
    if not os.path.basename(d).startswith(f"numpy-{numpy.__version__}."):
        print("removing stale", d); shutil.rmtree(d)
EOF

"$SRC_PY" -m pip freeze | sort > "$ENV_DIR/deep_field_freeze_after.txt"
if ! diff -q "$ENV_DIR/deep_field_freeze_before.txt" "$ENV_DIR/deep_field_freeze_after.txt" > /dev/null; then
    echo "ERROR: deep_field packages changed during setup" >&2
    diff "$ENV_DIR/deep_field_freeze_before.txt" "$ENV_DIR/deep_field_freeze_after.txt" >&2
    exit 1
fi

# Lock files for the new env, plus the documented difference from deep_field
"$NEW_PY" -m pip freeze | sort > "$ENV_DIR/pip_freeze.txt"
conda env export -n "$NEW_ENV" --no-builds > "$ENV_DIR/environment.yml"
diff "$ENV_DIR/deep_field_freeze_after.txt" "$ENV_DIR/pip_freeze.txt" > "$ENV_DIR/diff_vs_deep_field.txt" || true

"$NEW_PY" -c "import aeon, torch, einops, sklearn, numpy; print('aeon', aeon.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| sklearn', sklearn.__version__, '| numpy', numpy.__version__)"
echo "deep_field unchanged; $NEW_ENV ready at $NEW_PY"
