#!/bin/bash
set -e

# Install classical ML packages with LightGBM GPU (CUDA) support.
# This creates/populates the ml_field conda environment.
#
# Usage:
#   bash install_ml.sh              # full install (creates env if needed)
#   bash install_ml.sh --skip-lgbm  # skip LightGBM CUDA build

SKIP_LGBM_BUILD=false
if [ "${1:-}" = "--skip-lgbm" ]; then
    SKIP_LGBM_BUILD=true
fi

ENV_NAME="ml_field"
PYTHON_VERSION="3.11"

# ── Create conda env if it doesn't exist ──────────────────────────────
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "=== Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
else
    echo "=== Environment ${ENV_NAME} already exists ==="
fi

# Activate the environment and use its binaries directly (avoids PATH issues)
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
PIP="$CONDA_PREFIX/bin/pip"
PYTHON="$CONDA_PREFIX/bin/python3"
echo "Active env: $CONDA_DEFAULT_ENV"
echo "  pip:    $PIP"
echo "  python: $PYTHON"

# ── Install pip packages ──────────────────────────────────────────────
echo ""
echo "=== Installing classical ML packages ==="
$PIP install \
    "geopandas~=1.0.1" \
    "joblib~=1.4.2" \
    "matplotlib~=3.9.1" \
    "numpy~=1.26.4" \
    "optuna~=4.2.1" \
    "pandas~=2.2.2" \
    "polars~=1.25.2" \
    "psutil~=6.0.0" \
    "pyarrow~=17.0.0" \
    "rasterio~=1.4.3" \
    "scikit-learn~=1.5.1" \
    "seaborn~=0.13.2" \
    "shapely~=2.0.7" \
    "xgboost~=2.1.4" \
    "imbalanced-learn"

# ── Build LightGBM with CUDA support ─────────────────────────────────
if [ "$SKIP_LGBM_BUILD" = true ]; then
    echo ""
    echo "=== Skipping LightGBM CUDA build (installing CPU version) ==="
    $PIP install "lightgbm~=4.6.0"
else
    echo ""
    echo "=== Building LightGBM with CUDA support ==="

    # Install build deps (cmake >= 3.28 required by LightGBM v4.6)
    $PIP install "cmake>=3.28" scikit-build-core

    # Find CUDA toolkit
    if [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc &> /dev/null; then
        CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
    else
        echo "ERROR: nvcc not found. Ensure CUDA toolkit is installed and on PATH."
        exit 1
    fi
    echo "CUDA_HOME: $CUDA_HOME"
    export CUDA_HOME

    # Build from source via pip with CUDA cmake settings
    $PIP install --force-reinstall --no-binary lightgbm "lightgbm~=4.6.0" \
        --config-settings=cmake.define.USE_CUDA=ON \
        --config-settings=cmake.define.CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"

    echo "LightGBM CUDA build complete"
fi

# ── Verify ────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
$PYTHON -c "
import sklearn
print(f'scikit-learn {sklearn.__version__}')

import xgboost as xgb
print(f'XGBoost {xgb.__version__}')

import lightgbm as lgb
print(f'LightGBM {lgb.__version__}')

import optuna
print(f'Optuna {optuna.__version__}')

import pandas as pd
print(f'Pandas {pd.__version__}')

import geopandas as gpd
print(f'GeoPandas {gpd.__version__}')

# Check LightGBM CUDA support
try:
    params = {'device': 'cuda', 'verbose': -1}
    d = lgb.Dataset([[1,2],[3,4]], label=[0,1])
    b = lgb.train(params, d, num_boost_round=1, verbose_eval=False)
    print('LightGBM CUDA: OK')
except Exception as e:
    print(f'LightGBM CUDA: not available ({e})')
    print('  (CPU fallback will still work)')
"

echo ""
echo "Done. Activate with: conda activate ${ENV_NAME}"
echo "Run classical ML scripts from deep_learn/src/"
