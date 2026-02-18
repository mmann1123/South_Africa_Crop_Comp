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

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Active env: $CONDA_DEFAULT_ENV (Python $(python --version 2>&1))"

# ── Install pip packages ──────────────────────────────────────────────
echo ""
echo "=== Installing classical ML packages ==="
pip install \
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
    pip install "lightgbm~=4.6.0"
else
    echo ""
    echo "=== Building LightGBM with CUDA support ==="

    # Check prerequisites
    if ! command -v cmake &> /dev/null; then
        echo "ERROR: cmake not found. Install with: sudo apt install cmake"
        exit 1
    fi
    if ! command -v nvcc &> /dev/null; then
        echo "ERROR: nvcc not found. Ensure CUDA toolkit is installed and on PATH."
        exit 1
    fi

    BUILD_DIR=$(mktemp -d)
    echo "Build directory: $BUILD_DIR"

    cd "$BUILD_DIR"
    git clone --recursive --branch v4.6.0 --depth 1 https://github.com/microsoft/LightGBM.git
    cd LightGBM

    mkdir -p build && cd build
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

    cd ../python-package
    pip install --no-build-isolation .

    # Clean up
    cd /
    rm -rf "$BUILD_DIR"
    echo "LightGBM CUDA build complete"
fi

# ── Verify ────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
python -c "
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
