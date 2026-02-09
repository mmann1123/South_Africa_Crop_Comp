#!/bin/bash
set -e

# Detect CUDA version and install PyTorch + TensorFlow with matching GPU support.
# Falls back to CPU-only if no CUDA is found.
#
# Usage:
#   bash install.sh              # auto-detect CUDA
#   bash install.sh cpu          # force CPU-only
#   bash install.sh cu124        # force specific CUDA version

FORCE_VARIANT="${1:-}"

detect_cuda_version() {
    # Try nvidia-smi first (driver-level CUDA version)
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
        if [ -n "$CUDA_VER" ]; then
            echo "$CUDA_VER"
            return
        fi
    fi

    # Try nvcc (toolkit-level)
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || true)
        if [ -n "$CUDA_VER" ]; then
            echo "$CUDA_VER"
            return
        fi
    fi

    echo ""
}

# Map CUDA version to PyTorch index URL suffix
cuda_to_torch_variant() {
    local ver="$1"
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)

    if [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; }; then
        echo "cu126"
    elif [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 4 ]; }; then
        echo "cu124"
    elif [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 1 ]; }; then
        echo "cu121"
    elif [ "$major" -gt 11 ] || { [ "$major" -eq 11 ] && [ "$minor" -ge 8 ]; }; then
        echo "cu118"
    else
        echo "cpu"
    fi
}

# Determine variant
if [ -n "$FORCE_VARIANT" ]; then
    VARIANT="$FORCE_VARIANT"
    echo "Using forced variant: $VARIANT"
else
    CUDA_VERSION=$(detect_cuda_version)
    if [ -n "$CUDA_VERSION" ]; then
        VARIANT=$(cuda_to_torch_variant "$CUDA_VERSION")
        echo "Detected CUDA $CUDA_VERSION -> using PyTorch variant: $VARIANT"
    else
        VARIANT="cpu"
        echo "No CUDA detected -> using CPU-only"
    fi
fi

# Install TensorFlow first (PyTorch installed after so its nvidia packages win)
echo ""
echo "=== Installing TensorFlow ==="
if [ "$VARIANT" = "cpu" ]; then
    pip install "tensorflow~=2.18.0"
else
    pip install "tensorflow[and-cuda]~=2.18.0"
fi

# Install PyTorch with correct CUDA variant (after TF so its nvidia packages take precedence)
echo ""
echo "=== Installing PyTorch (${VARIANT}) ==="
if [ "$VARIANT" = "cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${VARIANT}"
fi

# Set up LD_LIBRARY_PATH for nvidia pip packages (needed by TensorFlow)
if [ "$VARIANT" != "cpu" ] && [ -n "$CONDA_PREFIX" ]; then
    echo ""
    echo "=== Setting up conda env activation for nvidia libraries ==="
    ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
    DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
    mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

    NVIDIA_LIB_DIRS=$(python3 -c "
import os, site
sp = site.getsitepackages()[0]
nvidia_dir = os.path.join(sp, 'nvidia')
if os.path.isdir(nvidia_dir):
    dirs = []
    for d in sorted(os.listdir(nvidia_dir)):
        lib_path = os.path.join(nvidia_dir, d, 'lib')
        if os.path.isdir(lib_path):
            dirs.append(lib_path)
    print(':'.join(dirs))
")

    cat > "$ACTIVATE_DIR/nvidia-libs.sh" << ACTIVATE_EOF
#!/bin/bash
export OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:\$LD_LIBRARY_PATH"
ACTIVATE_EOF

    cat > "$DEACTIVATE_DIR/nvidia-libs.sh" << DEACTIVATE_EOF
#!/bin/bash
export LD_LIBRARY_PATH="\$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
DEACTIVATE_EOF

    chmod +x "$ACTIVATE_DIR/nvidia-libs.sh" "$DEACTIVATE_DIR/nvidia-libs.sh"

    # Source it now for the verification step below
    source "$ACTIVATE_DIR/nvidia-libs.sh"
    echo "Created conda activation scripts for nvidia library paths"
fi

# Install remaining requirements
echo ""
echo "=== Installing other dependencies ==="
pip install \
    "geopandas~=1.0.1" \
    "joblib~=1.4.2" \
    "lightgbm~=4.6.0" \
    "matplotlib~=3.9.1" \
    "optuna~=4.2.1" \
    "pandas~=2.2.2" \
    "polars~=1.25.2" \
    "psutil~=6.0.0" \
    "pyarrow~=17.0.0" \
    "rasterio~=1.4.3" \
    "scikit-learn~=1.5.1" \
    "seaborn~=0.13.2" \
    "shapely~=2.0.7" \
    "torchtoolbox~=0.1.8.2" \
    "xgboost~=2.1.4" \
    "pytorch-tabnet"

# Verify
echo ""
echo "=== Verifying installation ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:   {torch.version.cuda}')
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow {tf.__version__}')
print(f'  GPUs: {len(gpus)}')
if gpus:
    for g in gpus:
        print(f'  Device: {g}')
"

echo ""
echo "Done. Run scripts from deep_learn/src/"
