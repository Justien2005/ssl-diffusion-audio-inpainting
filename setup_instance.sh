#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root after cloning thesisall:
#   bash setup_instance.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-venv}"
# RTX 50xx / Blackwell (sm_120, e.g. RTX 5090) needs PyTorch CUDA 12.8+ wheels.
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
AUDIO_MAE_FILE_ID="${AUDIO_MAE_FILE_ID:-18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq}"

cd "$REPO_DIR"

echo "==> Installing OS packages"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    python3.10 \
    python3.10-venv \
    unzip \
    wget
fi

echo "==> Creating Python environment: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"

echo "==> Installing CUDA PyTorch stack"
pip uninstall -y torch torchvision torchaudio || true
pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"

echo "==> Installing project requirements"
pip uninstall -y visqol visqol-python pyvisqol || true
python - <<'PY'
import os
import shutil
import site
import glob

for root in site.getsitepackages():
    paths = [
        os.path.join(root, "visqol"),
        os.path.join(root, "pyvisqol"),
        os.path.join(root, "visqol_lib_py.so"),
        os.path.join(root, "visqol_config_pb2.py"),
        os.path.join(root, "similarity_result_pb2.py"),
    ]
    for pattern in [
        os.path.join(root, "visqol-*.dist-info"),
        os.path.join(root, "visqol_python-*.dist-info"),
        os.path.join(root, "pyvisqol-*.dist-info"),
    ]:
        paths.extend(glob.glob(pattern))
    for path in paths:
        if os.path.isdir(path):
            print("Removing stale ViSQOL directory:", path)
            shutil.rmtree(path)
        elif os.path.isfile(path):
            print("Removing stale ViSQOL file:", path)
            os.remove(path)
PY
pip install -r requirements.txt

echo "==> Re-confirming CUDA PyTorch stack after requirements"
pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"
python -m pip install --force-reinstall "setuptools<82"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
PY

echo "==> Ensuring external repositories exist"
mkdir -p external
if [ ! -d external/CQTdiff/.git ]; then
  git clone https://github.com/eloimoliner/CQTdiff.git external/CQTdiff
fi
if [ ! -d external/AudioMAE/.git ]; then
  git clone https://github.com/facebookresearch/AudioMAE.git external/AudioMAE
fi
if [ ! -d external/DDPM-Midi2Performance-Model/.git ]; then
  git clone https://github.com/FlyToYourMooN/DDPM-Midi2Performance-Model.git external/DDPM-Midi2Performance-Model
fi

echo "==> Downloading CQT-Diff+ weights"
if [ ! -f external/CQTdiff/experiments/cqt/cqt_weights.pt ]; then
  (cd external/CQTdiff && bash download_weights_and_examples.sh)
else
  echo "CQT-Diff+ weights already present."
fi

echo "==> Downloading AudioMAE checkpoint"
mkdir -p external/AudioMAE/ckpt
if [ ! -f external/AudioMAE/ckpt/pretrained.pth ]; then
  gdown "https://drive.google.com/uc?id=${AUDIO_MAE_FILE_ID}" \
    -O external/AudioMAE/ckpt/pretrained.pth
else
  echo "AudioMAE checkpoint already present."
fi

echo "==> Writing environment helper"
cat > env_instance.sh <<'EOF'
#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/venv/bin/activate"
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_CPU_THREADS="${PIPELINE_CPU_THREADS:-1}"
export PIPELINE_TORCH_THREADS="${PIPELINE_TORCH_THREADS:-1}"
export PIPELINE_NUM_WORKERS="${PIPELINE_NUM_WORKERS:-2}"
export ENCODER_PRECOMPUTE_NUM_WORKERS="${ENCODER_PRECOMPUTE_NUM_WORKERS:-0}"
export OMP_NUM_THREADS="$PIPELINE_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$PIPELINE_CPU_THREADS"
export MKL_NUM_THREADS="$PIPELINE_CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$PIPELINE_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$PIPELINE_CPU_THREADS"
export CQT_DIFF_DIR="$PROJECT_ROOT/external/CQTdiff"
export AUDIO_MAE_DIR="$PROJECT_ROOT/external/AudioMAE"
export MIDI2PERFORMANCE_DIR="$PROJECT_ROOT/external/DDPM-Midi2Performance-Model"
export CQTDIFF_WEIGHTS="$CQT_DIFF_DIR/experiments/cqt/cqt_weights.pt"
export AUDIOMAE_CHECKPOINT="$AUDIO_MAE_DIR/ckpt/pretrained.pth"
export OFFICIAL_CQTDIFF_ADAPTER=official_cqtdiff_adapter
export MAID_ADAPTER=official_maid_adapter
EOF
chmod +x env_instance.sh

echo "==> Compile check"
source env_instance.sh
python - <<'PY'
import numpy as np
import visqol
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol import VisqolApi

print("visqol package:", visqol.__file__)
print("visqol native:", visqol_lib_py.__file__)
print("visqol config:", visqol_config_pb2.__file__)

sr = 48000
t = np.arange(sr, dtype=np.float64) / sr
ref = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
deg = ref.copy()

api = VisqolApi()
api.create(mode="audio")
result = api.measure_from_arrays(ref, deg, sample_rate=sr)
print("visqol-python api create/measure: ok", float(result.moslqo))
PY
python -m py_compile code_it_v2.py code_final_run_v2.py official_cqtdiff_adapter.py official_maid_adapter.py

echo "==> Setup complete"
echo "Next:"
echo "  source env_instance.sh"
echo "  python code_it_v2.py      # smoke test config"
echo "  python code_final_run_v2.py"
