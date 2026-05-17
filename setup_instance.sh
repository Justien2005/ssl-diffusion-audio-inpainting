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
    build-essential \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    openjdk-17-jdk \
    protobuf-compiler \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    swig \
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
pip install -r requirements.txt

echo "==> Installing ViSQOL fallback from google/visqol"
if ! command -v bazel >/dev/null 2>&1; then
  curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 \
    -o /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel
fi
export PYTHON_BIN_PATH="$(python -c 'import sys; print(sys.executable)')"
export PYTHON_LIB_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')"
python -c "import numpy; print('numpy for ViSQOL build:', numpy.__version__, numpy.get_include())"
pip install --no-build-isolation --no-cache-dir "git+https://github.com/google/visqol.git"

echo "==> Re-confirming CUDA PyTorch stack after requirements"
pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"
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
export CQT_DIFF_DIR="$PROJECT_ROOT/external/CQTdiff"
export AUDIO_MAE_DIR="$PROJECT_ROOT/external/AudioMAE"
export CQTDIFF_WEIGHTS="$CQT_DIFF_DIR/experiments/cqt/cqt_weights.pt"
export AUDIOMAE_CHECKPOINT="$AUDIO_MAE_DIR/ckpt/pretrained.pth"
export OFFICIAL_CQTDIFF_ADAPTER=official_cqtdiff_adapter
export MAID_ADAPTER=official_maid_adapter
# Optional override. The pipeline auto-detects the model packaged by google/visqol.
# export VISQOL_MODEL_PATH="$PROJECT_ROOT/path/to/libsvm_nu_svr_model.txt"
EOF
chmod +x env_instance.sh

echo "==> Compile check"
source env_instance.sh
python - <<'PY'
import importlib
import subprocess
import sys

def import_first(candidates):
    last_exc = None
    for name in candidates:
        try:
            module = importlib.import_module(name)
            print(f"import ok: {name}")
            return module
        except ImportError as exc:
            print(f"import failed: {name}: {exc}")
            last_exc = exc
    print("\nInstalled visqol files:")
    subprocess.run([sys.executable, "-m", "pip", "show", "-f", "visqol"], check=False)
    raise last_exc

visqol_lib_py = import_first([
    "visqol.visqol_lib_py",
    "python.visqol_lib_py",
    "visqol_lib_py",
])
visqol_config_pb2 = import_first([
    "visqol.pb2.visqol_config_pb2",
    "visqol_config_pb2",
])
print("visqol:", visqol_lib_py.__file__)
print("visqol config:", visqol_config_pb2.VisqolConfig().__class__.__name__)
PY
python -m py_compile code_it_v2.py code_final_run_v2.py official_cqtdiff_adapter.py official_maid_adapter.py

echo "==> Setup complete"
echo "Next:"
echo "  source env_instance.sh"
echo "  python code_it_v2.py      # smoke test config"
echo "  python code_final_run_v2.py"
