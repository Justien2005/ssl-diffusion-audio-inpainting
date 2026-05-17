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
    gcc-12 \
    g++-12 \
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
python -m pip install "protobuf==4.25.8"

echo "==> Installing CUDA PyTorch stack"
pip uninstall -y torch torchvision torchaudio || true
pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"

echo "==> Installing project requirements"
pip install -r requirements.txt
python -m pip install --force-reinstall "protobuf==4.25.8"

echo "==> Installing ViSQOL fallback from google/visqol"
if ! command -v bazel >/dev/null 2>&1; then
  curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 \
    -o /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel
fi
export PYTHON_BIN_PATH="$(python -c 'import sys; print(sys.executable)')"
export PYTHON_LIB_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')"
if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
  export CC="$(command -v gcc-12)"
  export CXX="$(command -v g++-12)"
fi
python -c "import numpy; print('numpy for ViSQOL build:', numpy.__version__, numpy.get_include())"
python -c "import google.protobuf; print('protobuf for ViSQOL build:', google.protobuf.__version__)"
if [ ! -d external/visqol/.git ]; then
  git clone https://github.com/google/visqol.git external/visqol
fi
(
  cd external/visqol
  bazel clean --expunge || true
  bazel build -c opt \
    --repo_env=CC="$CC" \
    --repo_env=CXX="$CXX" \
    --action_env=CC="$CC" \
    --action_env=CXX="$CXX" \
    --copt=-Wno-array-bounds \
    --copt=-Wno-stringop-overflow \
    --copt=-Wno-maybe-uninitialized \
    --host_copt=-Wno-array-bounds \
    --host_copt=-Wno-stringop-overflow \
    --host_copt=-Wno-maybe-uninitialized \
    //:similarity_result_py_pb2 //:visqol_config_py_pb2 //python:visqol_lib_py.so
)
pip install --no-deps --no-build-isolation --no-cache-dir "git+https://github.com/google/visqol.git" || true
VISQOL_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$VISQOL_SITE/visqol/pb2" "$VISQOL_SITE/visqol/model"
cp external/visqol/bazel-bin/python/visqol_lib_py.so "$VISQOL_SITE/visqol/visqol_lib_py.so"
cp external/visqol/bazel-bin/python/visqol_lib_py.so "$VISQOL_SITE/visqol_lib_py.so"
cp external/visqol/bazel-bin/similarity_result_pb2.py "$VISQOL_SITE/visqol/pb2/similarity_result_pb2.py"
cp external/visqol/bazel-bin/visqol_config_pb2.py "$VISQOL_SITE/visqol/pb2/visqol_config_pb2.py"
cp external/visqol/bazel-bin/similarity_result_pb2.py "$VISQOL_SITE/similarity_result_pb2.py"
cp external/visqol/bazel-bin/visqol_config_pb2.py "$VISQOL_SITE/visqol_config_pb2.py"
cp external/visqol/model/libsvm_nu_svr_model.txt "$VISQOL_SITE/visqol/model/libsvm_nu_svr_model.txt"
cp external/visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite \
  "$VISQOL_SITE/visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
touch "$VISQOL_SITE/visqol/__init__.py" "$VISQOL_SITE/visqol/pb2/__init__.py"

echo "==> Re-confirming CUDA PyTorch stack after requirements"
pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"
python -m pip install --force-reinstall "setuptools<82" "protobuf==4.25.8"
python - <<'PY'
import torch
import google.protobuf
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("protobuf:", google.protobuf.__version__)
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
from google.protobuf import message_factory

if (
    hasattr(message_factory, "MessageFactory")
    and hasattr(message_factory, "GetMessageClass")
    and not hasattr(message_factory.MessageFactory, "GetPrototype")
):
    message_factory.MessageFactory.GetPrototype = (
        lambda self, descriptor: message_factory.GetMessageClass(descriptor)
    )

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
    "visqol_lib_py",
    "visqol.visqol_lib_py",
    "python.visqol_lib_py",
])
visqol_config_pb2 = import_first([
    "visqol_config_pb2",
    "visqol.pb2.visqol_config_pb2",
])
print("visqol:", visqol_lib_py.__file__)
print("visqol config:", visqol_config_pb2.VisqolConfig().__class__.__name__)
config = visqol_config_pb2.VisqolConfig()
config.audio.sample_rate = 48000
config.options.use_speech_scoring = False
config.options.svr_model_path = "visqol/model/libsvm_nu_svr_model.txt"
api = visqol_lib_py.VisqolApi()
api.Create(config)
print("visqol api create: ok")
PY
python -m py_compile code_it_v2.py code_final_run_v2.py official_cqtdiff_adapter.py official_maid_adapter.py

echo "==> Setup complete"
echo "Next:"
echo "  source env_instance.sh"
echo "  python code_it_v2.py      # smoke test config"
echo "  python code_final_run_v2.py"
