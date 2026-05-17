#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root after cloning thesisall:
#   bash setup_instance.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-venv}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
AUDIO_MAE_FILE_ID="${AUDIO_MAE_FILE_ID:-18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq}"

cd "$REPO_DIR"

echo "==> Installing OS packages"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git wget unzip ffmpeg libsndfile1 python3.10 python3.10-venv
fi

echo "==> Creating Python environment: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "==> Installing CUDA PyTorch stack"
pip uninstall -y torch torchvision torchaudio || true
pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"

echo "==> Installing project requirements"
pip install -r requirements.txt

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
EOF
chmod +x env_instance.sh

echo "==> Compile check"
source env_instance.sh
python -m py_compile code_it_v2.py code_final_run_v2.py official_cqtdiff_adapter.py official_maid_adapter.py

echo "==> Setup complete"
echo "Next:"
echo "  source env_instance.sh"
echo "  python code_it_v2.py      # smoke test config"
echo "  python code_final_run_v2.py"
