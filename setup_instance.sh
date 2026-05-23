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
    autoconf \
    automake \
    build-essential \
    ffmpeg \
    git \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    gtk-doc-tools \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libtool \
    libsndfile1 \
    pkg-config \
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
if [ ! -d external/audio-inpainting-diffusion/.git ]; then
  git clone https://github.com/eloimoliner/audio-inpainting-diffusion.git external/audio-inpainting-diffusion
fi
if [ ! -d external/AudioMAE/.git ]; then
  git clone https://github.com/facebookresearch/AudioMAE.git external/AudioMAE
fi
if [ ! -d external/DDPM-Midi2Performance-Model/.git ]; then
  git clone https://github.com/FlyToYourMooN/DDPM-Midi2Performance-Model.git external/DDPM-Midi2Performance-Model
fi
if [ ! -d external/gstpeaq/.git ]; then
  git clone https://github.com/HSU-ANT/gstpeaq.git external/gstpeaq
fi

echo "==> Patching CQTdiff for compatibility (numpy 2.x + autograd safety)"
# nsgfwin*.py: numpy 2.x casting rules (harmless on 1.x, prevents future breakage)
for NSGFWIN in \
  "external/CQTdiff/src/nsgt/nsgfwin.py" \
  "external/CQTdiff/src/nsgt/nsgfwin_sl.py"
do
  if grep -q 'np.clip(M, min_win, np.inf, out=M)' "$NSGFWIN" 2>/dev/null; then
    sed -i 's/    np.clip(M, min_win, np.inf, out=M)/    M = np.clip(M, min_win, np.inf).astype(int)/' "$NSGFWIN"
    echo "  Patched: $NSGFWIN (numpy clip compatibility)"
  else
    echo "  Already patched or not needed: $NSGFWIN"
  fi
done

# nsigtf.py: replace in-place overlap-add with out-of-place index_add
# Prevents autograd errors if backbone is ever unfrozen for experiments
NSIGTF="external/CQTdiff/src/nsgt/nsigtf.py"
if [ -f "$NSIGTF" ]; then
  python - "$NSIGTF" <<'PATCH'
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
original_text = text

# Remove temp0 pre-allocation
text = text.replace(
    "    temp0 = torch.empty(*cseq_shape[:2], maxLg, dtype=fr.dtype, device=torch.device(device))  # pre-allocation\n",
    "",
)

# Patch matrixform branch
old_matrix = """            t1 = temp0[:, :, :r]
            t2 = temp0[:, :, Lg-l:Lg]

            t1[:, :, :] = t[:, :, :r]
            t2[:, :, :] = t[:, :, maxLg-l:maxLg]

            temp0[:, :, :Lg] *= gdiis[i, :Lg] 
            temp0[:, :, :Lg] *= maxLg

            fr[:, :, wr1] += t2
            fr[:, :, wr2] += t1
"""
new_matrix = """            if Lg - r - l > 0:
                middle = torch.zeros(*cseq_shape[:2], Lg - r - l, dtype=fr.dtype, device=torch.device(device))
                temp = torch.cat([t[:, :, :r], middle, t[:, :, maxLg-l:maxLg]], dim=-1)
            else:
                temp = torch.cat([t[:, :, :r], t[:, :, maxLg-l:maxLg]], dim=-1)
            temp = (temp * gdiis[i, :Lg] * maxLg).to(dtype=fr.dtype)

            wr1_idx = torch.as_tensor(wr1, dtype=torch.long, device=torch.device(device))
            wr2_idx = torch.as_tensor(wr2, dtype=torch.long, device=torch.device(device))
            fr = torch.index_add(fr, 2, wr1_idx, temp[:, :, Lg-l:Lg])
            fr = torch.index_add(fr, 2, wr2_idx, temp[:, :, :r])
"""

# Patch bucket branch
old_bucket = """                t1 = temp0[:, :, :r]
                t2 = temp0[:, :, Lg-l:Lg]

                t1[:, :, :] = t[:, :, :r]
                t2[:, :, :] = t[:, :, Lg-l:Lg]

                temp0[:, :, :Lg] *= gdiis[freq_idx, :Lg] 
                temp0[:, :, :Lg] *= Lg

                fr[:, :, wr1] += t2
                fr[:, :, wr2] += t1
"""
new_bucket = """                if Lg - r - l > 0:
                    middle = torch.zeros(*cseq_shape[:2], Lg - r - l, dtype=fr.dtype, device=torch.device(device))
                    temp = torch.cat([t[:, :, :r], middle, t[:, :, Lg-l:Lg]], dim=-1)
                else:
                    temp = torch.cat([t[:, :, :r], t[:, :, Lg-l:Lg]], dim=-1)
                temp = (temp * gdiis[freq_idx, :Lg] * Lg).to(dtype=fr.dtype)

                wr1_idx = torch.as_tensor(wr1, dtype=torch.long, device=torch.device(device))
                wr2_idx = torch.as_tensor(wr2, dtype=torch.long, device=torch.device(device))
                fr = torch.index_add(fr, 2, wr1_idx, temp[:, :, Lg-l:Lg])
                fr = torch.index_add(fr, 2, wr2_idx, temp[:, :, :r])
"""

if old_matrix in text and old_bucket in text:
    text = text.replace(old_matrix, new_matrix).replace(old_bucket, new_bucket)

# Upgrade earlier versions of this patch that produced ComplexDouble temp tensors
# when multiplying ComplexFloat data with float64 synthesis windows.
text = text.replace(
    "            temp = temp * gdiis[i, :Lg] * maxLg\n",
    "            temp = (temp * gdiis[i, :Lg] * maxLg).to(dtype=fr.dtype)\n",
)
text = text.replace(
    "                temp = temp * gdiis[freq_idx, :Lg] * Lg\n",
    "                temp = (temp * gdiis[freq_idx, :Lg] * Lg).to(dtype=fr.dtype)\n",
)

if text != original_text:
    with open(path, "w") as f:
        f.write(text)
    print(f"  Patched: {path} (autograd-safe index_add + dtype compatibility)")
else:
    print(f"  Already patched or not needed: {path}")
PATCH
else
  echo "  Missing, skip patch: $NSIGTF"
fi

echo "==> Building GstPEAQ"
if [ ! -x external/gstpeaq/src/peaq ] || [ ! -f external/gstpeaq/src/.libs/libgstpeaq.so ]; then
  (cd external/gstpeaq && ./autogen.sh && make -j"$(nproc)")
else
  echo "GstPEAQ binary/plugin already built."
fi

echo "==> Downloading CQT-Diff+ weights"
if [ ! -f external/CQTdiff/experiments/cqt/cqt_weights.pt ]; then
  (cd external/CQTdiff && bash download_weights_and_examples.sh)
else
  echo "CQT-Diff+ weights already present."
fi

echo "==> Downloading audio-inpainting MusicNet CQTdiff+ weights"
mkdir -p external/audio-inpainting-diffusion/experiments
if [ ! -f external/audio-inpainting-diffusion/experiments/musicnet_44k_4s-560000.pt ]; then
  wget -O external/audio-inpainting-diffusion/experiments/musicnet_44k_4s-560000.pt \
    https://huggingface.co/Eloimoliner/audio-inpainting-diffusion/resolve/main/musicnet_44k_4s-560000.pt
else
  echo "Audio-inpainting MusicNet CQTdiff+ weights already present."
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
export AUDIO_INPAINTING_DIR="$PROJECT_ROOT/external/audio-inpainting-diffusion"
export AUDIO_MAE_DIR="$PROJECT_ROOT/external/AudioMAE"
export MIDI2PERFORMANCE_DIR="$PROJECT_ROOT/external/DDPM-Midi2Performance-Model"
export CQTDIFF_WEIGHTS="$CQT_DIFF_DIR/experiments/cqt/cqt_weights.pt"
export AUDIO_INPAINTING_CQTDIFF_WEIGHTS="$AUDIO_INPAINTING_DIR/experiments/musicnet_44k_4s-560000.pt"
export AUDIOMAE_CHECKPOINT="$AUDIO_MAE_DIR/ckpt/pretrained.pth"
export GSTPEAQ_DIR="$PROJECT_ROOT/external/gstpeaq"
export GSTPEAQ_BIN="$GSTPEAQ_DIR/src/peaq"
export GSTPEAQ_PLUGIN="$GSTPEAQ_DIR/src/.libs/libgstpeaq.so"
export OFFICIAL_CQTDIFF_ADAPTER=official_audio_inpainting_cqtdiff_adapter
export MAID_ADAPTER=official_maid_adapter
export PIPELINE_TARGET_SR="${PIPELINE_TARGET_SR:-44100}"
export PIPELINE_SEGMENT_SAMPLES="${PIPELINE_SEGMENT_SAMPLES:-184184}"

# MAID: train from scratch (no pretrained DDPM-Midi2Performance checkpoint required)
export ALLOW_RANDOM_MAID=1

# MusicNet CQTdiff+ diffusion sampling config
export CQTDIFF_DIFFUSION_STEPS="${CQTDIFF_DIFFUSION_STEPS:-35}"
export CQTDIFF_DIFFUSION_XI="${CQTDIFF_DIFFUSION_XI:-0}"
export CQTDIFF_SIGMA_MIN="${CQTDIFF_SIGMA_MIN:-1e-4}"
export CQTDIFF_SIGMA_MAX="${CQTDIFF_SIGMA_MAX:-1.0}"
export CQTDIFF_SIGMA_DATA="${CQTDIFF_SIGMA_DATA:-0.063}"
export CQTDIFF_SCHURN="${CQTDIFF_SCHURN:-10}"
EOF
chmod +x env_instance.sh

echo "==> Compile check"
source env_instance.sh
python - <<'PY'
import numpy as np
import visqol
from visqol import VisqolApi

print("visqol package:", visqol.__file__)

sr = 48000
t = np.arange(sr, dtype=np.float64) / sr
ref = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
deg = ref.copy()

api = VisqolApi()
api.create(mode="audio")
result = api.measure_from_arrays(ref, deg, sample_rate=sr)
print("visqol-python api create/measure: ok", float(result.moslqo))
PY
python -m py_compile code_final_run_v2.py official_cqtdiff_adapter.py official_audio_inpainting_cqtdiff_adapter.py official_maid_adapter.py

echo "==> Audio-inpainting MusicNet CQTdiff+ adapter smoke check"
python - <<'PY'
import os, sys, torch
os.environ["CQTDIFF_DIFFUSION_STEPS"] = "3"
sys.path.insert(0, os.environ["PROJECT_ROOT"])
sys.path.insert(0, os.environ["AUDIO_INPAINTING_DIR"])

from official_audio_inpainting_cqtdiff_adapter import build_cqtdiff_decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = build_cqtdiff_decoder(
    device=device, target_sr=44100, segment_samples=184184,
    gap_durations_ms=[500], audio_inpainting_dir=os.environ["AUDIO_INPAINTING_DIR"],
)
decoder.eval()
print(f"Audio-inpainting MusicNet CQTdiff+ adapter loaded OK on {device}")
print(f"  Diffusion steps: {decoder.DIFFUSION_STEPS}")
print(f"  Native SR: {decoder.native_sr}, Native len: {decoder.native_len}")
print(f"  Backbone params: {sum(p.numel() for p in decoder.backbone.parameters()):,}")

# Quick inpaint test (synthetic)
import numpy as np
audio = np.random.randn(184184).astype(np.float32) * 0.1
mask = np.zeros(184184, dtype=bool)
gap_samples = int(round(44100 * 500 / 1000))
gap_start = len(audio) // 2 - gap_samples // 2
gap_end = gap_start + gap_samples
mask[gap_start:gap_end] = True
audio[gap_start:gap_end] = 0.0

with torch.inference_mode():
    mt = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    mk = torch.from_numpy(mask).unsqueeze(0).to(device)
    recon = decoder.inpaint(mt, mk, conditioning=None)
    rms = np.sqrt(np.mean(recon[gap_start:gap_end] ** 2))
    print(f"  Smoke inpaint (3 steps): gap RMS = {rms:.6f} {'OK' if rms > 0.001 else 'FAIL'}")
PY

echo "==> Setup complete"
echo "Next:"
echo "  source env_instance.sh"
echo "  python code_it_v2.py      # smoke test config"
echo "  python code_final_run_v2.py"
