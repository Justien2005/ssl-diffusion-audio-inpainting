# Generated from: code_it.ipynb
# Converted at: 2026-05-16T16:11:45.272Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 🎵 Music Audio Inpainting Pipeline v2
# 
# Notebook ini mengimplementasikan pipeline hybrid SSL + Diffusion untuk music audio inpainting.
# 
# ## Struktur Notebook
# | Cell | Isi |
# |------|-----|
# | 1 | Cek GPU & Install dependencies |
# | 2 | Mount Google Drive & setup folder |
# | 3 | Download & Preprocessing MusicNet |
# | 4 | Definisi FiLM Layer (improved init) |
# | 5 | Definisi Fungsi Evaluasi (LSD gap-restricted dB, FAD, VISQOL_ODG) |
# | 6 | Helper functions (memory management, checkpoint, group-aware split) |
# | 6.5 | Dataset, DataLoader & Shared Utilities (mask, crossfade) |
# | 6.6 | Training Loop (Reconstruction-based + CFG dropout) |
# | 6.7 | Shared hybrid training helpers, shared decoder builders, checkpoint utilities |
# | 7 | **BASELINE: CQT-Diff+ standalone TRAINED (tanpa encoder)** |
# | 8A | Training CLAP + CQT-Diff+ |
# | 8 | Evaluasi CLAP + CQT-Diff+ |
# | 9A | Training CLAP + MAID |
# | 9 | Evaluasi CLAP + MAID |
# | 10A | Training AudioMAE + CQT-Diff+ |
# | 10 | Evaluasi AudioMAE + CQT-Diff+ |
# | 11A | Training AudioMAE + MAID |
# | 11 | Evaluasi AudioMAE + MAID |
# | 12 | Gabungkan & visualisasikan semua hasil |
# 
# ## Model yang Dievaluasi
# - **Baseline**: CQT-Diff+ (tanpa encoder SSL, tanpa FiLM)
# - **Config 1**: CLAP + CQT-Diff+
# - **Config 2**: CLAP + MAID
# - **Config 3**: AudioMAE + CQT-Diff+
# - **Config 4**: AudioMAE + MAID
# 
# ## Alur Hybrid Terbaru
# - Jalankan cell training hybrid terlebih dahulu untuk menghasilkan checkpoint best per kombinasi.
# - Jika checkpoint sudah ada, cell training akan skip secara default kecuali `FORCE_RETRAIN = True`.
# - Cell evaluasi hybrid selalu mencoba memuat checkpoint terlatih.
# - Jika checkpoint hybrid belum ada, evaluasi akan berhenti dengan pesan yang jelas.
# 
# ## Gap Duration yang Dievaluasi
# 100ms, 300ms, 500ms, 750ms, 1200ms, 1700ms
# 
# ---
# ⚠️ **Pastikan Runtime → Change Runtime Type → GPU (T4) sebelum menjalankan!**


# ---
# ## CELL 1 — Cek GPU & Install Dependencies
# **Jalankan cell ini pertama kali setiap sesi Colab baru.**


# ============================================================
# CELL 1: CEK GPU & INSTALL DEPENDENCIES
# ============================================================
# Cara pakai: Jalankan sekali di awal setiap sesi.
# Estimasi waktu install: ~3-5 menit.
# ============================================================

import subprocess
import sys
import shutil
import platform
import os

CPU_THREAD_LIMIT = os.environ.get("PIPELINE_CPU_THREADS", "1")
for _thread_env in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_env, CPU_THREAD_LIMIT)


def log_gpu_environment():
    """Log environment GPU lengkap untuk dokumentasi run training."""
    print("\n" + "="*70)
    print("GPU / ENVIRONMENT LOG")
    print("="*70)
    print("$ nvidia-smi -q  # notebook equivalent: !nvidia-smi -q")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "-q"], check=False)
    else:
        print("nvidia-smi tidak ditemukan di environment ini.")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch as _torch
        print(f"PyTorch: {_torch.__version__}")
        print(f"CUDA available: {_torch.cuda.is_available()}")
        print(f"CUDA version: {_torch.version.cuda if _torch.version.cuda else 'not available'}")
        if _torch.cuda.is_available():
            props = _torch.cuda.get_device_properties(0)
            print(f"GPU name: {props.name}")
            print(f"VRAM: {props.total_memory / (1024**3):.2f} GiB")
    except Exception as e:
        print(f"Torch/CUDA detail belum bisa dibaca: {e}")
    print("="*70 + "\n")


log_gpu_environment()

# --- Cek GPU ---
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU aktif: {gpu_name}")
        print(f"✅ VRAM tersedia: {vram:.1f} GB")
    else:
        print("❌ GPU tidak aktif! Pergi ke Runtime → Change Runtime Type → GPU")
        sys.exit()
except ImportError:
    print("PyTorch belum terinstall, melanjutkan instalasi...")

print("\n📦 Menginstall dependencies...")

packages = [
    # "librosa",       # Load audio, CQT, Mel-spectrogram
    # "soundfile",     # Baca/tulis file audio
    # "audioread",     # Backend untuk librosa
    # "transformers",  # Load CLAP dan AudioMAE dari HuggingFace
    # "accelerate",    # Loading model besar lebih efisien
    # "einops",        # Operasi tensor yang lebih mudah dibaca
    # "timm",          # Library model vision, dipakai AudioMAE
    # "tqdm",          # Progress bar
    # "pandas",        # Simpan hasil evaluasi ke tabel
    # "matplotlib",    # Visualisasi hasil
    # "scipy",         # Operasi sinyal
    # "numpy",         # Operasi array numerik
    # "resampy",       # Resampling audio berkualitas tinggi
    # "torchvggish",   # VGGish embeddings untuk FAD legacy
    # "git+https://github.com/ashvala/AQUA-tk.git",  # Legacy perceptual package, tidak dipakai untuk metrik final
    # "visqol-lib-py", # ViSQOL perceptual quality metric
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], capture_output=True)

# Clone repo CQT-Diff+ dari GitHub ke folder portabel.
# Repo resmi: https://github.com/eloimoliner/CQTdiff
# Vast.ai/local: gunakan PROJECT_ROOT (default: current working directory), bukan path Colab.
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")
CQT_DIFF_DIR = os.environ.get("CQT_DIFF_DIR", os.path.join(EXTERNAL_DIR, "CQTdiff"))
AUDIO_INPAINTING_DIR = os.environ.get(
    "AUDIO_INPAINTING_DIR", os.path.join(EXTERNAL_DIR, "audio-inpainting-diffusion")
)
AUDIO_MAE_DIR = os.environ.get("AUDIO_MAE_DIR", os.path.join(EXTERNAL_DIR, "AudioMAE"))
MIDI2PERFORMANCE_DIR = os.environ.get(
    "MIDI2PERFORMANCE_DIR",
    os.path.join(EXTERNAL_DIR, "DDPM-Midi2Performance-Model"),
)
os.makedirs(EXTERNAL_DIR, exist_ok=True)

if os.path.exists(os.path.join(CQT_DIFF_DIR, ".git")):
    print(f"  CQT-Diff+ repository sudah ada: {CQT_DIFF_DIR}")
else:
    print(f"  Cloning CQT-Diff+ repository ke {CQT_DIFF_DIR}...")
    subprocess.run(
        ["git", "clone", "https://github.com/eloimoliner/CQTdiff.git", CQT_DIFF_DIR],
        check=False
    )
if os.path.exists(os.path.join(AUDIO_INPAINTING_DIR, ".git")):
    print(f"  Audio-inpainting CQTdiff+ repository sudah ada: {AUDIO_INPAINTING_DIR}")
else:
    print(f"  Cloning Audio-inpainting CQTdiff+ repository ke {AUDIO_INPAINTING_DIR}...")
    subprocess.run(
        ["git", "clone", "https://github.com/eloimoliner/audio-inpainting-diffusion.git", AUDIO_INPAINTING_DIR],
        check=False
    )

# Tambahkan repo ke Python path agar bisa di-import
if CQT_DIFF_DIR not in sys.path:
    sys.path.insert(0, CQT_DIFF_DIR)
if AUDIO_INPAINTING_DIR not in sys.path:
    sys.path.insert(0, AUDIO_INPAINTING_DIR)
if os.path.isdir(AUDIO_MAE_DIR) and AUDIO_MAE_DIR not in sys.path:
    sys.path.insert(0, AUDIO_MAE_DIR)
MIDI2PERFORMANCE_MAIN_DIR = os.path.join(MIDI2PERFORMANCE_DIR, "main")
if os.path.isdir(MIDI2PERFORMANCE_MAIN_DIR) and MIDI2PERFORMANCE_MAIN_DIR not in sys.path:
    sys.path.insert(0, MIDI2PERFORMANCE_MAIN_DIR)

print("\nDependencies siap.")
print("\nMode final: semua komponen model harus memakai implementasi asli.")
print("   Proxy/replika dinonaktifkan; pipeline akan berhenti jika model asli belum dikonfigurasi.")

# ---
# ## CELL 2 — Mount Google Drive & Setup Folder


# ============================================================
# CELL 2: MOUNT GOOGLE DRIVE & SETUP FOLDER
# ============================================================
# Cara pakai:
# - IS_LOCAL = True  -> Jalankan di lokal (tidak perlu Google Drive)
# - IS_LOCAL = False -> Jalankan di Google Colab dengan Google Drive
# ============================================================

import os

# --- PARAMETER -----------------------------------------------
PIPELINE_STAGE_NAME = "code_v4_musicnet_cqtdiffplus_44k"  # Native audio-inpainting CQTdiff+: 44.1 kHz, 184184 samples.
IS_LOCAL = True   # Vast.ai/local default. Ganti ke False hanya jika menggunakan Google Colab.
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
BASE_LOCAL_ROOT = os.environ.get("MUSIC_INPAINTING_ROOT", os.path.join(PROJECT_ROOT, "music_inpainting"))
LOCAL_ROOT = os.path.join(BASE_LOCAL_ROOT, "training_stages", PIPELINE_STAGE_NAME)
# -------------------------------------------------------------

if IS_LOCAL:
    BASE_DATA_ROOT = BASE_LOCAL_ROOT
    DATA_ROOT = LOCAL_ROOT
    print(f"Mode lokal aktif. Base dataset folder: {BASE_DATA_ROOT}")
else:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DATA_ROOT = "/content/drive/MyDrive/music_inpainting"
    DATA_ROOT = os.path.join(BASE_DATA_ROOT, "training_stages", PIPELINE_STAGE_NAME)
    print(f"Google Drive terpasang. Base dataset folder: {BASE_DATA_ROOT}")

print(f"Training stage: {PIPELINE_STAGE_NAME}")
print(f"Stage output root: {DATA_ROOT}")

PATHS = {
    # Dataset mentah dibagi antar stage agar tidak perlu download ulang.
    "dataset":      os.path.join(BASE_DATA_ROOT, "dataset"),
    # Artefak berikut diisolasi per stage agar pipeline lama tidak tertimpa.
    "preprocessed": os.path.join(DATA_ROOT, "preprocessed"),
    "masked":       os.path.join(DATA_ROOT, "masked"),
    "outputs":      os.path.join(DATA_ROOT, "outputs"),
    "results":      os.path.join(DATA_ROOT, "results"),
    "checkpoints":  os.path.join(DATA_ROOT, "checkpoints"),
    "logs":         os.path.join(DATA_ROOT, "logs"),
    "plots":        os.path.join(DATA_ROOT, "plots"),
}

# Final thesis guardrail: do not use local proxy/reimplementation models.
# CQTdiff+ and MAID need thin adapter modules because their official code does
# not expose the exact FiLM-training interface used by this notebook.
OFFICIAL_MODELS_ONLY = True
OFFICIAL_CQTDIFF_ADAPTER = os.environ.get(
    "OFFICIAL_CQTDIFF_ADAPTER", "official_audio_inpainting_cqtdiff_adapter"
)
MAID_ADAPTER = os.environ.get("MAID_ADAPTER", "official_maid_adapter")

# Diagnostic experiment: train the official CQTdiff+ backbone together with
# the adapter reconstruction head. This is heavier, but helps test whether the
# silent/noisy gap comes from the frozen backbone + small head setup.
CQTDIFF_TRAIN_BACKBONE_EXPERIMENT = False
os.environ["CQTDIFF_TRAIN_BACKBONE"] = "1" if CQTDIFF_TRAIN_BACKBONE_EXPERIMENT else "0"
print(f"MusicNet CQTdiff+ train backbone experiment: CQTDIFF_TRAIN_BACKBONE={os.environ['CQTDIFF_TRAIN_BACKBONE']}")


def validate_official_model_configuration():
    if not OFFICIAL_MODELS_ONLY:
        raise RuntimeError("Final pipeline harus berjalan dengan OFFICIAL_MODELS_ONLY=True.")

    import importlib.util

    required_adapters = {
        "MusicNet CQTdiff+ original": OFFICIAL_CQTDIFF_ADAPTER,
        "MAID original DDPM-Midi2Performance": MAID_ADAPTER,
    }
    missing = [
        f"{label}: module '{module_name}'"
        for label, module_name in required_adapters.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise RuntimeError(
            "Final pipeline disetel official-only, tetapi adapter model asli belum tersedia:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\n\nBuat module adapter tersebut di PYTHONPATH atau set env var "
              "OFFICIAL_CQTDIFF_ADAPTER/MAID_ADAPTER/MIDI2PERFORMANCE_DIR ke nilai yang benar."
        )


validate_official_model_configuration()

for name, path in PATHS.items():
    os.makedirs(path, exist_ok=True)
    print(f"Folder '{name}': {path}")

# Buat subfolder output untuk setiap model (termasuk baseline)
ALL_MODELS = ["baseline_cqtdiff", "baseline_cqtdiff_finetuned", "clap_cqtdiff", "clap_maid", "audiomae_cqtdiff", "audiomae_maid"]
for model in ALL_MODELS:
    os.makedirs(os.path.join(PATHS["outputs"], model), exist_ok=True)
    os.makedirs(os.path.join(PATHS["checkpoints"], model), exist_ok=True)
    os.makedirs(os.path.join(PATHS["plots"], model), exist_ok=True)


# Persistent Vast.ai-safe logging and environment capture.
import sys
import subprocess
import shutil
import platform
import json
import logging
import atexit
from datetime import datetime

RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
TRAINING_LOG_PATH = os.path.join(PATHS["logs"], "training_progress.log")
ERROR_LOG_PATH = os.path.join(PATHS["logs"], "errors.log")
VALIDATION_LOG_PATH = os.path.join(PATHS["logs"], "validation_metrics.log")
GPU_LOG_PATH = os.path.join(PATHS["logs"], "gpu_environment.json")
NVIDIA_SMI_Q_PATH = os.path.join(PATHS["logs"], "nvidia_smi_q.txt")


class TeeStream:
    """Mirror notebook stdout/stderr to persistent log files."""
    _pipeline_tee = True

    def __init__(self, stream, log_handle):
        self.stream = stream
        self.log_handle = log_handle

    def write(self, data):
        self.stream.write(data)
        self.log_handle.write(data)
        self.log_handle.flush()

    def flush(self):
        self.stream.flush()
        self.log_handle.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


def configure_persistent_logging():
    """Persist prints, errors, and logger records outside notebook output."""
    global _TRAINING_LOG_HANDLE, _ERROR_LOG_HANDLE
    os.makedirs(PATHS["logs"], exist_ok=True)

    if not getattr(sys.stdout, "_pipeline_tee", False):
        _TRAINING_LOG_HANDLE = open(TRAINING_LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(sys.stdout, _TRAINING_LOG_HANDLE)
        atexit.register(_TRAINING_LOG_HANDLE.close)

    if not getattr(sys.stderr, "_pipeline_tee", False):
        _ERROR_LOG_HANDLE = open(ERROR_LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stderr = TeeStream(sys.stderr, _ERROR_LOG_HANDLE)
        atexit.register(_ERROR_LOG_HANDLE.close)

    logger = logging.getLogger("music_inpainting")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    existing_paths = {getattr(h, "baseFilename", None) for h in logger.handlers}
    for log_path in [TRAINING_LOG_PATH, VALIDATION_LOG_PATH, ERROR_LOG_PATH]:
        if log_path not in existing_paths:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            handler.setLevel(logging.INFO if log_path != ERROR_LOG_PATH else logging.ERROR)
            logger.addHandler(handler)
    return logger


LOGGER = configure_persistent_logging()
LOGGER.info("Run started | stage=%s | run_id=%s", PIPELINE_STAGE_NAME, RUN_ID)


def capture_environment_snapshot():
    """Save nvidia-smi -q plus Python/CUDA/Torch/GPU details to logs/."""
    env = {
        "run_id": RUN_ID,
        "stage": PIPELINE_STAGE_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": None,
        "cuda_version": None,
        "torch_version": None,
        "gpu_name": None,
        "vram_gib": None,
    }

    try:
        import torch as _torch
        env["torch_version"] = _torch.__version__
        env["cuda_available"] = bool(_torch.cuda.is_available())
        env["cuda_version"] = _torch.version.cuda
        if _torch.cuda.is_available():
            props = _torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["vram_gib"] = props.total_memory / (1024 ** 3)
    except Exception as exc:
        env["torch_error"] = repr(exc)

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        with open(NVIDIA_SMI_Q_PATH, "w", encoding="utf-8") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
        env["nvidia_smi_q_path"] = NVIDIA_SMI_Q_PATH
        env["nvidia_smi_returncode"] = result.returncode
    else:
        with open(NVIDIA_SMI_Q_PATH, "w", encoding="utf-8") as f:
            f.write("nvidia-smi not found in this environment.\n")
        env["nvidia_smi_q_path"] = NVIDIA_SMI_Q_PATH
        env["nvidia_smi_returncode"] = None

    with open(GPU_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    LOGGER.info("Environment snapshot saved: %s", GPU_LOG_PATH)
    print(f"Persistent logs: {PATHS['logs']}")
    print(f"Environment snapshot: {GPU_LOG_PATH}")
    print(f"nvidia-smi -q log: {NVIDIA_SMI_Q_PATH}")
    return env


ENVIRONMENT_SNAPSHOT = capture_environment_snapshot()

print("\nSemua folder stage siap!")
print(f"Root folder stage: {DATA_ROOT}")


# ---
# ## CELL 3 — Download & Preprocessing MusicNet
# 
# ⚠️ **Jalankan SEKALI saja.** Hasil disimpan ke Drive dan tidak perlu diulang.


# ============================================================
# CELL 3: DOWNLOAD & PREPROCESSING MUSICNET
# ============================================================
# Cara pakai:
# - Jalankan SEKALI saja. Hasil disimpan ke Google Drive.
# - Set SKIP_IF_EXISTS = True jika preprocessing sudah pernah
#   dilakukan sebelumnya untuk menghemat waktu.
# - Estimasi waktu: ~15-30 menit
# ============================================================

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import urllib.request
import random
import time

# ============================================================
# KONFIGURASI
# ============================================================

# Set True jika preprocessing sudah pernah dijalankan
SKIP_IF_EXISTS = True

# Seed tetap untuk semua sampling dataset agar eksperimen reproducible
DATASET_RANDOM_SEED = 42

# Native MusicNet CQTdiff+ configuration from audio-inpainting-diffusion.
# The default backbone is trained at 44.1 kHz and 184184 samples (~4.18 s).
CQT_NATIVE_SR = int(os.environ.get("PIPELINE_TARGET_SR", "44100"))
CQT_NATIVE_SAMPLES = int(os.environ.get("PIPELINE_SEGMENT_SAMPLES", "184184"))
TARGET_SR = CQT_NATIVE_SR
SEGMENT_SAMPLES = CQT_NATIVE_SAMPLES
SEGMENT_DURATION = SEGMENT_SAMPLES / TARGET_SR
EXPERIMENT_CONFIG_ID = (
    f"musicnet_cqtdiffplus_sr{TARGET_SR}_n{SEGMENT_SAMPLES}_"
    f"dur{SEGMENT_DURATION:.6f}s"
)

# Gap duration yang dievaluasi (dalam milidetik) — DIPERBARUI
GAP_DURATIONS_MS = [100, 300, 500, 750, 1200, 1700]

# Persentase dataset yang digunakan
DATASET_FRACTION = 0.5

# Jumlah sampel evaluasi (bisa override via env var untuk test run cepat)
N_EVAL_SAMPLES = int(os.environ.get("N_EVAL_SAMPLES", "10"))
# Eval final harus fresh by default: hapus WAV rekonstruksi lama sebelum inpaint,
# lalu generate ulang semua output untuk checkpoint/model yang sedang diload.
EVAL_REUSE_RECONSTRUCTIONS = os.environ.get("EVAL_REUSE_RECONSTRUCTIONS", "0").lower() in {"1", "true", "yes", "on"}
EVAL_CLEAR_RECONSTRUCTIONS = os.environ.get("EVAL_CLEAR_RECONSTRUCTIONS", "1").lower() in {"1", "true", "yes", "on"}
EVAL_GAP_POSITION = os.environ.get("EVAL_GAP_POSITION", "center").strip().lower()
if EVAL_GAP_POSITION not in {"center", "random"}:
    raise ValueError("EVAL_GAP_POSITION harus 'center' atau 'random'.")
EVAL_RANDOM_GAP_MIN_CONTEXT_MS = int(os.environ.get("EVAL_RANDOM_GAP_MIN_CONTEXT_MS", "250"))
EVAL_GAP_WINDOW_PERCEPTUAL = os.environ.get("EVAL_GAP_WINDOW_PERCEPTUAL", "0").lower() in {"1", "true", "yes", "on"}
EVAL_GAP_WINDOW_PAD_MS = int(os.environ.get("EVAL_GAP_WINDOW_PAD_MS", "250"))

# Jumlah segmen maksimal per lagu
MAX_SEGMENTS_PER_FILE = 5

# Metadata MusicNet dipakai untuk stratified sampling composer + instrument
MUSICNET_METADATA_URL = "https://zenodo.org/record/5120004/files/musicnet_metadata.csv"


def gap_context_seconds(gap_ms, segment_samples=SEGMENT_SAMPLES, sr=TARGET_SR):
    """Return left/right context for center-gap evaluation in seconds."""
    gap_samples = int(round(sr * gap_ms / 1000))
    context_samples = max(0, segment_samples - gap_samples)
    return (context_samples / 2) / sr


def log_native_experiment_setup():
    print("\n" + "=" * 70)
    print("NATIVE CQT-DIFF EXPERIMENT SETUP")
    print("=" * 70)
    print(f"Config id        : {EXPERIMENT_CONFIG_ID}")
    print(f"Target SR        : {TARGET_SR} Hz")
    print(f"Segment samples  : {SEGMENT_SAMPLES}")
    print(f"Segment duration : {SEGMENT_DURATION:.3f} s")
    print(f"Eval gap position: {EVAL_GAP_POSITION}")
    print("Effective center-gap context per side:")
    for gap_ms in GAP_DURATIONS_MS:
        print(f"  - {gap_ms:4d} ms gap -> {gap_context_seconds(gap_ms):.3f} s left/right")
    print("=" * 70 + "\n")


log_native_experiment_setup()


# ============================================================
# FUNGSI DOWNLOAD
# ============================================================

def download_musicnet_metadata():
    """
    Download metadata MusicNet untuk mengambil label composer dan instrument/ensemble.
    Jika download gagal, pipeline tetap jalan dengan stratifikasi fallback "unknown".
    """
    dataset_dir = PATHS["dataset"]
    metadata_path = os.path.join(dataset_dir, "musicnet_metadata.csv")

    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)

    try:
        print("📥 Downloading MusicNet metadata...")
        urllib.request.urlretrieve(MUSICNET_METADATA_URL, metadata_path)
        return pd.read_csv(metadata_path)
    except Exception as e:
        print(f"⚠️ Metadata MusicNet tidak bisa didownload ({e}).")
        print("   Stratified sampling akan fallback ke label composer/instrument='unknown'.")
        return pd.DataFrame()


def download_musicnet():
    """
    Download dataset MusicNet dari Zenodo.
    MusicNet berisi 330 rekaman musik klasik.
    Kita hanya pakai subset sesuai DATASET_FRACTION untuk efisiensi komputasi.
    """
    dataset_dir = PATHS["dataset"]
    audio_dir = os.path.join(dataset_dir, "audio")

    if os.path.exists(audio_dir) and len(os.listdir(audio_dir)) > 10:
        print("✅ Dataset sudah ada di Drive, skip download.")
        return audio_dir

    os.makedirs(audio_dir, exist_ok=True)
    MUSICNET_URL = "https://zenodo.org/record/5120004/files/musicnet.tar.gz"
    tar_path = os.path.join(dataset_dir, "musicnet.tar.gz")
    

    if os.path.exists(tar_path) and os.path.getsize(tar_path) > 1e10:
        print(f"Menggunakan archive MusicNet yang sudah ada: {tar_path}")
    else:
        print("?? Downloading MusicNet audio files...")
        print("   Estimasi ukuran: ~11GB full archive")

        def progress_hook(count, block_size, total_size):
            percent = min(count * block_size * 100 / total_size, 100)
            print(f"\r  Progress: {percent:.1f}%", end="")

        urllib.request.urlretrieve(MUSICNET_URL, tar_path, progress_hook)
        print("\n  Download selesai.")

    print("  Mengekstrak MusicNet audio...")
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tar:
        wav_members = [m for m in tar.getmembers() if m.name.endswith('.wav')]
        for member in tqdm(wav_members, desc="Extracting"):
            tar.extract(member, audio_dir)

    print(f"? Dataset berhasil diekstrak ke {audio_dir}")
    return audio_dir


# ============================================================
# FUNGSI STRATIFIED SAMPLING
# ============================================================

def _track_id_from_path(audio_path):
    """Ambil MusicNet track id dari nama file audio, misal 1727.wav -> 1727."""
    return os.path.splitext(os.path.basename(audio_path))[0]


def _normalise_musicnet_metadata(metadata_df):
    """Rapikan nama kolom metadata agar robust terhadap variasi source."""
    if metadata_df is None or metadata_df.empty:
        return pd.DataFrame()

    meta = metadata_df.copy()
    meta.columns = [str(c).strip().lower() for c in meta.columns]

    if "id" not in meta.columns:
        return pd.DataFrame()

    meta["track_id"] = meta["id"].astype(str)
    if "composer" not in meta.columns:
        meta["composer"] = "unknown"

    # MusicNet metadata umum memakai ensemble; beberapa mirror menyediakan instrument.
    instrument_col = None
    for col in ["instrument", "instruments", "ensemble"]:
        if col in meta.columns:
            instrument_col = col
            break
    meta["instrument"] = meta[instrument_col] if instrument_col else "unknown"

    meta["composer"] = meta["composer"].fillna("unknown").astype(str)
    meta["instrument"] = meta["instrument"].fillna("unknown").astype(str)
    return meta[["track_id", "composer", "instrument"]].drop_duplicates("track_id")


def stratified_sample_dataframe(df, n_samples, stratify_cols, seed=DATASET_RANDOM_SEED):
    """
    Ambil sample deterministic dengan proporsi strata sebisa mungkin terjaga.
    Dipakai untuk stratifikasi composer + instrument pada pemilihan file dan eval.
    """
    if len(df) == 0 or n_samples <= 0:
        return df.iloc[0:0].copy()

    n_samples = min(int(n_samples), len(df))
    rng = np.random.default_rng(seed)

    work = df.copy().reset_index(drop=True)
    work["_sample_row_id"] = np.arange(len(work))
    for col in stratify_cols:
        if col not in work.columns:
            work[col] = "unknown"
        work[col] = work[col].fillna("unknown").astype(str)

    grouped = list(work.groupby(stratify_cols, dropna=False, sort=True))
    quotas = []
    for group_key, group in grouped:
        expected = len(group) * n_samples / len(work)
        base = int(np.floor(expected))
        quotas.append({
            "key": group_key,
            "group": group,
            "quota": min(base, len(group)),
            "fractional": expected - base,
            "tie": rng.random(),
        })

    remaining = n_samples - sum(q["quota"] for q in quotas)
    for q in sorted(quotas, key=lambda item: (-item["fractional"], item["tie"])):
        if remaining <= 0:
            break
        capacity = len(q["group"]) - q["quota"]
        if capacity > 0:
            q["quota"] += 1
            remaining -= 1

    sampled_parts = []
    for offset, q in enumerate(quotas):
        if q["quota"] > 0:
            sampled_parts.append(q["group"].sample(q["quota"], random_state=seed + offset))

    if sampled_parts:
        sampled = pd.concat(sampled_parts, ignore_index=True)
    else:
        sampled = work.iloc[0:0].copy()

    if len(sampled) < n_samples:
        missing = n_samples - len(sampled)
        sampled_ids = set(sampled["_sample_row_id"]) if "_sample_row_id" in sampled.columns else set()
        unsampled = work[~work["_sample_row_id"].isin(sampled_ids)]
        if len(unsampled) > 0:
            sampled = pd.concat([
                sampled,
                unsampled.sample(min(missing, len(unsampled)), random_state=seed + 999)
            ], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=seed).drop(columns=["_sample_row_id"], errors="ignore").reset_index(drop=True)


def select_stratified_audio_files(all_audio_files, metadata_df, fraction, seed=DATASET_RANDOM_SEED):
    """Pilih subset file audio dengan seed 42 dan stratifikasi composer + instrument."""
    audio_df = pd.DataFrame({"audio_path": sorted(all_audio_files)})
    audio_df["source_file"] = audio_df["audio_path"].apply(os.path.basename)
    audio_df["track_id"] = audio_df["audio_path"].apply(_track_id_from_path)

    meta = _normalise_musicnet_metadata(metadata_df)
    if not meta.empty:
        audio_df = audio_df.merge(meta, on="track_id", how="left")

    for col in ["composer", "instrument"]:
        if col not in audio_df.columns:
            audio_df[col] = "unknown"
        audio_df[col] = audio_df[col].fillna("unknown").astype(str)

    n_files = max(1, int(round(len(audio_df) * fraction)))
    selected = stratified_sample_dataframe(
        audio_df,
        n_samples=n_files,
        stratify_cols=["composer", "instrument"],
        seed=seed,
    )
    return selected


# ============================================================
# FUNGSI PREPROCESSING
# ============================================================

def preprocess_audio(audio_path):
    """
    Preprocessing standar untuk satu file audio:
    1. Load audio
    2. Konversi ke mono
    3. Resample ke TARGET_SR (44.1 kHz, native MusicNet CQTdiff+)
    4. Normalisasi RMS ke target level (-23 dBFS approx)

    Menggunakan RMS normalization alih-alih peak normalization
    agar dynamic range antar segmen tetap terjaga — penting untuk
    ViSQOL yang sensitif terhadap loudness statistics.
    """
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 0.07  # approx -23 dBFS
    if rms > 1e-6:
        audio = audio * (target_rms / rms)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def split_into_segments(audio, file_seed=0):
    """
    Potong audio panjang jadi segmen native MusicNet CQTdiff+ (~4.18 detik).
    Ambil maksimal MAX_SEGMENTS_PER_FILE secara random
    agar dataset lebih beragam.

    file_seed: per-file seed agar hasil reproducible di setiap run.
    """
    rng = random.Random(file_seed)
    if len(audio) < SEGMENT_SAMPLES:
        return []
    possible_starts = list(range(0, len(audio) - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES))
    n_segments = min(MAX_SEGMENTS_PER_FILE, len(possible_starts))
    selected_starts = rng.sample(possible_starts, n_segments)
    return [audio[s : s + SEGMENT_SAMPLES] for s in selected_starts]


def compute_gap_bounds(audio_length, gap_ms, sr=TARGET_SR, gap_start=None):
    """Compute sample-exact gap bounds within one native segment."""
    gap_samples = int(round(sr * gap_ms / 1000))
    if gap_samples <= 0 or gap_samples >= audio_length:
        raise ValueError(
            f"Gap {gap_ms}ms tidak valid untuk audio_length={audio_length}, sr={sr}."
        )
    if gap_start is None:
        center = audio_length // 2
        gap_start = center - gap_samples // 2
    gap_start = int(gap_start)
    gap_end = gap_start + gap_samples
    if gap_start < 0 or gap_end > audio_length:
        raise ValueError(
            f"Gap bounds keluar audio: start={gap_start}, end={gap_end}, length={audio_length}."
        )
    return gap_start, gap_end


def build_gap_mask_array(audio_length, gap_ms, sr=TARGET_SR, gap_start=None):
    gap_start, gap_end = compute_gap_bounds(audio_length, gap_ms, sr=sr, gap_start=gap_start)
    mask = np.zeros(audio_length, dtype=bool)
    mask[gap_start:gap_end] = True
    return mask, gap_start, gap_end


def apply_gap_mask(audio_segment, gap_ms, sr=TARGET_SR):
    """
    Buat versi audio dengan gap di tengah segmen.

    Gap ditempatkan di tengah agar model punya konteks
    yang seimbang di kiri dan kanan.

    Menggunakan int(round(...)) agar gap_samples selalu konsisten
    antara preprocessing dan inference — menghindari off-by-one.

    Returns:
        masked_audio : audio dengan gap diisi nol
        mask         : boolean array (True = posisi gap)
        gap_start    : indeks awal gap
        gap_end      : indeks akhir gap
    """
    mask, gap_start, gap_end = build_gap_mask_array(len(audio_segment), gap_ms, sr=sr)

    masked_audio = audio_segment.copy()
    masked_audio[gap_start:gap_end] = 0.0

    return masked_audio, mask, gap_start, gap_end


# ============================================================
# JALANKAN PREPROCESSING
# ============================================================



def write_preprocessing_timing(status, total_seconds, total_segments=0):
    row = {
        "stage": globals().get("PIPELINE_STAGE_NAME", "code_v3"),
        "status": status,
        "dataset_fraction": DATASET_FRACTION,
        "total_segments": int(total_segments or 0),
        "preprocessing_seconds": float(total_seconds or 0.0),
        "preprocessing_time": f"{float(total_seconds or 0.0):.1f}s",
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    timing_path = os.path.join(PATHS["results"], "preprocessing_timing.csv")
    pd.DataFrame([row]).to_csv(timing_path, index=False)
    print(f"Preprocessing timing saved: {timing_path}")


def _current_preprocessing_config():
    return {
        "config_id": EXPERIMENT_CONFIG_ID,
        "target_sr": int(TARGET_SR),
        "segment_samples": int(SEGMENT_SAMPLES),
        "segment_duration": float(SEGMENT_DURATION),
        "gap_durations_ms": list(GAP_DURATIONS_MS),
        "dataset_fraction": float(DATASET_FRACTION),
        "max_segments_per_file": int(MAX_SEGMENTS_PER_FILE),
        "seed": int(DATASET_RANDOM_SEED),
    }


def _preprocessing_config_matches(config_path, metadata_path):
    """Return True only if cached preprocessing belongs to this native CQT setup."""
    if not os.path.exists(config_path) or not os.path.exists(metadata_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        expected = _current_preprocessing_config()
        keys = ["config_id", "target_sr", "segment_samples", "gap_durations_ms", "seed"]
        if any(cached.get(key) != expected.get(key) for key in keys):
            return False
        meta = pd.read_csv(metadata_path)
        if meta.empty:
            return False
        if "sample_rate" in meta.columns and not (meta["sample_rate"].astype(int) == TARGET_SR).all():
            return False
        if "n_samples" in meta.columns and not (meta["n_samples"].astype(int) == SEGMENT_SAMPLES).all():
            return False
        return True
    except Exception as exc:
        print(f"⚠️ Cached preprocessing config tidak valid ({exc}); preprocessing akan dibuat ulang.")
        return False


preprocessing_start = time.perf_counter()

preprocessed_flag = os.path.join(PATHS["preprocessed"], ".done")
preprocessed_metadata = os.path.join(PATHS["preprocessed"], "metadata.csv")
preprocessed_config = os.path.join(PATHS["preprocessed"], "preprocessing_config.json")
expected_mask_dirs = [
    os.path.join(PATHS["masked"], f"gap_{gap_ms}ms")
    for gap_ms in GAP_DURATIONS_MS
]
preprocessed_artifacts_ready = (
    _preprocessing_config_matches(preprocessed_config, preprocessed_metadata)
    and all(os.path.isdir(path) for path in expected_mask_dirs)
)

if SKIP_IF_EXISTS and preprocessed_artifacts_ready:
    print("✅ Preprocessing sudah selesai sebelumnya, skip.")
    print("   Set SKIP_IF_EXISTS = False untuk memaksa preprocessing ulang.")
    if not os.path.exists(preprocessed_flag):
        with open(preprocessed_flag, "w") as f:
            f.write("done")
    write_preprocessing_timing("skipped", time.perf_counter() - preprocessing_start, 0)
else:
    if os.path.exists(preprocessed_metadata) and not _preprocessing_config_matches(preprocessed_config, preprocessed_metadata):
        print("♻️ Preprocessing lama tidak cocok dengan native CQT config; membersihkan artifact stage.")
        for stale_dir in [PATHS["preprocessed"], PATHS["masked"]]:
            if os.path.isdir(stale_dir):
                shutil.rmtree(stale_dir)
            os.makedirs(stale_dir, exist_ok=True)

    audio_dir = download_musicnet()
    metadata_df = download_musicnet_metadata()

    all_audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.endswith('.wav') or f.endswith('.flac'):
                all_audio_files.append(os.path.join(root, f))

    selected_table = select_stratified_audio_files(
        all_audio_files,
        metadata_df=metadata_df,
        fraction=DATASET_FRACTION,
        seed=DATASET_RANDOM_SEED,
    )

    print(f"\n📊 Total file audio: {len(all_audio_files)}")
    print(f"📊 File yang dipakai ({DATASET_FRACTION:.0%}): {len(selected_table)}")
    print(f"📊 Sample rate target: {TARGET_SR} Hz")
    print(f"📊 Random seed: {DATASET_RANDOM_SEED} | Stratified by composer + instrument")
    print(f"📊 Gap durations: {GAP_DURATIONS_MS} ms")
    if {"composer", "instrument"}.issubset(selected_table.columns):
        print("\n📊 Distribusi strata terpilih (top 10):")
        print(selected_table.groupby(["composer", "instrument"]).size().sort_values(ascending=False).head(10).to_string())

    segment_metadata = []
    segment_id = 0

    print("\n🔄 Memulai preprocessing...")
    for file_idx, (_, file_row) in enumerate(tqdm(selected_table.iterrows(), total=len(selected_table), desc="Preprocessing files")):
        filepath = file_row["audio_path"]
        try:
            audio = preprocess_audio(filepath)
            segments = split_into_segments(audio, file_seed=DATASET_RANDOM_SEED + file_idx)

            for segment in segments:
                clean_filename = f"seg_{segment_id:05d}.wav"
                clean_path = os.path.join(PATHS["preprocessed"], clean_filename)
                sf.write(clean_path, segment, TARGET_SR)

                for gap_ms in GAP_DURATIONS_MS:
                    masked_audio, mask, gap_start, gap_end = apply_gap_mask(segment, gap_ms)
                    masked_dir = os.path.join(PATHS["masked"], f"gap_{gap_ms}ms")
                    os.makedirs(masked_dir, exist_ok=True)
                    sf.write(os.path.join(masked_dir, clean_filename), masked_audio, TARGET_SR)

                composer = str(file_row.get("composer", "unknown"))
                instrument = str(file_row.get("instrument", "unknown"))
                segment_metadata.append({
                    "segment_id": segment_id,
                    "source_file": os.path.basename(filepath),
                    "track_id": file_row.get("track_id", _track_id_from_path(filepath)),
                    "composer": composer,
                    "instrument": instrument,
                    "stratify_key": f"{composer}__{instrument}",
                    "clean_path": clean_path,
                    "duration_s": SEGMENT_DURATION,
                    "sample_rate": TARGET_SR,
                    "n_samples": len(segment),
                })
                segment_id += 1

        except Exception as e:
            print(f"\n⚠️ Gagal memproses {filepath}: {e}")
            continue

    meta_df = pd.DataFrame(segment_metadata)
    meta_path = os.path.join(PATHS["preprocessed"], "metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    with open(preprocessed_flag, 'w') as f:
        f.write("done")
    with open(preprocessed_config, "w", encoding="utf-8") as f:
        json.dump(_current_preprocessing_config(), f, indent=2)

    print(f"\n✅ Preprocessing selesai! Total segmen: {segment_id}")
    print(f"   Metadata: {meta_path}")

# ---
# ## CELL 4 — Definisi FiLM Layer


# ============================================================
# CELL 4: DEFINISI FILM LAYER
# ============================================================
# FiLM (Feature-wise Linear Modulation) adalah jembatan antara
# encoder SSL dan decoder diffusion.
#
# Cara kerja:
#   output = gamma * fitur_decoder + beta
#   gamma dan beta dihasilkan dari latent encoder
#
# CATATAN: FiLM hanya digunakan oleh Cell 8-11 (kombinasi hybrid).
# Cell 7 (baseline) tidak menggunakan FiLM sama sekali.
# ============================================================

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer.

    Menyuntikkan informasi encoder ke dalam decoder
    dengan memodulasi fitur-fitur internal decoder.

    Support 2D (B, D) dan 3D (B, T, D) decoder features.
    Untuk 3D, gamma/beta di-broadcast ke semua timestep.

    Init: gamma=1, beta=0 (identity transform) tapi gradien non-zero.
    """

    def __init__(self, encoder_dim: int, decoder_feature_dim: int, hidden_dim: int = None):
        super(FiLMLayer, self).__init__()

        if hidden_dim is None:
            hidden_dim = (encoder_dim + decoder_feature_dim) // 2

        self.decoder_feature_dim = decoder_feature_dim

        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * decoder_feature_dim),
        )

        nn.init.zeros_(self.proj[-1].weight)
        with torch.no_grad():
            self.proj[-1].bias[:decoder_feature_dim].fill_(1.0)
            self.proj[-1].bias[decoder_feature_dim:].fill_(0.0)

    def forward(self, encoder_latent: torch.Tensor, decoder_features: torch.Tensor):
        """
        Args:
            encoder_latent  : (B, encoder_dim)
            decoder_features: (B, D) atau (B, T, D) — last dim = decoder_feature_dim
        Returns:
            Fitur decoder yang sudah dimodulasi, shape sama dengan input
        """
        # gamma, beta: (B, decoder_feature_dim)
        gamma, beta = self.proj(encoder_latent).chunk(2, dim=-1)

        # Broadcast ke sequence dimension kalau decoder_features 3D
        if decoder_features.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)    # (B, 1, D)

        return gamma * decoder_features + beta


# Konfigurasi dimensi FiLM per kombinasi
# encoder_dim        : ukuran output latent encoder
# decoder_feature_dim: ukuran fitur internal decoder yang dimodulasi
FILM_CONFIGS = {
    "baseline_cqtdiff_finetuned": {"encoder_dim": 512, "decoder_feature_dim": 256},
    "clap_cqtdiff":     {"encoder_dim": 512, "decoder_feature_dim": 256},
    "clap_maid":        {"encoder_dim": 512, "decoder_feature_dim": 512},
    "audiomae_cqtdiff": {"encoder_dim": 768, "decoder_feature_dim": 256},
    "audiomae_maid":    {"encoder_dim": 768, "decoder_feature_dim": 512},
    # baseline_cqtdiff tidak ada di sini karena tidak pakai FiLM
}

print("✅ FiLMLayer berhasil didefinisikan!")
print("\n📋 Konfigurasi FiLM per kombinasi:")
for combo, cfg in FILM_CONFIGS.items():
    note = " (zero encoder — ablation)" if combo == "baseline_cqtdiff_finetuned" else ""
    print(f"   {combo}: encoder_dim={cfg['encoder_dim']}, "
          f"decoder_feature_dim={cfg['decoder_feature_dim']}{note}")
print("   baseline_cqtdiff: tidak menggunakan FiLM (pretrained only)")

# ---
# ## CELL 5 — Fungsi Evaluasi (LSD, FAD, VISQOL_ODG)


# ============================================================
# CELL 5: FUNGSI EVALUASI
# ============================================================
# Mendefinisikan 3 metrik evaluasi:
#
# 1. LSD (Log Spectral Distance)
#    - Mengukur perbedaan spektral antara audio asli vs rekonstruksi
#    - Lebih rendah = lebih baik
#    - Range: 0 (sempurna) hingga ~5 (buruk)
#
# 2. FAD (Frechet Audio Distance)
#    - Mengukur jarak distribusi audio asli vs rekonstruksi
#    - Lebih rendah = lebih baik
#    - Dihitung per set, bukan per sample
#
# 3. VISQOL_ODG (ViSQOL Objective Difference Grade)
#    - Mengukur kualitas perseptual berdasarkan reference vs degraded audio
#    - Skala ODG-like: 0 (imperceptible) hingga -4 (sangat buruk)
#    - Lebih tinggi (mendekati 0) = lebih baik
# ============================================================

import numpy as np
import librosa
import pandas as pd

_VISQOL_FALLBACK_WARNED = False
# Aktifkan akselerasi GPU untuk metrik yang sudah punya implementasi torch.
# FAD/ViSQOL/GstPEAQ tetap memakai backend resmi CPU agar definisi metrik tidak berubah.
EVAL_USE_GPU = os.environ.get("EVAL_USE_GPU", "1").lower() in {"1", "true", "yes", "on"}
EVAL_VISQOL_BACKEND = os.environ.get("EVAL_VISQOL_BACKEND", "visqol")
EVAL_USE_GSTPEAQ = os.environ.get("EVAL_USE_GSTPEAQ", "1").lower() in {"1", "true", "yes"}
FAD_USE_VGGISH_PCA = os.environ.get("FAD_USE_VGGISH_PCA", "0").lower() in {"1", "true", "yes", "on"}
FAD_DEBUG_STATS = os.environ.get("FAD_DEBUG_STATS", "1").lower() in {"1", "true", "yes", "on"}
GSTPEAQ_DIR = os.environ.get("GSTPEAQ_DIR", os.path.join(EXTERNAL_DIR, "gstpeaq"))
GSTPEAQ_BIN = os.environ.get("GSTPEAQ_BIN", "")
GSTPEAQ_PLUGIN = os.environ.get("GSTPEAQ_PLUGIN", "")
GSTPEAQ_ADVANCED = os.environ.get("GSTPEAQ_ADVANCED", "0").lower() in {"1", "true", "yes"}


def compute_lsd(original: np.ndarray, reconstructed: np.ndarray,
                sr: int = TARGET_SR, n_fft: int = 2048, hop_length: int = 512,
                gap_start: int = None, gap_end: int = None, frame_pad: int = 2):
    """
    Hitung Log Spectral Distance (LSD) dalam dB.

    LSD dihitung hanya pada gap region (+ frame_pad frame di tiap sisi)
    agar metrik benar-benar mengukur kualitas inpainting, bukan bagian
    non-gap yang sudah diketahui.
    """
    n = min(len(original), len(reconstructed))
    o, r = original[:n], reconstructed[:n]

    O = np.abs(librosa.stft(o, n_fft=n_fft, hop_length=hop_length)) ** 2
    R = np.abs(librosa.stft(r, n_fft=n_fft, hop_length=hop_length)) ** 2

    eps = max(1e-10, 1e-6 * O.max())
    log_diff = 10.0 * (np.log10(O + eps) - np.log10(R + eps))  # dB

    if gap_start is not None and gap_end is not None:
        f_start = max(0, gap_start // hop_length - frame_pad)
        f_end = min(O.shape[1], gap_end // hop_length + frame_pad + 1)
        log_diff = log_diff[:, f_start:f_end]

    lsd = np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0)))
    return float(lsd)


def _metric_gap_bounds(audio_len, gap_ms, sr=TARGET_SR, region=None):
    if region is not None:
        return int(region["gap_start"]), int(region["gap_end"])
    return compute_gap_bounds(audio_len, gap_ms, sr=sr)


def _slice_metric_region(original, reconstructed, gap_start, gap_end):
    n = min(len(original), len(reconstructed))
    gap_start = max(0, min(int(gap_start), n))
    gap_end = max(gap_start, min(int(gap_end), n))
    ref = np.asarray(original[:n], dtype=np.float64)[gap_start:gap_end]
    est = np.asarray(reconstructed[:n], dtype=np.float64)[gap_start:gap_end]
    return ref, est


def compute_gap_snr(original, reconstructed, gap_start, gap_end):
    ref, est = _slice_metric_region(original, reconstructed, gap_start, gap_end)
    if len(ref) == 0:
        return np.nan
    noise = ref - est
    return float(10.0 * np.log10((np.sum(ref ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def compute_gap_si_sdr(original, reconstructed, gap_start, gap_end):
    ref, est = _slice_metric_region(original, reconstructed, gap_start, gap_end)
    if len(ref) == 0:
        return np.nan
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.sum(ref ** 2) + 1e-12
    target = (np.sum(est * ref) / ref_energy) * ref
    error = est - target
    return float(10.0 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(error ** 2) + 1e-12)))


def compute_gap_mel_distance(original, reconstructed, sr=TARGET_SR, gap_start=None, gap_end=None,
                             n_fft=1024, hop_length=256, n_mels=64):
    ref, est = _slice_metric_region(original, reconstructed, gap_start, gap_end)
    if len(ref) == 0:
        return np.nan
    if len(ref) < n_fft:
        pad = n_fft - len(ref)
        ref = np.pad(ref, (0, pad))
        est = np.pad(est, (0, pad))
    ref_mel = librosa.feature.melspectrogram(
        y=ref.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    est_mel = librosa.feature.melspectrogram(
        y=est.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    ref_peak = max(float(np.max(ref_mel)), 1e-10)
    ref_db = librosa.power_to_db(ref_mel, ref=ref_peak)
    est_db = librosa.power_to_db(est_mel, ref=ref_peak)
    frames = min(ref_db.shape[-1], est_db.shape[-1])
    return float(np.mean(np.abs(ref_db[..., :frames] - est_db[..., :frames])))


def compute_gap_window_visqol_odg(original, reconstructed, sr=TARGET_SR, gap_start=None, gap_end=None,
                                  pad_ms=EVAL_GAP_WINDOW_PAD_MS):
    pad = int(round(sr * pad_ms / 1000))
    n = min(len(original), len(reconstructed))
    start = max(0, int(gap_start) - pad)
    end = min(n, int(gap_end) + pad)
    if end <= start:
        return np.nan
    try:
        return compute_visqol_odg(original[start:end], reconstructed[start:end], sr)
    except Exception as exc:
        print(f"    Gap-window ViSQOL gagal ({exc}); nilai diisi NaN.")
        return np.nan


def extract_fad_features(audio_list: list, sr: int = TARGET_SR):
    """
    Ekstrak fitur untuk Frechet Audio Distance.

    FAD selalu menggunakan VGGish embeddings di CPU agar skala metrik konsisten
    dengan pipeline FAD legacy. Tidak ada fallback log-mel untuk FAD.
    Semua patch embedding VGGish dari seluruh set dipakai sebagai sampel FAD;
    embedding tidak dirata-rata per file.
    """
    features = []
    try:
        import inspect
        import torch as _torch
        import torchvggish
        try:
            from torchvggish import vggish_input as _vggish_input
        except Exception:
            _vggish_input = getattr(torchvggish, "vggish_input", None)

        _device = _torch.device("cpu")
        vggish_kwargs = {}
        try:
            sig = inspect.signature(torchvggish.vggish)
            if "postprocess" in sig.parameters:
                # FAD libraries commonly use raw VGGish embeddings
                # (equivalent to use_pca=False). The PCA+8-bit YouTube-8M
                # postprocess space has a 0..255 scale and can inflate FAD
                # into tens of thousands for otherwise reasonable audio.
                vggish_kwargs["postprocess"] = bool(FAD_USE_VGGISH_PCA)
        except (TypeError, ValueError):
            pass

        _vggish = torchvggish.vggish(**vggish_kwargs).to(_device).eval()
        postprocess_active = bool(vggish_kwargs.get("postprocess", False))
        postprocess_active = postprocess_active or bool(getattr(_vggish, "postprocess", False))
        postprocess_active = postprocess_active or bool(getattr(_vggish, "pproc", None) is not None)
        if postprocess_active and not FAD_USE_VGGISH_PCA:
            raise RuntimeError(
                "torchvggish tetap mengaktifkan VGGish post-processing walau "
                "FAD_USE_VGGISH_PCA=0. Ini berpotensi membuat skala FAD meledak. "
                "Gunakan package torchvggish yang mendukung vggish(postprocess=False), "
                "atau set FAD_USE_VGGISH_PCA=1 hanya jika memang ingin FAD pada "
                "embedding PCA+8-bit legacy."
            )
        if _vggish_input is None:
            raise RuntimeError("torchvggish.vggish_input tidak tersedia")

        with _torch.inference_mode():
            for audio in audio_list:
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1, dtype=np.float32)
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                audio_16k = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=16000,
                )
                examples = _vggish_input.waveform_to_examples(audio_16k, 16000)
                if len(examples) == 0:
                    raise RuntimeError("VGGish tidak menghasilkan example untuk salah satu audio.")
                examples = _torch.as_tensor(examples, dtype=_torch.float32, device=_device)
                emb = _vggish(examples)
                emb_np = emb.detach().cpu().numpy()
                if emb_np.dtype == np.uint8:
                    emb_np = emb_np.astype(np.float32)
                features.append(emb_np)
    except Exception as exc:
        raise RuntimeError(
            "FAD wajib memakai VGGish CPU. Install/konfigurasi torchvggish sebelum evaluasi FAD. "
            f"Detail: {exc}"
        ) from exc

    features = np.concatenate(features, axis=0).astype(np.float64, copy=False)
    if features.ndim != 2 or features.shape[0] < 2:
        raise RuntimeError(f"Embedding VGGish FAD tidak cukup: shape={features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("Embedding VGGish FAD berisi NaN/Inf.")
    if FAD_DEBUG_STATS:
        print(
            "    FAD VGGish features: "
            f"shape={features.shape}, mean={features.mean():.4f}, std={features.std():.4f}, "
            f"min={features.min():.4f}, max={features.max():.4f}"
        )
    return features


def compute_fad(original_audios: list, reconstructed_audios: list, sr: int = TARGET_SR):
    """
    Hitung Frechet Audio Distance (FAD) sebagai metrik distribusional per-set.

    FAD tetap terpisah dari VISQOL_ODG: FAD menjawab kemiripan distribusi
    embedding VGGish, sedangkan VISQOL_ODG menjawab kualitas perseptual per
    pasangan audio.
    """
    orig_features = extract_fad_features(original_audios, sr)
    recon_features = extract_fad_features(reconstructed_audios, sr)

    if orig_features.shape[1] != recon_features.shape[1]:
        raise RuntimeError(
            f"Dimensi embedding FAD tidak cocok: original={orig_features.shape}, recon={recon_features.shape}"
        )

    mu1 = np.mean(orig_features, axis=0)
    mu2 = np.mean(recon_features, axis=0)
    d = orig_features.shape[1]

    if len(orig_features) < 2 or len(recon_features) < 2:
        raise RuntimeError(
            f"FAD butuh minimal 2 embedding per set: original={len(orig_features)}, recon={len(recon_features)}"
        )

    sigma1 = np.cov(orig_features, rowvar=False) + 1e-6 * np.eye(d)
    sigma2 = np.cov(recon_features, rowvar=False) + 1e-6 * np.eye(d)
    sigma1 = (sigma1 + sigma1.T) * 0.5
    sigma2 = (sigma2 + sigma2.T) * 0.5

    diff = mu1 - mu2
    mean_diff = np.dot(diff, diff)
    if FAD_DEBUG_STATS:
        if len(orig_features) <= d or len(recon_features) <= d:
            print(
                "    ⚠️ FAD sample count lebih kecil/sama dari dimensi embedding "
                f"(orig={len(orig_features)}, recon={len(recon_features)}, dim={d}); "
                "covariance FAD bisa sangat noisy. Naikkan N_EVAL_SAMPLES untuk hasil final."
            )
        print(f"    FAD mean term: {mean_diff:.4f}")

    def _psd_matrix_sqrt(mat, eps=1e-10):
        mat = (mat + mat.T) * 0.5
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, eps, None)
        return (vecs * np.sqrt(vals)) @ vecs.T

    try:
        sqrt_sigma1 = _psd_matrix_sqrt(sigma1)
        covmean = _psd_matrix_sqrt(sqrt_sigma1 @ sigma2 @ sqrt_sigma1)
    except np.linalg.LinAlgError:
        offset = np.eye(d) * 1e-5
        sqrt_sigma1 = _psd_matrix_sqrt(sigma1 + offset)
        covmean = _psd_matrix_sqrt(sqrt_sigma1 @ (sigma2 + offset) @ sqrt_sigma1)

    fad = mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    if not np.isfinite(fad):
        raise RuntimeError("FAD menghasilkan NaN/Inf.")
    return float(max(0.0, np.real(fad)))


def _eval_device():
    import torch
    return torch.device("cuda" if EVAL_USE_GPU and torch.cuda.is_available() else "cpu")


def _stack_audio_gpu(audio_list, device):
    import torch
    min_len = min(len(audio) for audio in audio_list)
    arr = np.stack([np.asarray(audio[:min_len], dtype=np.float32) for audio in audio_list], axis=0)
    return torch.as_tensor(arr, dtype=torch.float32, device=device), min_len


def _torch_logmel(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    import torch
    import torchaudio

    window = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(
        audio.float(),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    power = spec.abs().pow(2.0)
    mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sr / 2,
        n_mels=n_mels,
        sample_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    ).to(device=audio.device, dtype=torch.float32).T.contiguous()
    mel_power = torch.einsum("mf,bft->bmt", mel_basis, power).clamp_min(1e-10)
    mel_db = 10.0 * torch.log10(mel_power)
    mel_db = mel_db - mel_db.amax(dim=(1, 2), keepdim=True)
    return torch.clamp(mel_db, min=-80.0)


def compute_lsd_batch_gpu(original_audios, reconstructed_audios, gap_ms, sr=TARGET_SR,
                          n_fft=2048, hop_length=512, frame_pad=2):
    import torch

    device = _eval_device()
    if device.type != "cuda":
        raise RuntimeError("GPU tidak tersedia untuk compute_lsd_batch_gpu")

    originals, n = _stack_audio_gpu(original_audios, device)
    recons, _ = _stack_audio_gpu(reconstructed_audios, device)
    originals = originals[:, :n]
    recons = recons[:, :n]

    window = torch.hann_window(n_fft, device=device)
    o_power = torch.stft(originals, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs().pow(2)
    r_power = torch.stft(recons, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs().pow(2)
    eps = torch.clamp(1e-6 * o_power.amax(dim=(1, 2), keepdim=True), min=1e-10)
    log_diff = 10.0 * (torch.log10(o_power + eps) - torch.log10(r_power + eps))

    gap_samples = int(round(sr * gap_ms / 1000))
    center = n // 2
    gap_start = center - gap_samples // 2
    gap_end = gap_start + gap_samples
    f_start = max(0, gap_start // hop_length - frame_pad)
    f_end = min(o_power.shape[-1], gap_end // hop_length + frame_pad + 1)
    log_diff = log_diff[:, :, f_start:f_end]

    scores = torch.sqrt(torch.mean(log_diff.pow(2), dim=1)).mean(dim=1)
    return scores.detach().cpu().numpy().astype(np.float64)


def extract_fad_features_gpu(audio_list, sr=TARGET_SR):
    raise RuntimeError("FAD tidak memakai fitur GPU/log-mel; gunakan VGGish CPU via extract_fad_features().")


def compute_fad_gpu(original_audios, reconstructed_audios, sr=TARGET_SR):
    """Backward-compatible wrapper: FAD tetap dihitung dengan VGGish di CPU."""
    return compute_fad(original_audios, reconstructed_audios, sr)


def compute_nsim_odg_batch_gpu(original_audios, reconstructed_audios, sr=TARGET_SR):
    import torch

    device = _eval_device()
    if device.type != "cuda":
        raise RuntimeError("GPU tidak tersedia untuk compute_nsim_odg_batch_gpu")
    originals, n = _stack_audio_gpu(original_audios, device)
    recons, _ = _stack_audio_gpu(reconstructed_audios, device)
    originals = originals[:, :n]
    recons = recons[:, :n]

    with torch.inference_mode():
        orig_log = _torch_logmel(originals, sr)
        recon_log = _torch_logmel(recons, sr)
        frames = min(orig_log.shape[-1], recon_log.shape[-1])
        orig_log = orig_log[..., :frames]
        recon_log = recon_log[..., :frames]

        c1, c2 = 1e-4, 1e-4
        mu_o = orig_log.mean(dim=1)
        mu_r = recon_log.mean(dim=1)
        sig_o = orig_log.std(dim=1, unbiased=False)
        sig_r = recon_log.std(dim=1, unbiased=False)
        sig_or = ((orig_log - mu_o.unsqueeze(1)) * (recon_log - mu_r.unsqueeze(1))).mean(dim=1)
        ssim = ((2 * mu_o * mu_r + c1) * (2 * sig_or + c2)) / (
            (mu_o.pow(2) + mu_r.pow(2) + c1) * (sig_o.pow(2) + sig_r.pow(2) + c2)
        )
        mean_ssim = torch.clamp(ssim, 0, 1).mean(dim=1)
        odg = torch.clamp(-4.0 * (1.0 - torch.sqrt(mean_ssim)), min=-4.0, max=0.0)
    return odg.detach().cpu().numpy().astype(np.float64)


def _moslqo_to_odg(moslqo):
    return float(np.clip(float(moslqo) - 5.0, -4.0, 0.0))


def _compute_visqol_python_odg(original, reconstructed, sr):
    """ViSQOL fallback via visqol-python pure Python API."""
    import visqol
    from visqol import VisqolApi

    orig_48k = librosa.resample(original, orig_sr=sr, target_sr=48000).astype(np.float64)
    recon_48k = librosa.resample(reconstructed, orig_sr=sr, target_sr=48000).astype(np.float64)
    n = min(len(orig_48k), len(recon_48k))
    orig_48k, recon_48k = orig_48k[:n], recon_48k[:n]

    try:
        api = VisqolApi()
        api.create(mode="audio")
        result = api.measure_from_arrays(orig_48k, recon_48k, sample_rate=48000)
    except Exception as exc:
        raise RuntimeError(
            "visqol-python gagal. Kemungkinan ada stale/mixed ViSQOL files di venv. "
            f"visqol={getattr(visqol, '__file__', None)}"
        ) from exc
    return _moslqo_to_odg(result.moslqo)


def _compute_visqol_odg(original, reconstructed, sr):
    """Fallback perceptual ODG via ViSQOL audio mode."""
    return _compute_visqol_python_odg(original, reconstructed, sr)


def _compute_nsim_odg(original, reconstructed, sr):
    """Last-resort fallback: NSIM pada log-mel, dipetakan ke ODG-like scale."""
    orig_mel = librosa.feature.melspectrogram(
        y=original, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    recon_mel = librosa.feature.melspectrogram(
        y=reconstructed, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )

    orig_log = librosa.power_to_db(orig_mel, ref=np.max)
    recon_log = librosa.power_to_db(recon_mel, ref=np.max)

    C1, C2 = 1e-4, 1e-4
    mu_o = np.mean(orig_log, axis=0)
    mu_r = np.mean(recon_log, axis=0)
    sig_o = np.std(orig_log, axis=0)
    sig_r = np.std(recon_log, axis=0)
    sig_or = np.mean((orig_log - mu_o) * (recon_log - mu_r), axis=0)

    ssim = ((2 * mu_o * mu_r + C1) * (2 * sig_or + C2)) / \
           ((mu_o**2 + mu_r**2 + C1) * (sig_o**2 + sig_r**2 + C2))
    mean_ssim = float(np.mean(np.clip(ssim, 0, 1)))

    odg = -4.0 * (1.0 - mean_ssim ** 0.5)
    return float(np.clip(odg, -4.0, 0.0))


def compute_visqol_odg(original: np.ndarray, reconstructed: np.ndarray, sr: int = TARGET_SR):
    """
    Hitung ViSQOL perceptual quality sebagai Objective Difference Grade-like score.

    Primary path memakai ViSQOL audio mode. Untuk hasil paper, kegagalan ViSQOL
    harus fail-fast; NSIM hanya boleh dipakai untuk smoke test eksplisit.
    """
    global _VISQOL_FALLBACK_WARNED

    min_len = min(len(original), len(reconstructed))
    original = np.asarray(original[:min_len], dtype=np.float64)
    reconstructed = np.asarray(reconstructed[:min_len], dtype=np.float64)

    visqol_sr = sr
    if visqol_sr not in (44100, 48000):
        original = librosa.resample(original, orig_sr=sr, target_sr=44100)
        reconstructed = librosa.resample(reconstructed, orig_sr=sr, target_sr=44100)
        visqol_sr = 44100

    try:
        return _compute_visqol_odg(original, reconstructed, visqol_sr)
    except Exception as exc:
        allow_nsim = os.environ.get("ALLOW_NSIM_VISQOL_FALLBACK", "0").lower() in {"1", "true", "yes"}
        if allow_nsim:
            if not _VISQOL_FALLBACK_WARNED:
                print(f"⚠️ ViSQOL gagal ({exc}); fallback ke NSIM ODG-like untuk smoke test.")
                _VISQOL_FALLBACK_WARNED = True
            return _compute_nsim_odg(original, reconstructed, visqol_sr)
        raise RuntimeError(
            "ViSQOL gagal, sehingga VISQOL_ODG tidak boleh diisi dengan proxy NSIM untuk hasil paper. "
            "Perbaiki instalasi ViSQOL atau set ALLOW_NSIM_VISQOL_FALLBACK=1 hanya untuk smoke test."
        ) from exc


def _find_gstpeaq_binary():
    import shutil

    candidates = [
        GSTPEAQ_BIN,
        os.path.join(GSTPEAQ_DIR, "src", "peaq"),
        os.path.join(GSTPEAQ_DIR, "src", "peaq.exe"),
        shutil.which("peaq"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Executable GstPEAQ 'peaq' tidak ditemukan. Build external/gstpeaq terlebih dahulu "
        "atau set GSTPEAQ_BIN=/path/to/peaq. Set EVAL_USE_GSTPEAQ=0 hanya jika ingin skip PEAQ_ODG."
    )


def _find_gstpeaq_plugin():
    candidates = [
        GSTPEAQ_PLUGIN,
        os.path.join(GSTPEAQ_DIR, "src", ".libs", "libgstpeaq.so"),
        os.path.join(GSTPEAQ_DIR, "src", ".libs", "libgstpeaq.dylib"),
        os.path.join(GSTPEAQ_DIR, "src", ".libs", "gstpeaq.dll"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return None


def _write_peaq_wav(path, audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, 48000, subtype="PCM_16")


def compute_gstpeaq_odg(original: np.ndarray, reconstructed: np.ndarray, sr: int = TARGET_SR):
    """Hitung ODG PEAQ asli via GstPEAQ CLI. Ini metrik terpisah dari VISQOL_ODG."""
    import re
    import tempfile

    peaq_bin = _find_gstpeaq_binary()
    peaq_plugin = _find_gstpeaq_plugin()

    min_len = min(len(original), len(reconstructed))
    original = np.asarray(original[:min_len], dtype=np.float32)
    reconstructed = np.asarray(reconstructed[:min_len], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="gstpeaq_", dir=PATHS["outputs"]) as tmpdir:
        ref_path = os.path.join(tmpdir, "ref.wav")
        test_path = os.path.join(tmpdir, "test.wav")
        _write_peaq_wav(ref_path, original, sr)
        _write_peaq_wav(test_path, reconstructed, sr)

        cmd = [peaq_bin, "--gst-disable-segtrap"]
        if peaq_plugin:
            cmd.append(f"--gst-plugin-load={peaq_plugin}")
        if GSTPEAQ_ADVANCED:
            cmd.append("--advanced")
        cmd.extend([ref_path, test_path])

        env = os.environ.copy()
        env["LC_ALL"] = "C"
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        output = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode != 0:
            raise RuntimeError(f"GstPEAQ gagal dengan exit code {proc.returncode}: {output.strip()}")

    match = re.search(r"Objective Difference Grade:\s*([-+]?\d+(?:\.\d+)?)", output)
    if not match:
        raise RuntimeError(f"Output GstPEAQ tidak memuat ODG: {output.strip()}")
    odg = float(match.group(1))
    if not np.isfinite(odg):
        raise RuntimeError("GstPEAQ menghasilkan ODG NaN/Inf.")
    return odg


# Backward-compatible aliases for the actual GstPEAQ metric.
compute_peaq_odg = compute_gstpeaq_odg
compute_peaq = compute_gstpeaq_odg


def _regions_are_centered(regions, audio_len, gap_ms, sr=TARGET_SR):
    if not regions:
        return True
    center_start, center_end = compute_gap_bounds(audio_len, gap_ms, sr=sr)
    return all(
        int(region.get("gap_start", -1)) == center_start
        and int(region.get("gap_end", -1)) == center_end
        for region in regions
    )


def evaluate_all_gaps(original_audios: list, reconstructed_dict: dict, sr: int = TARGET_SR,
                      gap_regions_by_gap: dict = None):
    """
    Evaluasi semua gap duration untuk satu model.

    Returns:
        DataFrame dengan full-clip metrics dan gap-focused metrics.
    """
    results = []
    use_gpu_metrics = _eval_device().type == "cuda"
    if use_gpu_metrics:
        print(
            "  Fast GPU evaluation aktif: LSD batch di CUDA; "
            f"VISQOL_ODG backend={EVAL_VISQOL_BACKEND}; FAD tetap VGGish CPU."
        )

    for gap_ms, recon_audios in reconstructed_dict.items():
        print(f"  Evaluating gap {gap_ms}ms...")

        regions = (gap_regions_by_gap or {}).get(gap_ms)
        if not regions:
            regions = [
                {"gap_start": compute_gap_bounds(len(orig), gap_ms, sr=sr)[0],
                 "gap_end": compute_gap_bounds(len(orig), gap_ms, sr=sr)[1],
                 "gap_position": "center"}
                for orig in original_audios
            ]
        centered_regions = _regions_are_centered(regions, len(original_audios[0]), gap_ms, sr)

        # Hitung gap indices (konsisten dengan apply_gap_mask)
        gap_samples = int(round(sr * gap_ms / 1000))
        fad_score = compute_fad(original_audios, recon_audios, sr)

        if use_gpu_metrics and centered_regions:
            try:
                lsd_scores = compute_lsd_batch_gpu(original_audios, recon_audios, gap_ms, sr)
                lsd_gap_only_scores = compute_lsd_batch_gpu(
                    original_audios, recon_audios, gap_ms, sr, frame_pad=0
                )
                if EVAL_VISQOL_BACKEND in {"fast_gpu", "gpu", "nsim"}:
                    visqol_odg_scores = compute_nsim_odg_batch_gpu(original_audios, recon_audios, sr)
                else:
                    visqol_odg_scores = [compute_visqol_odg(orig, recon, sr) for orig, recon in zip(original_audios, recon_audios)]
            except Exception as exc:
                print(f"⚠️ Fast GPU evaluation gagal ({exc}); fallback ke CPU metric path.")
                lsd_scores = []
                lsd_gap_only_scores = []
                visqol_odg_scores = []
                for orig, recon in zip(original_audios, recon_audios):
                    center = len(orig) // 2
                    gap_start = center - gap_samples // 2
                    gap_end = gap_start + gap_samples

                    lsd_scores.append(compute_lsd(orig, recon, sr,
                                                  gap_start=gap_start, gap_end=gap_end))
                    lsd_gap_only_scores.append(compute_lsd(
                        orig, recon, sr, gap_start=gap_start, gap_end=gap_end, frame_pad=0
                    ))
                    visqol_odg_scores.append(compute_visqol_odg(orig, recon, sr))
        else:
            lsd_scores = []
            lsd_gap_only_scores = []
            visqol_odg_scores = []
            for idx, (orig, recon) in enumerate(zip(original_audios, recon_audios)):
                gap_start, gap_end = _metric_gap_bounds(
                    len(orig), gap_ms, sr=sr, region=regions[idx] if idx < len(regions) else None
                )

                lsd_scores.append(compute_lsd(orig, recon, sr,
                                              gap_start=gap_start, gap_end=gap_end))
                lsd_gap_only_scores.append(compute_lsd(
                    orig, recon, sr, gap_start=gap_start, gap_end=gap_end, frame_pad=0
                ))
                visqol_odg_scores.append(compute_visqol_odg(orig, recon, sr))

        gap_snr_scores = []
        gap_si_sdr_scores = []
        gap_mel_scores = []
        gap_window_visqol_scores = []
        for idx, (orig, recon) in enumerate(zip(original_audios, recon_audios)):
            gap_start, gap_end = _metric_gap_bounds(
                len(orig), gap_ms, sr=sr, region=regions[idx] if idx < len(regions) else None
            )
            gap_snr_scores.append(compute_gap_snr(orig, recon, gap_start, gap_end))
            gap_si_sdr_scores.append(compute_gap_si_sdr(orig, recon, gap_start, gap_end))
            gap_mel_scores.append(compute_gap_mel_distance(
                orig, recon, sr=sr, gap_start=gap_start, gap_end=gap_end
            ))
            if EVAL_GAP_WINDOW_PERCEPTUAL:
                gap_window_visqol_scores.append(compute_gap_window_visqol_odg(
                    orig, recon, sr=sr, gap_start=gap_start, gap_end=gap_end
                ))

        peaq_odg_scores = []
        if EVAL_USE_GSTPEAQ:
            print("    Computing PEAQ_ODG via GstPEAQ...")
            for orig, recon in zip(original_audios, recon_audios):
                peaq_odg_scores.append(compute_gstpeaq_odg(orig, recon, sr))

        results.append({
            "gap_ms": gap_ms,
            "gap_position": EVAL_GAP_POSITION,
            "LSD": round(np.mean(lsd_scores), 4),
            "LSD_GAP_ONLY": round(np.mean(lsd_gap_only_scores), 4),
            "GAP_LSD": round(np.mean(lsd_gap_only_scores), 4),
            "GAP_SI_SDR": round(np.nanmean(gap_si_sdr_scores), 4),
            "GAP_SNR": round(np.nanmean(gap_snr_scores), 4),
            "GAP_MEL_DISTANCE": round(np.nanmean(gap_mel_scores), 4),
            "FAD": round(fad_score, 4),
            "VISQOL_ODG": round(np.mean(visqol_odg_scores), 4),
            "PEAQ_ODG": round(np.mean(peaq_odg_scores), 4) if peaq_odg_scores else np.nan,
            "GAP_WINDOW_VISQOL_ODG": round(np.nanmean(gap_window_visqol_scores), 4)
            if gap_window_visqol_scores else np.nan,
        })

    return pd.DataFrame(results)


print("✅ Fungsi evaluasi (LSD, FAD, VISQOL_ODG, PEAQ_ODG) berhasil didefinisikan!")

# ---
# ## CELL 6 — Helper Functions


# ============================================================
# CELL 6: HELPER FUNCTIONS
# ============================================================
# Fungsi pendukung untuk:
# - Monitoring penggunaan VRAM
# - Membersihkan memori GPU setelah selesai satu model
# - Menyimpan hasil ke Google Drive
# - Mengecek apakah model sudah pernah dijalankan
# - Loading data yang sudah dipreprocess
# ============================================================

import torch
import gc
import os
import pandas as pd
import soundfile as sf
import random
import json
import logging
from datetime import datetime


def print_gpu_usage(label: str = ""):
    """
    Tampilkan penggunaan VRAM saat ini.

    Cara pakai:
        print_gpu_usage("Sebelum load encoder")
        # load model...
        print_gpu_usage("Sesudah load encoder")
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        free      = total - reserved
        label_str = f"[{label}] " if label else ""
        print(f"🖥️  GPU {label_str}| Terpakai: {allocated:.2f}GB | "
              f"Reserved: {reserved:.2f}GB | Bebas: {free:.2f}GB / {total:.2f}GB")


def clear_gpu_memory(*models):
    """
    Bebaskan VRAM setelah selesai menggunakan model.

    Cara pakai:
        clear_gpu_memory(encoder, decoder, film_layer)
        # atau untuk baseline:
        clear_gpu_memory(decoder)
    """
    print_gpu_usage("Sebelum clear")
    for model in models:
        if model is not None:
            del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print_gpu_usage("Sesudah clear")
    print("✅ GPU memory berhasil dibersihkan\n")


def save_results(results_df: pd.DataFrame, model_name: str):
    """
    Simpan hasil evaluasi ke Google Drive.

    Menyimpan dua file:
    1. File per model: {model_name}_results.csv
    2. Master file: all_results.csv (gabungan semua model)

    Args:
        results_df : DataFrame hasil evaluasi
        model_name : Nama model (misal: "baseline_cqtdiff", "clap_cqtdiff")
    """
    results_df = results_df.copy()
    results_df["model"] = model_name
    results_df["experiment_config_id"] = EXPERIMENT_CONFIG_ID
    results_df["target_sr"] = TARGET_SR
    results_df["segment_samples"] = SEGMENT_SAMPLES
    if "gap_position" not in results_df.columns:
        results_df["gap_position"] = EVAL_GAP_POSITION
    results_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Simpan file per model
    result_artifact_name = evaluation_artifact_name(model_name)
    model_path = os.path.join(PATHS["results"], f"{result_artifact_name}_results.csv")
    results_df.to_csv(model_path, index=False)
    print(f"💾 Hasil {model_name} disimpan: {model_path}")

    # Update master file
    master_path = os.path.join(PATHS["results"], "all_results.csv")
    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        if "gap_position" not in existing.columns:
            existing["gap_position"] = "center"
        existing = existing[
            ~((existing["model"] == model_name) & (existing["gap_position"] == EVAL_GAP_POSITION))
        ]  # Hapus hasil lama untuk model + eval gap position yang sama
        combined = pd.concat([existing, results_df], ignore_index=True)
    else:
        combined = results_df

    combined.to_csv(master_path, index=False)
    print(f"💾 Master file diupdate: {master_path}")
    update_experiment_summary()



EXPECTED_MODEL_CONFIGS = [
    "baseline_cqtdiff",
    "baseline_cqtdiff_finetuned",
    "clap_cqtdiff",
    "clap_maid",
    "audiomae_cqtdiff",
    "audiomae_maid",
]
TRAINING_TIMING_SUMMARY = []


def format_duration(seconds):
    """Format durasi detik menjadi string ringkas."""
    seconds = float(seconds or 0.0)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{seconds:.1f}s"





def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def make_json_safe(value):
    """Convert numpy/pandas scalars so experiment summaries can be dumped as JSON."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def get_peak_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_peak_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _read_single_timing_seconds(path, column):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if not df.empty and column in df.columns:
                return safe_float(df[column].iloc[-1], 0.0)
        except Exception:
            return 0.0
    return 0.0


def model_name_from_label(label):
    raw = str(label).strip()
    if raw in EXPECTED_MODEL_CONFIGS or raw in ALL_MODELS:
        return raw
    text = raw.lower().replace("+", " ").replace("-", " ")
    text = " ".join(text.split())
    explicit = {
        "baseline cqtdiff": "baseline_cqtdiff",
        "baseline cqt diff": "baseline_cqtdiff",
        "baseline fine tuned no ssl": "baseline_cqtdiff_finetuned",
        "baseline finetuned no ssl": "baseline_cqtdiff_finetuned",
        "fine tuned baseline no ssl": "baseline_cqtdiff_finetuned",
        "finetuned baseline no ssl": "baseline_cqtdiff_finetuned",
        "clap cqtdiff": "clap_cqtdiff",
        "clap cqt diff": "clap_cqtdiff",
        "clap maid": "clap_maid",
        "audiomae cqtdiff": "audiomae_cqtdiff",
        "audiomae cqt diff": "audiomae_cqtdiff",
        "audiomae maid": "audiomae_maid",
    }
    if text in explicit:
        return explicit[text]
    if "baseline" in text and ("fine tuned" in text or "finetuned" in text or "no ssl" in text):
        return "baseline_cqtdiff_finetuned"
    if "baseline" in text:
        return "baseline_cqtdiff"
    if "clap" in text and "maid" in text:
        return "clap_maid"
    if "clap" in text and "cqt" in text:
        return "clap_cqtdiff"
    if "audiomae" in text and "maid" in text:
        return "audiomae_maid"
    if "audiomae" in text and "cqt" in text:
        return "audiomae_cqtdiff"
    return str(label).strip().lower().replace(" ", "_")


def save_training_history_artifacts(model_name, history):
    """Persist train/validation loss history plus PNG plot for each model."""
    if not history:
        return None

    os.makedirs(PATHS["logs"], exist_ok=True)
    os.makedirs(PATHS["plots"], exist_ok=True)
    hist_df = pd.DataFrame(history)
    history_path = os.path.join(PATHS["logs"], f"{model_name}_training_history.csv")
    hist_df.to_csv(history_path, index=False)

    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        if "train_loss" in hist_df.columns:
            ax1.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="train_loss")
        if "val_loss" in hist_df.columns and hist_df["val_loss"].notna().any():
            val_df = hist_df.dropna(subset=["val_loss"])
            ax1.plot(val_df["epoch"], val_df["val_loss"], marker="s", label="val_loss")
        ax1.set_title(f"Training History - {model_name}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")
        fig.tight_layout()
        plot_path = os.path.join(PATHS["plots"], f"{model_name}_training_history.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        plot_path = None
        print(f"Could not save training plot for {model_name}: {exc}")

    logging.getLogger("music_inpainting").info("Training history saved for %s: %s", model_name, history_path)
    return {"history_csv": history_path, "plot_png": plot_path}


def update_experiment_summary():
    """Write CSV/JSON summary for all configured baselines and hybrids."""
    rows = []
    timing_path = os.path.join(PATHS["results"], "training_timing_summary.csv")
    eval_timing_path = os.path.join(PATHS["results"], "evaluation_timing_summary.csv")
    results_path = os.path.join(PATHS["results"], "all_results.csv")
    preprocessing_path = os.path.join(PATHS["results"], "preprocessing_timing.csv")

    timing_df = pd.read_csv(timing_path) if os.path.exists(timing_path) else pd.DataFrame()
    eval_df = pd.read_csv(eval_timing_path) if os.path.exists(eval_timing_path) else pd.DataFrame()
    results_df = pd.read_csv(results_path) if os.path.exists(results_path) else pd.DataFrame()
    preprocessing_seconds = _read_single_timing_seconds(preprocessing_path, "preprocessing_seconds")

    for model_name in EXPECTED_MODEL_CONFIGS:
        row = {
            "stage": globals().get("PIPELINE_STAGE_NAME", "code_v3"),
            "model": model_name,
            "dataset_fraction": globals().get("DATASET_FRACTION", None),
            "experiment_config_id": EXPERIMENT_CONFIG_ID,
            "target_sr": TARGET_SR,
            "segment_samples": SEGMENT_SAMPLES,
            "segment_duration": SEGMENT_DURATION,
            "eval_gap_position": EVAL_GAP_POSITION,
            "batch_size": None,
            "epochs": None,
            "training_seconds": None,
            "training_time": None,
            "preprocessing_seconds": preprocessing_seconds,
            "evaluation_seconds": None,
            "evaluation_time": None,
            "peak_vram_gb": None,
            "checkpoint_path": None,
            "final_LSD_mean": None,
            "final_LSD_GAP_ONLY_mean": None,
            "final_GAP_LSD_mean": None,
            "final_GAP_SI_SDR_mean": None,
            "final_GAP_SNR_mean": None,
            "final_GAP_MEL_DISTANCE_mean": None,
            "final_FAD_mean": None,
            "final_VISQOL_ODG_mean": None,
            "final_PEAQ_ODG_mean": None,
            "status": "pending",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not timing_df.empty and "model" in timing_df.columns:
            m = timing_df[timing_df["model"] == model_name]
            if not m.empty:
                last = m.iloc[-1]
                row.update({
                    "batch_size": last.get("batch_size"),
                    "epochs": last.get("num_epochs"),
                    "training_seconds": safe_float(last.get("total_seconds")),
                    "training_time": last.get("total_time"),
                    "peak_vram_gb": safe_float(last.get("peak_vram_gb")),
                    "checkpoint_path": last.get("checkpoint_path"),
                    "status": last.get("status", "trained"),
                })

        if not eval_df.empty and "model" in eval_df.columns:
            e = eval_df[eval_df["model"] == model_name]
            if "eval_gap_position" in e.columns:
                e = e[e["eval_gap_position"].fillna("center") == EVAL_GAP_POSITION]
            if not e.empty:
                last = e.iloc[-1]
                row["evaluation_seconds"] = safe_float(last.get("evaluation_seconds"))
                row["evaluation_time"] = last.get("evaluation_time")
                row["status"] = "evaluated"

        if not results_df.empty and "model" in results_df.columns:
            r = results_df[results_df["model"] == model_name]
            if "gap_position" in r.columns:
                r = r[r["gap_position"].fillna("center") == EVAL_GAP_POSITION]
            if not r.empty:
                for metric in [
                    "LSD", "LSD_GAP_ONLY", "GAP_LSD", "GAP_SI_SDR", "GAP_SNR",
                    "GAP_MEL_DISTANCE", "FAD", "VISQOL_ODG", "PEAQ_ODG",
                    "GAP_WINDOW_VISQOL_ODG",
                ]:
                    if metric in r.columns:
                        row[f"final_{metric}_mean"] = safe_float(r[metric].mean())
                row["status"] = "evaluated"

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(PATHS["results"], "experiment_summary.csv")
    json_path = os.path.join(PATHS["results"], "experiment_summary.json")
    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(rows), f, indent=2)
    print(f"Experiment summary saved: {csv_path}")
    print(f"Experiment summary JSON saved: {json_path}")
    return summary_df


def record_evaluation_timing(model_name, total_seconds, n_eval_samples=None):
    row = {
        "stage": globals().get("PIPELINE_STAGE_NAME", "code_v3"),
        "model": model_name,
        "n_eval_samples": n_eval_samples,
        "experiment_config_id": EXPERIMENT_CONFIG_ID,
        "target_sr": TARGET_SR,
        "segment_samples": SEGMENT_SAMPLES,
        "eval_gap_position": EVAL_GAP_POSITION,
        "evaluation_seconds": float(total_seconds or 0.0),
        "evaluation_time": format_duration(total_seconds or 0.0),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    timing_path = os.path.join(PATHS["results"], "evaluation_timing_summary.csv")
    if os.path.exists(timing_path):
        timing_df = pd.read_csv(timing_path)
        if "eval_gap_position" not in timing_df.columns:
            timing_df["eval_gap_position"] = "center"
        timing_df = timing_df[
            ~((timing_df["model"] == model_name) & (timing_df["eval_gap_position"] == EVAL_GAP_POSITION))
        ]
        timing_df = pd.concat([timing_df, pd.DataFrame([row])], ignore_index=True)
    else:
        timing_df = pd.DataFrame([row])
    timing_df.to_csv(timing_path, index=False)
    logging.getLogger("music_inpainting").info("Evaluation timing saved for %s: %.2fs", model_name, total_seconds)
    update_experiment_summary()
    return row

def check_batch_size_memory(batch_size, min_expected_vram_gb=8.0):
    """Cetak warning jika VRAM runtime terlihat terlalu kecil untuk batch size stage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU wajib aktif untuk pipeline ini. Aktifkan GPU runtime sebelum menjalankan notebook.")

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"GPU check: {props.name} | VRAM {total_gb:.2f} GiB | batch_size={batch_size}")
    if total_gb < min_expected_vram_gb:
        print(
            f"VRAM terdeteksi {total_gb:.2f} GiB, di bawah estimasi aman "
            f"{min_expected_vram_gb:.1f} GiB untuk batch size {batch_size}. "
            "Jika OOM, gunakan smoke stage atau GPU dengan VRAM lebih besar."
        )
        return False
    print(f"? VRAM memenuhi estimasi minimum batch size {batch_size}.")
    return True


def record_training_timing(model_name, total_seconds, num_epochs, lr, batch_size=None,
                           dataset_fraction=None, best_val_loss=None, status="trained",
                           checkpoint_path=None, epoch_times=None, peak_vram_gb=None):
    """Simpan ringkasan waktu training per model ke memory dan CSV stage."""
    batch_size = batch_size if batch_size is not None else globals().get("BATCH_SIZE")
    dataset_fraction = dataset_fraction if dataset_fraction is not None else globals().get("DATASET_FRACTION")
    epoch_times = list(epoch_times or [])
    avg_epoch_seconds = float(np.mean(epoch_times)) if epoch_times else 0.0
    row = {
        "stage": globals().get("PIPELINE_STAGE_NAME", "code_v3"),
        "model": model_name,
        "status": status,
        "experiment_config_id": EXPERIMENT_CONFIG_ID,
        "target_sr": TARGET_SR,
        "segment_samples": SEGMENT_SAMPLES,
        "segment_duration": SEGMENT_DURATION,
        "dataset_fraction": dataset_fraction,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "total_seconds": float(total_seconds or 0.0),
        "total_time": format_duration(total_seconds or 0.0),
        "avg_epoch_seconds": avg_epoch_seconds,
        "avg_epoch_time": format_duration(avg_epoch_seconds),
        "epoch_times_json": json.dumps(epoch_times),
        "best_val_loss": best_val_loss,
        "peak_vram_gb": peak_vram_gb if peak_vram_gb is not None else get_peak_vram_gb(),
        "checkpoint_path": checkpoint_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    TRAINING_TIMING_SUMMARY.append(row)

    timing_path = os.path.join(PATHS["results"], "training_timing_summary.csv")
    if os.path.exists(timing_path):
        timing_df = pd.read_csv(timing_path)
        timing_df = timing_df[timing_df["model"] != model_name]
        timing_df = pd.concat([timing_df, pd.DataFrame([row])], ignore_index=True)
    else:
        timing_df = pd.DataFrame([row])
    timing_df.to_csv(timing_path, index=False)
    logging.getLogger("music_inpainting").info("Training timing saved for %s: %.2fs", model_name, total_seconds)
    print(f"Timing summary updated: {timing_path}")
    update_experiment_summary()


def print_training_timing_summary(expected_models=None):
    """Print timing summary untuk baseline + semua kombinasi hybrid."""
    expected_models = expected_models or EXPECTED_MODEL_CONFIGS
    timing_path = os.path.join(PATHS["results"], "training_timing_summary.csv")
    if os.path.exists(timing_path):
        timing_df = pd.read_csv(timing_path)
    else:
        timing_df = pd.DataFrame(TRAINING_TIMING_SUMMARY)

    print("\n" + "="*70)
    print("TRAINING TIME SUMMARY")
    print("="*70)
    print(f"Total expected configurations: {len(expected_models)}")
    print("Baseline models: baseline_cqtdiff")
    print("Hybrid combinations: clap_cqtdiff, clap_maid, audiomae_cqtdiff, audiomae_maid")

    if timing_df.empty:
        print("Belum ada timing training yang tercatat.")
        return

    cols = ["model", "status", "dataset_fraction", "batch_size", "num_epochs", "total_time", "best_val_loss"]
    available_cols = [c for c in cols if c in timing_df.columns]
    print(timing_df[available_cols].to_string(index=False))

    completed = set(timing_df["model"].tolist()) if "model" in timing_df.columns else set()
    missing = [m for m in expected_models if m not in completed]
    if missing:
        print(f"Timing belum ada untuk: {missing}")
    else:
        print("? Timing tersedia untuk semua 5 konfigurasi.")


def print_final_training_summary(available_models=None):
    """Ringkasan akhir untuk baseline + 4 kombinasi model."""
    available_models = set(list(available_models or []))
    print("\n" + "="*70)
    print("FINAL TRAINING SUMMARY")
    print("="*70)
    print(f"Stage: {globals().get('PIPELINE_STAGE_NAME', 'code_v3')}")
    print(f"Dataset fraction: {globals().get('DATASET_FRACTION', 'unknown')}")
    print(f"Total configurations: {len(EXPECTED_MODEL_CONFIGS)}")
    for model_name in EXPECTED_MODEL_CONFIGS:
        kind = "baseline" if model_name.startswith("baseline_") else "hybrid"
        status = "evaluated" if model_name in available_models else "pending/no results yet"
        print(f"  - {model_name:<20} | {kind:<8} | {status}")
    print_training_timing_summary(EXPECTED_MODEL_CONFIGS)

def check_if_done(model_name: str):
    """
    Cek apakah model ini sudah pernah dijalankan.

    Berguna saat Colab crash: model yang sudah selesai
    tidak perlu diulang.

    Returns:
        True  : sudah selesai, bisa di-skip
        False : belum selesai, perlu dijalankan
    """
    result_path = os.path.join(PATHS["results"], f"{evaluation_artifact_name(model_name)}_results.csv")
    if os.path.exists(result_path):
        print(f"✅ {model_name} sudah selesai sebelumnya.")
        print(f"   Untuk menjalankan ulang, hapus: {result_path}")
        return True
    return False


def _stratified_sample_table(df, n_samples, seed=DATASET_RANDOM_SEED,
                             stratify_cols=("composer", "instrument")):
    """Sampling deterministic dengan proporsi composer + instrument sebisa mungkin terjaga."""
    if len(df) == 0 or n_samples <= 0:
        return df.iloc[0:0].copy()

    work = df.copy().reset_index(drop=True)
    work["_sample_row_id"] = np.arange(len(work))
    n_samples = min(int(n_samples), len(work))
    for col in stratify_cols:
        if col not in work.columns:
            work[col] = "unknown"
        work[col] = work[col].fillna("unknown").astype(str)

    rng = np.random.default_rng(seed)
    grouped = list(work.groupby(list(stratify_cols), dropna=False, sort=True))
    quotas = []
    for _, group in grouped:
        expected = len(group) * n_samples / len(work)
        base = int(np.floor(expected))
        quotas.append({
            "group": group,
            "quota": min(base, len(group)),
            "fractional": expected - base,
            "tie": rng.random(),
        })

    remaining = n_samples - sum(q["quota"] for q in quotas)
    for q in sorted(quotas, key=lambda item: (-item["fractional"], item["tie"])):
        if remaining <= 0:
            break
        capacity = len(q["group"]) - q["quota"]
        if capacity > 0:
            q["quota"] += 1
            remaining -= 1

    parts = []
    for offset, q in enumerate(quotas):
        if q["quota"] > 0:
            parts.append(q["group"].sample(q["quota"], random_state=seed + offset))

    sampled = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()
    if len(sampled) < n_samples:
        missing = n_samples - len(sampled)
        sampled_ids = set(sampled["_sample_row_id"]) if "_sample_row_id" in sampled.columns else set()
        unsampled = work[~work["_sample_row_id"].isin(sampled_ids)]
        if len(unsampled) > 0:
            sampled = pd.concat([
                sampled,
                unsampled.sample(min(missing, len(unsampled)), random_state=seed + 999)
            ], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=seed).drop(columns=["_sample_row_id"], errors="ignore").reset_index(drop=True)


def get_data_splits(meta_df=None):
    """
    Buat group-aware train/val/test split berdasarkan source_file.

    Split dilakukan pada level source_file (lagu), bukan segment,
    agar tidak ada segment dari lagu yang sama muncul di train dan test
    (mencegah data leakage). Pemilihan source_file memakai seed 42 dan
    stratified sampling berdasarkan composer + instrument.

    Returns:
        dict: {"train": DataFrame, "val": DataFrame, "test": DataFrame}
    """
    if meta_df is None:
        meta_path = os.path.join(PATHS["preprocessed"], "metadata.csv")
        meta_df = pd.read_csv(meta_path)

    source_cols = ["source_file"]
    for col in ["composer", "instrument"]:
        if col in meta_df.columns:
            source_cols.append(col)

    source_df = meta_df[source_cols].drop_duplicates("source_file").reset_index(drop=True)
    for col in ["composer", "instrument"]:
        if col not in source_df.columns:
            source_df[col] = "unknown"
        source_df[col] = source_df[col].fillna("unknown").astype(str)

    n_test = max(1, int(round(len(source_df) * 0.15)))
    n_val = max(1, int(round(len(source_df) * 0.15)))

    test_sources_df = _stratified_sample_table(source_df, n_test, seed=DATASET_RANDOM_SEED)
    remaining_sources = source_df[~source_df["source_file"].isin(test_sources_df["source_file"])].reset_index(drop=True)
    val_sources_df = _stratified_sample_table(remaining_sources, n_val, seed=DATASET_RANDOM_SEED + 1)
    train_sources_df = remaining_sources[~remaining_sources["source_file"].isin(val_sources_df["source_file"])].reset_index(drop=True)

    test_files = set(test_sources_df["source_file"])
    val_files = set(val_sources_df["source_file"])
    train_files = set(train_sources_df["source_file"])

    splits = {
        "train": meta_df[meta_df["source_file"].isin(train_files)].reset_index(drop=True),
        "val": meta_df[meta_df["source_file"].isin(val_files)].reset_index(drop=True),
        "test": meta_df[meta_df["source_file"].isin(test_files)].reset_index(drop=True),
    }

    for name, df in splits.items():
        n_sources = df["source_file"].nunique()
        print(f"  {name:5s}: {len(df)} segments from {n_sources} source files")

    print(f"  split seed: {DATASET_RANDOM_SEED} | stratified by composer + instrument")
    return splits

def load_preprocessed_data(n_samples: int = 50, split: str = "test", gap_position: str = None):
    """
    Load data yang sudah dipreprocess dari Drive.

    Menggunakan group-aware split untuk mencegah data leakage.
    Default menggunakan split "test" untuk evaluasi.

    Args:
        n_samples: Jumlah sampel untuk evaluasi.
        split: "train", "val", atau "test"

    Returns:
        original_audios : List audio ground truth
        masked_by_gap   : {gap_ms: [list audio masked]}
        gap_regions     : {gap_ms: [{"gap_start", "gap_end", "gap_position"}]}
    """
    meta_path = os.path.join(PATHS["preprocessed"], "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Metadata tidak ditemukan! Jalankan Cell 3 (preprocessing) terlebih dahulu."
        )

    meta_df = pd.read_csv(meta_path)
    splits = get_data_splits(meta_df)
    split_df = splits[split]

    selected = _stratified_sample_table(split_df, min(n_samples, len(split_df)), seed=DATASET_RANDOM_SEED + 2)

    gap_position = (gap_position or EVAL_GAP_POSITION).strip().lower()
    if gap_position not in {"center", "random"}:
        raise ValueError("gap_position harus 'center' atau 'random'.")

    print(f"📂 Loading {len(selected)} sampel dari Drive (split={split}, gap_position={gap_position})...")

    original_audios = []
    masked_by_gap = {gap_ms: [] for gap_ms in GAP_DURATIONS_MS}
    gap_regions = {gap_ms: [] for gap_ms in GAP_DURATIONS_MS}

    for sample_index, (_, row) in enumerate(selected.iterrows()):
        orig_audio, sr_loaded = sf.read(row["clean_path"])
        if int(sr_loaded) != int(TARGET_SR):
            raise RuntimeError(f"SR preprocessed tidak cocok: {sr_loaded} != {TARGET_SR} pada {row['clean_path']}")
        orig_audio = np.asarray(orig_audio, dtype=np.float32)
        original_audios.append(orig_audio)

        filename = os.path.basename(row["clean_path"])
        for gap_ms in GAP_DURATIONS_MS:
            if gap_position == "center":
                masked_path = os.path.join(PATHS["masked"], f"gap_{gap_ms}ms", filename)
                masked_audio, sr_masked = sf.read(masked_path)
                if int(sr_masked) != int(TARGET_SR):
                    raise RuntimeError(f"SR masked tidak cocok: {sr_masked} != {TARGET_SR} pada {masked_path}")
                mask, gap_start, gap_end = make_gap_mask(len(orig_audio), gap_ms)
                masked_audio = np.asarray(masked_audio, dtype=np.float32)
            else:
                mask, gap_start, gap_end = make_eval_gap_mask(
                    len(orig_audio), gap_ms, sample_index=sample_index, sr=TARGET_SR
                )
                masked_audio = orig_audio.copy()
                masked_audio[gap_start:gap_end] = 0.0
            masked_by_gap[gap_ms].append(masked_audio)
            gap_regions[gap_ms].append({
                "gap_start": int(gap_start),
                "gap_end": int(gap_end),
                "gap_position": gap_position,
            })

    print(f"✅ {len(original_audios)} sampel siap dievaluasi.")
    return original_audios, masked_by_gap, gap_regions


print("✅ Helper functions berhasil didefinisikan!")

# ---
# ## CELL 6.5 — Dataset, DataLoader & Shared Utilities
# 
# Definisi Dataset/DataLoader untuk training, serta fungsi utilitas
# yang dipakai di semua cell inference (mask creation, boundary cross-fade).


# ============================================================
# CELL 6.5: DATASET, DATALOADER & SHARED UTILITIES
# ============================================================
# 1. MusicGapDataset: PyTorch Dataset untuk training
# 2. make_gap_mask(): fungsi mask yang konsisten antara
#    preprocessing dan inference (menghindari off-by-one)
# 3. crossfade_boundary(): half-cosine crossfade untuk
#    menghilangkan click artifact di boundary gap
# 4. DataLoader optimized untuk Vast.ai GPU training
# ============================================================

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import os
import time
import random
from tqdm import tqdm

# Vast.ai/Linux: use a small worker pool. SSL encoder precompute stays single-process
# because AudioMAE/timm/torchaudio can fan out OpenMP threads inside each worker.
CPU_COUNT = os.cpu_count() or 4
AUTO_NUM_WORKERS = int(os.environ.get(
    "PIPELINE_NUM_WORKERS",
    0 if os.name == "nt" else min(2, max(1, CPU_COUNT // 4)),
))
ENCODER_PRECOMPUTE_NUM_WORKERS = int(os.environ.get("ENCODER_PRECOMPUTE_NUM_WORKERS", "0"))
DATALOADER_PREFETCH_FACTOR = int(os.environ.get("PIPELINE_PREFETCH_FACTOR", "2"))
CACHE_AUDIO_IN_MEMORY = True
PRECOMPUTE_ENCODER_LATENTS = True

try:
    torch.set_num_threads(int(os.environ.get("PIPELINE_TORCH_THREADS", CPU_THREAD_LIMIT)))
    torch.set_num_interop_threads(1)
except Exception:
    pass


def make_gap_mask(audio_length, gap_ms, sr=TARGET_SR):
    return build_gap_mask_array(audio_length, gap_ms, sr=sr)


def make_eval_gap_mask(audio_length, gap_ms, sample_index, sr=TARGET_SR):
    """Center gap by default; deterministic non-center random gap for robustness eval."""
    if EVAL_GAP_POSITION == "center":
        return make_gap_mask(audio_length, gap_ms, sr=sr)

    gap_samples = int(round(sr * gap_ms / 1000))
    min_context = int(round(sr * EVAL_RANDOM_GAP_MIN_CONTEXT_MS / 1000))
    min_start = min_context
    max_start = audio_length - gap_samples - min_context
    if max_start < min_start:
        min_start = 0
        max_start = audio_length - gap_samples
    if max_start < min_start:
        raise ValueError(
            f"Random gap {gap_ms}ms tidak valid untuk audio_length={audio_length}, sr={sr}."
        )
    rng = np.random.default_rng(DATASET_RANDOM_SEED + 100_000 + int(sample_index) * 997 + int(gap_ms))
    gap_start = int(rng.integers(min_start, max_start + 1))
    return build_gap_mask_array(audio_length, gap_ms, sr=sr, gap_start=gap_start)


def crossfade_boundary(original, reconstructed, gap_start, gap_end,
                       sr=TARGET_SR, fade_ms=30):
    fade_n = int(fade_ms * 1e-3 * sr)
    if fade_n <= 0:
        return reconstructed

    output = reconstructed.copy()
    fade = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_n)))

    left_start = max(0, gap_start - fade_n)
    left_len = gap_start - left_start
    if left_len > 0:
        f = fade[-left_len:]
        output[left_start:gap_start] = (
            original[left_start:gap_start] * (1 - f)
            + reconstructed[left_start:gap_start] * f
        )

    right_end = min(len(output), gap_end + fade_n)
    right_len = right_end - gap_end
    if right_len > 0:
        f = fade[:right_len]
        output[gap_end:right_end] = (
            reconstructed[gap_end:right_end] * (1 - f)
            + original[gap_end:right_end] * f
        )

    return output


def evaluation_artifact_name(model_name: str):
    """Separate optional robustness-eval artifacts from main center-gap artifacts."""
    if EVAL_GAP_POSITION == "center":
        return model_name
    return f"{model_name}_{EVAL_GAP_POSITION}gap"


def prepare_reconstructed_outputs(model_name: str, clear: bool = False):
    """Siapkan folder reconstructed audio; hapus hanya jika diminta eksplisit."""
    import shutil

    out_dir = os.path.join(PATHS["outputs"], evaluation_artifact_name(model_name))
    if clear and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def reset_reconstructed_outputs(model_name: str):
    """Backward-compatible helper untuk mengosongkan reconstructed audio."""
    return prepare_reconstructed_outputs(model_name, clear=True)


def reconstructed_output_path(model_name: str, gap_ms: int, sample_index: int):
    artifact_name = evaluation_artifact_name(model_name)
    out_dir = os.path.join(PATHS["outputs"], artifact_name, f"gap_{gap_ms}ms")
    filename = f"{artifact_name}_gap{gap_ms}ms_sample{sample_index:04d}_reconstructed.wav"
    return os.path.join(out_dir, filename)


CURRENT_RECONSTRUCTION_CACHE_TAG = None
EVAL_CACHE_CODE_VERSION = "eval_cache_v4_musicnet_cqtdiffplus_gapaware"


def _file_fingerprint(path: str):
    if not path or not os.path.exists(path):
        return None
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _decoder_cache_identity(decoder):
    return {
        "class": decoder.__class__.__name__,
        "architecture": getattr(decoder, "architecture_name", None),
        "official_weights_path": getattr(decoder, "official_weights_path", None),
        "official_weights_type": getattr(decoder, "official_weights_type", None),
        "official_weights_fingerprint": _file_fingerprint(getattr(decoder, "official_weights_path", None)),
    }


def set_reconstruction_cache_context(model_name: str, decoder, checkpoint_path: str = None):
    """Cache tag supaya evaluator tidak memakai WAV lama dari checkpoint/arsitektur berbeda."""
    global CURRENT_RECONSTRUCTION_CACHE_TAG
    context = {
        "version": EVAL_CACHE_CODE_VERSION,
        "stage": globals().get("PIPELINE_STAGE_NAME", "code_v3"),
        "experiment_config_id": EXPERIMENT_CONFIG_ID,
        "model": model_name,
        "checkpoint": _file_fingerprint(checkpoint_path),
        "decoder": _decoder_cache_identity(decoder),
        "target_sr": int(TARGET_SR),
        "segment_samples": int(SEGMENT_SAMPLES),
        "gap_durations_ms": list(GAP_DURATIONS_MS),
        "eval_gap_position": EVAL_GAP_POSITION,
        "eval_random_gap_min_context_ms": int(EVAL_RANDOM_GAP_MIN_CONTEXT_MS),
    }
    CURRENT_RECONSTRUCTION_CACHE_TAG = json.dumps(context, sort_keys=True)
    return CURRENT_RECONSTRUCTION_CACHE_TAG


def _reconstructed_sidecar_path(path: str):
    return f"{path}.meta.json"


def save_reconstructed_output(model_name: str, gap_ms: int, sample_index: int,
                              reconstructed, sr: int = TARGET_SR):
    """Simpan final reconstructed waveform yang dipakai oleh metrik evaluasi."""
    path = reconstructed_output_path(model_name, gap_ms, sample_index)
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)

    audio = np.asarray(reconstructed, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    sf.write(path, audio, sr, subtype="FLOAT")
    sidecar = {
        "model": model_name,
        "gap_ms": int(gap_ms),
        "sample_index": int(sample_index),
        "sr": int(sr),
        "n_samples": int(len(audio)),
        "experiment_config_id": EXPERIMENT_CONFIG_ID,
        "eval_gap_position": EVAL_GAP_POSITION,
        "cache_tag": CURRENT_RECONSTRUCTION_CACHE_TAG,
        "saved_at": pd.Timestamp.now().isoformat(),
    }
    with open(_reconstructed_sidecar_path(path), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    return path


def load_reconstructed_output_if_available(model_name: str, gap_ms: int, sample_index: int,
                                           expected_n_samples: int = None, sr: int = TARGET_SR):
    """Load reconstructed WAV yang sudah ada; return (audio, path) atau (None, path)."""
    path = reconstructed_output_path(model_name, gap_ms, sample_index)
    if not EVAL_REUSE_RECONSTRUCTIONS or not os.path.exists(path):
        return None, path

    try:
        if CURRENT_RECONSTRUCTION_CACHE_TAG is not None:
            sidecar_path = _reconstructed_sidecar_path(path)
            if not os.path.exists(sidecar_path):
                print(f"    Reconstruct cache ignored (missing cache metadata): {path}")
                return None, path
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            if sidecar.get("cache_tag") != CURRENT_RECONSTRUCTION_CACHE_TAG:
                print(f"    Reconstruct cache ignored (checkpoint/model cache tag mismatch): {path}")
                return None, path
            if sidecar.get("model") != model_name or int(sidecar.get("gap_ms", -1)) != int(gap_ms):
                print(f"    Reconstruct cache ignored (model/gap metadata mismatch): {path}")
                return None, path
            if int(sidecar.get("sample_index", -1)) != int(sample_index):
                print(f"    Reconstruct cache ignored (sample metadata mismatch): {path}")
                return None, path
        info = sf.info(path)
        if int(info.samplerate) != int(sr):
            print(f"    Reconstruct cache ignored (SR mismatch): {path}")
            return None, path
        if expected_n_samples is not None and int(info.frames) != int(expected_n_samples):
            print(f"    Reconstruct cache ignored (length mismatch): {path}")
            return None, path
        audio = _read_audio_float32(path)
        if not np.isfinite(audio).all():
            print(f"    Reconstruct cache ignored (non-finite audio): {path}")
            return None, path
        return audio, path
    except Exception as exc:
        print(f"    Reconstruct cache ignored ({exc}): {path}")
        return None, path


def summarize_reconstruction_cache(model_name: str, n_eval_samples: int):
    """Print ringkasan cache rekonstruksi yang akan dipakai evaluator."""
    total_expected = len(GAP_DURATIONS_MS) * int(n_eval_samples)
    existing = 0
    missing_examples = []

    for gap_ms in GAP_DURATIONS_MS:
        for sample_index in range(int(n_eval_samples)):
            path = reconstructed_output_path(model_name, gap_ms, sample_index)
            if os.path.exists(path):
                existing += 1
            elif len(missing_examples) < 3:
                missing_examples.append(path)

    print(
        f"  Reconstruction cache probe: {existing}/{total_expected} WAV ditemukan "
        f"di {os.path.join(PATHS['outputs'], evaluation_artifact_name(model_name))}"
    )
    if missing_examples:
        print("  Contoh path yang belum ada:")
        for path in missing_examples:
            print(f"   - {path}")
    return existing, total_expected


def save_reconstruction_manifest(model_name: str, rows: list):
    out_dir = os.path.join(PATHS["outputs"], evaluation_artifact_name(model_name))
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"💾 Reconstructed output manifest: {manifest_path}")
    return manifest_path


def validate_masked_gap_alignment(original, masked, mask, gap_ms, sr=TARGET_SR,
                                  atol=2e-4, expected_gap_start=None, expected_gap_end=None):
    """Fail-fast jika file masked tidak memiliki gap di posisi yang sama dengan mask evaluasi."""
    original = np.asarray(original, dtype=np.float32)
    masked = np.asarray(masked, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    n = min(len(original), len(masked), len(mask))
    original = original[:n]
    masked = masked[:n]
    mask = mask[:n]

    gap_samples = int(round(sr * gap_ms / 1000))
    if expected_gap_start is None or expected_gap_end is None:
        expected_start, expected_end = compute_gap_bounds(n, gap_ms, sr=sr)
    else:
        expected_start, expected_end = int(expected_gap_start), int(expected_gap_end)
    gap_idx = np.flatnonzero(mask)
    if len(gap_idx) != gap_samples or gap_idx[0] != expected_start or gap_idx[-1] + 1 != expected_end:
        raise RuntimeError(
            f"Mask gap tidak sesuai untuk {gap_ms}ms: expected=({expected_start},{expected_end}), "
            f"actual=({gap_idx[0] if len(gap_idx) else None},{gap_idx[-1] + 1 if len(gap_idx) else None}), "
            f"gap_samples={len(gap_idx)}"
        )

    gap_abs = float(np.max(np.abs(masked[mask]))) if np.any(mask) else 0.0
    known_abs = float(np.max(np.abs(masked[~mask] - original[~mask]))) if np.any(~mask) else 0.0
    if gap_abs > atol:
        raise RuntimeError(f"Masked audio gap {gap_ms}ms tidak nol penuh: max_abs_gap={gap_abs:.6g}")
    if known_abs > atol:
        raise RuntimeError(f"Masked audio non-gap berubah dari original: max_abs_known_diff={known_abs:.6g}")

    return {
        "gap_start": int(expected_start),
        "gap_end": int(expected_end),
        "gap_samples": int(gap_samples),
        "masked_gap_max_abs": gap_abs,
        "known_region_max_abs_diff": known_abs,
    }


def _gap_region_stats(original, reconstructed, gap_ms, sr=TARGET_SR, gap_start=None, gap_end=None):
    n = min(len(original), len(reconstructed))
    original = np.asarray(original[:n], dtype=np.float32)
    reconstructed = np.asarray(reconstructed[:n], dtype=np.float32)
    if gap_start is None or gap_end is None:
        gap_start, gap_end = compute_gap_bounds(n, gap_ms, sr=sr)
    gap_start, gap_end = int(gap_start), int(gap_end)
    ref_gap = original[gap_start:gap_end]
    rec_gap = reconstructed[gap_start:gap_end]
    eps = 1e-12
    ref_rms = float(np.sqrt(np.mean(ref_gap ** 2) + eps))
    rec_rms = float(np.sqrt(np.mean(rec_gap ** 2) + eps))
    return {
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
        "ref_gap_rms": ref_rms,
        "recon_gap_rms": rec_rms,
        "gap_gain_db": float(20.0 * np.log10((rec_rms + eps) / (ref_rms + eps))),
        "ref_full_rms": float(np.sqrt(np.mean(original ** 2) + eps)),
        "recon_full_rms": float(np.sqrt(np.mean(reconstructed ** 2) + eps)),
        "gap_peak_abs": float(np.max(np.abs(rec_gap))) if len(rec_gap) else 0.0,
        "gap_zero_fraction": float(np.mean(np.abs(rec_gap) < 1e-6)) if len(rec_gap) else 1.0,
    }


def save_reconstruction_diagnostics(model_name: str, original_audios: list, reconstructed_dict: dict,
                                    alignment_rows: list = None, conditioning_rows: list = None,
                                    gap_regions_by_gap: dict = None):
    """Simpan diagnostik gain/posisi/conditioning untuk audit hasil evaluasi."""
    out_dir = os.path.join(PATHS["outputs"], evaluation_artifact_name(model_name))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for gap_ms, recon_audios in reconstructed_dict.items():
        regions = (gap_regions_by_gap or {}).get(gap_ms, [])
        per_sample = []
        for idx, (orig, recon) in enumerate(zip(original_audios, recon_audios)):
            region = regions[idx] if idx < len(regions) else {}
            per_sample.append(
                _gap_region_stats(
                    orig, recon, gap_ms, TARGET_SR,
                    gap_start=region.get("gap_start"),
                    gap_end=region.get("gap_end"),
                )
            )
        row = {"model": model_name, "gap_ms": int(gap_ms), "n_samples": int(len(per_sample))}
        for key in per_sample[0].keys():
            values = np.asarray([item[key] for item in per_sample], dtype=np.float64)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        rows.append(row)

    gain_path = os.path.join(out_dir, "diagnostics_gain_by_gap.csv")
    pd.DataFrame(rows).to_csv(gain_path, index=False)
    print(f"💾 Gain diagnostics: {gain_path}")

    if alignment_rows:
        align_path = os.path.join(out_dir, "diagnostics_gap_alignment.csv")
        pd.DataFrame(alignment_rows).to_csv(align_path, index=False)
        print(f"💾 Gap alignment diagnostics: {align_path}")

    if conditioning_rows:
        cond_path = os.path.join(out_dir, "diagnostics_conditioning.csv")
        pd.DataFrame(conditioning_rows).to_csv(cond_path, index=False)
        print(f"💾 Conditioning diagnostics: {cond_path}")


def _read_audio_float32(path):
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


class MusicGapDataset(Dataset):
    """
    Dataset optimized for GPU training:
    - metadata columns are lists, avoiding pandas iloc per sample
    - clean audio can be cached in RAM once, removing repeated disk reads
    - mask creation is simple slicing and matches apply_gap_mask semantics
    """
    def __init__(self, meta_df, gap_ms_choices, sr=TARGET_SR, cache_audio=CACHE_AUDIO_IN_MEMORY):
        self.meta = meta_df.reset_index(drop=True)
        self.clean_paths = self.meta["clean_path"].astype(str).tolist()
        self.gap_ms_choices = list(gap_ms_choices)
        self.sr = sr
        self.cache_audio = bool(cache_audio)
        self._audio_cache = None

        if self.cache_audio:
            cache_start = time.perf_counter()
            self._audio_cache = [_read_audio_float32(path) for path in self.clean_paths]
            mb = sum(audio.nbytes for audio in self._audio_cache) / (1024 ** 2)
            print(f"   Cached {len(self._audio_cache)} audio segments ({mb:.1f} MB) in {time.perf_counter() - cache_start:.1f}s")

    def __len__(self):
        return len(self.clean_paths) * len(self.gap_ms_choices)

    def _get_clean_audio(self, seg_i):
        if self._audio_cache is not None:
            return self._audio_cache[seg_i]
        return _read_audio_float32(self.clean_paths[seg_i])

    def __getitem__(self, idx):
        seg_i, g_i = divmod(int(idx), len(self.gap_ms_choices))
        gap_ms = self.gap_ms_choices[g_i]

        clean = self._get_clean_audio(seg_i)
        mask, gs, ge = make_gap_mask(len(clean), gap_ms, self.sr)
        masked = clean.copy()
        masked[gs:ge] = 0.0

        return {
            "clean": torch.from_numpy(clean),
            "masked": torch.from_numpy(masked),
            "mask": torch.from_numpy(mask),
            "gap_start": gs,
            "gap_end": ge,
            "gap_ms": gap_ms,
        }


class EncoderCachedDataset(Dataset):
    """Dataset wrapper that adds precomputed frozen SSL encoder latents."""
    def __init__(self, base_dataset, encoder_latents):
        self.base_dataset = base_dataset
        self.encoder_latents = encoder_latents.cpu().float()
        if len(self.encoder_latents) != len(self.base_dataset):
            raise ValueError("encoder_latents length must match base_dataset length")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item["encoder_latent"] = self.encoder_latents[int(idx)]
        return item


def _seed_worker(worker_id):
    worker_seed = DATASET_RANDOM_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)


def build_audio_loader(ds, batch_size, shuffle, num_workers=AUTO_NUM_WORKERS):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (int(num_workers) > 0),
        "worker_init_fn": _seed_worker if int(num_workers) > 0 else None,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR
    return DataLoader(ds, **kwargs)


def make_dataloaders(batch_size=16, num_workers=AUTO_NUM_WORKERS, cache_audio=CACHE_AUDIO_IN_MEMORY):
    """Buat DataLoader untuk train/val/test dengan group-aware split."""
    meta_path = os.path.join(PATHS["preprocessed"], "metadata.csv")
    meta_df = pd.read_csv(meta_path)
    splits = get_data_splits(meta_df)

    if num_workers is None:
        num_workers = AUTO_NUM_WORKERS
    print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, pin_memory={torch.cuda.is_available()}, prefetch={DATALOADER_PREFETCH_FACTOR if int(num_workers) > 0 else 0}, cache_audio={cache_audio}")

    loaders = {}
    for name, df in splits.items():
        ds = MusicGapDataset(df, GAP_DURATIONS_MS, sr=TARGET_SR, cache_audio=cache_audio)
        loaders[name] = build_audio_loader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        )
    return loaders


def _encoder_cache_path(model_name, split_name, dataset_len):
    encoder_key = str(model_name).split("_")[0]
    gaps = "-".join(str(g) for g in GAP_DURATIONS_MS)
    config_key = EXPERIMENT_CONFIG_ID.replace(".", "p")
    cache_dir = os.path.join(PATHS["preprocessed"], "encoder_latents")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"{encoder_key}_{split_name}_n{dataset_len}_{config_key}_gaps{gaps}.pt",
    )


def precompute_encoder_latents_for_dataset(base_dataset, encoder_fn, device, batch_size, num_workers,
                                           model_name, split_name):
    """Precompute frozen CLAP/AudioMAE latents so training loop no longer runs CPU preprocessing."""
    num_workers = ENCODER_PRECOMPUTE_NUM_WORKERS if num_workers is None else min(
        int(num_workers),
        ENCODER_PRECOMPUTE_NUM_WORKERS,
    )
    cache_path = _encoder_cache_path(model_name, split_name, len(base_dataset))
    if os.path.exists(cache_path):
        print(f"   Loading cached encoder latents [{split_name}]: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"   Precomputing encoder latents [{split_name}] ({len(base_dataset)} items, num_workers={num_workers})...")
    precompute_loader = build_audio_loader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    latents = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(precompute_loader, desc=f"Precompute {model_name}:{split_name}", leave=False):
            masked = batch["masked"].to(device, non_blocking=True)
            z = encoder_fn(masked)
            latents.append(z.detach().cpu().float())
    latents = torch.cat(latents, dim=0)
    torch.save(latents, cache_path)
    print(f"   Saved encoder latents [{split_name}] in {format_duration(time.perf_counter() - start)}: {cache_path}")
    return latents


def add_encoder_cache_to_loaders(loaders, encoder_fn, device, model_name, num_workers=AUTO_NUM_WORKERS,
                                 splits=("train", "val")):
    if not PRECOMPUTE_ENCODER_LATENTS:
        return loaders

    cached = dict(loaders)
    for split_name in splits:
        if split_name not in loaders:
            continue
        loader = loaders[split_name]
        latents = precompute_encoder_latents_for_dataset(
            loader.dataset,
            encoder_fn,
            device,
            batch_size=loader.batch_size,
            num_workers=num_workers,
            model_name=model_name,
            split_name=split_name,
        )
        cached_ds = EncoderCachedDataset(loader.dataset, latents)
        cached[split_name] = build_audio_loader(
            cached_ds,
            batch_size=loader.batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )
    return cached


print("✅ Dataset, DataLoader & shared utilities berhasil didefinisikan!")
print(f"   Gap durations: {GAP_DURATIONS_MS} ms")
print(f"   AUTO_NUM_WORKERS: {AUTO_NUM_WORKERS}")
print(f"   ENCODER_PRECOMPUTE_NUM_WORKERS: {ENCODER_PRECOMPUTE_NUM_WORKERS}")
print(f"   CPU thread env limit: {CPU_THREAD_LIMIT}")


# ---
# ## CELL 6.6 — Training Loop (Reconstruction-Based)
# 
# **Perubahan penting dari versi sebelumnya:**
# Versi lama memakai DDPM epsilon prediction untuk training, tapi saat inference
# langsung pakai output model sebagai audio. Ini menyebabkan **mismatch fatal**:
# model dilatih memprediksi noise, tapi output-nya dipakai sebagai rekonstruksi.
# 
# Versi baru ini membedakan objective per decoder:
# - CQT-Diff hybrid: SSL latent menjadi conditioning denoiser diffusion
# - Fallback/STFT decoder: prediksi complex STFT clean audio pada gap frames
# - Training dan inference dijaga selaras per decoder
# 
# Fitur:
# - Gap-only reconstruction loss (L1 pada complex STFT real+imag di gap region)
# - Full audio auxiliary loss (0.1x bobot, buat stabilitas)
# - Classifier-free guidance dropout (CFG)
# - Mixed precision training (AMP) + gradient clipping
# - Separate trainer untuk baseline (tanpa encoder) dan hybrid (dengan encoder)


# ============================================================
# CELL 6.6: TRAINING LOOP (CONDITIONED DIFFUSION / RECONSTRUCTION)
# ============================================================
# Training untuk pipeline hybrid SSL + Decoder.
#
# PERBAIKAN UTAMA dari versi sebelumnya:
# - CQT-Diff hybrid memakai diffusion_loss sehingga SSL conditioning dipelajari
#   di dalam denoiser/sampler diffusion.
# - Decoder fallback tetap memakai rekonstruksi STFT yang selaras dengan
#   inference masing-masing.
# - Loss dihitung pada gap region.
# - CFG dropout tetap dipertahankan
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm
import time


def compute_stft_target(clean_audio, n_fft=2048, hop_length=512):
    """Hitung target complex STFT real+imag dari clean audio."""
    window = _get_hann_window(n_fft, clean_audio.device)
    spec = torch.stft(
        clean_audio, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True
    )
    spec_ri = torch.view_as_real(spec).permute(0, 1, 3, 2).contiguous()
    return spec_ri.reshape(spec.shape[0], spec.shape[1] * 2, spec.shape[2])  # (B, 2F, T)


def compute_frame_mask(sample_mask, n_frames, hop_length=512):
    """Konversi sample-level mask -> frame-level mask dengan satu GPU op."""
    mask_f = sample_mask.float().unsqueeze(1)
    pooled = F.avg_pool1d(mask_f, kernel_size=hop_length, stride=hop_length, ceil_mode=True).squeeze(1)
    if pooled.shape[1] < n_frames:
        pooled = F.pad(pooled, (0, n_frames - pooled.shape[1]))
    elif pooled.shape[1] > n_frames:
        pooled = pooled[:, :n_frames]
    return pooled > 0.5  # (B, T) boolean


CQT_WAVEFORM_GAP_LOSS_WEIGHT = float(os.environ.get("CQT_WAVEFORM_GAP_LOSS_WEIGHT", "0.1"))
CQT_ENERGY_LOSS_WEIGHT = float(os.environ.get("CQT_ENERGY_LOSS_WEIGHT", "0.05"))


def spec_features_to_waveform(pred_spec, audio_length, n_fft=2048, hop_length=512):
    """Convert predicted real+imag STFT features (B, T, 2F) back to waveform."""
    with torch.autocast(device_type="cuda" if pred_spec.device.type == "cuda" else "cpu", enabled=False):
        pred_spec = pred_spec.float()
        batch_size, n_frames, two_freq = pred_spec.shape
        freq_bins = two_freq // 2
        pred_pairs = pred_spec.reshape(batch_size, n_frames, freq_bins, 2).permute(0, 2, 1, 3).contiguous()
        complex_spec = torch.view_as_complex(pred_pairs)
        window = _get_hann_window(n_fft, pred_spec.device)
        return torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=audio_length,
        )


def compute_waveform_gap_losses(pred_spec, clean_audio, sample_mask):
    """Waveform-domain gap loss plus log-RMS energy loss to discourage silent gaps."""
    pred_wave = spec_features_to_waveform(pred_spec, clean_audio.shape[-1])
    mask = sample_mask.bool()
    waveform_gap_loss = F.l1_loss(pred_wave[mask], clean_audio[mask])

    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    pred_rms = torch.sqrt(((pred_wave * mask_f).pow(2).sum(dim=1) / denom).clamp_min(1e-10))
    target_rms = torch.sqrt(((clean_audio * mask_f).pow(2).sum(dim=1) / denom).clamp_min(1e-10))
    energy_loss = F.l1_loss(torch.log(pred_rms + 1e-5), torch.log(target_rms + 1e-5))
    return pred_wave, waveform_gap_loss, energy_loss


PROFILE_EVERY_N_STEPS = 25


class EpochProfiler:
    """Lightweight timing for data wait, H2D, preprocessing/encoder, and GPU compute."""
    def __init__(self, profile_every=PROFILE_EVERY_N_STEPS):
        self.profile_every = max(1, int(profile_every))
        self.data_load_seconds = 0.0
        self.h2d_seconds = 0.0
        self.preprocess_seconds = 0.0
        self.gpu_compute_seconds = 0.0
        self.profiled_steps = 0
        self.total_steps = 0

    def add_data_wait(self, seconds):
        self.data_load_seconds += float(seconds)
        self.total_steps += 1

    def should_profile(self, step_idx):
        return (step_idx % self.profile_every) == 0

    def add_step_profile(self, profile):
        if not profile:
            return
        self.profiled_steps += 1
        self.h2d_seconds += float(profile.get("h2d_seconds", 0.0))
        self.preprocess_seconds += float(profile.get("preprocess_seconds", 0.0))
        self.gpu_compute_seconds += float(profile.get("gpu_compute_seconds", 0.0))

    def summary(self):
        scale = (self.total_steps / self.profiled_steps) if self.profiled_steps else 0.0
        h2d = self.h2d_seconds * scale
        preprocess = self.preprocess_seconds * scale
        gpu_compute = self.gpu_compute_seconds * scale
        active = h2d + preprocess + gpu_compute
        total = self.data_load_seconds + active
        return {
            "steps": self.total_steps,
            "profiled_steps": self.profiled_steps,
            "data_load_seconds": self.data_load_seconds,
            "h2d_seconds_est": h2d,
            "preprocess_seconds_est": preprocess,
            "gpu_compute_seconds_est": gpu_compute,
            "gpu_active_share": (gpu_compute / total) if total > 0 else 0.0,
        }


def _sync_if_profile(profile_step):
    if profile_step and torch.cuda.is_available():
        torch.cuda.synchronize()


def get_gpu_utilization_snapshot():
    try:
        import subprocess
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=2).strip().splitlines()[0]
        gpu_util, mem_util, mem_used, mem_total = [part.strip() for part in out.split(",")]
        return f"nvidia-smi gpu={gpu_util}% mem_util={mem_util}% mem={mem_used}/{mem_total}MB"
    except Exception:
        return "nvidia-smi unavailable"


def print_epoch_profile(model_name, epoch, summary):
    print(
        f"  Profile {model_name} epoch {epoch}: "
        f"data_wait={summary['data_load_seconds']:.2f}s | "
        f"h2d~={summary['h2d_seconds_est']:.2f}s | "
        f"preproc/encoder~={summary['preprocess_seconds_est']:.2f}s | "
        f"gpu_compute~={summary['gpu_compute_seconds_est']:.2f}s | "
        f"gpu_active_share~={summary['gpu_active_share']:.1%} | "
        f"peak_vram={get_peak_vram_gb():.2f}GB | {get_gpu_utilization_snapshot()}"
    )


def train_step_reconstruction(decoder, encoder_fn, film, batch, optimizer, scaler,
                              cfg_drop=0.1, device="cuda", profile_step=False):
    """Satu step training hybrid: CQT uses conditioned diffusion loss, fallback uses STFT loss."""
    decoder.train()
    film.train()
    profile = {}

    _sync_if_profile(profile_step)
    t0 = time.perf_counter()
    clean = batch["clean"].to(device, non_blocking=True)
    masked = batch["masked"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    cached_z = batch.get("encoder_latent")
    if cached_z is not None:
        cached_z = cached_z.to(device, non_blocking=True)
    _sync_if_profile(profile_step)
    if profile_step:
        profile["h2d_seconds"] = time.perf_counter() - t0

    B = clean.size(0)

    _sync_if_profile(profile_step)
    t_pre = time.perf_counter()
    with torch.no_grad():
        z = cached_z if cached_z is not None else encoder_fn(masked)
    _sync_if_profile(profile_step)
    if profile_step:
        profile["preprocess_seconds"] = time.perf_counter() - t_pre

    drop = (torch.rand(B, device=device) < cfg_drop).view(-1, *([1] * (z.dim() - 1)))
    z = torch.where(drop, torch.zeros_like(z), z)

    _sync_if_profile(profile_step)
    t_gpu = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=scaler.is_enabled()):
        # SSL conditioning dipakai oleh diffusion denoiser untuk belajar,
        # bukan sebagai post-processing/refinement setelah hasil CQT-Diff keluar.
        features = decoder.get_features(masked, mask)
        cond_features = film(z.float(), features)
        if hasattr(decoder, "diffusion_loss"):
            loss_parts = decoder.diffusion_loss(
                clean,
                masked,
                mask=mask,
                conditioning=cond_features,
            )
            loss = loss_parts["loss"]
            gap_loss = loss_parts.get("gap_loss", loss)
            full_loss = loss_parts.get("full_loss", loss)
            waveform_gap_loss = loss_parts.get("waveform_gap_loss", torch.zeros_like(loss))
            energy_loss = loss_parts.get("energy_loss", torch.zeros_like(loss))
        else:
            pred_spec = decoder.decode_features(cond_features)
            target_spec = compute_stft_target(clean).permute(0, 2, 1)

            T_min = min(pred_spec.shape[1], target_spec.shape[1])
            pred_spec = pred_spec[:, :T_min, :]
            target_spec = target_spec[:, :T_min, :]

            frame_mask = compute_frame_mask(mask, T_min).unsqueeze(-1).expand_as(pred_spec)
            gap_loss = F.l1_loss(pred_spec[frame_mask], target_spec[frame_mask])
            full_loss = F.l1_loss(pred_spec, target_spec)
            _, waveform_gap_loss, energy_loss = compute_waveform_gap_losses(pred_spec, clean, mask)
            loss = (
                gap_loss
                + 0.1 * full_loss
                + CQT_WAVEFORM_GAP_LOSS_WEIGHT * waveform_gap_loss
                + CQT_ENERGY_LOSS_WEIGHT * energy_loss
            )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(decoder.parameters()) + list(film.parameters()), max_norm=1.0
    )
    scaler.step(optimizer)
    scaler.update()
    _sync_if_profile(profile_step)
    if profile_step:
        profile["gpu_compute_seconds"] = time.perf_counter() - t_gpu

    return {
        "loss": loss.item(),
        "gap_loss": gap_loss.item(),
        "full_loss": full_loss.item(),
        "waveform_gap_loss": waveform_gap_loss.item(),
        "energy_loss": energy_loss.item(),
        "_profile": profile,
    }


def train_step_baseline(decoder, batch, optimizer, scaler, device="cuda", profile_step=False):
    """Training step buat baseline (tanpa encoder, tanpa FiLM)."""
    decoder.train()
    profile = {}

    _sync_if_profile(profile_step)
    t0 = time.perf_counter()
    clean = batch["clean"].to(device, non_blocking=True)
    masked = batch["masked"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    _sync_if_profile(profile_step)
    if profile_step:
        profile["h2d_seconds"] = time.perf_counter() - t0
        profile["preprocess_seconds"] = 0.0

    _sync_if_profile(profile_step)
    t_gpu = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=scaler.is_enabled()):
        features = decoder.get_features(masked, mask)
        pred_spec = decoder.decode_features(features)
        target_spec = compute_stft_target(clean).permute(0, 2, 1)

        T_min = min(pred_spec.shape[1], target_spec.shape[1])
        pred_spec = pred_spec[:, :T_min, :]
        target_spec = target_spec[:, :T_min, :]

        frame_mask = compute_frame_mask(mask, T_min).unsqueeze(-1).expand_as(pred_spec)
        gap_loss = F.l1_loss(pred_spec[frame_mask], target_spec[frame_mask])
        full_loss = F.l1_loss(pred_spec, target_spec)
        _, waveform_gap_loss, energy_loss = compute_waveform_gap_losses(pred_spec, clean, mask)
        loss = (
            gap_loss
            + 0.1 * full_loss
            + CQT_WAVEFORM_GAP_LOSS_WEIGHT * waveform_gap_loss
            + CQT_ENERGY_LOSS_WEIGHT * energy_loss
        )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    _sync_if_profile(profile_step)
    if profile_step:
        profile["gpu_compute_seconds"] = time.perf_counter() - t_gpu

    return {
        "loss": loss.item(),
        "gap_loss": gap_loss.item(),
        "full_loss": full_loss.item(),
        "waveform_gap_loss": waveform_gap_loss.item(),
        "energy_loss": energy_loss.item(),
        "_profile": profile,
    }


def get_training_checkpoint_paths(checkpoint_dir, model_name):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return {
        "latest": os.path.join(checkpoint_dir, f"{model_name}_latest.pt"),
        "best": os.path.join(checkpoint_dir, f"{model_name}_best.pt"),
    }


def atomic_torch_save(payload, path):
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def save_training_checkpoint(decoder, optimizer, epoch, metric, checkpoint_dir, model_name,
                             film=None, scheduler=None, scaler=None, history=None,
                             best_val_loss=None, is_best=False, metric_name="val_loss"):
    paths = get_training_checkpoint_paths(checkpoint_dir, model_name)
    payload = {
        "model_name": model_name,
        "epoch": int(epoch),
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        metric_name: float(metric) if metric is not None and np.isfinite(metric) else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None and np.isfinite(best_val_loss) else None,
        "best_metrics": {"val_loss": float(best_val_loss)} if best_val_loss is not None and np.isfinite(best_val_loss) else {},
        "history": history or [],
        "saved_at": pd.Timestamp.now().isoformat(),
        "stage": globals().get("PIPELINE_STAGE_NAME", "code_v3"),
        "architecture": getattr(decoder, "architecture_name", decoder.__class__.__name__),
    }
    if film is not None:
        payload["film_state"] = film.state_dict()

    atomic_torch_save(payload, paths["latest"])
    if is_best:
        atomic_torch_save(payload, paths["best"])
        print(f"  Best checkpoint saved: {paths['best']}")
    print(f"  Latest checkpoint saved: {paths['latest']}")
    return paths


def load_training_checkpoint_if_available(decoder, optimizer, scheduler, scaler, checkpoint_dir,
                                          model_name, device, film=None):
    if not checkpoint_dir:
        return 0, float("inf"), []

    paths = get_training_checkpoint_paths(checkpoint_dir, model_name)
    latest_path = paths["latest"]
    if not os.path.exists(latest_path):
        return 0, float("inf"), []

    payload = torch.load(latest_path, map_location=device, weights_only=False)
    expected_arch = getattr(decoder, "architecture_name", None)
    checkpoint_arch = payload.get("architecture")
    if expected_arch is not None and checkpoint_arch != expected_arch:
        print(
            f"Checkpoint {latest_path} memakai arsitektur lama "
            f"({checkpoint_arch or 'unknown'}), training {model_name} dimulai ulang "
            f"dengan {expected_arch}."
        )
        return 0, float("inf"), []
    decoder.load_state_dict(payload["decoder_state"])
    if film is not None and payload.get("film_state") is not None:
        film.load_state_dict(payload["film_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])

    start_epoch = int(payload.get("epoch", -1)) + 1
    best_val_loss = payload.get("best_val_loss")
    if best_val_loss is None:
        best_val_loss = payload.get("best_metrics", {}).get("val_loss", payload.get("val_loss", float("inf")))
    if best_val_loss is None:
        best_val_loss = float("inf")
    history = payload.get("history", []) or []
    print(f"Resuming {model_name} from epoch {start_epoch + 1} using {latest_path}")
    return start_epoch, float(best_val_loss), history


EARLY_STOPPING_ENABLED = os.environ.get("EARLY_STOPPING_ENABLED", "1").lower() in {"1", "true", "yes"}
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "4"))
EARLY_STOPPING_MIN_DELTA = float(os.environ.get("EARLY_STOPPING_MIN_DELTA", "1e-4"))
EARLY_STOPPING_MIN_EPOCHS = int(os.environ.get("EARLY_STOPPING_MIN_EPOCHS", "30"))
VAL_EVERY_EPOCHS = int(os.environ.get("VAL_EVERY_EPOCHS", "5"))


def _count_stale_validations(history, best_val_loss,
                             min_delta=EARLY_STOPPING_MIN_DELTA):
    stale = 0
    for row in reversed(history or []):
        val = row.get("val_loss")
        if val is None or not np.isfinite(val):
            continue
        if float(val) <= float(best_val_loss) + float(min_delta):
            break
        stale += 1
    return stale


def should_early_stop(history, best_val_loss, current_epoch, model_name):
    if not EARLY_STOPPING_ENABLED:
        return False
    if current_epoch < EARLY_STOPPING_MIN_EPOCHS:
        return False
    stale = _count_stale_validations(history, best_val_loss)
    if stale >= EARLY_STOPPING_PATIENCE:
        print(
            f"  Early stopping {model_name}: tidak ada improvement val_loss > "
            f"{EARLY_STOPPING_MIN_DELTA:g} selama {stale} validasi "
            f"(patience={EARLY_STOPPING_PATIENCE})."
        )
        return True
    return False


def train_model(decoder, encoder_fn, film, train_loader, val_loader=None,
                num_epochs=50, lr=1e-4, device="cuda", checkpoint_dir=None,
                model_name="model", batch_size=None, dataset_fraction=None):
    """
    Training loop lengkap untuk hybrid model (encoder + FiLM + decoder).
    Saves latest checkpoints every epoch and resumes from *_latest.pt.
    """
    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(film.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    reset_peak_vram_stats()
    start_epoch, best_val_loss, history = load_training_checkpoint_if_available(
        decoder, optimizer, scheduler, scaler, checkpoint_dir, model_name, device, film=film
    )

    if start_epoch >= num_epochs:
        print(f"{model_name} already reached {num_epochs} epochs. No additional training needed.")
        save_training_history_artifacts(model_name, history)
        return decoder, film

    training_start = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        decoder.train()
        film.train()
        epoch_losses = []

        profiler = EpochProfiler()
        data_wait_start = time.perf_counter()
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            profiler.add_data_wait(time.perf_counter() - data_wait_start)
            metrics = train_step_reconstruction(
                decoder, encoder_fn, film, batch, optimizer, scaler, device=device,
                profile_step=profiler.should_profile(step_idx),
            )
            profiler.add_step_profile(metrics.get("_profile"))
            epoch_losses.append(metrics["loss"])
            data_wait_start = time.perf_counter()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_seconds = time.perf_counter() - epoch_start
        profile_summary = profiler.summary()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {lr_now:.2e} | Time: {format_duration(epoch_seconds)}")
        print_epoch_profile(model_name, epoch + 1, profile_summary)
        scheduler.step()

        val_loss = None
        is_best = False
        if val_loader is not None and (epoch + 1) % VAL_EVERY_EPOCHS == 0:
            val_loss = validate_model(decoder, encoder_fn, film, val_loader, device)
            print(f"  Val Loss: {val_loss:.6f}")
            logging.getLogger("music_inpainting").info("%s epoch=%s train_loss=%.6f val_loss=%.6f", model_name, epoch + 1, avg_loss, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
        else:
            logging.getLogger("music_inpainting").info("%s epoch=%s train_loss=%.6f", model_name, epoch + 1, avg_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
            "epoch_time": format_duration(epoch_seconds),
            "learning_rate": lr_now,
            "peak_vram_gb": get_peak_vram_gb(),
            **profile_summary,
        })

        if checkpoint_dir:
            save_training_checkpoint(
                decoder, optimizer, epoch, val_loss if val_loss is not None else avg_loss,
                checkpoint_dir, model_name, film=film, scheduler=scheduler, scaler=scaler,
                history=history, best_val_loss=best_val_loss, is_best=is_best,
            )
        save_training_history_artifacts(model_name, history)
        if val_loss is not None and should_early_stop(history, best_val_loss, epoch + 1, model_name):
            break

    if checkpoint_dir:
        paths = get_training_checkpoint_paths(checkpoint_dir, model_name)
        if not os.path.exists(paths["best"]):
            save_training_checkpoint(
                decoder, optimizer, num_epochs - 1, history[-1]["train_loss"],
                checkpoint_dir, model_name, film=film, scheduler=scheduler, scaler=scaler,
                history=history, best_val_loss=best_val_loss, is_best=True,
            )

    elapsed_this_run = time.perf_counter() - training_start
    total_seconds = float(sum(row.get("epoch_seconds", 0.0) for row in history))
    avg_epoch_seconds = float(np.mean([row.get("epoch_seconds", 0.0) for row in history])) if history else 0.0
    print(f"\nTraining time this run {model_name}: {format_duration(elapsed_this_run)}")
    print(f"Total recorded training time {model_name}: {format_duration(total_seconds)}")
    print(f"Average epoch time {model_name}: {format_duration(avg_epoch_seconds)}")
    record_training_timing(
        model_name, total_seconds, num_epochs, lr,
        batch_size=batch_size, dataset_fraction=dataset_fraction,
        best_val_loss=best_val_loss,
        checkpoint_path=os.path.join(checkpoint_dir, f"{model_name}_best.pt") if checkpoint_dir else None,
        epoch_times=[row.get("epoch_seconds", 0.0) for row in history],
        peak_vram_gb=get_peak_vram_gb(),
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    return decoder, film


def train_baseline_model(decoder, train_loader, val_loader=None,
                         num_epochs=50, lr=1e-4, device="cuda",
                         checkpoint_dir=None, model_name="baseline_cqtdiff",
                         batch_size=None, dataset_fraction=None):
    """
    Training loop buat baseline (tanpa encoder, tanpa FiLM).
    Saves latest checkpoints every epoch and resumes from *_latest.pt.
    """
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    reset_peak_vram_stats()
    start_epoch, best_val_loss, history = load_training_checkpoint_if_available(
        decoder, optimizer, scheduler, scaler, checkpoint_dir, model_name, device, film=None
    )

    if start_epoch >= num_epochs:
        print(f"{model_name} already reached {num_epochs} epochs. No additional training needed.")
        save_training_history_artifacts(model_name, history)
        return decoder

    training_start = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        decoder.train()
        epoch_losses = []

        profiler = EpochProfiler()
        data_wait_start = time.perf_counter()
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            profiler.add_data_wait(time.perf_counter() - data_wait_start)
            metrics = train_step_baseline(
                decoder, batch, optimizer, scaler, device=device,
                profile_step=profiler.should_profile(step_idx),
            )
            profiler.add_step_profile(metrics.get("_profile"))
            epoch_losses.append(metrics["loss"])
            data_wait_start = time.perf_counter()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_seconds = time.perf_counter() - epoch_start
        profile_summary = profiler.summary()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {lr_now:.2e} | Time: {format_duration(epoch_seconds)}")
        print_epoch_profile(model_name, epoch + 1, profile_summary)
        scheduler.step()

        val_loss = None
        is_best = False
        if val_loader is not None and (epoch + 1) % VAL_EVERY_EPOCHS == 0:
            val_loss = validate_baseline(decoder, val_loader, device)
            print(f"  Val Loss: {val_loss:.6f}")
            logging.getLogger("music_inpainting").info("%s epoch=%s train_loss=%.6f val_loss=%.6f", model_name, epoch + 1, avg_loss, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
        else:
            logging.getLogger("music_inpainting").info("%s epoch=%s train_loss=%.6f", model_name, epoch + 1, avg_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
            "epoch_time": format_duration(epoch_seconds),
            "learning_rate": lr_now,
            "peak_vram_gb": get_peak_vram_gb(),
            **profile_summary,
        })

        if checkpoint_dir:
            save_training_checkpoint(
                decoder, optimizer, epoch, val_loss if val_loss is not None else avg_loss,
                checkpoint_dir, model_name, scheduler=scheduler, scaler=scaler,
                history=history, best_val_loss=best_val_loss, is_best=is_best,
            )
        save_training_history_artifacts(model_name, history)
        if val_loss is not None and should_early_stop(history, best_val_loss, epoch + 1, model_name):
            break

    if checkpoint_dir:
        paths = get_training_checkpoint_paths(checkpoint_dir, model_name)
        if not os.path.exists(paths["best"]):
            save_training_checkpoint(
                decoder, optimizer, num_epochs - 1, history[-1]["train_loss"],
                checkpoint_dir, model_name, scheduler=scheduler, scaler=scaler,
                history=history, best_val_loss=best_val_loss, is_best=True,
            )

    elapsed_this_run = time.perf_counter() - training_start
    total_seconds = float(sum(row.get("epoch_seconds", 0.0) for row in history))
    avg_epoch_seconds = float(np.mean([row.get("epoch_seconds", 0.0) for row in history])) if history else 0.0
    print(f"\nTraining time this run {model_name}: {format_duration(elapsed_this_run)}")
    print(f"Total recorded training time {model_name}: {format_duration(total_seconds)}")
    print(f"Average epoch time {model_name}: {format_duration(avg_epoch_seconds)}")
    record_training_timing(
        model_name, total_seconds, num_epochs, lr,
        batch_size=batch_size, dataset_fraction=dataset_fraction,
        best_val_loss=best_val_loss,
        checkpoint_path=os.path.join(checkpoint_dir, f"{model_name}_best.pt") if checkpoint_dir else None,
        epoch_times=[row.get("epoch_seconds", 0.0) for row in history],
        peak_vram_gb=get_peak_vram_gb(),
    )
    print(f"\nBaseline training complete. Best val loss: {best_val_loss:.6f}")
    return decoder


def validate_model(decoder, encoder_fn, film, val_loader, device):
    """Validasi hybrid model (dengan encoder + FiLM)."""
    decoder.eval()
    film.eval()
    val_losses = []

    with torch.inference_mode():
        for batch in val_loader:
            clean = batch["clean"].to(device, non_blocking=True)
            masked = batch["masked"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            cached_z = batch.get("encoder_latent")
            z = cached_z.to(device, non_blocking=True) if cached_z is not None else encoder_fn(masked)
            features = decoder.get_features(masked, mask)
            cond_features = film(z.float(), features)
            if hasattr(decoder, "diffusion_loss"):
                loss_parts = decoder.diffusion_loss(
                    clean,
                    masked,
                    mask=mask,
                    conditioning=cond_features,
                    deterministic_sigma=True,
                )
                val_loss = loss_parts["loss"]
            else:
                pred_spec = decoder.decode_features(cond_features)

                target_spec = compute_stft_target(clean).permute(0, 2, 1)
                T_min = min(pred_spec.shape[1], target_spec.shape[1])
                pred_spec = pred_spec[:, :T_min, :]
                target_spec = target_spec[:, :T_min, :]

                frame_mask = compute_frame_mask(mask, T_min).unsqueeze(-1).expand_as(pred_spec)
                gap_loss = F.l1_loss(pred_spec[frame_mask], target_spec[frame_mask])
                full_loss = F.l1_loss(pred_spec, target_spec)
                _, waveform_gap_loss, energy_loss = compute_waveform_gap_losses(pred_spec, clean, mask)
                val_loss = (
                    gap_loss
                    + 0.1 * full_loss
                    + CQT_WAVEFORM_GAP_LOSS_WEIGHT * waveform_gap_loss
                    + CQT_ENERGY_LOSS_WEIGHT * energy_loss
                )
            val_losses.append(val_loss.item())

    return float(np.mean(val_losses))


def validate_baseline(decoder, val_loader, device):
    """Validasi baseline (tanpa encoder)."""
    decoder.eval()
    val_losses = []

    with torch.inference_mode():
        for batch in val_loader:
            clean = batch["clean"].to(device, non_blocking=True)
            masked = batch["masked"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            features = decoder.get_features(masked, mask)
            pred_spec = decoder.decode_features(features)

            target_spec = compute_stft_target(clean).permute(0, 2, 1)
            T_min = min(pred_spec.shape[1], target_spec.shape[1])
            pred_spec = pred_spec[:, :T_min, :]
            target_spec = target_spec[:, :T_min, :]

            frame_mask = compute_frame_mask(mask, T_min).unsqueeze(-1).expand_as(pred_spec)
            gap_loss = F.l1_loss(pred_spec[frame_mask], target_spec[frame_mask])
            full_loss = F.l1_loss(pred_spec, target_spec)
            _, waveform_gap_loss, energy_loss = compute_waveform_gap_losses(pred_spec, clean, mask)
            val_loss = (
                gap_loss
                + 0.1 * full_loss
                + CQT_WAVEFORM_GAP_LOSS_WEIGHT * waveform_gap_loss
                + CQT_ENERGY_LOSS_WEIGHT * energy_loss
            )
            val_losses.append(val_loss.item())

    return float(np.mean(val_losses))


def save_checkpoint(decoder, film, optimizer, epoch, val_loss, checkpoint_dir, model_name):
    """Simpan checkpoint model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    torch.save({
        "epoch": epoch,
        "decoder_state": decoder.state_dict(),
        "film_state": film.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, ckpt_path)
    print(f"  💾 Best checkpoint saved: {ckpt_path}")


print("✅ Training loop (conditioned diffusion / reconstruction) berhasil didefinisikan!")
print("   Komponen: train_step_reconstruction, train_step_baseline,")
print("             train_model, train_baseline_model, validate_model")


# ============================================================
# CELL 6.7: SHARED HYBRID TRAINING HELPERS
# ============================================================
# Helper bersama untuk:
# - builder encoder batch-capable (CLAP, AudioMAE)
# - builder decoder via adapters (CQT-Diff+ original, MAID original DDPM-Midi2Performance)
# - save/load checkpoint hybrid
# - trainer minimal-kompatibel untuk MAID
# - helper evaluasi hybrid yang selalu load checkpoint terlatih
#
# PERBAIKAN UTAMA:
# - proxy/replika decoder dinonaktifkan untuk final pipeline
# - Training & inference aligned sesuai objective decoder
# - Mask-aware: model tahu lokasi dan ukuran gap
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import torchaudio
import time

HYBRID_CKPT_SUFFIX = "_best.pt"


def get_model_checkpoint_dir(model_name: str):
    ckpt_dir = os.path.join(PATHS["checkpoints"], model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def get_model_checkpoint_path(model_name: str):
    return os.path.join(get_model_checkpoint_dir(model_name), f"{model_name}{HYBRID_CKPT_SUFFIX}")


def hybrid_checkpoint_exists(model_name: str):
    return os.path.exists(get_model_checkpoint_path(model_name))


def reset_training_checkpoints_if_requested(model_name: str, force_retrain: bool):
    """Remove latest/best checkpoints so FORCE_RETRAIN starts from epoch 1."""
    if not force_retrain:
        return

    ckpt_dir = get_model_checkpoint_dir(model_name)
    paths = get_training_checkpoint_paths(ckpt_dir, model_name)
    removed = []
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)

    if removed:
        print(f"♻️ FORCE_RETRAIN aktif. Checkpoint lama dihapus untuk {model_name}:")
        for path in removed:
            print(f"   - {path}")


def load_hybrid_checkpoint(model_name: str, decoder: nn.Module, film_layer: nn.Module, device):
    ckpt_path = get_model_checkpoint_path(model_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint untuk {model_name} belum ada: {ckpt_path}. Jalankan cell training-nya terlebih dahulu."
        )

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    expected_arch = getattr(decoder, "architecture_name", None)
    checkpoint_arch = payload.get("architecture")
    if expected_arch is not None and checkpoint_arch != expected_arch:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} memakai arsitektur lama "
            f"({checkpoint_arch or 'unknown'}), sedangkan model sekarang {expected_arch}. "
            "Retrain model ini agar SSL menjadi conditioning diffusion, bukan refinement lama."
        )
    if expected_arch is not None and checkpoint_arch is None:
        decoder.load_state_dict(payload["decoder_state"], strict=False)
    else:
        decoder.load_state_dict(payload["decoder_state"])
    film_layer.load_state_dict(payload["film_state"])
    print(f"✅ Checkpoint diload: {ckpt_path}")
    return payload


def load_baseline_checkpoint(decoder: nn.Module, device):
    """Load checkpoint baseline (tanpa FiLM)."""
    ckpt_dir = get_model_checkpoint_dir("baseline_cqtdiff")
    ckpt_path = os.path.join(ckpt_dir, "baseline_cqtdiff_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Baseline checkpoint belum ada: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    expected_arch = getattr(decoder, "architecture_name", None)
    checkpoint_arch = payload.get("architecture")
    if expected_arch is not None and checkpoint_arch not in {None, expected_arch}:
        raise RuntimeError(
            f"Baseline checkpoint {ckpt_path} memakai arsitektur {checkpoint_arch}, "
            f"sedangkan model sekarang {expected_arch}."
        )
    if expected_arch is not None and checkpoint_arch is None:
        decoder.load_state_dict(payload["decoder_state"], strict=False)
    else:
        decoder.load_state_dict(payload["decoder_state"])
    print(f"✅ Baseline checkpoint diload: {ckpt_path}")
    return payload


_MEL_FILTER_CACHE = {}
_WINDOW_CACHE = {}
_INVERSE_MEL_CACHE = {}
_GRIFFINLIM_CACHE = {}


def _get_hann_window(n_fft, device):
    key = (n_fft, str(device))
    window = _WINDOW_CACHE.get(key)
    if window is None or window.device != device:
        window = torch.hann_window(n_fft, device=device)
        _WINDOW_CACHE[key] = window
    return window


def _get_mel_filter(sr, n_fft, n_mels, device):
    key = (sr, n_fft, n_mels, str(device))
    mel = _MEL_FILTER_CACHE.get(key)
    if mel is None or mel.device != device:
        # TorchAudio returns (freq, mel); transpose once and cache on-device.
        mel = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sr / 2,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        ).to(device=device, dtype=torch.float32).transpose(0, 1).contiguous()
        _MEL_FILTER_CACHE[key] = mel
    return mel


def _module_device(module, fallback):
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return fallback


def _get_inverse_mel_scale(sr, n_fft, n_mels, device):
    key = (sr, n_fft, n_mels, str(device))
    inverse = _INVERSE_MEL_CACHE.get(key)
    if inverse is None or _module_device(inverse, device) != device:
        inverse = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        _INVERSE_MEL_CACHE[key] = inverse
    return inverse


def _get_griffinlim(n_fft, hop_length, device):
    key = (n_fft, hop_length, str(device))
    griffinlim = _GRIFFINLIM_CACHE.get(key)
    if griffinlim is None or _module_device(griffinlim, device) != device:
        griffinlim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=64,
            win_length=n_fft,
            hop_length=hop_length,
            power=1.0,
            momentum=0.99,
        ).to(device)
        _GRIFFINLIM_CACHE[key] = griffinlim
    return griffinlim


def torch_audio_to_mel_batch(audio_batch, sr: int = TARGET_SR, n_mels: int = 128,
                             n_fft: int = 2048, hop_length: int = 512, device=None):
    """GPU mel-power/log-mel path matching librosa power_to_db(ref=np.max) semantics."""
    if isinstance(audio_batch, torch.Tensor):
        if device is None:
            device = audio_batch.device
        audio = audio_batch.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        audio = torch.as_tensor(np.stack(ensure_audio_list(audio_batch), axis=0), dtype=torch.float32, device=device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() == 3:
        audio = audio.squeeze(1)

    autocast_device = "cuda" if audio.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, enabled=False):
        audio = audio.float()
        window = _get_hann_window(n_fft, audio.device)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel_basis = _get_mel_filter(sr, n_fft, n_mels, audio.device)
        mel_power = torch.einsum("mf,bft->bmt", mel_basis, power).clamp_min(1e-10)

        mel_db_raw = 10.0 * torch.log10(mel_power)
        ref_db = mel_db_raw.amax(dim=(1, 2), keepdim=True)
        mel_db = torch.clamp(mel_db_raw - ref_db, min=-80.0)
        mel_norm = (mel_db + 40.0) / 40.0
    return mel_db, mel_norm





def ensure_audio_list(audio_batch):
    if isinstance(audio_batch, torch.Tensor):
        audio_batch = audio_batch.detach().cpu().numpy()

    if isinstance(audio_batch, np.ndarray):
        if audio_batch.ndim == 1:
            return [audio_batch.astype(np.float32)]
        return [sample.astype(np.float32) for sample in audio_batch]

    if isinstance(audio_batch, (list, tuple)):
        return [np.asarray(sample, dtype=np.float32) for sample in audio_batch]

    raise TypeError(f"Tipe audio batch tidak didukung: {type(audio_batch)}")



def build_film_layer(model_name: str, device):
    cfg = FILM_CONFIGS[model_name]
    return FiLMLayer(
        encoder_dim=cfg["encoder_dim"],
        decoder_feature_dim=cfg["decoder_feature_dim"],
    ).to(device)


def _load_official_adapter(module_name: str, builder_name: str, model_label: str):
    """Load an official-model adapter and fail loudly if it is not configured."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"{model_label} harus memakai model asli, tetapi adapter '{module_name}' "
            f"belum tersedia/importable. Buat module adapter tersebut atau set env var "
            f"yang sesuai sebelum menjalankan pipeline final."
        ) from exc

    builder = getattr(module, builder_name, None)
    if not callable(builder):
        raise RuntimeError(
            f"Adapter '{module_name}' tidak punya fungsi callable '{builder_name}' "
            f"untuk membangun {model_label} asli."
        )
    return builder


def _validate_decoder_interface(decoder, required_methods, model_label: str):
    missing = [name for name in required_methods if not hasattr(decoder, name)]
    if missing:
        raise TypeError(
            f"{model_label} adapter tidak kompatibel dengan training/evaluasi pipeline ini. "
            f"Method yang belum ada: {missing}"
        )
    return decoder



def build_hybrid_cqtdiff_decoder(device):
    builder = _load_official_adapter(
        OFFICIAL_CQTDIFF_ADAPTER,
        "build_cqtdiff_decoder",
        "MusicNet CQTdiff+ original",
    )
    decoder = builder(
        device=device,
        target_sr=TARGET_SR,
        segment_samples=SEGMENT_SAMPLES,
        gap_durations_ms=GAP_DURATIONS_MS,
        cqt_diff_dir=CQT_DIFF_DIR,
    )
    decoder = _validate_decoder_interface(
        decoder,
        ["get_features", "decode_features", "inpaint", "parameters", "state_dict", "load_state_dict", "train", "eval"],
        "MusicNet CQTdiff+ original",
    )
    decoder.eval()
    print("MusicNet CQTdiff+ original loaded via official audio-inpainting adapter.")
    return decoder



def build_maid_decoder(device):
    builder = _load_official_adapter(
        MAID_ADAPTER,
        "build_maid_decoder",
        "MAID original DDPM-Midi2Performance",
    )
    decoder = builder(
        device=device,
        target_sr=TARGET_SR,
        segment_samples=SEGMENT_SAMPLES,
        gap_durations_ms=GAP_DURATIONS_MS,
        midi2performance_dir=MIDI2PERFORMANCE_DIR,
    )
    decoder = _validate_decoder_interface(
        decoder,
        ["get_features", "predict_mel_norm", "audio_to_mel_batch", "mask_to_frame_mask", "inpaint",
         "diffusion_loss", "parameters", "state_dict", "load_state_dict", "train", "eval"],
        "MAID original DDPM-Midi2Performance",
    )
    decoder.eval()
    print("MAID original DDPM-Midi2Performance loaded via adapter.")
    return decoder



def build_clap_encoder(device):
    from transformers import ClapModel, ClapProcessor

    torch_dtype = torch.float16
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained(
        "laion/clap-htsat-unfused",
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    def encode(audio_batch, sr: int = TARGET_SR):
        audio_list = []
        for audio in ensure_audio_list(audio_batch):
            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            audio_list.append(audio.astype(np.float32))

        inputs = processor(audio=audio_list, sampling_rate=48000, return_tensors="pt", padding=True)
        inputs = {
            key: value.to(device=device, dtype=torch_dtype if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            audio_features = model.get_audio_features(**inputs)

        if isinstance(audio_features, torch.Tensor):
            return audio_features.float()
        if hasattr(audio_features, "pooler_output"):
            return audio_features.pooler_output.float()
        raise TypeError(f"Output CLAP tidak dikenali: {type(audio_features)}")

    return model, encode



def build_audiomae_encoder(device):
    if not os.path.isdir(AUDIO_MAE_DIR):
        raise RuntimeError(f"Repo AudioMAE tidak ditemukan: {AUDIO_MAE_DIR}")
    if AUDIO_MAE_DIR not in sys.path:
        sys.path.insert(0, AUDIO_MAE_DIR)
    # AudioMAE repo targets older PyTorch where torch._six still existed.
    try:
        import types
        import math as _math
        import collections.abc as _container_abcs
        import numpy as _np
        if not hasattr(_np, "float"):
            _np.float = float
        if "torch._six" not in sys.modules:
            _six = types.ModuleType("torch._six")
            _six.inf = _math.inf
            _six.container_abcs = _container_abcs
            sys.modules["torch._six"] = _six
    except Exception:
        pass

    try:
        import models_mae
    except Exception as exc:
        raise RuntimeError(
            f"AudioMAE harus memakai repo asli di {AUDIO_MAE_DIR}, tetapi import models_mae gagal."
        ) from exc
    # AudioMAE was written for older timm where Block accepted qk_scale.
    try:
        from timm.models.vision_transformer import Block as _TimmBlock

        class _AudioMAECompatBlock(_TimmBlock):
            def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                         qk_scale=None, norm_layer=nn.LayerNorm, **kwargs):
                kwargs.pop("qk_scale", None)
                super().__init__(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    **kwargs,
                )

        models_mae.Block = _AudioMAECompatBlock
    except Exception as exc:
        raise RuntimeError("Gagal memasang compatibility shim timm Block untuk AudioMAE.") from exc

    model = models_mae.mae_vit_base_patch16(
        norm_pix_loss=False,
        in_chans=1,
        audio_exp=True,
        img_size=(1024, 128),
        alpha=0.0,
        mode=0,
        use_custom_patch=False,
        split_pos=False,
        pos_trainable=False,
        use_nce=False,
        decoder_mode=0,
        mask_2d=False,
        mask_t_prob=0.6,
        mask_f_prob=0.5,
        no_shift=False,
    ).to(device)

    ckpt_candidates = [
        os.environ.get("AUDIOMAE_CHECKPOINT"),
        os.path.join(AUDIO_MAE_DIR, "ckpt", "pretrained.pth"),
        os.path.join(AUDIO_MAE_DIR, "ckpt", "finetuned.pth"),
        os.path.join(AUDIO_MAE_DIR, "pretrained.pth"),
        os.path.join(AUDIO_MAE_DIR, "finetuned.pth"),
    ]
    ckpt_path = next((p for p in ckpt_candidates if p and os.path.exists(p)), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            "Checkpoint AudioMAE asli belum ditemukan. Letakkan checkpoint di "
            f"{os.path.join(AUDIO_MAE_DIR, 'ckpt', 'pretrained.pth')} atau set AUDIOMAE_CHECKPOINT."
        )

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"AudioMAE asli dari repo berhasil diload: {ckpt_path}")
    print(f"AudioMAE load_state_dict: {msg}")

    fbank_mean = -4.2677393
    fbank_std = 4.5689974

    def _audio_to_audiomae_input(audio_list, sr):
        fbanks = []
        for audio in audio_list:
            wav = torch.as_tensor(audio, dtype=torch.float32).view(1, -1).cpu()
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav - wav.mean()
            fbank = torchaudio.compliance.kaldi.fbank(
                wav,
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
            if fbank.shape[0] < 1024:
                fbank = F.pad(fbank, (0, 0, 0, 1024 - fbank.shape[0]))
            elif fbank.shape[0] > 1024:
                fbank = fbank[:1024, :]
            fbank = (fbank - fbank_mean) / (fbank_std * 2.0)
            fbanks.append(fbank)
        return torch.stack(fbanks, dim=0).unsqueeze(1).to(device, non_blocking=True)

    def encode(audio_batch, sr: int = TARGET_SR):
        audio_list = ensure_audio_list(audio_batch)

        with torch.inference_mode():
            inputs = _audio_to_audiomae_input(audio_list, sr)
            embeddings = model.forward_encoder_no_mask(inputs)
            return embeddings[:, 0, :].float()

    return model, encode



def train_maid_step(decoder, encoder_fn, film, batch, optimizer, scaler, cfg_drop=0.1,
                    device="cuda", profile_step=False):
    """Training step MAID: reconstruction loss di mel domain, with GPU mel extraction."""
    decoder.train()
    film.train()
    profile = {}

    _sync_if_profile(profile_step)
    t0 = time.perf_counter()
    clean = batch["clean"].to(device, non_blocking=True)
    masked = batch["masked"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    cached_z = batch.get("encoder_latent")
    if cached_z is not None:
        cached_z = cached_z.to(device, non_blocking=True)
    _sync_if_profile(profile_step)
    if profile_step:
        profile["h2d_seconds"] = time.perf_counter() - t0

    batch_size = clean.size(0)

    _sync_if_profile(profile_step)
    t_pre = time.perf_counter()
    with torch.no_grad():
        z = cached_z if cached_z is not None else encoder_fn(masked)
    _sync_if_profile(profile_step)
    if profile_step:
        profile["preprocess_seconds"] = time.perf_counter() - t_pre

    drop = (torch.rand(batch_size, device=device) < cfg_drop).view(-1, *([1] * (z.dim() - 1)))
    z = torch.where(drop, torch.zeros_like(z), z)

    _sync_if_profile(profile_step)
    t_gpu = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        decoder_features = decoder.get_features(masked, mask)
        conditioned_features = film(z.float(), decoder_features)
        if hasattr(decoder, "diffusion_loss"):
            loss_parts = decoder.diffusion_loss(
                clean,
                masked,
                mask=mask,
                conditioning=conditioned_features,
            )
            loss = loss_parts["loss"]
            gap_loss = loss_parts.get("gap_loss", loss)
            full_loss = loss_parts.get("full_loss", loss)
        else:
            pred_mel_norm = decoder.predict_mel_norm(masked, conditioning=conditioned_features)
            _, clean_mel_norm = decoder.audio_to_mel_batch(clean)
            clean_mel_norm = clean_mel_norm.permute(0, 2, 1)

            frame_mask = decoder.mask_to_frame_mask(mask, pred_mel_norm.shape[1])
            expanded_mask = frame_mask.unsqueeze(-1).expand_as(pred_mel_norm)

            gap_loss = F.l1_loss(pred_mel_norm[expanded_mask], clean_mel_norm[expanded_mask])
            full_loss = F.l1_loss(pred_mel_norm, clean_mel_norm)
            loss = gap_loss + 0.1 * full_loss

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(decoder.parameters()) + list(film.parameters()), max_norm=1.0
    )
    scaler.step(optimizer)
    scaler.update()
    _sync_if_profile(profile_step)
    if profile_step:
        profile["gpu_compute_seconds"] = time.perf_counter() - t_gpu

    return {
        "loss": loss.item(),
        "gap_loss": gap_loss.item() if isinstance(gap_loss, torch.Tensor) else float(gap_loss),
        "full_loss": full_loss.item() if isinstance(full_loss, torch.Tensor) else float(full_loss),
        "_profile": profile,
    }


def train_maid_model(decoder, encoder_fn, film, train_loader, val_loader=None,
                     num_epochs=20, lr=1e-4, device="cuda", checkpoint_dir=None,
                     model_name="model", batch_size=None, dataset_fraction=None):
    """Training loop MAID (mel reconstruction) with resume-safe checkpointing."""
    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(film.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    reset_peak_vram_stats()
    start_epoch, best_val_loss, history = load_training_checkpoint_if_available(
        decoder, optimizer, scheduler, scaler, checkpoint_dir, model_name, device, film=film
    )
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt") if checkpoint_dir else None

    if start_epoch >= num_epochs:
        print(f"{model_name} already reached {num_epochs} epochs. No additional training needed.")
        save_training_history_artifacts(model_name, history)
        return decoder, film

    training_start = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        decoder.train()
        film.train()
        epoch_losses = []

        profiler = EpochProfiler()
        data_wait_start = time.perf_counter()
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            profiler.add_data_wait(time.perf_counter() - data_wait_start)
            metrics = train_maid_step(
                decoder, encoder_fn, film, batch, optimizer, scaler, device=device,
                profile_step=profiler.should_profile(step_idx),
            )
            profiler.add_step_profile(metrics.get("_profile"))
            epoch_losses.append(metrics["loss"])
            data_wait_start = time.perf_counter()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_seconds = time.perf_counter() - epoch_start
        profile_summary = profiler.summary()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {lr_now:.2e} | Time: {format_duration(epoch_seconds)}")
        print_epoch_profile(model_name, epoch + 1, profile_summary)
        scheduler.step()

        val_loss = None
        is_best = False
        if val_loader is not None and (epoch + 1) % VAL_EVERY_EPOCHS == 0:
            decoder.eval()
            film.eval()
            val_losses = []

            with torch.inference_mode():
                for batch in val_loader:
                    clean = batch["clean"].to(device, non_blocking=True)
                    masked = batch["masked"].to(device, non_blocking=True)
                    mask_b = batch["mask"].to(device, non_blocking=True)
                    cached_z = batch.get("encoder_latent")
                    z = cached_z.to(device, non_blocking=True) if cached_z is not None else encoder_fn(masked)
                    decoder_features = decoder.get_features(masked, mask_b)
                    conditioned_features = film(z.float(), decoder_features)
                    if hasattr(decoder, "diffusion_loss"):
                        loss_parts = decoder.diffusion_loss(
                            clean,
                            masked,
                            mask=mask_b,
                            conditioning=conditioned_features,
                            deterministic_sigma=True,
                        )
                        batch_val_loss = loss_parts.get("gap_loss", loss_parts["loss"])
                    else:
                        pred_mel_norm = decoder.predict_mel_norm(masked, conditioning=conditioned_features)
                        _, clean_mel_norm = decoder.audio_to_mel_batch(clean)
                        clean_mel_norm = clean_mel_norm.permute(0, 2, 1)
                        frame_mask = decoder.mask_to_frame_mask(mask_b, pred_mel_norm.shape[1])
                        expanded_mask = frame_mask.unsqueeze(-1).expand_as(pred_mel_norm)
                        batch_val_loss = F.l1_loss(pred_mel_norm[expanded_mask], clean_mel_norm[expanded_mask])
                    val_losses.append(batch_val_loss.item())

            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            print(f"  Val Loss: {val_loss:.6f}")
            logging.getLogger("music_inpainting").info("%s epoch=%s train_loss=%.6f val_loss=%.6f", model_name, epoch + 1, avg_loss, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
        else:
            logging.getLogger("music_inpainting").info("%s epoch=%s train_loss=%.6f", model_name, epoch + 1, avg_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
            "epoch_time": format_duration(epoch_seconds),
            "learning_rate": lr_now,
            "peak_vram_gb": get_peak_vram_gb(),
            **profile_summary,
        })

        if checkpoint_dir:
            save_training_checkpoint(
                decoder, optimizer, epoch, val_loss if val_loss is not None else avg_loss,
                checkpoint_dir, model_name, film=film, scheduler=scheduler, scaler=scaler,
                history=history, best_val_loss=best_val_loss, is_best=is_best,
            )
        save_training_history_artifacts(model_name, history)
        if val_loss is not None and should_early_stop(history, best_val_loss, epoch + 1, model_name):
            break

    if checkpoint_dir:
        paths = get_training_checkpoint_paths(checkpoint_dir, model_name)
        if not os.path.exists(paths["best"]):
            save_training_checkpoint(
                decoder, optimizer, num_epochs - 1, history[-1]["train_loss"],
                checkpoint_dir, model_name, film=film, scheduler=scheduler, scaler=scaler,
                history=history, best_val_loss=best_val_loss, is_best=True,
            )

    elapsed_this_run = time.perf_counter() - training_start
    total_seconds = float(sum(row.get("epoch_seconds", 0.0) for row in history))
    avg_epoch_seconds = float(np.mean([row.get("epoch_seconds", 0.0) for row in history])) if history else 0.0
    print(f"\nTraining time this run {model_name}: {format_duration(elapsed_this_run)}")
    print(f"Total recorded training time {model_name}: {format_duration(total_seconds)}")
    print(f"Average epoch time {model_name}: {format_duration(avg_epoch_seconds)}")
    record_training_timing(
        model_name, total_seconds, num_epochs, lr,
        batch_size=batch_size, dataset_fraction=dataset_fraction,
        best_val_loss=best_val_loss, checkpoint_path=checkpoint_path,
        epoch_times=[row.get("epoch_seconds", 0.0) for row in history],
        peak_vram_gb=get_peak_vram_gb(),
    )
    print(f"\nMAID training complete. Best val loss: {best_val_loss:.6f}")
    return decoder, film


def run_hybrid_inpainting_evaluation(model_label: str, encoder_fn, decoder, film_layer, device,
                                     n_eval_samples: int = 50, model_name: str = None):
    """
    Evaluasi hybrid model (encoder + FiLM + decoder).

    Evaluasi hybrid memakai decoder asli melalui adapter resmi.
    FiLM conditioning di-inject ke fitur decoder sesuai interface adapter.
    """
    evaluation_start = time.perf_counter()
    model_name = model_name or model_name_from_label(model_label)
    if model_name not in EXPECTED_MODEL_CONFIGS and model_name not in ALL_MODELS:
        raise ValueError(f"Model name evaluasi tidak dikenal: {model_name}")
    cache_tag = set_reconstruction_cache_context(
        model_name,
        decoder,
        checkpoint_path=get_model_checkpoint_path(model_name) if hybrid_checkpoint_exists(model_name) else None,
    )
    prepare_reconstructed_outputs(model_name, clear=EVAL_CLEAR_RECONSTRUCTIONS)
    output_manifest_rows = []
    alignment_rows = []
    conditioning_rows = []
    original_audios, masked_by_gap, gap_regions_by_gap = load_preprocessed_data(
        n_eval_samples, gap_position=EVAL_GAP_POSITION
    )
    reconstructed_dict = {}
    reused_count = 0
    generated_count = 0

    print(f"\n🎵 Menjalankan inpainting {model_label}...")
    print(f"  Cache tag aktif: {cache_tag[:96]}...")
    if EVAL_REUSE_RECONSTRUCTIONS:
        print("  Resume rekonstruksi aktif: WAV yang sudah ada akan dipakai ulang.")
        summarize_reconstruction_cache(model_name, n_eval_samples)
    if EVAL_CLEAR_RECONSTRUCTIONS:
        print("  EVAL_CLEAR_RECONSTRUCTIONS aktif: output lama dibersihkan sebelum eval.")

    for gap_ms in GAP_DURATIONS_MS:
        print(f"\n  Gap {gap_ms}ms...")
        reconstructed_list = []

        for index, (orig_audio, masked_audio) in enumerate(zip(original_audios, masked_by_gap[gap_ms])):
            gap_region = gap_regions_by_gap[gap_ms][index]
            if (index + 1) % 10 == 0:
                print(f"    Sample {index+1}/{n_eval_samples}")

            reconstructed, output_path = load_reconstructed_output_if_available(
                model_name, gap_ms, index, expected_n_samples=len(orig_audio), sr=TARGET_SR
            )

            if reconstructed is not None:
                reused_count += 1
            else:
                with torch.inference_mode():
                    # Encode masked audio pakai SSL encoder
                    encoder_latent = encoder_fn(masked_audio)
                    if not torch.isfinite(encoder_latent).all():
                        raise RuntimeError(f"Encoder latent {model_name} mengandung NaN/Inf pada gap={gap_ms}, sample={index}.")

                    masked_tensor = torch.from_numpy(masked_audio).float().unsqueeze(0).to(device)
                    mask, gap_start, gap_end = build_gap_mask_array(
                        len(masked_audio), gap_ms, gap_start=gap_region["gap_start"]
                    )
                    align = validate_masked_gap_alignment(
                        orig_audio, masked_audio, mask, gap_ms,
                        expected_gap_start=gap_start, expected_gap_end=gap_end,
                    )
                    align.update({
                        "model": model_name,
                        "gap_ms": int(gap_ms),
                        "sample_index": int(index),
                        "gap_position": gap_region["gap_position"],
                    })
                    alignment_rows.append(align)
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

                    has_diffusion = hasattr(decoder, "_inpaint_diffusion")
                    import inspect
                    sig = inspect.signature(decoder.get_features)
                    has_mask_param = 'mask' in sig.parameters

                    if has_diffusion:
                        # CQT-Diff+: SSL representations condition the diffusion
                        # sampler directly. Tidak ada post-processing/refinement
                        # setelah baseline diffusion selesai.
                        decoder_features = decoder.get_features(masked_tensor, mask_tensor)
                        conditioned_features = film_layer(encoder_latent.float(), decoder_features)
                        if not torch.isfinite(decoder_features).all():
                            raise RuntimeError(f"Decoder features {model_name} mengandung NaN/Inf pada gap={gap_ms}, sample={index}.")
                        if not torch.isfinite(conditioned_features).all():
                            raise RuntimeError(f"FiLM conditioning {model_name} mengandung NaN/Inf pada gap={gap_ms}, sample={index}.")
                        conditioning_rows.append({
                            "model": model_name,
                            "gap_ms": int(gap_ms),
                            "sample_index": int(index),
                            "encoder_shape": tuple(encoder_latent.shape),
                            "decoder_features_shape": tuple(decoder_features.shape),
                            "conditioned_features_shape": tuple(conditioned_features.shape),
                            "encoder_min": float(encoder_latent.float().min().item()),
                            "encoder_max": float(encoder_latent.float().max().item()),
                            "encoder_mean": float(encoder_latent.float().mean().item()),
                            "encoder_std": float(encoder_latent.float().std(unbiased=False).item()),
                            "conditioning_min": float(conditioned_features.float().min().item()),
                            "conditioning_max": float(conditioned_features.float().max().item()),
                            "conditioning_mean": float(conditioned_features.float().mean().item()),
                            "conditioning_std": float(conditioned_features.float().std(unbiased=False).item()),
                        })

                        reconstructed = decoder.inpaint(
                            masked_tensor,
                            mask_tensor,
                            conditioning=conditioned_features,
                        )
                    else:
                        # MAID: diffusion handled internally by decoder.inpaint()
                        decoder_features = decoder.get_features(masked_tensor, mask_tensor)
                        conditioned_features = film_layer(encoder_latent.float(), decoder_features)
                        if not torch.isfinite(decoder_features).all():
                            raise RuntimeError(f"Decoder features {model_name} mengandung NaN/Inf pada gap={gap_ms}, sample={index}.")
                        if not torch.isfinite(conditioned_features).all():
                            raise RuntimeError(f"FiLM conditioning {model_name} mengandung NaN/Inf pada gap={gap_ms}, sample={index}.")
                        conditioning_rows.append({
                            "model": model_name,
                            "gap_ms": int(gap_ms),
                            "sample_index": int(index),
                            "encoder_shape": tuple(encoder_latent.shape),
                            "decoder_features_shape": tuple(decoder_features.shape),
                            "conditioned_features_shape": tuple(conditioned_features.shape),
                            "encoder_min": float(encoder_latent.float().min().item()),
                            "encoder_max": float(encoder_latent.float().max().item()),
                            "encoder_mean": float(encoder_latent.float().mean().item()),
                            "encoder_std": float(encoder_latent.float().std(unbiased=False).item()),
                            "conditioning_min": float(conditioned_features.float().min().item()),
                            "conditioning_max": float(conditioned_features.float().max().item()),
                            "conditioning_mean": float(conditioned_features.float().mean().item()),
                            "conditioning_std": float(conditioned_features.float().std(unbiased=False).item()),
                        })

                        reconstructed = decoder.inpaint(
                            masked_tensor,
                            mask_tensor,
                            conditioning=conditioned_features,
                        )

                    # Crossfade buat menghilangkan click artifacts
                    reconstructed = crossfade_boundary(
                        orig_audio, reconstructed, gap_start, gap_end,
                    )
                output_path = save_reconstructed_output(model_name, gap_ms, index, reconstructed, TARGET_SR)
                generated_count += 1

            reconstructed_list.append(reconstructed)
            output_manifest_rows.append({
                "model": model_name,
                "gap_ms": gap_ms,
                "sample_index": index,
                "sr": TARGET_SR,
                "n_samples": int(len(reconstructed)),
                "duration_seconds": float(len(reconstructed) / TARGET_SR),
                "gap_start": int(gap_region["gap_start"]),
                "gap_end": int(gap_region["gap_end"]),
                "gap_position": gap_region["gap_position"],
                "reconstructed_path": output_path,
            })

        reconstructed_dict[gap_ms] = reconstructed_list

    save_reconstruction_manifest(model_name, output_manifest_rows)
    save_reconstruction_diagnostics(
        model_name, original_audios, reconstructed_dict, alignment_rows,
        conditioning_rows, gap_regions_by_gap=gap_regions_by_gap,
    )
    print(f"  Rekonstruksi reused/generated: {reused_count}/{generated_count}")
    results_df = evaluate_all_gaps(original_audios, reconstructed_dict, TARGET_SR, gap_regions_by_gap)
    record_evaluation_timing(model_name, time.perf_counter() - evaluation_start, n_eval_samples)
    return results_df


def run_baseline_inpainting_evaluation(decoder, device, n_eval_samples: int = 50):
    """
    Evaluasi baseline (tanpa encoder, tanpa FiLM).
    Decoder dipakai langsung tanpa conditioning.
    """
    baseline_eval_start = time.perf_counter()
    model_name = "baseline_cqtdiff"
    cache_tag = set_reconstruction_cache_context(model_name, decoder, checkpoint_path=None)
    prepare_reconstructed_outputs(model_name, clear=EVAL_CLEAR_RECONSTRUCTIONS)
    output_manifest_rows = []
    alignment_rows = []
    original_audios, masked_by_gap, gap_regions_by_gap = load_preprocessed_data(
        n_eval_samples, gap_position=EVAL_GAP_POSITION
    )
    reconstructed_dict = {}
    reused_count = 0
    generated_count = 0

    print("\n🎵 Menjalankan baseline CQT-Diff+ (tanpa SSL encoder)...")
    print(f"  Cache tag aktif: {cache_tag[:96]}...")
    if EVAL_REUSE_RECONSTRUCTIONS:
        print("  Resume rekonstruksi aktif: WAV yang sudah ada akan dipakai ulang.")
        summarize_reconstruction_cache(model_name, n_eval_samples)
    if EVAL_CLEAR_RECONSTRUCTIONS:
        print("  EVAL_CLEAR_RECONSTRUCTIONS aktif: output lama dibersihkan sebelum eval.")

    for gap_ms in GAP_DURATIONS_MS:
        print(f"\n  Gap {gap_ms}ms...")
        reconstructed_list = []

        for i, (orig_audio, masked_audio) in enumerate(zip(original_audios, masked_by_gap[gap_ms])):
            gap_region = gap_regions_by_gap[gap_ms][i]
            if (i + 1) % 10 == 0:
                print(f"    Sample {i+1}/{n_eval_samples}")

            reconstructed, output_path = load_reconstructed_output_if_available(
                model_name, gap_ms, i, expected_n_samples=len(orig_audio), sr=TARGET_SR
            )

            if reconstructed is not None:
                reused_count += 1
            else:
                with torch.inference_mode():
                    masked_tensor = torch.from_numpy(masked_audio).float().unsqueeze(0).to(device)
                    mask, gap_start, gap_end = build_gap_mask_array(
                        len(masked_audio), gap_ms, gap_start=gap_region["gap_start"]
                    )
                    align = validate_masked_gap_alignment(
                        orig_audio, masked_audio, mask, gap_ms,
                        expected_gap_start=gap_start, expected_gap_end=gap_end,
                    )
                    align.update({
                        "model": model_name,
                        "gap_ms": int(gap_ms),
                        "sample_index": int(i),
                        "gap_position": gap_region["gap_position"],
                    })
                    alignment_rows.append(align)
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

                    # Baseline: conditioning=None
                    reconstructed = decoder.inpaint(
                        masked_tensor, mask_tensor, conditioning=None,
                    )
                    reconstructed = crossfade_boundary(
                        orig_audio, reconstructed, gap_start, gap_end,
                    )
                output_path = save_reconstructed_output(model_name, gap_ms, i, reconstructed, TARGET_SR)
                generated_count += 1

            reconstructed_list.append(reconstructed)
            output_manifest_rows.append({
                "model": model_name,
                "gap_ms": gap_ms,
                "sample_index": i,
                "sr": TARGET_SR,
                "n_samples": int(len(reconstructed)),
                "duration_seconds": float(len(reconstructed) / TARGET_SR),
                "gap_start": int(gap_region["gap_start"]),
                "gap_end": int(gap_region["gap_end"]),
                "gap_position": gap_region["gap_position"],
                "reconstructed_path": output_path,
            })

        reconstructed_dict[gap_ms] = reconstructed_list

    save_reconstruction_manifest(model_name, output_manifest_rows)
    save_reconstruction_diagnostics(
        model_name, original_audios, reconstructed_dict, alignment_rows, None,
        gap_regions_by_gap=gap_regions_by_gap,
    )
    print(f"  Rekonstruksi reused/generated: {reused_count}/{generated_count}")
    results_df = evaluate_all_gaps(original_audios, reconstructed_dict, TARGET_SR, gap_regions_by_gap)
    record_evaluation_timing(model_name, time.perf_counter() - baseline_eval_start, n_eval_samples)
    return results_df


print("✅ Shared hybrid training helpers berhasil didefinisikan!")
print("   Tersedia: builder encoder asli, adapter decoder asli, checkpoint helpers, trainer MAID,")
print("   run_hybrid_inpainting_evaluation, run_baseline_inpainting_evaluation")


def _parse_csv_arg(value, default_items=None):
    if value is None or str(value).strip() == "":
        return list(default_items or [])
    text = str(value).strip()
    if text.lower() == "all":
        return list(default_items or [])
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_run_selection():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", "--run-phase", dest="phase", default=os.environ.get("RUN_PHASE", "all"))
    parser.add_argument("--models", "--run-models", dest="models", default=os.environ.get("RUN_MODELS", "all"))
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        default=os.environ.get("AUTO_STOP_INSTANCE", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Stop/power off the instance after all selected phase/model targets finish successfully.",
    )
    parser.add_argument(
        "--auto-stop-command",
        default=os.environ.get("AUTO_STOP_COMMAND", "sudo shutdown -h now"),
        help="Command used when --auto-stop is enabled. Default: sudo shutdown -h now",
    )
    args, _ = parser.parse_known_args()

    requested_phases = set(_parse_csv_arg(args.phase, ["all"]))
    if "all" in requested_phases:
        phases = {"train", "eval", "summary"}
    else:
        phases = requested_phases

    valid_phases = {"train", "eval", "summary"}
    invalid_phases = sorted(phases - valid_phases)
    if invalid_phases:
        raise ValueError(f"RUN_PHASE/--phase tidak dikenali: {invalid_phases}. Pilihan: all, train, eval, summary.")

    models = set(_parse_csv_arg(args.models, EXPECTED_MODEL_CONFIGS))
    invalid_models = sorted(models - set(EXPECTED_MODEL_CONFIGS))
    if invalid_models:
        raise ValueError(f"RUN_MODELS/--models tidak dikenali: {invalid_models}. Pilihan: {EXPECTED_MODEL_CONFIGS}")

    return phases, models, bool(args.auto_stop), str(args.auto_stop_command).strip()


RUN_PHASES, RUN_MODEL_SELECTION, AUTO_STOP_INSTANCE, AUTO_STOP_COMMAND = _parse_run_selection()


def should_run_train(model_name):
    return "train" in RUN_PHASES and model_name in RUN_MODEL_SELECTION


def should_run_eval(model_name):
    return "eval" in RUN_PHASES and model_name in RUN_MODEL_SELECTION


def should_run_summary():
    return "summary" in RUN_PHASES


print("\nRun selection:")
print(f"   phases: {sorted(RUN_PHASES)}")
print(f"   models: {sorted(RUN_MODEL_SELECTION)}")
print(f"   auto_stop: {AUTO_STOP_INSTANCE}")


def _summary_artifact_exists():
    expected = [
        os.path.join(PATHS["results"], "experiment_summary.json"),
        os.path.join(PATHS["results"], "experiment_summary.csv"),
    ]
    return any(os.path.exists(path) for path in expected)


def _missing_selected_run_targets():
    missing = []

    if "train" in RUN_PHASES:
        for model_name in sorted(RUN_MODEL_SELECTION):
            # Baseline pakai pretrained weights, gak perlu checkpoint training
            if model_name == "baseline_cqtdiff":
                continue
            ckpt_path = get_model_checkpoint_path(model_name)
            if not os.path.exists(ckpt_path):
                missing.append(f"train:{model_name} -> {ckpt_path}")

    if "eval" in RUN_PHASES:
        for model_name in sorted(RUN_MODEL_SELECTION):
            result_path = os.path.join(PATHS["results"], f"{evaluation_artifact_name(model_name)}_results.csv")
            if not os.path.exists(result_path):
                missing.append(f"eval:{model_name} -> {result_path}")

    if "summary" in RUN_PHASES and not _summary_artifact_exists():
        missing.append(f"summary -> {os.path.join(PATHS['results'], 'experiment_summary.json')}")

    return missing


def maybe_auto_stop_instance():
    if not AUTO_STOP_INSTANCE:
        return

    missing = _missing_selected_run_targets()
    if missing:
        print("\nAuto-stop diminta, tapi target run yang dipilih belum lengkap. Instance tidak dimatikan.")
        for item in missing:
            print(f"  - missing {item}")
        return

    if not AUTO_STOP_COMMAND:
        print("\nAuto-stop diminta, tapi AUTO_STOP_COMMAND kosong. Instance tidak dimatikan.")
        return

    print(f"\nAuto-stop: semua target phase/model terpilih selesai. Menjalankan: {AUTO_STOP_COMMAND}")
    try:
        import shlex
        import subprocess
        subprocess.Popen(shlex.split(AUTO_STOP_COMMAND))
    except Exception as exc:
        print(f"Auto-stop gagal dijalankan: {exc}")


# ---
# ## CELL 7 — BASELINE: CQT-Diff+ Standalone (Pretrained)
#
# Baseline pretrained mengukur kemampuan checkpoint CQT-Diff+ asli tanpa
# adaptation ke MusicNet dan tanpa SSL. Klaim kontribusi SSL tidak boleh
# hanya dibandingkan ke baseline ini; gunakan Cell 7B
# `baseline_cqtdiff_finetuned` sebagai pembanding fair no-SSL.


# ============================================================
# CELL 7: BASELINE — CQT-Diff+ STANDALONE (PRETRAINED DIFFUSION)
# ============================================================
# Baseline = CQT-Diff+ original pakai pretrained weights + proper
# multi-step reverse diffusion sampling buat inpainting.
#
# TIDAK PERLU TRAINING TAMBAHAN karena:
# - CQT-Diff+ sudah di-pretrain pada diffusion denoising objective
# - Inpainting dilakukan via multi-step reverse diffusion (T=35 steps)
#   dengan data consistency (replacement method) — sesuai paper asli
# - Baseline menunjukkan kemampuan murni CQT-Diff+ tanpa SSL conditioning
# - Hybrid models (Cell 8-11) menambahkan SSL encoder + FiLM di atas ini
#
# Untuk fairness thesis:
# - baseline_cqtdiff: pretrained, tanpa fine-tuning, tanpa SSL
# - baseline_cqtdiff_finetuned: fine-tuned adapter no-SSL
# - clap/audiomae_cqtdiff: fine-tuned adapter dengan SSL-guided residual conditioning
# ============================================================

import torch
import numpy as np
import os

MODEL_NAME = "baseline_cqtdiff"
FORCE_REEVAL = True

print(f"Config stage: {PIPELINE_STAGE_NAME} | model: {MODEL_NAME} | mode: pretrained diffusion (no training)")

# Hapus hasil lama kalau FORCE_REEVAL aktif
if should_run_eval(MODEL_NAME) and FORCE_REEVAL:
    old_result = os.path.join(PATHS["results"], f"{evaluation_artifact_name(MODEL_NAME)}_results.csv")
    if os.path.exists(old_result):
        os.remove(old_result)
        print(f"Hasil lama dihapus: {old_result}")

if not should_run_eval(MODEL_NAME):
    print(f"Skip {MODEL_NAME}: tidak dipilih oleh RUN_PHASE/RUN_MODELS.")
else:
    device = torch.device("cuda")

    if should_run_eval(MODEL_NAME):
        if check_if_done(MODEL_NAME):
            print(f"Baseline {MODEL_NAME} sudah selesai. Lewati evaluasi.")
        else:
            # ============================================================
            # EVALUASI BASELINE (pretrained CQT-Diff+ diffusion sampling)
            # ============================================================
            print("\nLoading pretrained CQT-Diff+ untuk baseline evaluasi...")
            print("   Inpainting via multi-step reverse diffusion (bukan single-pass).")
            print("   Tidak perlu training tambahan — pretrained weights sudah cukup.\n")

            baseline_model = build_hybrid_cqtdiff_decoder(device)
            baseline_model.eval()
            print_gpu_usage("Setelah load baseline")

            print("\nMengevaluasi baseline...")
            results_df = run_baseline_inpainting_evaluation(baseline_model, device, n_eval_samples=N_EVAL_SAMPLES)

            print(f"\nHasil evaluasi BASELINE (CQT-Diff+ pretrained diffusion sampling):")
            print(results_df.to_string(index=False))

            save_results(results_df, MODEL_NAME)

            print("\nMembersihkan memori GPU...")
            clear_gpu_memory(baseline_model)

            print(f"\nBASELINE {MODEL_NAME} selesai!")
    else:
        print(f"Skip evaluasi {MODEL_NAME}: RUN_PHASE tidak memuat eval atau model tidak dipilih.")


# ---
# ## CELL 7B — BASELINE FINE-TUNED: Conditioning Head tanpa SSL
#
# Ablation baseline: conditioning head yang sama (FiLM + spec_decoder +
# condition_gate + sigma_scale_net) di-train pada MusicNet, tapi
# TANPA SSL encoder (input encoder selalu zeros).
#
# Tujuan: memisahkan efek domain adaptation (fine-tuning pada MusicNet)
# dari kontribusi spesifik SSL encoder (CLAP/AudioMAE).
#
# Perbandingan yang fair:
# - baseline_cqtdiff:           pretrained, tanpa fine-tuning, tanpa SSL
# - baseline_cqtdiff_finetuned: fine-tuned conditioning head, tanpa SSL
# - clap_cqtdiff:               fine-tuned conditioning head, DENGAN SSL
#
# Jika clap_cqtdiff > baseline_cqtdiff_finetuned → SSL memberikan
# kontribusi nyata di atas fine-tuning.


# ============================================================
# CELL 7B-A: TRAINING — BASELINE FINE-TUNED (tanpa SSL encoder)
# ============================================================

MODEL_NAME = "baseline_cqtdiff_finetuned"
FORCE_RETRAIN = True
BATCH_SIZE = 16
NUM_WORKERS = AUTO_NUM_WORKERS
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

print(f"Config stage: {PIPELINE_STAGE_NAME} | model: {MODEL_NAME} | "
      f"dataset_fraction={DATASET_FRACTION:.0%} | batch_size={BATCH_SIZE} | epochs={NUM_EPOCHS}")

assert NUM_EPOCHS >= 5, "NUM_EPOCHS minimal 5 agar checkpoint best tervalidasi bisa tersimpan."

if should_run_train(MODEL_NAME) and FORCE_RETRAIN:
    reset_training_checkpoints_if_requested(MODEL_NAME, FORCE_RETRAIN)

ckpt_path = get_model_checkpoint_path(MODEL_NAME)
if not should_run_train(MODEL_NAME):
    print(f"Skip training {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif hybrid_checkpoint_exists(MODEL_NAME) and not FORCE_RETRAIN:
    print(f"Checkpoint sudah ada. Skip training: {ckpt_path}")
else:
    device = torch.device("cuda")
    cqtdiff_model = None
    film_layer = None

    try:
        print(f"Device: {device}")
        check_batch_size_memory(BATCH_SIZE, min_expected_vram_gb=8.0)

        loaders = make_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        _ft_encoder_dim = FILM_CONFIGS[MODEL_NAME]["encoder_dim"]

        def _zero_encoder_fn(audio_input):
            """Dummy encoder: selalu return zeros (tanpa SSL)."""
            if isinstance(audio_input, np.ndarray):
                return torch.zeros(1, _ft_encoder_dim, device=device)
            if isinstance(audio_input, torch.Tensor):
                return torch.zeros(audio_input.shape[0], _ft_encoder_dim, device=device)
            return torch.zeros(1, _ft_encoder_dim, device=device)

        loaders = add_encoder_cache_to_loaders(
            loaders, _zero_encoder_fn, device, MODEL_NAME, num_workers=NUM_WORKERS
        )
        cqtdiff_model = build_hybrid_cqtdiff_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)

        print_gpu_usage("Sebelum training")
        train_model(
            cqtdiff_model,
            _zero_encoder_fn,
            film_layer,
            loaders["train"],
            val_loader=loaders["val"],
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            checkpoint_dir=get_model_checkpoint_dir(MODEL_NAME),
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            dataset_fraction=DATASET_FRACTION,
        )

        if not hybrid_checkpoint_exists(MODEL_NAME):
            raise RuntimeError(f"Checkpoint {MODEL_NAME} tidak tersimpan.")

        print(f"Training selesai. Checkpoint siap dipakai: {ckpt_path}")
    finally:
        clear_gpu_memory(cqtdiff_model, film_layer)


# ============================================================
# CELL 7B: EVALUASI — BASELINE FINE-TUNED (tanpa SSL encoder)
# ============================================================

MODEL_NAME = "baseline_cqtdiff_finetuned"
FORCE_REEVAL = True

if should_run_eval(MODEL_NAME) and FORCE_REEVAL:
    old_result = os.path.join(PATHS["results"], f"{evaluation_artifact_name(MODEL_NAME)}_results.csv")
    if os.path.exists(old_result):
        os.remove(old_result)
        print(f"Hasil lama dihapus: {old_result}")

if not should_run_eval(MODEL_NAME):
    print(f"Skip evaluasi {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif check_if_done(MODEL_NAME):
    print(f"Model {MODEL_NAME} sudah selesai. Lewati cell ini.")
else:
    ckpt_path = get_model_checkpoint_path(MODEL_NAME)
    if not hybrid_checkpoint_exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Checkpoint {MODEL_NAME} belum ditemukan di {ckpt_path}. "
            "Jalankan CELL 7B-A terlebih dahulu."
        )

    device = torch.device("cuda")
    cqtdiff_model = None
    film_layer = None

    try:
        print_gpu_usage("Awal")
        print(f"Menggunakan checkpoint: {ckpt_path}")

        _ft_encoder_dim = FILM_CONFIGS[MODEL_NAME]["encoder_dim"]

        def _zero_encoder_fn(audio_input):
            """Dummy encoder: selalu return zeros (tanpa SSL)."""
            if isinstance(audio_input, np.ndarray):
                return torch.zeros(1, _ft_encoder_dim, device=device)
            if isinstance(audio_input, torch.Tensor):
                return torch.zeros(audio_input.shape[0], _ft_encoder_dim, device=device)
            return torch.zeros(1, _ft_encoder_dim, device=device)

        cqtdiff_model = build_hybrid_cqtdiff_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)
        load_hybrid_checkpoint(MODEL_NAME, cqtdiff_model, film_layer, device)
        print_gpu_usage("Setelah load model terlatih")

        results_df = run_hybrid_inpainting_evaluation(
            "Baseline Fine-tuned (no SSL)",
            _zero_encoder_fn,
            cqtdiff_model,
            film_layer,
            device,
            n_eval_samples=N_EVAL_SAMPLES,
            model_name=MODEL_NAME,
        )

        print(f"\nHasil {MODEL_NAME}:")
        print(results_df.to_string(index=False))
        save_results(results_df, MODEL_NAME)
    finally:
        clear_gpu_memory(cqtdiff_model, film_layer)

    print(f"\nBASELINE FINE-TUNED {MODEL_NAME} selesai!")


# ---
# ## CELL 8 — Kombinasi 1: CLAP + CQT-Diff+
#
# Model hybrid pertama. Dibandingkan dengan baseline di Cell 7,
# perbedaannya hanya penambahan **CLAP encoder + FiLM conditioning**.


# ============================================================
# CELL 8A: TRAINING — CLAP + CQT-Diff+
# ============================================================
# Default: skip training jika checkpoint sudah ada.
# Set FORCE_RETRAIN = True untuk melatih ulang.
# ============================================================

# ============================================================
# CELL 8A: TRAINING — CLAP + CQT-Diff+
# ============================================================
# Set FORCE_RETRAIN = True untuk melatih ulang.
# PENTING: Set True setelah update arsitektur model!
# ============================================================

MODEL_NAME = "clap_cqtdiff"
FORCE_RETRAIN = True   # <-- True karena arsitektur model berubah!
# Stage override: instance memory/throughput test uses batch size 8.
BATCH_SIZE = 16
NUM_WORKERS = AUTO_NUM_WORKERS
# Stage override: instance test keeps the current 10 training epochs.
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

print(f"Config stage: {PIPELINE_STAGE_NAME} | dataset_fraction={DATASET_FRACTION:.0%} | batch_size={BATCH_SIZE} | epochs={NUM_EPOCHS}")

assert NUM_EPOCHS >= 5, "NUM_EPOCHS minimal 5 agar checkpoint best tervalidasi bisa tersimpan."

if should_run_train(MODEL_NAME) and FORCE_RETRAIN:
    reset_training_checkpoints_if_requested(MODEL_NAME, FORCE_RETRAIN)

ckpt_path = get_model_checkpoint_path(MODEL_NAME)
if not should_run_train(MODEL_NAME):
    print(f"⏭️ Skip training {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif hybrid_checkpoint_exists(MODEL_NAME) and not FORCE_RETRAIN:
    print(f"✅ Checkpoint sudah ada. Skip training: {ckpt_path}")
else:
    device = torch.device("cuda")
    clap_model = None
    cqtdiff_model = None
    film_layer = None

    try:
        print(f"🔧 Device: {device}")
        check_batch_size_memory(BATCH_SIZE, min_expected_vram_gb=8.0)

        loaders = make_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        clap_model, encoder_fn = build_clap_encoder(device)
        loaders = add_encoder_cache_to_loaders(loaders, encoder_fn, device, MODEL_NAME, num_workers=NUM_WORKERS)
        cqtdiff_model = build_hybrid_cqtdiff_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)

        print_gpu_usage("Sebelum training")
        train_model(
            cqtdiff_model,
            encoder_fn,
            film_layer,
            loaders["train"],
            val_loader=loaders["val"],
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            checkpoint_dir=get_model_checkpoint_dir(MODEL_NAME),
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            dataset_fraction=DATASET_FRACTION,
        )

        if not hybrid_checkpoint_exists(MODEL_NAME):
            raise RuntimeError(f"Checkpoint {MODEL_NAME} tidak tersimpan.")

        print(f"✅ Training selesai. Checkpoint siap dipakai: {ckpt_path}")
    finally:
        clear_gpu_memory(clap_model, cqtdiff_model, film_layer)


# ============================================================
# CELL 8: KOMBINASI 1 — CLAP + CQT-Diff+
# ============================================================
# Evaluasi hybrid selalu memakai checkpoint terlatih.
# Jika checkpoint belum ada, jalankan CELL 8A terlebih dahulu.
# ============================================================

MODEL_NAME = "clap_cqtdiff"
FORCE_REEVAL = True  # <-- True buat re-evaluasi setelah update arsitektur

if should_run_eval(MODEL_NAME) and FORCE_REEVAL:
    old_result = os.path.join(PATHS["results"], f"{evaluation_artifact_name(MODEL_NAME)}_results.csv")
    if os.path.exists(old_result):
        os.remove(old_result)

if not should_run_eval(MODEL_NAME):
    print(f"⏭️ Skip evaluasi {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif check_if_done(MODEL_NAME):
    print(f"Model {MODEL_NAME} sudah selesai. Lewati cell ini.")
else:
    ckpt_path = get_model_checkpoint_path(MODEL_NAME)
    if not hybrid_checkpoint_exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Checkpoint {MODEL_NAME} belum ditemukan di {ckpt_path}. Jalankan CELL 8A terlebih dahulu."
        )

    device = torch.device("cuda")
    clap_model = None
    cqtdiff_model = None
    film_layer = None

    try:
        print_gpu_usage("Awal")
        print(f"📦 Menggunakan checkpoint: {ckpt_path}")

        clap_model, encoder_fn = build_clap_encoder(device)
        cqtdiff_model = build_hybrid_cqtdiff_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)
        load_hybrid_checkpoint(MODEL_NAME, cqtdiff_model, film_layer, device)
        print_gpu_usage("Setelah load model terlatih")

        results_df = run_hybrid_inpainting_evaluation(
            "CLAP + CQT-Diff+",
            encoder_fn,
            cqtdiff_model,
            film_layer,
            device,
            n_eval_samples=N_EVAL_SAMPLES,
            model_name=MODEL_NAME,
        )

        print(f"\n📋 Hasil {MODEL_NAME}:")
        print(results_df.to_string(index=False))
        save_results(results_df, MODEL_NAME)
    finally:
        clear_gpu_memory(clap_model, cqtdiff_model, film_layer)

    print(f"\n✅ {MODEL_NAME} selesai!")

# ---
# ## CELL 9 — Kombinasi 2: CLAP + MAID


# ============================================================
# CELL 9A: TRAINING — CLAP + MAID
# ============================================================
# Default: skip training jika checkpoint sudah ada.
# Set FORCE_RETRAIN = True untuk melatih ulang.
# ============================================================

MODEL_NAME = "clap_maid"
FORCE_RETRAIN = True   # <-- True karena arsitektur/training berubah!
# Stage override: instance memory/throughput test uses batch size 8.
BATCH_SIZE = 16
NUM_WORKERS = AUTO_NUM_WORKERS
# Stage override: instance test keeps the current 10 training epochs.
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

print(f"Config stage: {PIPELINE_STAGE_NAME} | dataset_fraction={DATASET_FRACTION:.0%} | batch_size={BATCH_SIZE} | epochs={NUM_EPOCHS}")

assert NUM_EPOCHS >= 5, "NUM_EPOCHS minimal 5 agar checkpoint best tervalidasi bisa tersimpan."

if should_run_train(MODEL_NAME) and FORCE_RETRAIN:
    reset_training_checkpoints_if_requested(MODEL_NAME, FORCE_RETRAIN)

ckpt_path = get_model_checkpoint_path(MODEL_NAME)
if not should_run_train(MODEL_NAME):
    print(f"⏭️ Skip training {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif hybrid_checkpoint_exists(MODEL_NAME) and not FORCE_RETRAIN:
    print(f"✅ Checkpoint sudah ada. Skip training: {ckpt_path}")
else:
    device = torch.device("cuda")
    clap_model = None
    maid_model = None
    film_layer = None

    try:
        print(f"🔧 Device: {device}")
        check_batch_size_memory(BATCH_SIZE, min_expected_vram_gb=8.0)

        loaders = make_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        clap_model, encoder_fn = build_clap_encoder(device)
        loaders = add_encoder_cache_to_loaders(loaders, encoder_fn, device, MODEL_NAME, num_workers=NUM_WORKERS)
        maid_model = build_maid_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)

        print_gpu_usage("Sebelum training")
        train_maid_model(
            maid_model,
            encoder_fn,
            film_layer,
            loaders["train"],
            val_loader=loaders["val"],
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            checkpoint_dir=get_model_checkpoint_dir(MODEL_NAME),
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            dataset_fraction=DATASET_FRACTION,
        )

        if not hybrid_checkpoint_exists(MODEL_NAME):
            raise RuntimeError(f"Checkpoint {MODEL_NAME} tidak tersimpan.")

        print(f"✅ Training selesai. Checkpoint siap dipakai: {ckpt_path}")
    finally:
        clear_gpu_memory(clap_model, maid_model, film_layer)


# ============================================================
# CELL 9: KOMBINASI 2 — CLAP + MAID
# ============================================================
# Evaluasi hybrid selalu memakai checkpoint terlatih.
# Jika checkpoint belum ada, jalankan CELL 9A terlebih dahulu.
# ============================================================

MODEL_NAME = "clap_maid"
FORCE_REEVAL = True  # <-- True buat re-evaluasi setelah update

if should_run_eval(MODEL_NAME) and FORCE_REEVAL:
    old_result = os.path.join(PATHS["results"], f"{evaluation_artifact_name(MODEL_NAME)}_results.csv")
    if os.path.exists(old_result):
        os.remove(old_result)

if not should_run_eval(MODEL_NAME):
    print(f"⏭️ Skip evaluasi {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif check_if_done(MODEL_NAME):
    print(f"Model {MODEL_NAME} sudah selesai. Lewati cell ini.")
else:
    ckpt_path = get_model_checkpoint_path(MODEL_NAME)
    if not hybrid_checkpoint_exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Checkpoint {MODEL_NAME} belum ditemukan di {ckpt_path}. Jalankan CELL 9A terlebih dahulu."
        )

    device = torch.device("cuda")
    clap_model = None
    maid_model = None
    film_layer = None

    try:
        print_gpu_usage("Awal")
        print(f"📦 Menggunakan checkpoint: {ckpt_path}")

        clap_model, encoder_fn = build_clap_encoder(device)
        maid_model = build_maid_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)
        load_hybrid_checkpoint(MODEL_NAME, maid_model, film_layer, device)
        print_gpu_usage("Setelah load model terlatih")

        results_df = run_hybrid_inpainting_evaluation(
            "CLAP + MAID",
            encoder_fn,
            maid_model,
            film_layer,
            device,
            n_eval_samples=N_EVAL_SAMPLES,
            model_name=MODEL_NAME,
        )

        print(f"\n📋 Hasil {MODEL_NAME}:")
        print(results_df.to_string(index=False))
        save_results(results_df, MODEL_NAME)
    finally:
        clear_gpu_memory(clap_model, maid_model, film_layer)

    print(f"\n✅ {MODEL_NAME} selesai!")

# ---
# ## CELL 10 — Kombinasi 3: AudioMAE + CQT-Diff+


# ============================================================
# CELL 10A: TRAINING — AudioMAE + CQT-Diff+
# ============================================================
# Default: skip training jika checkpoint sudah ada.
# Set FORCE_RETRAIN = True untuk melatih ulang.
# ============================================================

MODEL_NAME = "audiomae_cqtdiff"
FORCE_RETRAIN = True   # <-- True karena arsitektur model berubah!
# Stage override: instance memory/throughput test uses batch size 8.
BATCH_SIZE = 16
NUM_WORKERS = AUTO_NUM_WORKERS
# Stage override: instance test keeps the current 10 training epochs.
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

print(f"Config stage: {PIPELINE_STAGE_NAME} | dataset_fraction={DATASET_FRACTION:.0%} | batch_size={BATCH_SIZE} | epochs={NUM_EPOCHS}")

assert NUM_EPOCHS >= 5, "NUM_EPOCHS minimal 5 agar checkpoint best tervalidasi bisa tersimpan."

if should_run_train(MODEL_NAME) and FORCE_RETRAIN:
    reset_training_checkpoints_if_requested(MODEL_NAME, FORCE_RETRAIN)

ckpt_path = get_model_checkpoint_path(MODEL_NAME)
if not should_run_train(MODEL_NAME):
    print(f"⏭️ Skip training {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif hybrid_checkpoint_exists(MODEL_NAME) and not FORCE_RETRAIN:
    print(f"✅ Checkpoint sudah ada. Skip training: {ckpt_path}")
else:
    device = torch.device("cuda")
    audiomae_model = None
    cqtdiff_model = None
    film_layer = None

    try:
        print(f"🔧 Device: {device}")
        check_batch_size_memory(BATCH_SIZE, min_expected_vram_gb=8.0)

        loaders = make_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        audiomae_model, encoder_fn = build_audiomae_encoder(device)
        loaders = add_encoder_cache_to_loaders(loaders, encoder_fn, device, MODEL_NAME, num_workers=NUM_WORKERS)
        cqtdiff_model = build_hybrid_cqtdiff_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)

        print_gpu_usage("Sebelum training")
        train_model(
            cqtdiff_model,
            encoder_fn,
            film_layer,
            loaders["train"],
            val_loader=loaders["val"],
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            checkpoint_dir=get_model_checkpoint_dir(MODEL_NAME),
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            dataset_fraction=DATASET_FRACTION,
        )

        if not hybrid_checkpoint_exists(MODEL_NAME):
            raise RuntimeError(f"Checkpoint {MODEL_NAME} tidak tersimpan.")

        print(f"✅ Training selesai. Checkpoint siap dipakai: {ckpt_path}")
    finally:
        clear_gpu_memory(audiomae_model, cqtdiff_model, film_layer)


# ============================================================
# CELL 10: KOMBINASI 3 — AudioMAE + CQT-Diff+
# ============================================================
# Evaluasi hybrid selalu memakai checkpoint terlatih.
# Jika checkpoint belum ada, jalankan CELL 10A terlebih dahulu.
# ============================================================

MODEL_NAME = "audiomae_cqtdiff"
FORCE_REEVAL = True  # <-- True buat re-evaluasi setelah update arsitektur

if should_run_eval(MODEL_NAME) and FORCE_REEVAL:
    old_result = os.path.join(PATHS["results"], f"{evaluation_artifact_name(MODEL_NAME)}_results.csv")
    if os.path.exists(old_result):
        os.remove(old_result)

if not should_run_eval(MODEL_NAME):
    print(f"⏭️ Skip evaluasi {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif check_if_done(MODEL_NAME):
    print(f"Model {MODEL_NAME} sudah selesai. Lewati cell ini.")
else:
    ckpt_path = get_model_checkpoint_path(MODEL_NAME)
    if not hybrid_checkpoint_exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Checkpoint {MODEL_NAME} belum ditemukan di {ckpt_path}. Jalankan CELL 10A terlebih dahulu."
        )

    device = torch.device("cuda")
    audiomae_model = None
    cqtdiff_model = None
    film_layer = None

    try:
        print_gpu_usage("Awal")
        print(f"📦 Menggunakan checkpoint: {ckpt_path}")

        audiomae_model, encoder_fn = build_audiomae_encoder(device)
        cqtdiff_model = build_hybrid_cqtdiff_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)
        load_hybrid_checkpoint(MODEL_NAME, cqtdiff_model, film_layer, device)
        print_gpu_usage("Setelah load model terlatih")

        results_df = run_hybrid_inpainting_evaluation(
            "AudioMAE + CQT-Diff+",
            encoder_fn,
            cqtdiff_model,
            film_layer,
            device,
            n_eval_samples=N_EVAL_SAMPLES,
            model_name=MODEL_NAME,
        )

        print(f"\n📋 Hasil {MODEL_NAME}:")
        print(results_df.to_string(index=False))
        save_results(results_df, MODEL_NAME)
    finally:
        clear_gpu_memory(audiomae_model, cqtdiff_model, film_layer)

    print(f"\n✅ {MODEL_NAME} selesai!")

# ---
# ## CELL 11 — Kombinasi 4: AudioMAE + MAID


# ============================================================
# CELL 11A: TRAINING — AudioMAE + MAID
# ============================================================
# Default: skip training jika checkpoint sudah ada.
# Set FORCE_RETRAIN = True untuk melatih ulang.
# ============================================================

MODEL_NAME = "audiomae_maid"
FORCE_RETRAIN = True   # <-- True karena arsitektur/training berubah!
# Stage override: instance memory/throughput test uses batch size 8.
BATCH_SIZE = 16
NUM_WORKERS = AUTO_NUM_WORKERS
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

print(f"Config stage: {PIPELINE_STAGE_NAME} | dataset_fraction={DATASET_FRACTION:.0%} | batch_size={BATCH_SIZE} | epochs={NUM_EPOCHS}")

assert NUM_EPOCHS >= 5, "NUM_EPOCHS minimal 5 agar checkpoint best tervalidasi bisa tersimpan."

if should_run_train(MODEL_NAME) and FORCE_RETRAIN:
    reset_training_checkpoints_if_requested(MODEL_NAME, FORCE_RETRAIN)

ckpt_path = get_model_checkpoint_path(MODEL_NAME)
if not should_run_train(MODEL_NAME):
    print(f"⏭️ Skip training {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif hybrid_checkpoint_exists(MODEL_NAME) and not FORCE_RETRAIN:
    print(f"✅ Checkpoint sudah ada. Skip training: {ckpt_path}")
else:
    device = torch.device("cuda")
    audiomae_model = None
    maid_model = None
    film_layer = None

    try:
        print(f"🔧 Device: {device}")
        check_batch_size_memory(BATCH_SIZE, min_expected_vram_gb=8.0)

        loaders = make_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        audiomae_model, encoder_fn = build_audiomae_encoder(device)
        loaders = add_encoder_cache_to_loaders(loaders, encoder_fn, device, MODEL_NAME, num_workers=NUM_WORKERS)
        maid_model = build_maid_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)

        print_gpu_usage("Sebelum training")
        train_maid_model(
            maid_model,
            encoder_fn,
            film_layer,
            loaders["train"],
            val_loader=loaders["val"],
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            checkpoint_dir=get_model_checkpoint_dir(MODEL_NAME),
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            dataset_fraction=DATASET_FRACTION,
        )

        if not hybrid_checkpoint_exists(MODEL_NAME):
            raise RuntimeError(f"Checkpoint {MODEL_NAME} tidak tersimpan.")

        print(f"✅ Training selesai. Checkpoint siap dipakai: {ckpt_path}")
    finally:
        clear_gpu_memory(audiomae_model, maid_model, film_layer)


# ============================================================
# CELL 11: KOMBINASI 4 — AudioMAE + MAID
# ============================================================
# Evaluasi hybrid selalu memakai checkpoint terlatih.
# Jika checkpoint belum ada, jalankan CELL 11A terlebih dahulu.
# ============================================================

MODEL_NAME = "audiomae_maid"
FORCE_REEVAL = True  # <-- True buat re-evaluasi setelah update

if should_run_eval(MODEL_NAME) and FORCE_REEVAL:
    old_result = os.path.join(PATHS["results"], f"{evaluation_artifact_name(MODEL_NAME)}_results.csv")
    if os.path.exists(old_result):
        os.remove(old_result)

if not should_run_eval(MODEL_NAME):
    print(f"⏭️ Skip evaluasi {MODEL_NAME}: RUN_PHASE/RUN_MODELS tidak memilih blok ini.")
elif check_if_done(MODEL_NAME):
    print(f"Model {MODEL_NAME} sudah selesai. Lewati cell ini.")
else:
    ckpt_path = get_model_checkpoint_path(MODEL_NAME)
    if not hybrid_checkpoint_exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Checkpoint {MODEL_NAME} belum ditemukan di {ckpt_path}. Jalankan CELL 11A terlebih dahulu."
        )

    device = torch.device("cuda")
    audiomae_model = None
    maid_model = None
    film_layer = None

    try:
        print_gpu_usage("Awal")
        print(f"📦 Menggunakan checkpoint: {ckpt_path}")

        audiomae_model, encoder_fn = build_audiomae_encoder(device)
        maid_model = build_maid_decoder(device)
        film_layer = build_film_layer(MODEL_NAME, device)
        load_hybrid_checkpoint(MODEL_NAME, maid_model, film_layer, device)
        print_gpu_usage("Setelah load model terlatih")

        results_df = run_hybrid_inpainting_evaluation(
            "AudioMAE + MAID",
            encoder_fn,
            maid_model,
            film_layer,
            device,
            n_eval_samples=N_EVAL_SAMPLES,
            model_name=MODEL_NAME,
        )

        print(f"\n📋 Hasil {MODEL_NAME}:")
        print(results_df.to_string(index=False))
        save_results(results_df, MODEL_NAME)
    finally:
        clear_gpu_memory(audiomae_model, maid_model, film_layer)

    print(f"\n✅ {MODEL_NAME} selesai!")

# ---
# ## CELL 12 — Gabungkan & Visualisasikan Semua Hasil
# 
# Jalankan setelah baseline dan seluruh kombinasi hybrid selesai dievaluasi.
# Untuk hybrid, pastikan cell training dan cell evaluasinya sudah dijalankan sehingga checkpoint dan CSV hasil tersedia.
# Grafik akan menampilkan **baseline vs 4 model hybrid** untuk perbandingan langsung.


# ============================================================
# CELL 12: VISUALISASI HASIL LENGKAP
# ============================================================
# Menampilkan perbandingan semua model:
# - Baseline: CQT-Diff+ standalone (garis putus-putus)
# - 4 kombinasi hybrid (garis solid)
#
# Gap durations: 100, 300, 500, 750, 1200, 1700 ms
# ============================================================

if not should_run_summary():
    print("⏭️ Skip summary/visualisasi: RUN_PHASE tidak memuat summary.")
    sys.exit(0)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 10

master_path = os.path.join(PATHS["results"], "all_results.csv")

if not os.path.exists(master_path):
    print("❌ File hasil belum ada. Pastikan Cell 7-11 sudah dijalankan.")
else:
    all_results = pd.read_csv(master_path)

    # Backward compatibility untuk file hasil lama.
    if "VISQOL_ODG" not in all_results.columns:
        if "VISQOL" in all_results.columns:
            all_results["VISQOL_ODG"] = all_results["VISQOL"]
    if "gap_position" not in all_results.columns:
        all_results["gap_position"] = "center"
    all_results = all_results[all_results["gap_position"].fillna("center") == EVAL_GAP_POSITION].copy()
    if all_results.empty:
        print(f"❌ Tidak ada hasil untuk gap_position={EVAL_GAP_POSITION}.")
        sys.exit(0)

    # Cek model mana yang sudah selesai
    available_models = all_results["model"].unique()
    print(f"📊 Model yang tersedia: {list(available_models)}")

    gap_durations = sorted(all_results["gap_ms"].unique())

    # ============================================================
    # TABEL PERBANDINGAN
    # ============================================================
    print("\n" + "="*70)
    print("TABEL PERBANDINGAN LENGKAP")
    print("="*70)

    metric_order = [
        m for m in [
            "LSD", "LSD_GAP_ONLY", "GAP_SI_SDR", "GAP_SNR", "GAP_MEL_DISTANCE",
            "FAD", "VISQOL_ODG", "PEAQ_ODG", "GAP_WINDOW_VISQOL_ODG",
        ]
        if m in all_results.columns
    ]
    for metric in metric_order:
        if metric in ["LSD", "LSD_GAP_ONLY", "GAP_LSD", "GAP_MEL_DISTANCE", "FAD"]:
            direction = "↓ lebih rendah = lebih baik"
        elif metric in ["GAP_SI_SDR", "GAP_SNR"]:
            direction = "↑ lebih tinggi = lebih baik"
        else:
            direction = "↑ mendekati 0 = lebih baik"
        print(f"\n{metric} ({direction}):")
        pivot = all_results.pivot(index="gap_ms", columns="model", values=metric)
        # Urutkan kolom: baseline dulu, lalu hybrid
        ordered_cols = [c for c in [
            "baseline_cqtdiff", "baseline_cqtdiff_finetuned",
            "clap_cqtdiff", "audiomae_cqtdiff", "clap_maid", "audiomae_maid",
        ] if c in pivot.columns]
        print(pivot[ordered_cols].to_string())


    # ============================================================
    # VISUALISASI
    # ============================================================
    # Style per model
    # Baseline: garis putus-putus hitam untuk mudah dibedakan
    # Hybrid: garis solid berwarna
    styles = {
        "baseline_cqtdiff":  {"color": "#000000", "marker": "x", "linestyle": "--",
                               "label": "Baseline: CQT-Diff+ pretrained", "linewidth": 2.5, "zorder": 10},
        "baseline_cqtdiff_finetuned": {"color": "#666666", "marker": "P", "linestyle": "-.",
                               "label": "Baseline: CQT-Diff+ fine-tuned no SSL", "linewidth": 2.2, "zorder": 9},
        "clap_cqtdiff":      {"color": "#2196F3", "marker": "o", "linestyle": "-",
                               "label": "CLAP + CQT-Diff+", "linewidth": 1.5, "zorder": 5},
        "clap_maid":         {"color": "#4CAF50", "marker": "s", "linestyle": "-",
                               "label": "CLAP + MAID", "linewidth": 1.5, "zorder": 5},
        "audiomae_cqtdiff":  {"color": "#FF9800", "marker": "^", "linestyle": "-",
                               "label": "AudioMAE + CQT-Diff+", "linewidth": 1.5, "zorder": 5},
        "audiomae_maid":     {"color": "#F44336", "marker": "D", "linestyle": "-",
                               "label": "AudioMAE + MAID", "linewidth": 1.5, "zorder": 5},
    }

    metrics_info = {
        "LSD": {"title": "Log Spectral Distance (LSD)",
                "ylabel": "LSD (dB)",
                "note": "↓ lebih rendah = lebih baik"},
        "LSD_GAP_ONLY": {"title": "Gap-only Log Spectral Distance",
                "ylabel": "Gap LSD (dB)",
                "note": "↓ lebih rendah = lebih baik"},
        "GAP_SI_SDR": {"title": "Gap SI-SDR",
                "ylabel": "SI-SDR (dB)",
                "note": "↑ lebih tinggi = lebih baik"},
        "GAP_SNR": {"title": "Gap SNR",
                "ylabel": "SNR (dB)",
                "note": "↑ lebih tinggi = lebih baik"},
        "GAP_MEL_DISTANCE": {"title": "Gap Mel Spectral Distance",
                "ylabel": "Mean |Mel dB diff|",
                "note": "↓ lebih rendah = lebih baik"},
        "FAD": {"title": "Frechet Audio Distance (FAD)",
                "ylabel": "FAD Score",
                "note": "↓ lebih rendah = lebih baik"},
        "VISQOL_ODG": {"title": "ViSQOL Objective Difference Grade",
                       "ylabel": "VISQOL_ODG Score",
                       "note": "↑ mendekati 0 = lebih baik"},
        "PEAQ_ODG": {"title": "GstPEAQ Objective Difference Grade",
                     "ylabel": "PEAQ_ODG Score",
                     "note": "↑ mendekati 0 = lebih baik"},
        "GAP_WINDOW_VISQOL_ODG": {"title": "Gap-window ViSQOL ODG",
                     "ylabel": "Gap-window VISQOL_ODG",
                     "note": "↑ mendekati 0 = lebih baik"},
    }
    metrics_info = {k: v for k, v in metrics_info.items() if k in metric_order}

    fig, axes = plt.subplots(1, len(metrics_info), figsize=(6 * len(metrics_info), 6))
    if len(metrics_info) == 1:
        axes = [axes]
    fig.suptitle(
        "Music Audio Inpainting — Baseline vs Hybrid SSL+Diffusion Models\n"
        f"Native MusicNet CQTdiff+: {TARGET_SR} Hz, {SEGMENT_SAMPLES} samples ({SEGMENT_DURATION:.2f}s), "
        f"gap={EVAL_GAP_POSITION}",
        fontsize=13, fontweight='bold'
    )

    for ax, (metric, info) in zip(axes, metrics_info.items()):
        # Plot baseline dan hybrid
        # Urutan plot: hybrid dulu, baseline paling atas (zorder lebih tinggi)
        plot_order = [m for m in [
            "clap_cqtdiff", "audiomae_cqtdiff", "clap_maid", "audiomae_maid",
            "baseline_cqtdiff_finetuned", "baseline_cqtdiff",
        ] if m in available_models]

        for model_name in plot_order:
            model_data = all_results[all_results["model"] == model_name].sort_values("gap_ms")
            s = styles.get(model_name, {"color": "gray", "marker": "x",
                                         "linestyle": "-", "label": model_name,
                                         "linewidth": 1.5, "zorder": 1})
            ax.plot(
                model_data["gap_ms"],
                model_data[metric],
                color=s["color"],
                marker=s["marker"],
                linestyle=s["linestyle"],
                label=s["label"],
                linewidth=s["linewidth"],
                markersize=7,
                zorder=s["zorder"]
            )

        ax.set_title(info["title"], fontsize=11, fontweight='bold')
        ax.set_xlabel("Gap Duration (ms)", fontsize=10)
        ax.set_ylabel(info["ylabel"], fontsize=10)
        ax.set_xticks(gap_durations)
        ax.set_xticklabels([str(g) for g in gap_durations], rotation=45)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, info["note"],
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', style='italic', color='gray')

    plt.tight_layout()

    # Simpan grafik
    plot_path = os.path.join(PATHS["plots"], "comparison_plot.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"\n💾 Grafik disimpan: {plot_path}")


    # ============================================================
    # RANGKUMAN: IMPROVEMENT HYBRID vs BASELINE
    # ============================================================
    if "baseline_cqtdiff" in available_models:
        print("\n" + "="*72)
        print("📈 IMPROVEMENT HYBRID vs PRETRAINED BASELINE (per gap duration)")
        print("   Positif = lebih baik dari baseline")
        print("="*72)

        baseline_data = all_results[all_results["model"] == "baseline_cqtdiff"]

        for gap_ms in gap_durations:
            bl = baseline_data[baseline_data["gap_ms"] == gap_ms].iloc[0]
            print(f"\n  Gap {gap_ms}ms:")

            header_cols = ["Model", "ΔLSD"]
            if "FAD" in all_results.columns:
                header_cols.append("ΔFAD")
            if "VISQOL_ODG" in all_results.columns:
                header_cols.append("ΔVISQOL_ODG")
            if "PEAQ_ODG" in all_results.columns:
                header_cols.append("ΔPEAQ_ODG")
            print(f"  {header_cols[0]:<25} " + " ".join(f"{h:>10}" for h in header_cols[1:]))
            print(f"  {'-'*65}")

            for model_name in ["clap_cqtdiff", "clap_maid", "audiomae_cqtdiff", "audiomae_maid"]:
                if model_name not in available_models:
                    continue
                hybrid = all_results[
                    (all_results["model"] == model_name) &
                    (all_results["gap_ms"] == gap_ms)
                ].iloc[0]

                # Positif selalu berarti hybrid lebih baik.
                deltas = [bl["LSD"] - hybrid["LSD"]]
                if "FAD" in all_results.columns:
                    deltas.append(bl["FAD"] - hybrid["FAD"])
                if "VISQOL_ODG" in all_results.columns:
                    deltas.append(hybrid["VISQOL_ODG"] - bl["VISQOL_ODG"])
                if "PEAQ_ODG" in all_results.columns:
                    deltas.append(hybrid["PEAQ_ODG"] - bl["PEAQ_ODG"])

                print(f"  {model_name:<25} " + " ".join(f"{d:>+10.4f}" for d in deltas))

    if "baseline_cqtdiff_finetuned" in available_models:
        print("\n" + "="*78)
        print("📈 CQT HYBRID vs FINE-TUNED NO-SSL BASELINE (fair SSL contribution check)")
        print("   Positif = hybrid lebih baik dari baseline fine-tuned tanpa SSL")
        print("="*78)

        ft_data = all_results[all_results["model"] == "baseline_cqtdiff_finetuned"]
        fair_metrics = [
            m for m in ["LSD_GAP_ONLY", "GAP_SI_SDR", "GAP_SNR", "GAP_MEL_DISTANCE", "FAD", "PEAQ_ODG"]
            if m in all_results.columns
        ]
        for gap_ms in gap_durations:
            ft = ft_data[ft_data["gap_ms"] == gap_ms].iloc[0]
            print(f"\n  Gap {gap_ms}ms:")
            print(f"  {'Model':<25} " + " ".join(f"Δ{m:>14}" for m in fair_metrics))
            print(f"  {'-'*90}")
            for model_name in ["clap_cqtdiff", "audiomae_cqtdiff"]:
                if model_name not in available_models:
                    continue
                hybrid = all_results[
                    (all_results["model"] == model_name) &
                    (all_results["gap_ms"] == gap_ms)
                ].iloc[0]
                deltas = []
                for metric in fair_metrics:
                    if metric in ["LSD", "LSD_GAP_ONLY", "GAP_LSD", "GAP_MEL_DISTANCE", "FAD"]:
                        deltas.append(ft[metric] - hybrid[metric])
                    else:
                        deltas.append(hybrid[metric] - ft[metric])
                print(f"  {model_name:<25} " + " ".join(f"{d:>+15.4f}" for d in deltas))

    summary_df = update_experiment_summary()
    print(f"\n✅ Semua hasil tersimpan di: {PATHS['results']}")
    print(f"Plots tersimpan di: {PATHS['plots']}")


maybe_auto_stop_instance()
