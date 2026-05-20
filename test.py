"""
Standalone test: CQT-Diff+ baseline diffusion inpainting.
Load WAV masked dari folder test_recon_this/, jalankan inpainting, simpan hasilnya.
"""

import os
import sys
import time

import numpy as np
import torch
from scipy.io import wavfile

# Setup path biar bisa import adapter
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CQT_DIFF_DIR = os.path.join(PROJECT_ROOT, "external", "CQTdiff")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CQT_DIFF_DIR)

# Kurangi steps buat tes cepat (35 = full quality, 10 = quick check)
os.environ["CQTDIFF_DIFFUSION_STEPS"] = "35"

# ============================================================
# Config
# ============================================================
INPUT_DIR   = os.path.join(PROJECT_ROOT, "test_recon_this")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "test_outputs", "recon_results")
TARGET_SR   = 44100
SEGMENT_LEN = 176400   # 4 detik @ 44100 Hz
GAP_START   = 50700    # sample mulai silence
GAP_END     = 125500   # sample akhir silence
GAP_MS      = int((GAP_END - GAP_START) / TARGET_SR * 1000)  # ~1700 ms

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_wav(path):
    """Load WAV jadi float32 mono, normalized ke [-1, 1]."""
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0
    return audio, sr


def save_wav(path, audio, sr):
    """Simpan float32 array ke WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, audio)
    print(f"  Saved: {path}")


def build_gap_mask(audio_len, gap_start, gap_end):
    """Buat boolean mask: True = gap area yang mau di-inpaint."""
    mask = np.zeros(audio_len, dtype=bool)
    mask[gap_start:gap_end] = True
    return mask


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("TEST: CQT-Diff+ Inpainting pada file masked WAV")
    print("=" * 60)
    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Gap       : sample {GAP_START}-{GAP_END} (~{GAP_MS}ms)")
    print(f"Steps     : {os.environ.get('CQTDIFF_DIFFUSION_STEPS', '35')}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device    : {torch.cuda.get_device_name(0)}\n")
    else:
        device = torch.device("cpu")
        print("WARNING: GPU tidak tersedia, pakai CPU (akan lambat!)\n")

    # Load adapter
    print("Loading CQT-Diff+ adapter...")
    from official_cqtdiff_adapter import build_cqtdiff_decoder

    decoder = build_cqtdiff_decoder(
        device=device,
        target_sr=TARGET_SR,
        segment_samples=SEGMENT_LEN,
        gap_durations_ms=[GAP_MS],
        cqt_diff_dir=CQT_DIFF_DIR,
    )
    decoder.eval()
    print("Model loaded OK.\n")

    wav_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")])
    print(f"Total file: {len(wav_files)}\n")

    results_summary = []

    for fname in wav_files:
        in_path  = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname.replace(".wav", "_recon.wav"))

        masked_audio, sr = load_wav(in_path)
        assert sr == TARGET_SR, f"SR mismatch: {sr} != {TARGET_SR}"

        if len(masked_audio) < SEGMENT_LEN:
            masked_audio = np.pad(masked_audio, (0, SEGMENT_LEN - len(masked_audio)))
        elif len(masked_audio) > SEGMENT_LEN:
            masked_audio = masked_audio[:SEGMENT_LEN]

        mask = build_gap_mask(len(masked_audio), GAP_START, GAP_END)

        t_start = time.perf_counter()
        with torch.inference_mode():
            masked_tensor = torch.from_numpy(masked_audio).float().unsqueeze(0).to(device)
            mask_tensor   = torch.from_numpy(mask).unsqueeze(0).to(device)
            recon = decoder.inpaint(masked_tensor, mask_tensor, conditioning=None)
        elapsed = time.perf_counter() - t_start

        rms_gap_in   = np.sqrt(np.mean(masked_audio[GAP_START:GAP_END] ** 2))
        rms_gap_out  = np.sqrt(np.mean(recon[GAP_START:GAP_END] ** 2))
        rms_ctx_left = np.sqrt(np.mean(masked_audio[max(0, GAP_START - 2205):GAP_START] ** 2))
        status = "OK" if rms_gap_out > rms_ctx_left * 0.2 else "QUIET"

        print(f"{fname} | {elapsed:.1f}s | "
              f"rms_in={rms_gap_in:.5f} rms_out={rms_gap_out:.5f} "
              f"ctx={rms_ctx_left:.5f} [{status}]")

        save_wav(out_path, recon, TARGET_SR)
        results_summary.append((fname, elapsed, rms_gap_out, rms_ctx_left, status))

    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    ok_count = sum(1 for r in results_summary if r[4] == "OK")
    print(f"Berhasil (gap punya energi): {ok_count}/{len(results_summary)}")
    print(f"\nOutput files ada di: {OUTPUT_DIR}/")
    print("Nama file: <original>_recon.wav")


if __name__ == "__main__":
    main()
