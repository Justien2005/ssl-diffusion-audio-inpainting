import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STAGE_NAME = "code_v3_final_run_native_cqt"
DEFAULT_MODEL_NAME = "baseline_cqtdiff"
DEFAULT_SR = 22050


def make_gap_bounds(audio_length, gap_ms, sr=DEFAULT_SR):
    gap_samples = int(round(sr * gap_ms / 1000))
    center = audio_length // 2
    gap_start = center - gap_samples // 2
    gap_end = gap_start + gap_samples
    return max(0, gap_start), min(audio_length, gap_end)


def rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def read_audio_float32(path):
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        return audio, sr
    except ModuleNotFoundError:
        try:
            from scipy.io import wavfile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Butuh package soundfile atau scipy untuk membaca WAV. "
                "Install salah satu: pip install soundfile atau pip install scipy"
            ) from exc

        sr, audio = wavfile.read(path)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
        else:
            audio = audio.astype(np.float32)
        return audio, sr


def write_audio_float32(path, audio, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    try:
        import soundfile as sf

        sf.write(path, audio, sr, subtype="FLOAT")
    except ModuleNotFoundError:
        try:
            from scipy.io import wavfile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Butuh package soundfile atau scipy untuk menulis WAV. "
                "Install salah satu: pip install soundfile atau pip install scipy"
            ) from exc

        wavfile.write(path, sr, audio)


def default_manifest_path(model_name):
    project_root = Path(os.environ.get("PROJECT_ROOT", os.getcwd()))
    base_root = Path(os.environ.get("MUSIC_INPAINTING_ROOT", project_root / "music_inpainting"))
    stage_name = os.environ.get("PIPELINE_STAGE_NAME", DEFAULT_STAGE_NAME)
    return base_root / "training_stages" / stage_name / "outputs" / model_name / "manifest.csv"


def build_gain_envelope(length, gain, fade_samples):
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    envelope = np.full(length, gain, dtype=np.float32)
    fade = min(int(fade_samples), length // 2)
    if fade > 0:
        ramp_up = np.linspace(1.0, gain, fade, dtype=np.float32)
        ramp_down = np.linspace(gain, 1.0, fade, dtype=np.float32)
        envelope[:fade] = ramp_up
        envelope[-fade:] = ramp_down
    return envelope


def gain_match_manifest(
    manifest_path,
    output_dir=None,
    max_gain=1000.0,
    target_ratio=1.0,
    context_ms=500.0,
    fade_ms=30.0,
    peak_limit=0.98,
):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"reconstructed_path", "gap_ms", "sample_index"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Kolom manifest kurang: {sorted(missing)}")

    model_name = manifest_path.parent.name
    if output_dir is None:
        output_dir = manifest_path.parent.parent / f"{model_name}_gainmatched"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    adjusted_manifest_rows = []

    for _, row in manifest.iterrows():
        in_path = Path(str(row["reconstructed_path"]))
        gap_ms = int(row["gap_ms"])
        sample_index = int(row["sample_index"])

        audio, sr = read_audio_float32(in_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32)

        gap_start, gap_end = make_gap_bounds(len(audio), gap_ms, sr)
        gap = audio[gap_start:gap_end]

        context_n = max(1, int(round(sr * context_ms / 1000)))
        left = audio[max(0, gap_start - context_n):gap_start]
        right = audio[gap_end:min(len(audio), gap_end + context_n)]
        context = np.concatenate([left, right])

        gap_rms_before = rms(gap)
        context_rms = rms(context)

        if gap_rms_before > 0 and context_rms > 0:
            raw_gain = (context_rms * float(target_ratio)) / gap_rms_before
            applied_gain = min(float(raw_gain), float(max_gain))
        else:
            raw_gain = np.nan
            applied_gain = 1.0

        out_audio = audio.copy()
        fade_samples = int(round(sr * fade_ms / 1000))
        envelope = build_gain_envelope(gap_end - gap_start, applied_gain, fade_samples)
        out_audio[gap_start:gap_end] = gap * envelope

        peak = float(np.max(np.abs(out_audio))) if out_audio.size else 0.0
        limiter_gain = 1.0
        if peak > float(peak_limit) > 0:
            limiter_gain = float(peak_limit) / peak
            out_audio *= limiter_gain

        out_name = f"{model_name}_gainmatched_gap{gap_ms}ms_sample{sample_index:04d}.wav"
        out_path = output_dir / f"gap_{gap_ms}ms" / out_name
        write_audio_float32(out_path, out_audio, sr)

        gap_after = out_audio[gap_start:gap_end]
        rows.append(
            {
                "gap_ms": gap_ms,
                "sample_index": sample_index,
                "rms_gap_before": gap_rms_before,
                "rms_gap_after": rms(gap_after),
                "rms_context": context_rms,
                "raw_gain": raw_gain,
                "applied_gain": applied_gain,
                "limiter_gain": limiter_gain,
                "output_path": str(out_path),
                "input_path": str(in_path),
            }
        )

        adjusted_manifest_rows.append(
            {
                "model": f"{model_name}_gainmatched",
                "gap_ms": gap_ms,
                "sample_index": sample_index,
                "sr": int(sr),
                "n_samples": int(len(out_audio)),
                "duration_seconds": float(len(out_audio) / sr),
                "reconstructed_path": str(out_path),
                "source_reconstructed_path": str(in_path),
            }
        )

    detail = pd.DataFrame(rows)
    adjusted_manifest = pd.DataFrame(adjusted_manifest_rows)

    detail_path = output_dir / "gain_match_detail.csv"
    manifest_out_path = output_dir / "manifest.csv"
    summary_path = output_dir / "gain_match_summary.csv"

    detail.to_csv(detail_path, index=False)
    adjusted_manifest.to_csv(manifest_out_path, index=False)

    summary = detail.groupby("gap_ms", as_index=False).agg(
        rms_gap_before_mean=("rms_gap_before", "mean"),
        rms_gap_after_mean=("rms_gap_after", "mean"),
        rms_context_mean=("rms_context", "mean"),
        raw_gain_median=("raw_gain", "median"),
        applied_gain_median=("applied_gain", "median"),
        applied_gain_max=("applied_gain", "max"),
        n_samples=("sample_index", "count"),
    )
    summary.to_csv(summary_path, index=False)

    print(f"Input manifest: {manifest_path}")
    print(f"Output dir: {output_dir}")
    print(f"Adjusted manifest: {manifest_out_path}")
    print(f"Detail CSV: {detail_path}")
    print(f"Summary CSV: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nCatatan: ini hanya patch diagnostik. Kalau isi gap jadi terdengar, berarti masalah utamanya amplitudo/energy collapse.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gain-match area gap terhadap RMS konteks kiri/kanan untuk diagnosis output inpainting silent."
    )
    parser.add_argument("--manifest", default=None, help="Path ke manifest.csv input.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Nama model output. Default: baseline_cqtdiff.")
    parser.add_argument("--output-dir", default=None, help="Folder output WAV gain-matched.")
    parser.add_argument("--max-gain", type=float, default=1000.0, help="Batas gain maksimum pada gap.")
    parser.add_argument("--target-ratio", type=float, default=1.0, help="Target RMS gap terhadap RMS konteks.")
    parser.add_argument("--context-ms", type=float, default=500.0, help="Durasi konteks kiri/kanan untuk hitung RMS.")
    parser.add_argument("--fade-ms", type=float, default=30.0, help="Fade gain di awal/akhir gap.")
    parser.add_argument("--peak-limit", type=float, default=0.98, help="Limiter peak global setelah gain.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = Path(args.manifest) if args.manifest else default_manifest_path(args.model)
    gain_match_manifest(
        manifest,
        output_dir=args.output_dir,
        max_gain=args.max_gain,
        target_ratio=args.target_ratio,
        context_ms=args.context_ms,
        fade_ms=args.fade_ms,
        peak_limit=args.peak_limit,
    )
