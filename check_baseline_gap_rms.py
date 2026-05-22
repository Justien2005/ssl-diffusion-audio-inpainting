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


def peak(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


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
            max_abs = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_abs
        else:
            audio = audio.astype(np.float32)
        return audio, sr


def default_manifest_path(model_name):
    project_root = Path(os.environ.get("PROJECT_ROOT", os.getcwd()))
    base_root = Path(os.environ.get("MUSIC_INPAINTING_ROOT", project_root / "music_inpainting"))
    stage_name = os.environ.get("PIPELINE_STAGE_NAME", DEFAULT_STAGE_NAME)
    return base_root / "training_stages" / stage_name / "outputs" / model_name / "manifest.csv"


def analyze_manifest(manifest_path, output_csv=None, max_rows=20):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"reconstructed_path", "gap_ms", "sample_index"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Kolom manifest kurang: {sorted(missing)}")

    rows = []
    for _, row in manifest.iterrows():
        path = Path(str(row["reconstructed_path"]))
        gap_ms = int(row["gap_ms"])
        sample_index = int(row["sample_index"])

        audio, sr = read_audio_float32(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1, dtype=np.float32)

        gap_start, gap_end = make_gap_bounds(len(audio), gap_ms, sr)
        gap_audio = audio[gap_start:gap_end]
        context_len = max(1, gap_end - gap_start)
        left_context = audio[max(0, gap_start - context_len):gap_start]
        right_context = audio[gap_end:min(len(audio), gap_end + context_len)]

        gap_rms = rms(gap_audio)
        left_rms = rms(left_context)
        right_rms = rms(right_context)
        context_rms = rms(np.concatenate([left_context, right_context]))

        rows.append(
            {
                "gap_ms": gap_ms,
                "sample_index": sample_index,
                "sr": int(sr),
                "gap_start": int(gap_start),
                "gap_end": int(gap_end),
                "rms_gap": gap_rms,
                "rms_left_context": left_rms,
                "rms_right_context": right_rms,
                "rms_context": context_rms,
                "gap_to_context_ratio": gap_rms / context_rms if context_rms > 0 else np.nan,
                "peak_gap": peak(gap_audio),
                "peak_full": peak(audio),
                "reconstructed_path": str(path),
            }
        )

    detail = pd.DataFrame(rows)
    summary = detail.groupby("gap_ms", as_index=False).agg(
        rms_gap_mean=("rms_gap", "mean"),
        rms_gap_median=("rms_gap", "median"),
        rms_gap_min=("rms_gap", "min"),
        rms_gap_max=("rms_gap", "max"),
        rms_context_mean=("rms_context", "mean"),
        ratio_mean=("gap_to_context_ratio", "mean"),
        ratio_median=("gap_to_context_ratio", "median"),
        peak_gap_mean=("peak_gap", "mean"),
        peak_gap_max=("peak_gap", "max"),
        n_samples=("sample_index", "count"),
    )

    if output_csv is None:
        output_csv = manifest_path.with_name("baseline_gap_rms_detail.csv")
    output_csv = Path(output_csv)
    detail.to_csv(output_csv, index=False)

    summary_csv = output_csv.with_name(output_csv.stem.replace("_detail", "") + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print(f"Manifest: {manifest_path}")
    print(f"Detail CSV: {output_csv}")
    print(f"Summary CSV: {summary_csv}")
    print("\nSummary per gap_ms:")
    print(summary.to_string(index=False))

    print(f"\n{max_rows} sample dengan rms_gap terkecil:")
    cols = ["gap_ms", "sample_index", "rms_gap", "rms_context", "gap_to_context_ratio", "peak_gap"]
    print(detail.sort_values("rms_gap")[cols].head(max_rows).to_string(index=False))

    print("\nInterpretasi cepat:")
    print("- Jika rms_gap dan peak_gap mendekati 0, area gap memang hampir silent.")
    print("- Jika gap_to_context_ratio sangat kecil, gap jauh lebih pelan daripada konteks kiri/kanan.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cek RMS/amplitudo area gap pada output evaluasi baseline_cqtdiff."
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path ke manifest.csv. Default mengikuti PATHS outputs baseline dari code_final_run_v2.py.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Nama model output. Default: baseline_cqtdiff.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path CSV detail output. Default: sebelah manifest.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Jumlah sample terendah yang ditampilkan.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = Path(args.manifest) if args.manifest else default_manifest_path(args.model)
    analyze_manifest(manifest, output_csv=args.output_csv, max_rows=args.max_rows)
