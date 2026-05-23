#!/usr/bin/env python
import argparse
import json
import os
import random
import shutil
import tarfile
import time
import urllib.request

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


PIPELINE_STAGE_NAME = os.environ.get(
    "PIPELINE_STAGE_NAME", "code_v4_musicnet_cqtdiffplus_44k"
)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
BASE_LOCAL_ROOT = os.environ.get(
    "MUSIC_INPAINTING_ROOT", os.path.join(PROJECT_ROOT, "music_inpainting")
)
DATA_ROOT = os.path.join(BASE_LOCAL_ROOT, "training_stages", PIPELINE_STAGE_NAME)

PATHS = {
    "dataset": os.path.join(BASE_LOCAL_ROOT, "dataset"),
    "preprocessed": os.path.join(DATA_ROOT, "preprocessed"),
    "masked": os.path.join(DATA_ROOT, "masked"),
    "results": os.path.join(DATA_ROOT, "results"),
}

TARGET_SR = int(os.environ.get("PIPELINE_TARGET_SR", "44100"))
SEGMENT_SAMPLES = int(os.environ.get("PIPELINE_SEGMENT_SAMPLES", "184184"))
SEGMENT_DURATION = SEGMENT_SAMPLES / TARGET_SR
EXPERIMENT_CONFIG_ID = (
    f"musicnet_cqtdiffplus_sr{TARGET_SR}_n{SEGMENT_SAMPLES}_"
    f"dur{SEGMENT_DURATION:.6f}s"
)

GAP_DURATIONS_MS = [100, 300, 500, 750, 1200, 1700]
DATASET_RANDOM_SEED = int(os.environ.get("DATASET_RANDOM_SEED", "42"))
DATASET_FRACTION = float(os.environ.get("DATASET_FRACTION", "0.5"))
MAX_SEGMENTS_PER_FILE = int(os.environ.get("MAX_SEGMENTS_PER_FILE", "5"))

MUSICNET_URL = "https://zenodo.org/record/5120004/files/musicnet.tar.gz"
MUSICNET_METADATA_URL = "https://zenodo.org/record/5120004/files/musicnet_metadata.csv"


def ensure_dirs():
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)


def _track_id_from_path(audio_path):
    return os.path.splitext(os.path.basename(audio_path))[0]


def _normalise_musicnet_metadata(metadata_df):
    if metadata_df is None or metadata_df.empty:
        return pd.DataFrame()

    meta = metadata_df.copy()
    meta.columns = [str(c).strip().lower() for c in meta.columns]
    if "id" not in meta.columns:
        return pd.DataFrame()

    meta["track_id"] = meta["id"].astype(str)
    if "composer" not in meta.columns:
        meta["composer"] = "unknown"

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

    quotas = []
    for group_key, group in work.groupby(stratify_cols, dropna=False, sort=True):
        expected = len(group) * n_samples / len(work)
        base = int(np.floor(expected))
        quotas.append(
            {
                "key": group_key,
                "group": group,
                "quota": min(base, len(group)),
                "fractional": expected - base,
                "tie": rng.random(),
            }
        )

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
    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else work.iloc[0:0].copy()

    if len(sampled) < n_samples:
        missing = n_samples - len(sampled)
        sampled_ids = set(sampled["_sample_row_id"]) if "_sample_row_id" in sampled.columns else set()
        unsampled = work[~work["_sample_row_id"].isin(sampled_ids)]
        if len(unsampled) > 0:
            sampled = pd.concat(
                [sampled, unsampled.sample(min(missing, len(unsampled)), random_state=seed + 999)],
                ignore_index=True,
            )

    return (
        sampled.sample(frac=1.0, random_state=seed)
        .drop(columns=["_sample_row_id"], errors="ignore")
        .reset_index(drop=True)
    )


def select_stratified_audio_files(all_audio_files, metadata_df, fraction, seed=DATASET_RANDOM_SEED):
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
    return stratified_sample_dataframe(
        audio_df,
        n_samples=n_files,
        stratify_cols=["composer", "instrument"],
        seed=seed,
    )


def download_musicnet_metadata(allow_download=False):
    metadata_path = os.path.join(PATHS["dataset"], "musicnet_metadata.csv")
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)
    if not allow_download:
        print("Metadata MusicNet belum ada; lanjut dengan strata composer/instrument='unknown'.")
        return pd.DataFrame()

    print("Downloading MusicNet metadata...")
    urllib.request.urlretrieve(MUSICNET_METADATA_URL, metadata_path)
    return pd.read_csv(metadata_path)


def find_or_download_musicnet_audio(allow_download=False):
    audio_dir = os.path.join(PATHS["dataset"], "audio")
    if os.path.exists(audio_dir):
        existing = [
            name for name in os.listdir(audio_dir)
            if name.endswith((".wav", ".flac")) or os.path.isdir(os.path.join(audio_dir, name))
        ]
        if len(existing) > 0:
            print(f"Dataset audio sudah ada, skip download: {audio_dir}")
            return audio_dir

    if not allow_download:
        raise FileNotFoundError(
            f"Dataset audio belum ditemukan di {audio_dir}. "
            "Jalankan setup/download lama dulu, atau pakai --download-audio kalau memang mau download ulang."
        )

    os.makedirs(audio_dir, exist_ok=True)
    tar_path = os.path.join(PATHS["dataset"], "musicnet.tar.gz")
    if not (os.path.exists(tar_path) and os.path.getsize(tar_path) > 1e10):
        print("Downloading MusicNet audio archive (~11GB)...")
        urllib.request.urlretrieve(MUSICNET_URL, tar_path)
    print("Extracting MusicNet audio...")
    with tarfile.open(tar_path, "r:gz") as tar:
        wav_members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
        for member in tqdm(wav_members, desc="Extracting"):
            tar.extract(member, audio_dir)
    return audio_dir


def preprocess_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 0.07
    if rms > 1e-6:
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)


def split_into_segments(audio, file_seed=0):
    rng = random.Random(file_seed)
    if len(audio) < SEGMENT_SAMPLES:
        return []
    possible_starts = list(range(0, len(audio) - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES))
    n_segments = min(MAX_SEGMENTS_PER_FILE, len(possible_starts))
    selected_starts = rng.sample(possible_starts, n_segments)
    return [audio[start : start + SEGMENT_SAMPLES] for start in selected_starts]


def compute_gap_bounds(audio_length, gap_ms, sr=TARGET_SR):
    gap_samples = int(round(sr * gap_ms / 1000))
    if gap_samples <= 0 or gap_samples >= audio_length:
        raise ValueError(f"Gap {gap_ms}ms tidak valid untuk audio_length={audio_length}, sr={sr}.")
    center = audio_length // 2
    gap_start = center - gap_samples // 2
    gap_end = gap_start + gap_samples
    if gap_start < 0 or gap_end > audio_length:
        raise ValueError(f"Gap bounds keluar audio: start={gap_start}, end={gap_end}, length={audio_length}.")
    return gap_start, gap_end


def apply_gap_mask(audio_segment, gap_ms, sr=TARGET_SR):
    gap_start, gap_end = compute_gap_bounds(len(audio_segment), gap_ms, sr=sr)
    mask = np.zeros(len(audio_segment), dtype=bool)
    mask[gap_start:gap_end] = True
    masked_audio = audio_segment.copy()
    masked_audio[gap_start:gap_end] = 0.0
    return masked_audio, mask, gap_start, gap_end


def current_preprocessing_config():
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


def preprocessing_config_matches(config_path, metadata_path):
    if not os.path.exists(config_path) or not os.path.exists(metadata_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        expected = current_preprocessing_config()
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
        print(f"Cached preprocessing config tidak valid ({exc}); preprocessing akan dibuat ulang.")
        return False


def artifacts_ready():
    config_path = os.path.join(PATHS["preprocessed"], "preprocessing_config.json")
    metadata_path = os.path.join(PATHS["preprocessed"], "metadata.csv")
    expected_mask_dirs = [os.path.join(PATHS["masked"], f"gap_{gap_ms}ms") for gap_ms in GAP_DURATIONS_MS]
    return preprocessing_config_matches(config_path, metadata_path) and all(
        os.path.isdir(path) for path in expected_mask_dirs
    )


def clear_preprocessed_artifacts():
    for key in ["preprocessed", "masked"]:
        path = PATHS[key]
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def write_preprocessing_timing(status, total_seconds, total_segments=0):
    os.makedirs(PATHS["results"], exist_ok=True)
    row = {
        "stage": PIPELINE_STAGE_NAME,
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


def run_preprocessing(force=False, skip_if_exists=True, download_audio=False, download_metadata=False):
    ensure_dirs()
    start_time = time.perf_counter()

    print("=" * 72)
    print("MusicNet 44k CQTdiff+ Preprocessing")
    print("=" * 72)
    print(f"Stage           : {PIPELINE_STAGE_NAME}")
    print(f"Dataset root    : {PATHS['dataset']}")
    print(f"Preprocessed    : {PATHS['preprocessed']}")
    print(f"Masked          : {PATHS['masked']}")
    print(f"Target SR       : {TARGET_SR}")
    print(f"Segment samples : {SEGMENT_SAMPLES} ({SEGMENT_DURATION:.3f}s)")
    print(f"Dataset fraction: {DATASET_FRACTION:.0%}")
    print(f"Skip if exists  : {skip_if_exists}")
    print("=" * 72)

    if force:
        print("Force enabled: membersihkan artifact preprocessed/masked.")
        clear_preprocessed_artifacts()
    elif skip_if_exists and artifacts_ready():
        print("Preprocessing 44k sudah lengkap; skip.")
        done_path = os.path.join(PATHS["preprocessed"], ".done")
        if not os.path.exists(done_path):
            with open(done_path, "w", encoding="utf-8") as f:
                f.write("done")
        write_preprocessing_timing("skipped", time.perf_counter() - start_time, 0)
        return
    else:
        metadata_path = os.path.join(PATHS["preprocessed"], "metadata.csv")
        config_path = os.path.join(PATHS["preprocessed"], "preprocessing_config.json")
        if os.path.exists(metadata_path) or os.path.exists(config_path):
            print("Cache preprocessing lama/tidak lengkap ditemukan; regenerate preprocessed + masked saja.")
            clear_preprocessed_artifacts()

    audio_dir = find_or_download_musicnet_audio(allow_download=download_audio)
    metadata_df = download_musicnet_metadata(allow_download=download_metadata)

    all_audio_files = []
    for root, _, files in os.walk(audio_dir):
        for filename in files:
            if filename.endswith((".wav", ".flac")):
                all_audio_files.append(os.path.join(root, filename))
    if not all_audio_files:
        raise RuntimeError(f"Tidak ada file .wav/.flac di {audio_dir}")

    selected_table = select_stratified_audio_files(
        all_audio_files,
        metadata_df=metadata_df,
        fraction=DATASET_FRACTION,
        seed=DATASET_RANDOM_SEED,
    )

    print(f"Total file audio : {len(all_audio_files)}")
    print(f"File dipakai     : {len(selected_table)}")
    print(f"Gap durations    : {GAP_DURATIONS_MS} ms")

    segment_metadata = []
    segment_id = 0

    for file_idx, (_, file_row) in enumerate(
        tqdm(selected_table.iterrows(), total=len(selected_table), desc="Preprocessing files")
    ):
        filepath = file_row["audio_path"]
        try:
            audio = preprocess_audio(filepath)
            segments = split_into_segments(audio, file_seed=DATASET_RANDOM_SEED + file_idx)
            for segment in segments:
                clean_filename = f"seg_{segment_id:05d}.wav"
                clean_path = os.path.join(PATHS["preprocessed"], clean_filename)
                sf.write(clean_path, segment, TARGET_SR)

                for gap_ms in GAP_DURATIONS_MS:
                    masked_audio, _, _, _ = apply_gap_mask(segment, gap_ms)
                    masked_dir = os.path.join(PATHS["masked"], f"gap_{gap_ms}ms")
                    os.makedirs(masked_dir, exist_ok=True)
                    sf.write(os.path.join(masked_dir, clean_filename), masked_audio, TARGET_SR)

                composer = str(file_row.get("composer", "unknown"))
                instrument = str(file_row.get("instrument", "unknown"))
                segment_metadata.append(
                    {
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
                    }
                )
                segment_id += 1
        except Exception as exc:
            print(f"Gagal memproses {filepath}: {exc}")

    meta_df = pd.DataFrame(segment_metadata)
    if meta_df.empty:
        raise RuntimeError("Preprocessing selesai tanpa segmen. Cek dataset audio dan SEGMENT_SAMPLES.")

    meta_path = os.path.join(PATHS["preprocessed"], "metadata.csv")
    config_path = os.path.join(PATHS["preprocessed"], "preprocessing_config.json")
    done_path = os.path.join(PATHS["preprocessed"], ".done")
    meta_df.to_csv(meta_path, index=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(current_preprocessing_config(), f, indent=2)
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("done")

    elapsed = time.perf_counter() - start_time
    write_preprocessing_timing("completed", elapsed, segment_id)
    print(f"Preprocessing selesai. Total segmen: {segment_id}")
    print(f"Metadata: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone MusicNet preprocessing untuk default 44k MusicNet CQTdiff+ pipeline."
    )
    parser.add_argument("--force", action="store_true", help="Regenerate preprocessed/masked walau cache valid.")
    parser.add_argument(
        "--no-skip-if-exists",
        action="store_true",
        help="Abaikan cache-ready check tanpa menghapus terlebih dahulu kecuali cache tidak cocok.",
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        help="Download MusicNet audio jika dataset/audio belum ada. Default: tidak download.",
    )
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="Download metadata jika belum ada. Default: fallback unknown tanpa download.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        force=args.force,
        skip_if_exists=not args.no_skip_if_exists,
        download_audio=args.download_audio,
        download_metadata=args.download_metadata,
    )
