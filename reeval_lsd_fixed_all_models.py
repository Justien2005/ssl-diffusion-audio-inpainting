"""
Re-evaluate LSD for all reconstructed model outputs with the corrected formula.

This script intentionally does not run model inference. It reloads the existing
reconstructed WAV files produced by code_final_run_v2.py, recomputes only:
    LSD, LSD_GAP_ONLY, GAP_LSD
and writes the same per-model result CSVs plus results/all_results.csv.

Correct LSD formula used here:
    magnitude STFT, no power spectrum
    20 * log10(magnitude ratio)
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf


PIPELINE_STAGE_NAME = "code_v4_musicnet_cqtdiffplus_44k"
COLAB_DRIVE_ROOT = Path(os.environ.get("COLAB_DRIVE_ROOT", "/content/drive/MyDrive"))
DEFAULT_DRIVE_DATA_ROOT = COLAB_DRIVE_ROOT / "THESIS CODE"

TARGET_SR = int(os.environ.get("PIPELINE_TARGET_SR", "44100"))
SEGMENT_SAMPLES = int(os.environ.get("PIPELINE_SEGMENT_SAMPLES", "184184"))
SEGMENT_DURATION = SEGMENT_SAMPLES / TARGET_SR
EXPERIMENT_CONFIG_ID = (
    f"musicnet_cqtdiffplus_sr{TARGET_SR}_n{SEGMENT_SAMPLES}_"
    f"dur{SEGMENT_DURATION:.6f}s"
)
GAP_DURATIONS_MS = [100, 300, 500, 750, 1200, 1700]
DATASET_RANDOM_SEED = 42
N_EVAL_SAMPLES = int(os.environ.get("N_EVAL_SAMPLES", "100"))
EVAL_GAP_POSITION = os.environ.get("EVAL_GAP_POSITION", "center").strip().lower()
EVAL_RANDOM_GAP_MIN_CONTEXT_MS = int(os.environ.get("EVAL_RANDOM_GAP_MIN_CONTEXT_MS", "250"))

EXPECTED_MODEL_CONFIGS = [
    "baseline_cqtdiff",
    "baseline_cqtdiff_finetuned",
    "clap_cqtdiff",
    "clap_maid",
    "audiomae_cqtdiff",
    "audiomae_maid",
]
RUN_MODEL_SELECTION = [
    item.strip()
    for item in os.environ.get("RUN_MODELS", ",".join(EXPECTED_MODEL_CONFIGS)).split(",")
    if item.strip()
]
if RUN_MODEL_SELECTION == ["all"]:
    RUN_MODEL_SELECTION = EXPECTED_MODEL_CONFIGS


def info(message: str) -> None:
    print(f"[fixed-lsd] {message}", flush=True)


def fail(message: str) -> None:
    raise SystemExit(f"\nERROR: {message}\n")


def is_colab_runtime() -> bool:
    return (
        "COLAB_GPU" in os.environ
        or "google.colab" in sys.modules
        or importlib.util.find_spec("google.colab") is not None
    )


def maybe_mount_google_drive() -> None:
    if not is_colab_runtime():
        return
    if COLAB_DRIVE_ROOT.exists():
        info(f"Google Drive already mounted: {COLAB_DRIVE_ROOT}")
        return
    try:
        from google.colab import drive
    except Exception as exc:
        fail(f"Runtime Colab terdeteksi, tapi Drive tidak bisa dimount. Detail: {exc}")
    info("Mounting Google Drive at /content/drive")
    drive.mount("/content/drive")


def project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()


def data_root() -> Path:
    if "MUSIC_INPAINTING_ROOT" in os.environ:
        return Path(os.environ["MUSIC_INPAINTING_ROOT"]).resolve()
    if is_colab_runtime():
        return DEFAULT_DRIVE_DATA_ROOT.resolve()
    return (project_root() / "music_inpainting").resolve()


def paths() -> dict[str, Path]:
    base = data_root()
    stage = base / "training_stages" / PIPELINE_STAGE_NAME
    return {
        "base": base,
        "source_preprocessed": Path(
            os.environ.get("BASELINE_EVAL_PREPROCESSED_DIR", base / "preprocessed")
        ).resolve(),
        "preprocessed": (stage / "preprocessed").resolve(),
        "outputs": (stage / "outputs").resolve(),
        "results": (stage / "results").resolve(),
    }


def evaluation_artifact_name(model_name: str) -> str:
    if EVAL_GAP_POSITION == "center":
        return model_name
    return f"{model_name}_{EVAL_GAP_POSITION}gap"


def compute_gap_bounds(audio_length: int, gap_ms: int, sr: int = TARGET_SR) -> tuple[int, int]:
    gap_samples = int(round(sr * gap_ms / 1000))
    center = int(audio_length) // 2
    gap_start = center - gap_samples // 2
    return int(gap_start), int(gap_start + gap_samples)


def build_gap_mask_array(audio_length: int, gap_ms: int, sr: int = TARGET_SR,
                         gap_start: int | None = None) -> tuple[np.ndarray, int, int]:
    gap_samples = int(round(sr * gap_ms / 1000))
    if gap_start is None:
        gap_start, gap_end = compute_gap_bounds(audio_length, gap_ms, sr=sr)
    else:
        gap_start = int(gap_start)
        gap_end = gap_start + gap_samples
    mask = np.zeros(audio_length, dtype=bool)
    mask[gap_start:gap_end] = True
    return mask, int(gap_start), int(gap_end)


def make_eval_gap_mask(audio_length: int, gap_ms: int, sample_index: int,
                       sr: int = TARGET_SR) -> tuple[np.ndarray, int, int]:
    if EVAL_GAP_POSITION == "center":
        return build_gap_mask_array(audio_length, gap_ms, sr=sr)

    gap_samples = int(round(sr * gap_ms / 1000))
    min_context = int(round(sr * EVAL_RANDOM_GAP_MIN_CONTEXT_MS / 1000))
    min_start = min_context
    max_start = audio_length - gap_samples - min_context
    if max_start < min_start:
        min_start = 0
        max_start = audio_length - gap_samples
    if max_start < min_start:
        raise ValueError(f"Random gap {gap_ms}ms tidak valid untuk audio_length={audio_length}.")
    rng = np.random.default_rng(DATASET_RANDOM_SEED + 100_000 + int(sample_index) * 997 + int(gap_ms))
    return build_gap_mask_array(audio_length, gap_ms, sr=sr, gap_start=int(rng.integers(min_start, max_start + 1)))


def _stratified_sample_table(df: pd.DataFrame, n_samples: int, seed: int = DATASET_RANDOM_SEED,
                             stratify_cols=("composer", "instrument")) -> pd.DataFrame:
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
                unsampled.sample(min(missing, len(unsampled)), random_state=seed + 999),
            ], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=seed).drop(
        columns=["_sample_row_id"], errors="ignore"
    ).reset_index(drop=True)


def get_data_splits(meta_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
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

    return {
        "train": meta_df[meta_df["source_file"].isin(set(train_sources_df["source_file"]))].reset_index(drop=True),
        "val": meta_df[meta_df["source_file"].isin(set(val_sources_df["source_file"]))].reset_index(drop=True),
        "test": meta_df[meta_df["source_file"].isin(set(test_sources_df["source_file"]))].reset_index(drop=True),
    }


def read_audio_float32(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if int(sr) != int(TARGET_SR):
        raise RuntimeError(f"SR tidak cocok: {sr} != {TARGET_SR} pada {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_eval_originals(preprocessed_dir: Path, n_samples: int) -> list[np.ndarray]:
    metadata_path = preprocessed_dir / "metadata.csv"
    if not metadata_path.exists():
        fail(f"metadata.csv tidak ditemukan: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)
    if "clean_path" not in meta_df.columns:
        fail(f"metadata.csv tidak punya kolom clean_path: {metadata_path}")
    if "source_file" not in meta_df.columns:
        fail(f"metadata.csv tidak punya kolom source_file: {metadata_path}")

    splits = get_data_splits(meta_df)
    selected = _stratified_sample_table(
        splits["test"], min(int(n_samples), len(splits["test"])), seed=DATASET_RANDOM_SEED + 2
    )
    originals = []
    for _, row in selected.iterrows():
        clean_path = Path(row["clean_path"])
        if not clean_path.exists():
            candidate = preprocessed_dir / clean_path.name
            if candidate.exists():
                clean_path = candidate
            else:
                fail(f"File clean tidak ditemukan: {row['clean_path']}")
        originals.append(read_audio_float32(clean_path))
    return originals


def load_manifest(model_name: str, p: dict[str, Path]) -> pd.DataFrame:
    manifest_path = p["outputs"] / evaluation_artifact_name(model_name) / "manifest.csv"
    if not manifest_path.exists():
        fail(f"Manifest rekonstruksi tidak ditemukan untuk {model_name}: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"gap_ms", "sample_index", "reconstructed_path"}
    missing = required - set(manifest.columns)
    if missing:
        fail(f"Manifest {manifest_path} tidak punya kolom: {sorted(missing)}")
    return manifest


def load_reconstructed_from_manifest(model_name: str, manifest: pd.DataFrame, gap_ms: int,
                                     p: dict[str, Path]) -> tuple[list[np.ndarray], list[dict]]:
    subset = manifest[manifest["gap_ms"].astype(int) == int(gap_ms)].sort_values("sample_index")
    if subset.empty:
        fail(f"Manifest tidak punya rows gap {gap_ms}ms.")

    recon_audios = []
    regions = []
    artifact_name = evaluation_artifact_name(model_name)
    for _, row in subset.iterrows():
        recon_path = Path(str(row["reconstructed_path"]))
        if not recon_path.exists():
            fallback = (
                p["outputs"]
                / artifact_name
                / f"gap_{int(gap_ms)}ms"
                / f"{artifact_name}_gap{int(gap_ms)}ms_sample{int(row['sample_index']):04d}_reconstructed.wav"
            )
            if fallback.exists():
                recon_path = fallback
            else:
                fail(
                    "Reconstructed WAV tidak ditemukan.\n"
                    f"  manifest path: {recon_path}\n"
                    f"  fallback path: {fallback}"
                )
        recon_audios.append(read_audio_float32(recon_path))
        regions.append({
            "gap_start": int(row["gap_start"]) if "gap_start" in row and pd.notna(row["gap_start"]) else None,
            "gap_end": int(row["gap_end"]) if "gap_end" in row and pd.notna(row["gap_end"]) else None,
            "gap_position": str(row.get("gap_position", EVAL_GAP_POSITION)),
        })
    return recon_audios, regions


def compute_lsd_fixed_cpu(original: np.ndarray, reconstructed: np.ndarray,
                          sr: int = TARGET_SR, n_fft: int = 2048,
                          hop_length: int = 512, gap_start: int | None = None,
                          gap_end: int | None = None, frame_pad: int = 2) -> float:
    import librosa

    n = min(len(original), len(reconstructed))
    o = np.asarray(original[:n], dtype=np.float32)
    r = np.asarray(reconstructed[:n], dtype=np.float32)

    o_mag = np.abs(librosa.stft(o, n_fft=n_fft, hop_length=hop_length))
    r_mag = np.abs(librosa.stft(r, n_fft=n_fft, hop_length=hop_length))
    eps = max(1e-10, 1e-6 * float(o_mag.max()))
    log_diff = 20.0 * (np.log10(o_mag + eps) - np.log10(r_mag + eps))

    if gap_start is not None and gap_end is not None:
        f_start = max(0, int(gap_start) // hop_length - frame_pad)
        f_end = min(o_mag.shape[1], int(gap_end) // hop_length + frame_pad + 1)
        log_diff = log_diff[:, f_start:f_end]

    return float(np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0))))


def _eval_device():
    import torch
    use_gpu = os.environ.get("EVAL_USE_GPU", "1").lower() in {"1", "true", "yes", "on"}
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def compute_lsd_fixed_batch_gpu(original_audios: list[np.ndarray], reconstructed_audios: list[np.ndarray],
                                regions: list[dict], frame_pad: int = 2,
                                n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    import torch

    device = _eval_device()
    if device.type != "cuda":
        raise RuntimeError("CUDA tidak tersedia")
    min_len = min(min(len(x) for x in original_audios), min(len(x) for x in reconstructed_audios))
    originals = torch.as_tensor(
        np.stack([np.asarray(x[:min_len], dtype=np.float32) for x in original_audios], axis=0),
        dtype=torch.float32,
        device=device,
    )
    recons = torch.as_tensor(
        np.stack([np.asarray(x[:min_len], dtype=np.float32) for x in reconstructed_audios], axis=0),
        dtype=torch.float32,
        device=device,
    )

    window = torch.hann_window(n_fft, device=device)
    o_mag = torch.stft(
        originals, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    ).abs()
    r_mag = torch.stft(
        recons, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    ).abs()
    eps = torch.clamp(1e-6 * o_mag.amax(dim=(1, 2), keepdim=True), min=1e-10)
    log_diff = 20.0 * (torch.log10(o_mag + eps) - torch.log10(r_mag + eps))

    scores = []
    for idx, region in enumerate(regions):
        gap_start = int(region["gap_start"])
        gap_end = int(region["gap_end"])
        f_start = max(0, gap_start // hop_length - frame_pad)
        f_end = min(o_mag.shape[-1], gap_end // hop_length + frame_pad + 1)
        sample_diff = log_diff[idx, :, f_start:f_end]
        scores.append(torch.sqrt(torch.mean(sample_diff.pow(2), dim=0)).mean())
    return torch.stack(scores).detach().cpu().numpy().astype(np.float64)


def compute_model_lsd(model_name: str, originals: list[np.ndarray], manifest: pd.DataFrame,
                      p: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for gap_ms in GAP_DURATIONS_MS:
        recon_audios, regions = load_reconstructed_from_manifest(model_name, manifest, gap_ms, p)
        n = min(len(originals), len(recon_audios))
        if n == 0:
            fail(f"Tidak ada pasangan original/reconstructed untuk gap {gap_ms}ms.")
        gap_regions = []
        for idx in range(n):
            region = regions[idx]
            if region["gap_start"] is None or region["gap_end"] is None:
                _, gap_start, gap_end = make_eval_gap_mask(len(originals[idx]), gap_ms, idx)
                region = {**region, "gap_start": gap_start, "gap_end": gap_end}
            gap_regions.append(region)

        try:
            lsd_scores = compute_lsd_fixed_batch_gpu(originals[:n], recon_audios[:n], gap_regions, frame_pad=2)
            lsd_gap_scores = compute_lsd_fixed_batch_gpu(originals[:n], recon_audios[:n], gap_regions, frame_pad=0)
        except Exception as exc:
            info(f"GPU LSD gagal untuk gap {gap_ms}ms ({exc}); fallback CPU.")
            lsd_scores = np.asarray([
                compute_lsd_fixed_cpu(
                    originals[idx], recon_audios[idx],
                    gap_start=gap_regions[idx]["gap_start"], gap_end=gap_regions[idx]["gap_end"],
                    frame_pad=2,
                )
                for idx in range(n)
            ], dtype=np.float64)
            lsd_gap_scores = np.asarray([
                compute_lsd_fixed_cpu(
                    originals[idx], recon_audios[idx],
                    gap_start=gap_regions[idx]["gap_start"], gap_end=gap_regions[idx]["gap_end"],
                    frame_pad=0,
                )
                for idx in range(n)
            ], dtype=np.float64)

        rows.append({
            "gap_ms": int(gap_ms),
            "gap_position": EVAL_GAP_POSITION,
            "LSD": round(float(np.mean(lsd_scores)), 4),
            "LSD_GAP_ONLY": round(float(np.mean(lsd_gap_scores)), 4),
            "GAP_LSD": round(float(np.mean(lsd_gap_scores)), 4),
        })
    return pd.DataFrame(rows)


def save_results_like_pipeline(results_df: pd.DataFrame, model_name: str, p: dict[str, Path]) -> None:
    p["results"].mkdir(parents=True, exist_ok=True)
    results_df = results_df.copy()
    results_df["model"] = model_name
    results_df["experiment_config_id"] = EXPERIMENT_CONFIG_ID
    results_df["target_sr"] = TARGET_SR
    results_df["segment_samples"] = SEGMENT_SAMPLES
    if "gap_position" not in results_df.columns:
        results_df["gap_position"] = EVAL_GAP_POSITION
    results_df["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    artifact_name = evaluation_artifact_name(model_name)
    model_path = p["results"] / f"{artifact_name}_results.csv"
    results_df.to_csv(model_path, index=False)
    info(f"Saved {model_name}: {model_path}")

    master_path = p["results"] / "all_results.csv"
    if master_path.exists():
        existing = pd.read_csv(master_path)
        if "gap_position" not in existing.columns:
            existing["gap_position"] = "center"
        existing = existing[
            ~((existing["model"] == model_name) & (existing["gap_position"] == EVAL_GAP_POSITION))
        ]
        combined = pd.concat([existing, results_df], ignore_index=True)
    else:
        combined = results_df
    combined.to_csv(master_path, index=False)
    info(f"Updated master: {master_path}")


def merge_lsd_into_existing(model_name: str, lsd_df: pd.DataFrame, p: dict[str, Path]) -> pd.DataFrame:
    model_path = p["results"] / f"{evaluation_artifact_name(model_name)}_results.csv"
    if model_path.exists():
        result_df = pd.read_csv(model_path)
    else:
        info(f"CSV lama tidak ditemukan untuk {model_name}; membuat CSV LSD-only.")
        result_df = lsd_df.copy()

    if "gap_ms" not in result_df.columns:
        fail(f"CSV hasil lama tidak punya kolom gap_ms: {model_path}")

    if "gap_position" not in result_df.columns:
        result_df["gap_position"] = EVAL_GAP_POSITION
    result_df = result_df.copy()
    for _, row in lsd_df.iterrows():
        mask = result_df["gap_ms"].astype(int) == int(row["gap_ms"])
        if "gap_position" in result_df.columns:
            mask = mask & (result_df["gap_position"].fillna("center") == EVAL_GAP_POSITION)
        if not mask.any():
            result_df = pd.concat([result_df, pd.DataFrame([row.to_dict()])], ignore_index=True)
            mask = result_df["gap_ms"].astype(int) == int(row["gap_ms"])
        for col in ["LSD", "LSD_GAP_ONLY", "GAP_LSD"]:
            result_df.loc[mask, col] = row[col]
    return result_df


def infer_n_eval_samples(model_manifests: dict[str, pd.DataFrame]) -> int:
    counts = []
    for manifest in model_manifests.values():
        per_gap = manifest.groupby("gap_ms")["sample_index"].nunique()
        if not per_gap.empty:
            counts.append(int(per_gap.min()))
    return min(counts) if counts else N_EVAL_SAMPLES


def main() -> None:
    maybe_mount_google_drive()
    p = paths()
    info(f"data root     : {p['base']}")
    info(f"preprocessed  : {p['preprocessed']}")
    info(f"outputs       : {p['outputs']}")
    info(f"results       : {p['results']}")

    if not p["preprocessed"].exists() and p["source_preprocessed"].exists():
        p["preprocessed"] = p["source_preprocessed"]
        info(f"using root preprocessed folder: {p['preprocessed']}")
    if not p["outputs"].exists():
        fail(f"Folder outputs tidak ditemukan: {p['outputs']}")

    model_manifests = {}
    for model_name in RUN_MODEL_SELECTION:
        model_manifests[model_name] = load_manifest(model_name, p)
    n_eval_samples = infer_n_eval_samples(model_manifests)
    info(f"n_eval_samples inferred from manifests: {n_eval_samples}")

    originals = load_eval_originals(p["preprocessed"], n_eval_samples)
    info(f"loaded originals: {len(originals)}")

    for model_name in RUN_MODEL_SELECTION:
        info(f"Re-evaluating fixed LSD: {model_name}")
        lsd_df = compute_model_lsd(model_name, originals, model_manifests[model_name], p)
        merged_df = merge_lsd_into_existing(model_name, lsd_df, p)
        save_results_like_pipeline(merged_df, model_name, p)

    info("Fixed-LSD re-evaluation complete.")


if __name__ == "__main__":
    main()
