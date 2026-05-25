"""
One-shot baseline CQTdiff+ evaluation runner.

Equivalent target run:
    python code_final_run_v2.py --phase eval --models baseline_cqtdiff

This file adds the missing preflight pieces for eval-only usage:
- validate dataset/preprocessed/masked artifacts before the main pipeline can
  fall back to preprocessing;
- install missing Python requirements;
- ensure external CQTdiff/audio-inpainting repos exist;
- download the MusicNet audio-inpainting CQTdiff+ checkpoint if missing.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import urllib.request


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

CKPT_URL = (
    "https://huggingface.co/Eloimoliner/audio-inpainting-diffusion/resolve/main/"
    "musicnet_44k_4s-560000.pt"
)
CKPT_FILENAME = "musicnet_44k_4s-560000.pt"
MUSICNET_URL = "https://zenodo.org/record/5120004/files/musicnet.tar.gz"
MUSICNET_METADATA_URL = "https://zenodo.org/record/5120004/files/musicnet_metadata.csv"

IMPORT_CHECKS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "scipy": "scipy",
    "resampy": "resampy",
    "tqdm": "tqdm",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "omegaconf": "omegaconf",
    "hydra": "hydra-core",
    "einops": "einops",
    "torchvggish": "torchvggish",
    "visqol": "visqol-python",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
}


def info(message: str) -> None:
    print(f"[baseline-eval] {message}", flush=True)


def fail(message: str) -> None:
    raise SystemExit(f"\nERROR: {message}\n")


def run_command(command: list[str], cwd: Path | None = None) -> None:
    info("Running: " + " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()


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
        fail(
            "Runtime terlihat seperti Colab, tapi google.colab.drive tidak bisa di-import.\n"
            f"Detail: {exc}"
        )
    info("Mounting Google Drive at /content/drive")
    drive.mount("/content/drive")


def default_data_root(root: Path) -> Path:
    if "MUSIC_INPAINTING_ROOT" in os.environ:
        return Path(os.environ["MUSIC_INPAINTING_ROOT"]).resolve()
    if is_colab_runtime():
        return DEFAULT_DRIVE_DATA_ROOT.resolve()
    return (root / "music_inpainting").resolve()


def default_external_root(root: Path) -> Path:
    if "BASELINE_EVAL_EXTERNAL_ROOT" in os.environ:
        return Path(os.environ["BASELINE_EVAL_EXTERNAL_ROOT"]).resolve()
    if is_colab_runtime():
        return Path("/content/baseline_cqtdiff_external").resolve()
    return (root / "external").resolve()


def path_config() -> dict[str, Path]:
    root = project_root()
    base_root = default_data_root(root)
    stage_root = base_root / "training_stages" / PIPELINE_STAGE_NAME
    source_preprocessed = Path(
        os.environ.get("BASELINE_EVAL_PREPROCESSED_DIR", base_root / "preprocessed")
    ).resolve()
    source_masked = Path(os.environ.get("BASELINE_EVAL_MASKED_DIR", base_root / "masked")).resolve()
    external_root = default_external_root(root)
    audio_inpainting_dir = Path(
        os.environ.get("AUDIO_INPAINTING_DIR", external_root / "audio-inpainting-diffusion")
    ).resolve()
    cqt_diff_dir = Path(os.environ.get("CQT_DIFF_DIR", external_root / "CQTdiff")).resolve()
    ckpt_path = Path(
        os.environ.get(
            "AUDIO_INPAINTING_CQTDIFF_WEIGHTS",
            (
                base_root / "official_checkpoints" / "audio_inpainting_cqtdiff" / CKPT_FILENAME
                if is_colab_runtime()
                else audio_inpainting_dir / "experiments" / CKPT_FILENAME
            ),
        )
    ).resolve()
    return {
        "root": root,
        "base_root": base_root,
        "stage_root": stage_root,
        "dataset": base_root / "dataset",
        "source_preprocessed": source_preprocessed,
        "source_masked": source_masked,
        "preprocessed": stage_root / "preprocessed",
        "masked": stage_root / "masked",
        "external": external_root,
        "audio_inpainting_dir": audio_inpainting_dir,
        "cqt_diff_dir": cqt_diff_dir,
        "ckpt_path": ckpt_path,
        "main_py": root / "code_final_run_v2.py",
    }


def ensure_python_requirements(root: Path) -> None:
    missing = [
        package_name
        for import_name, package_name in IMPORT_CHECKS.items()
        if importlib.util.find_spec(import_name) is None
    ]
    if not missing:
        info("Python requirements already importable; skip pip install.")
        return

    info("Missing imports: " + ", ".join(sorted(missing)))
    requirements = root / "requirements.txt"
    if not requirements.exists():
        fail(f"requirements.txt tidak ditemukan di {requirements}")

    if "torch" in missing:
        torch_index = os.environ.get("TORCH_CUDA_INDEX", "").strip()
        torch_cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        if torch_index:
            torch_cmd.extend(["--index-url", torch_index])
        run_command(torch_cmd, cwd=root)

    run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)], cwd=root)


def ensure_git_repo(path: Path, url: str) -> None:
    if (path / ".git").exists():
        info(f"Repository already exists; skip clone: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", url, str(path)], cwd=path.parent)


def ensure_external_repos(paths: dict[str, Path]) -> None:
    ensure_git_repo(paths["cqt_diff_dir"], "https://github.com/eloimoliner/CQTdiff.git")
    ensure_git_repo(
        paths["audio_inpainting_dir"],
        "https://github.com/eloimoliner/audio-inpainting-diffusion.git",
    )


def ensure_checkpoint(ckpt_path: Path) -> None:
    if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
        info(f"Checkpoint already exists; skip download: {ckpt_path}")
        return

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    info(f"Downloading checkpoint to {ckpt_path}")
    try:
        urllib.request.urlretrieve(CKPT_URL, tmp_path)
        tmp_path.replace(ckpt_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        fail(
            "Gagal download checkpoint MusicNet CQTdiff+.\n"
            f"URL: {CKPT_URL}\n"
            f"Target: {ckpt_path}\n"
            f"Detail: {exc}"
        )


def ensure_musicnet_dataset(dataset_dir: Path) -> None:
    """Same dataset location and download/extract behavior as code_final_run_v2.py."""
    audio_dir = dataset_dir / "audio"
    if audio_dir.exists() and len(list(audio_dir.iterdir())) > 10:
        info(f"MusicNet dataset already exists; skip download: {audio_dir}")
        ensure_musicnet_metadata(dataset_dir)
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dataset_dir / "musicnet.tar.gz"

    if tar_path.exists() and tar_path.stat().st_size > 10_000_000_000:
        info(f"Using existing MusicNet archive: {tar_path}")
    else:
        info("Downloading MusicNet audio files (~11GB archive)")

        def progress_hook(count: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            percent = min(count * block_size * 100 / total_size, 100)
            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(MUSICNET_URL, tar_path, progress_hook)
        print("\n  Download selesai.", flush=True)

    info(f"Extracting MusicNet WAV files to {audio_dir}")
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda items, desc=None: items

    with tarfile.open(tar_path, "r:gz") as tar:
        wav_members = [member for member in tar.getmembers() if member.name.endswith(".wav")]
        for member in tqdm(wav_members, desc="Extracting"):
            tar.extract(member, audio_dir)

    info(f"Dataset berhasil diekstrak ke {audio_dir}")
    ensure_musicnet_metadata(dataset_dir)


def ensure_musicnet_metadata(dataset_dir: Path) -> None:
    metadata_path = dataset_dir / "musicnet_metadata.csv"
    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        return
    try:
        info(f"Downloading MusicNet metadata to {metadata_path}")
        urllib.request.urlretrieve(MUSICNET_METADATA_URL, metadata_path)
    except Exception as exc:
        info(f"MusicNet metadata tidak bisa didownload; lanjut tanpa metadata. Detail: {exc}")


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def ensure_stage_link(expected: Path, source: Path, label: str) -> None:
    """Expose root-level preprocessed/masked folders at the stage path used by the main pipeline."""
    if not source.is_dir():
        fail(
            f"Folder {label} asli tidak ditemukan: {source}\n"
            "Sediakan folder ini di root Drive, atau set env "
            f"BASELINE_EVAL_{label.upper()}_DIR ke path yang benar."
        )
    if _same_path(expected, source):
        return

    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists() or expected.is_symlink():
        if expected.is_symlink() and _same_path(expected, source):
            info(f"Stage {label} link already OK: {expected} -> {source}")
            return
        if expected.is_dir() and not any(expected.iterdir()):
            expected.rmdir()
        else:
            info(f"Stage {label} path already exists; using it as-is: {expected}")
            return

    try:
        expected.symlink_to(source, target_is_directory=True)
        info(f"Created stage {label} link: {expected} -> {source}")
    except OSError as exc:
        fail(
            f"Gagal membuat symlink untuk {label}.\n"
            f"Source: {source}\nExpected by pipeline: {expected}\nDetail: {exc}"
        )


def prepare_stage_layout(paths: dict[str, Path]) -> None:
    ensure_stage_link(paths["preprocessed"], paths["source_preprocessed"], "preprocessed")
    ensure_stage_link(paths["masked"], paths["source_masked"], "masked")


def _first_missing(paths: list[Path], limit: int = 10) -> str:
    shown = "\n".join(f"  - {path}" for path in paths[:limit])
    if len(paths) > limit:
        shown += f"\n  ... dan {len(paths) - limit} path lain"
    return shown


def validate_preprocessing_config(config_path: Path, metadata_path: Path) -> None:
    if not config_path.exists():
        fail(
            "preprocessing_config.json tidak ditemukan. Eval-only dihentikan supaya "
            f"pipeline tidak membuat preprocessing baru.\nPath: {config_path}"
        )
    try:
        cached = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"preprocessing_config.json tidak bisa dibaca: {config_path}\nDetail: {exc}")

    expected = {
        "config_id": EXPERIMENT_CONFIG_ID,
        "target_sr": TARGET_SR,
        "segment_samples": SEGMENT_SAMPLES,
        "gap_durations_ms": GAP_DURATIONS_MS,
        "seed": DATASET_RANDOM_SEED,
    }
    mismatch = [
        f"{key}: cached={cached.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if cached.get(key) != value
    ]
    if mismatch:
        fail(
            "Preprocessed artifacts tidak cocok dengan konfigurasi eval baseline ini.\n"
            + "\n".join(f"  - {item}" for item in mismatch)
            + f"\nConfig: {config_path}\nMetadata: {metadata_path}"
        )


def read_metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    try:
        with metadata_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except UnicodeDecodeError:
        with metadata_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"metadata.csv kosong atau tidak valid: {metadata_path}")
    if "clean_path" not in rows[0]:
        fail(f"metadata.csv tidak memiliki kolom clean_path: {metadata_path}")
    return rows


def repair_metadata_clean_paths(metadata_path: Path, preprocessed_dir: Path) -> None:
    rows = read_metadata_rows(metadata_path)
    needs_repair = False
    repaired = []

    for row in rows:
        clean_path = Path(row["clean_path"])
        candidate = preprocessed_dir / clean_path.name
        if not clean_path.exists() and candidate.exists():
            row = dict(row)
            row["clean_path"] = str(candidate)
            needs_repair = True
        repaired.append(row)

    if not needs_repair:
        return

    backup_path = metadata_path.with_suffix(metadata_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(metadata_path.read_text(encoding="utf-8"), encoding="utf-8")

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(repaired[0].keys()))
        writer.writeheader()
        writer.writerows(repaired)
    info(f"metadata.csv clean_path repaired for Colab paths. Backup: {backup_path}")


def validate_eval_artifacts(paths: dict[str, Path]) -> None:
    required_dirs = [paths["dataset"], paths["preprocessed"], paths["masked"]]
    missing_dirs = [path for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        fail(
            "Folder dataset/preprocessed/masked yang dibutuhkan belum ada:\n"
            + _first_missing(missing_dirs)
            + "\nJalankan preprocessing/restore stage dulu, lalu ulangi eval baseline."
        )

    metadata_path = paths["preprocessed"] / "metadata.csv"
    config_path = paths["preprocessed"] / "preprocessing_config.json"
    if not metadata_path.exists():
        fail(f"metadata.csv preprocessed tidak ditemukan: {metadata_path}")
    validate_preprocessing_config(config_path, metadata_path)
    repair_metadata_clean_paths(metadata_path, paths["preprocessed"])

    gap_dirs = [paths["masked"] / f"gap_{gap_ms}ms" for gap_ms in GAP_DURATIONS_MS]
    missing_gap_dirs = [path for path in gap_dirs if not path.is_dir()]
    if missing_gap_dirs:
        fail("Folder masked per gap belum lengkap:\n" + _first_missing(missing_gap_dirs))

    rows = read_metadata_rows(metadata_path)
    missing_clean = []
    missing_masked = []
    for row in rows:
        clean_path = Path(row["clean_path"])
        if not clean_path.exists():
            missing_clean.append(clean_path)
        filename = clean_path.name
        for gap_ms in GAP_DURATIONS_MS:
            masked_path = paths["masked"] / f"gap_{gap_ms}ms" / filename
            if not masked_path.exists():
                missing_masked.append(masked_path)

    if missing_clean:
        fail("File clean preprocessed dari metadata tidak ditemukan:\n" + _first_missing(missing_clean))
    if missing_masked:
        fail("File masked yang dibutuhkan metadata tidak ditemukan:\n" + _first_missing(missing_masked))

    info(
        f"Eval artifacts OK: {len(rows)} clean segments, "
        f"{len(GAP_DURATIONS_MS)} gap folders."
    )


def configure_environment(paths: dict[str, Path]) -> None:
    os.environ["PROJECT_ROOT"] = str(paths["root"])
    os.environ["MUSIC_INPAINTING_ROOT"] = str(paths["base_root"])
    os.environ["CQT_DIFF_DIR"] = str(paths["cqt_diff_dir"])
    os.environ["AUDIO_INPAINTING_DIR"] = str(paths["audio_inpainting_dir"])
    os.environ["AUDIO_INPAINTING_CQTDIFF_WEIGHTS"] = str(paths["ckpt_path"])
    os.environ["OFFICIAL_CQTDIFF_ADAPTER"] = "official_audio_inpainting_cqtdiff_adapter"
    os.environ["RUN_PHASE"] = "eval"
    os.environ["RUN_MODELS"] = "baseline_cqtdiff"
    os.environ.setdefault("PIPELINE_CPU_THREADS", "1")
    os.environ.setdefault("PIPELINE_TORCH_THREADS", "1")
    os.environ.setdefault("PIPELINE_NUM_WORKERS", "2")
    os.environ.setdefault("CQTDIFF_DIFFUSION_STEPS", "35")
    os.environ.setdefault("CQTDIFF_SIGMA_MIN", "1e-4")
    os.environ.setdefault("CQTDIFF_SIGMA_MAX", "1.0")
    os.environ.setdefault("CQTDIFF_SIGMA_DATA", "0.063")
    os.environ.setdefault("CQTDIFF_SCHURN", "10")

    for value in [paths["root"], paths["cqt_diff_dir"], paths["audio_inpainting_dir"]]:
        text = str(value)
        if text not in sys.path:
            sys.path.insert(0, text)


def validate_cuda() -> None:
    try:
        import torch
    except Exception as exc:
        fail(f"PyTorch tidak bisa di-import setelah install dependency: {exc}")
    if not torch.cuda.is_available():
        fail(
            "CUDA GPU tidak aktif. code_final_run_v2.py baseline eval memakai device='cuda'. "
            "Aktifkan GPU/runtime CUDA dulu."
        )
    info(f"CUDA OK: {torch.cuda.get_device_name(0)}")


def print_path_summary(paths: dict[str, Path]) -> None:
    info("Path configuration:")
    info(f"  project root      : {paths['root']}")
    info(f"  Drive/data root   : {paths['base_root']}")
    info(f"  stage root        : {paths['stage_root']}")
    info(f"  raw dataset       : {paths['dataset']}")
    info(f"  source preprocessed: {paths['source_preprocessed']}")
    info(f"  source masked      : {paths['source_masked']}")
    info(f"  pipeline preprocessed: {paths['preprocessed']}")
    info(f"  pipeline masked      : {paths['masked']}")
    info(f"  official ckpt     : {paths['ckpt_path']}")
    info(f"  external repos    : {paths['external']}")


def run_baseline_eval(paths: dict[str, Path]) -> None:
    main_py = paths["main_py"]
    if not main_py.exists():
        fail(f"File pipeline utama tidak ditemukan: {main_py}")

    sys.argv = [str(main_py), "--phase", "eval", "--models", "baseline_cqtdiff"]
    info("Starting baseline eval: python code_final_run_v2.py --phase eval --models baseline_cqtdiff")
    runpy.run_path(str(main_py), run_name="__main__")


def main() -> None:
    maybe_mount_google_drive()
    paths = path_config()
    print_path_summary(paths)
    configure_environment(paths)
    ensure_musicnet_dataset(paths["dataset"])
    prepare_stage_layout(paths)
    validate_eval_artifacts(paths)
    ensure_python_requirements(paths["root"])
    ensure_external_repos(paths)
    ensure_checkpoint(paths["ckpt_path"])
    validate_cuda()
    run_baseline_eval(paths)


if __name__ == "__main__":
    main()
