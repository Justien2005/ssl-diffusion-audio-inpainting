# Thesis Audio Inpainting Vast.ai Bundle

This folder is the deployment bundle for running the thesis notebooks on a Vast.ai instance.

## Files

- `code_v3_smoke_test.ipynb`: quick end-to-end stage.
- `code_v3_instance_test.ipynb`: batch-size 8 instance test.
- `code_v3_final_pipeline.ipynb`: final 50% dataset, batch-size 8, 100 epochs.
- `code_v3.ipynb`: original current pipeline copy.
- `requirements.txt`: Python dependencies.
- `music_inpainting/`: default dataset/output root.
- `external/`: external repositories, including CQT-Diff after first notebook run.

## Vast.ai Setup

Run from inside this `thesisall` folder:

```bash
pip install -r requirements.txt
```

Then open Jupyter and run one notebook from top to bottom. The copied notebooks are patched to use:

```text
PROJECT_ROOT = current working directory
MUSIC_INPAINTING_ROOT = ./music_inpainting
CQT_DIFF_DIR = ./external/CQT_diff
```

You can override these with environment variables before launching Jupyter.

## Dataset Strategy

Recommended for Vast.ai: use a persistent volume or manual upload for the MusicNet archive/audio.

Supported locations:

- `music_inpainting/dataset/musicnet.tar.gz`
- `music_inpainting/dataset/audio/*.wav`

If neither exists, the notebook attempts to download MusicNet automatically from Zenodo. Auto-download is convenient but less reliable for long GPU runs because the archive is large.

## Persistence Warning

Before destroying a Vast.ai instance, back up:

- `music_inpainting/training_stages/*/checkpoints`
- `music_inpainting/training_stages/*/results`
- optionally `music_inpainting/training_stages/*/preprocessed`
- optionally `music_inpainting/training_stages/*/masked`

Checkpoints and results are not safe if they only live on ephemeral instance storage.

## Logging

The staged notebooks log:

- `nvidia-smi -q`
- GPU name, VRAM, CUDA version
- epoch training time
- total training time
- per-model timing summary for all 5 configurations
