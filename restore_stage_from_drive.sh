#!/usr/bin/env bash
set -euo pipefail

# Restore a previously exported training stage into this cloned repository.
#
# Usage:
#   bash setup_instance.sh
#   bash restore_stage_from_drive.sh
#
# Optional overrides:
#   STAGE_ZIP_URL="https://drive.google.com/..." bash restore_stage_from_drive.sh
#   STAGE_ZIP_PATH="/tmp/code_v3_final_run_native_cqt.zip" bash restore_stage_from_drive.sh
#   MUSIC_INPAINTING_ROOT="/workspace/music_inpainting" bash restore_stage_from_drive.sh
#   FORCE_OVERWRITE=1 bash restore_stage_from_drive.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_DIR}"
MUSIC_ROOT="${MUSIC_INPAINTING_ROOT:-$PROJECT_ROOT/music_inpainting}"
STAGE_NAME="${PIPELINE_STAGE_NAME:-code_v3_final_run_native_cqt}"
TARGET_STAGE_DIR="$MUSIC_ROOT/training_stages/$STAGE_NAME"
STAGE_ZIP_URL="${STAGE_ZIP_URL:-https://drive.google.com/file/d/1dO25s3cuKbXbnjJxy680WSr51H62yP2I/view?usp=sharing}"
STAGE_ZIP_PATH="${STAGE_ZIP_PATH:-$REPO_DIR/${STAGE_NAME}.zip}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

cd "$REPO_DIR"

if [ -d "$REPO_DIR/venv" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/venv/bin/activate"
fi

mkdir -p "$MUSIC_ROOT/training_stages"

if [ ! -f "$STAGE_ZIP_PATH" ]; then
  echo "==> Downloading stage zip"
  echo "    url: $STAGE_ZIP_URL"
  echo "    out: $STAGE_ZIP_PATH"
  DRIVE_FILE_ID=""
  if [[ "$STAGE_ZIP_URL" =~ /d/([^/]+) ]]; then
    DRIVE_FILE_ID="${BASH_REMATCH[1]}"
  elif [[ "$STAGE_ZIP_URL" =~ id=([^&]+) ]]; then
    DRIVE_FILE_ID="${BASH_REMATCH[1]}"
  fi
  if command -v gdown >/dev/null 2>&1; then
    if gdown --help 2>/dev/null | grep -q -- "--fuzzy"; then
      gdown --fuzzy "$STAGE_ZIP_URL" -O "$STAGE_ZIP_PATH"
    elif [ -n "$DRIVE_FILE_ID" ]; then
      gdown "https://drive.google.com/uc?id=$DRIVE_FILE_ID" -O "$STAGE_ZIP_PATH"
    else
      gdown "$STAGE_ZIP_URL" -O "$STAGE_ZIP_PATH"
    fi
  else
    if python -m gdown --help 2>/dev/null | grep -q -- "--fuzzy"; then
      python -m gdown --fuzzy "$STAGE_ZIP_URL" -O "$STAGE_ZIP_PATH"
    elif [ -n "$DRIVE_FILE_ID" ]; then
      python -m gdown "https://drive.google.com/uc?id=$DRIVE_FILE_ID" -O "$STAGE_ZIP_PATH"
    else
      python -m gdown "$STAGE_ZIP_URL" -O "$STAGE_ZIP_PATH"
    fi
  fi
else
  echo "==> Using existing zip: $STAGE_ZIP_PATH"
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "==> Extracting zip"
unzip -q "$STAGE_ZIP_PATH" -d "$TMP_DIR"

SOURCE_STAGE_DIR=""
if [ -d "$TMP_DIR/$STAGE_NAME" ]; then
  SOURCE_STAGE_DIR="$TMP_DIR/$STAGE_NAME"
elif [ -d "$TMP_DIR/training_stages/$STAGE_NAME" ]; then
  SOURCE_STAGE_DIR="$TMP_DIR/training_stages/$STAGE_NAME"
elif [ -f "$TMP_DIR/metadata.csv" ] || [ -d "$TMP_DIR/preprocessed" ]; then
  SOURCE_STAGE_DIR="$TMP_DIR"
else
  found="$(find "$TMP_DIR" -type d -name "$STAGE_NAME" | head -n 1 || true)"
  if [ -n "$found" ]; then
    SOURCE_STAGE_DIR="$found"
  fi
fi

if [ -z "$SOURCE_STAGE_DIR" ] || [ ! -d "$SOURCE_STAGE_DIR" ]; then
  echo "ERROR: tidak menemukan folder stage '$STAGE_NAME' di dalam zip."
  echo "Isi zip yang didukung:"
  echo "  - $STAGE_NAME/..."
  echo "  - training_stages/$STAGE_NAME/..."
  echo "  - langsung berisi preprocessed/, masked/, checkpoints/, dst."
  exit 1
fi

if [ "$FORCE_OVERWRITE" = "1" ] && [ -d "$TARGET_STAGE_DIR" ]; then
  echo "==> FORCE_OVERWRITE=1, deleting existing target stage"
  rm -rf "$TARGET_STAGE_DIR"
fi

mkdir -p "$TARGET_STAGE_DIR"

echo "==> Restoring stage"
echo "    from: $SOURCE_STAGE_DIR"
echo "    to  : $TARGET_STAGE_DIR"
cp -a "$SOURCE_STAGE_DIR"/. "$TARGET_STAGE_DIR"/

echo "==> Checking restored structure"
for required_dir in preprocessed masked checkpoints; do
  if [ ! -d "$TARGET_STAGE_DIR/$required_dir" ]; then
    echo "ERROR: missing required directory: $TARGET_STAGE_DIR/$required_dir"
    exit 1
  fi
done

for optional_dir in logs results plots outputs; do
  mkdir -p "$TARGET_STAGE_DIR/$optional_dir"
done

if [ ! -f "$TARGET_STAGE_DIR/preprocessed/metadata.csv" ]; then
  echo "ERROR: missing metadata: $TARGET_STAGE_DIR/preprocessed/metadata.csv"
  exit 1
fi

touch "$TARGET_STAGE_DIR/preprocessed/.done"

echo "==> Existing checkpoint folders"
find "$TARGET_STAGE_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf "    %f\n" | sort || true

echo "==> Restore complete"
echo "Stage directory:"
echo "  $TARGET_STAGE_DIR"
echo
echo "Next examples:"
echo "  source env_instance.sh"
echo "  python code_final_run_v2.py --phase train --models clap_maid"
echo "  python code_final_run_v2.py --phase train --models audiomae_cqtdiff"
echo "  python code_final_run_v2.py --phase train --models audiomae_maid"
