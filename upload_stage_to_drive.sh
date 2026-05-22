#!/usr/bin/env bash
set -euo pipefail

# Package and upload the full native CQT-Diff training stage to Google Drive.
#
# This uses rclone because Google Drive public-editor links are not reliable
# unauthenticated upload targets for CLI scripts. Configure rclone once:
#   rclone config
# then run:
#   bash upload_stage_to_drive.sh
#
# Optional overrides:
#   RCLONE_REMOTE=gdrive bash upload_stage_to_drive.sh
#   MUSIC_INPAINTING_ROOT=/workspace/music_inpainting bash upload_stage_to_drive.sh
#   PIPELINE_STAGE_NAME=code_v3_final_run_native_cqt bash upload_stage_to_drive.sh
#   DRIVE_FOLDER_ID=187wcDBfOmjZcVpJ_cu8tmh74EfpwEltp bash upload_stage_to_drive.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_DIR}"
MUSIC_ROOT="${MUSIC_INPAINTING_ROOT:-$PROJECT_ROOT/music_inpainting}"
STAGE_NAME="${PIPELINE_STAGE_NAME:-code_v3_final_run_native_cqt}"
STAGE_DIR="$MUSIC_ROOT/training_stages/$STAGE_NAME"
DRIVE_FOLDER_ID="${DRIVE_FOLDER_ID:-187wcDBfOmjZcVpJ_cu8tmh74EfpwEltp}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
EXPORT_DIR="${EXPORT_DIR:-$REPO_DIR/stage_exports}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ZIP_PATH="$EXPORT_DIR/${STAGE_NAME}_${TIMESTAMP}.zip"
SHA_PATH="$ZIP_PATH.sha256"

cd "$REPO_DIR"

if [ ! -d "$STAGE_DIR" ]; then
  echo "ERROR: stage directory tidak ditemukan:"
  echo "  $STAGE_DIR"
  echo "Pastikan MUSIC_INPAINTING_ROOT dan PIPELINE_STAGE_NAME benar."
  exit 1
fi

for required_dir in checkpoints preprocessed masked; do
  if [ ! -d "$STAGE_DIR/$required_dir" ]; then
    echo "ERROR: missing required directory: $STAGE_DIR/$required_dir"
    exit 1
  fi
done

mkdir -p "$EXPORT_DIR"

echo "==> Creating stage zip"
echo "    source: $STAGE_DIR"
echo "    zip   : $ZIP_PATH"

if command -v zip >/dev/null 2>&1; then
  (
    cd "$MUSIC_ROOT/training_stages"
    zip -qr "$ZIP_PATH" "$STAGE_NAME"
  )
else
  python - "$STAGE_DIR" "$ZIP_PATH" "$STAGE_NAME" <<'PY'
import os
import sys
import zipfile

stage_dir, zip_path, stage_name = sys.argv[1:4]
root = os.path.dirname(stage_dir)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for current, _, files in os.walk(stage_dir):
        for name in files:
            path = os.path.join(current, name)
            arcname = os.path.join(stage_name, os.path.relpath(path, stage_dir))
            zf.write(path, arcname)
PY
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ZIP_PATH" > "$SHA_PATH"
else
  python - "$ZIP_PATH" "$SHA_PATH" <<'PY'
import hashlib
import sys

zip_path, sha_path = sys.argv[1:3]
h = hashlib.sha256()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
with open(sha_path, "w", encoding="utf-8") as f:
    f.write(f"{h.hexdigest()}  {zip_path}\n")
PY
fi

echo "==> Zip created"
ls -lh "$ZIP_PATH" "$SHA_PATH"

if ! command -v rclone >/dev/null 2>&1; then
  echo
  echo "ERROR: rclone belum terinstall."
  echo "Install di Ubuntu/Vast.ai:"
  echo "  sudo apt-get update && sudo apt-get install -y rclone"
  echo "Lalu login/config remote:"
  echo "  rclone config"
  echo
  echo "Zip tetap sudah dibuat lokal:"
  echo "  $ZIP_PATH"
  exit 1
fi

if ! rclone listremotes | grep -Fxq "${RCLONE_REMOTE}:"; then
  echo
  echo "ERROR: rclone remote '${RCLONE_REMOTE}:' belum ada."
  echo "Buat dulu:"
  echo "  rclone config"
  echo "Atau set remote lain:"
  echo "  RCLONE_REMOTE=nama_remote bash upload_stage_to_drive.sh"
  exit 1
fi

echo "==> Uploading to Google Drive folder id: $DRIVE_FOLDER_ID"
rclone copy "$ZIP_PATH" "${RCLONE_REMOTE}:" --drive-root-folder-id "$DRIVE_FOLDER_ID" --progress
rclone copy "$SHA_PATH" "${RCLONE_REMOTE}:" --drive-root-folder-id "$DRIVE_FOLDER_ID" --progress

echo "==> Upload complete"
echo "Uploaded files:"
echo "  $(basename "$ZIP_PATH")"
echo "  $(basename "$SHA_PATH")"
echo
echo "Folder:"
echo "  https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID"
