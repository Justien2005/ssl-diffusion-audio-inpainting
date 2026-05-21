#!/usr/bin/env bash
# ============================================================
# TEST RUN: Validasi pipeline end-to-end sebelum full run.
#
# Konfigurasi minimal:
# - 5 epoch (cukup buat validasi loss turun)
# - 10 eval samples (cepat, cuma buat cek output non-silent)
# - Hanya baseline + 1 hybrid CQT-Diff+ model
# - DATASET_FRACTION tetap 0.5 (sesuai pipeline)
#
# Perkiraan waktu: ~15-30 menit (tergantung GPU)
# Jalankan: bash run_test.sh
# ============================================================

set -euo pipefail
source env_instance.sh

echo "============================================"
echo " TEST RUN: Pipeline Validation"
echo "============================================"
echo "Tujuan: Validasi bahwa baseline diffusion + hybrid SSL-conditioned diffusion bekerja."
echo ""

# Override config buat test run cepat
export CQTDIFF_DIFFUSION_STEPS=10  # 10 step (bukan 35) biar cepat
export N_EVAL_SAMPLES=10            # 10 samples eval (bukan 250)

echo "==> Phase 1: Baseline eval only (no training)"
python code_final_run_v2.py \
    --phase eval \
    --models baseline_cqtdiff

echo ""
echo "==> Phase 2: Train + eval clap_cqtdiff (10 epochs, reduced eval)"
python code_final_run_v2.py \
    --phase train,eval \
    --models clap_cqtdiff

echo ""
echo "============================================"
echo " TEST RUN SELESAI"
echo "============================================"
echo ""
echo "Cek hasilnya:"
echo "  1. Baseline: pastikan gap RMS non-zero di results CSV"
echo "  2. clap_cqtdiff: pastikan training loss turun + metrics > baseline"
echo ""
echo "Jika OK, jalankan full run:"
echo "  bash run_full.sh"
