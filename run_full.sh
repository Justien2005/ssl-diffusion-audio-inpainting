#!/usr/bin/env bash
# ============================================================
# FULL RUN: Training & evaluasi semua model untuk thesis.
#
# Konfigurasi:
# - 35 diffusion steps (full quality)
# - 250 eval samples
# - Semua 5 model: baseline + 4 hybrid
#
# Urutan:
# 1. Baseline CQT-Diff+ (eval only, pretrained)
# 2. CLAP + CQT-Diff+ (train + eval)
# 3. CLAP + MAID (train + eval)
# 4. AudioMAE + CQT-Diff+ (train + eval)
# 5. AudioMAE + MAID (train + eval)
#
# Perkiraan waktu: ~4-8 jam (tergantung GPU)
# Jalankan: bash run_full.sh
# ============================================================

set -euo pipefail
source env_instance.sh

echo "============================================"
echo " FULL RUN: All Models Training & Evaluation"
echo "============================================"
echo ""

# Full quality config
export CQTDIFF_DIFFUSION_STEPS=35
export N_EVAL_SAMPLES=250

python code_final_run_v2.py \
    --phase train,eval,summary \
    --models all \
    --auto-stop

echo ""
echo "============================================"
echo " FULL RUN SELESAI"
echo "============================================"
