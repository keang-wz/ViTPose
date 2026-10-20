#!/usr/bin/env bash
# Third round: seed=2. Prepared only, never executed by the audit.
# Running the entire file executes eight training jobs sequentially.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# TG3K
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  --work-dir work_dirs/vitpose_base_tg3k_256x192_seed2 \
  --seed 2

# acdc
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/vitpose_base_acdc_256x192_seed2 \
  --seed 2

# hc18
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/vitpose_base_hc18_256x192_seed2 \
  --seed 2

# oasis
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  --work-dir work_dirs/vitpose_base_oasis_256x192_seed2 \
  --seed 2

# Synapse-CT
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  --work-dir work_dirs/vitpose_base_synapse_ct_256x192_seed2 \
  --seed 2

# SKI10
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  --work-dir work_dirs/vitpose_base_ski10_256x192_seed2 \
  --seed 2

# ChestX-ray
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/vitpose_base_chestxray_256x192_seed2 \
  --seed 2

# LUNA16_26
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  --work-dir work_dirs/vitpose_base_luna16_26_256x192_seed2 \
  --seed 2
