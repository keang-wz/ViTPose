#!/usr/bin/env bash
# Seed=1 second run. Prepared only; not executed by the assistant.
# bash-ing the entire file runs tasks sequentially.
# To run concurrently, copy one command into each initialized terminal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# TG3K
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  --work-dir work_dirs/vitpose_base_tg3k_256x192_seed1 \
  --seed 1

# acdc
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/vitpose_base_acdc_256x192_seed1 \
  --seed 1

# hc18
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/vitpose_base_hc18_256x192_seed1 \
  --seed 1

# oasis
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  --work-dir work_dirs/vitpose_base_oasis_256x192_seed1 \
  --seed 1

# Synapse-CT
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  --work-dir work_dirs/vitpose_base_synapse_ct_256x192_seed1 \
  --seed 1

# SKI10
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  --work-dir work_dirs/vitpose_base_ski10_256x192_seed1 \
  --seed 1

# ChestX-ray
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/vitpose_base_chestxray_256x192_seed1 \
  --seed 1

# LUNA16_26
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  --work-dir work_dirs/vitpose_base_luna16_26_256x192_seed1 \
  --seed 1
