#!/usr/bin/env bash
# Prepared commands only. Running this file WILL start seven training jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
launch_dir="work_dirs/medical_baseline_launch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$launch_dir"
pids=()

# acdc: 516 training images, 14 keypoints.
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/vitpose_base_acdc_256x192 --seed 42 >"$launch_dir/acdc.log" 2>&1 &
pids+=("$!")

# hc18: 210 training images, 13 keypoints.
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/vitpose_base_hc18_256x192 --seed 42 >"$launch_dir/hc18.log" 2>&1 &
pids+=("$!")

# oasis: 942 training images, 8 keypoints.
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  --work-dir work_dirs/vitpose_base_oasis_256x192 --seed 42 >"$launch_dir/oasis.log" 2>&1 &
pids+=("$!")

# Synapse-CT: 2617 training images, 13 keypoints.
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  --work-dir work_dirs/vitpose_base_synapse_ct_256x192 --seed 42 >"$launch_dir/synapse_ct.log" 2>&1 &
pids+=("$!")

# SKI10: 2628 training images, 8 keypoints.
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  --work-dir work_dirs/vitpose_base_ski10_256x192 --seed 42 >"$launch_dir/ski10.log" 2>&1 &
pids+=("$!")

# ChestX-ray: 164 training images, 32 keypoints.
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/vitpose_base_chestxray_256x192 --seed 42 >"$launch_dir/chestxray.log" 2>&1 &
pids+=("$!")

# LUNA16_26: 3749 training images, 26 keypoints.
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  --work-dir work_dirs/vitpose_base_luna16_26_256x192 --seed 42 >"$launch_dir/luna16_26.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
