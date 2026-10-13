#!/usr/bin/env bash
# Running this file executes test inference/evaluation, never training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  work_dirs/tg3k_smoke/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/TG3K/annotations/test.json \
  --out work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_metrics.json
