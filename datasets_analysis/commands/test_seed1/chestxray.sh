#!/usr/bin/env bash
# Prepared test inference/evaluation commands only; not executed by the summarizer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  work_dirs/vitpose_base_chestxray_256x192_seed1/best_MRE_epoch_200.pth \
  --test-json datasets_analysis/coco_format/ChestX-ray/annotations/test.json \
  --out work_dirs/chestxray_test_evaluation_seed1/best_MRE_epoch_200_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/chestxray_test_evaluation_seed1/best_MRE_epoch_200_predictions.json \
  --gt datasets_analysis/coco_format/ChestX-ray/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/chestxray_test_evaluation_seed1/best_MRE_epoch_200_metrics.json
