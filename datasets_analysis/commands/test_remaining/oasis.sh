#!/usr/bin/env bash
# Test inference/evaluation only. This file was generated, not executed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  work_dirs/vitpose_base_oasis_256x192/best_MRE_epoch_200.pth \
  --test-json datasets_analysis/coco_format/oasis/annotations/test.json \
  --out work_dirs/oasis_test_evaluation/best_MRE_epoch_200_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/oasis_test_evaluation/best_MRE_epoch_200_predictions.json \
  --gt datasets_analysis/coco_format/oasis/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/oasis_test_evaluation/best_MRE_epoch_200_metrics.json
