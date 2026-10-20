#!/usr/bin/env bash
# Prepared test inference/evaluation commands only; not executed by the summarizer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  work_dirs/vitpose_base_synapse_ct_256x192_seed2/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/Synapse-CT/annotations/test.json \
  --out work_dirs/synapse_ct_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/synapse_ct_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/Synapse-CT/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/synapse_ct_test_evaluation_seed2/best_MRE_epoch_210_metrics.json
