#!/usr/bin/env bash
# Separate, manually invoked post-training inference + evaluation.
# Usage: bash infer_eval_seeds.sh <dataset> <physical_gpu> [seed ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
shopt -s nullglob

if (( $# < 2 )); then
  echo "Usage: bash $0 <acdc|hc18|TG3K|ChestX-ray> <gpu> [seed ...]" >&2
  exit 2
fi
dataset=$1
gpu=$2
shift 2
case "$dataset" in
  acdc) slug=acdc; best_key=MRE_mm ;;
  hc18) slug=hc18; best_key=MRE_mm ;;
  TG3K) slug=tg3k; best_key=MRE ;;
  ChestX-ray) slug=chestxray; best_key=MRE_mm ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPU must be a nonnegative integer" >&2; exit 2; }
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(1 2 3); fi
config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_${slug}_256x192.py"
gt="datasets_analysis/coco_format/painter3d/${dataset}/annotations/test.json"
declare -A checkpoints=()
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[123]$ ]] || { echo "Supported seeds: 1, 2, 3" >&2; exit 2; }
  [[ ! ${checkpoints[$seed]+exists} ]] || { echo "Duplicate seed: $seed" >&2; exit 2; }
  work_dir="work_dirs/painter3d_seed${seed}/${slug}"
  if [[ ! -f "$work_dir/training_complete.txt" || ! -f "$work_dir/epoch_210.pth" ]]; then
    echo "Full training has not completed via train_seeds.sh: $work_dir" >&2
    exit 1
  fi
  candidates=("$work_dir"/best_"$best_key"_epoch_*.pth)
  if (( ${#candidates[@]} != 1 )); then
    echo "Expected exactly one validation-best checkpoint in $work_dir; found ${#candidates[@]}" >&2
    exit 1
  fi
  checkpoints[$seed]=${candidates[0]}
  if [[ -e "$work_dir/test_evaluation" ]]; then
    echo "Refusing to overwrite existing evaluation: $work_dir/test_evaluation" >&2
    exit 1
  fi
done

for seed in "${seeds[@]}"; do
  output="work_dirs/painter3d_seed${seed}/${slug}/test_evaluation"
  mkdir "$output"
  echo "Test inference: Painter-3D $dataset, seed=$seed, ${checkpoints[$seed]}"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u datasets_analysis/tools/infer_medical_keypoints.py \
    "$config" "${checkpoints[$seed]}" \
    --test-json "$gt" --out "$output/predictions.json" \
    --device cuda:0 --batch-size 8 --workers 2 2>&1 | tee "$output/inference.log"
  .venv/bin/python -u datasets_analysis/tools/evaluate_medical_keypoints.py \
    --predictions "$output/predictions.json" --gt "$gt" \
    --config "$config" --metrics MRE MSE RMSE MAE \
    --out "$output/metrics.json" 2>&1 | tee "$output/evaluation.log"
done
echo "$dataset: requested seeds' test inference and evaluation completed."
