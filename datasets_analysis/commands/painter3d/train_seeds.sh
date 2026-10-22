#!/usr/bin/env bash
# Manual entry point only. This script does not create screen sessions or evaluate.
# Usage: bash train_seeds.sh <dataset> <physical_gpu> [seed ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

if (( $# < 2 )); then
  echo "Usage: bash $0 <acdc|hc18|TG3K|ChestX-ray> <gpu> [seed ...]" >&2
  exit 2
fi
dataset=$1
gpu=$2
shift 2
case "$dataset" in
  acdc) slug=acdc ;;
  hc18) slug=hc18 ;;
  TG3K) slug=tg3k ;;
  ChestX-ray) slug=chestxray ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPU must be a nonnegative integer" >&2; exit 2; }
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(1 2 3); fi
config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_${slug}_256x192.py"
declare -A seen=()
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[123]$ ]] || { echo "Supported seeds: 1, 2, 3" >&2; exit 2; }
  [[ ! ${seen[$seed]+exists} ]] || { echo "Duplicate seed: $seed" >&2; exit 2; }
  seen[$seed]=1
  work_dir="work_dirs/painter3d_seed${seed}/${slug}"
  if [[ -e "$work_dir" ]]; then
    echo "Refusing to overwrite existing run: $work_dir" >&2
    echo "To run only unfinished seeds, pass their numbers after the GPU argument." >&2
    exit 1
  fi
done

for seed in "${seeds[@]}"; do
  work_dir="work_dirs/painter3d_seed${seed}/${slug}"
  mkdir -p "$(dirname "$work_dir")"
  mkdir "$work_dir"
  echo "Training Painter-3D $dataset, seed=$seed, GPU=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u tools/train.py "$config" \
    --work-dir "$work_dir" --seed "$seed" 2>&1 | tee "$work_dir/train_console.log"
  # Only a successful full training command reaches this point (pipefail).
  printf 'dataset=%s\nseed=%s\nconfig=%s\n' "$dataset" "$seed" "$config" \
    > "$work_dir/training_complete.txt"
done
echo "$dataset: requested seeds completed. Inference/evaluation must be started manually."
