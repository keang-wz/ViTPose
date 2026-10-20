# Seed=2 第三轮 test 推理与评价

八个任务均完成210 epoch，配置与第一轮一致；日志和checkpoint seed=2正确。优化器步数匹配训练量，epoch10至210的157个参数张量全部发生更新（另有6个BatchNorm buffer，不计入参数数量），无训练错误记录。GT划分、关键点数量和原始图像路径通过检查。

checkpoint选择仍为验证集MRE最小，MAE作为主要展示指标。TG3K、LUNA16_26使用epoch190，acdc使用epoch110，其余使用epoch210；不能用latest替代这些best权重。

训练验证结果见 [第三轮报告](../results/seed2_training_summary/report.md)，真实参数更新核验见 [checkpoint_update_audit.json](../results/seed2_training_summary/checkpoint_update_audit.json)。下面命令尚未执行。test完成后才可与前两轮test指标汇总为三个seed的mean±std，ddof=1。

分别运行每个脚本（内部先推理、后评价）：

```bash
bash datasets_analysis/commands/test_seed2/tg3k.sh
bash datasets_analysis/commands/test_seed2/acdc.sh
bash datasets_analysis/commands/test_seed2/hc18.sh
bash datasets_analysis/commands/test_seed2/oasis.sh
bash datasets_analysis/commands/test_seed2/synapse_ct.sh
bash datasets_analysis/commands/test_seed2/ski10.sh
bash datasets_analysis/commands/test_seed2/chestxray.sh
bash datasets_analysis/commands/test_seed2/luna16_26.sh
```

顺序执行全部任务：

```bash
bash datasets_analysis/commands/test_seed2/all.sh
```

所有预测与评价另存 `work_dirs/<slug>_test_evaluation_seed2/`，已存在输出时拒绝覆盖。

## Test推理与医学评价命令

在仓库根目录运行：

```bash
cd /home/wzk/WorkSpace/ViTPose
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

所有输出位于独立的 *_test_evaluation_seed2 目录，不覆盖其他轮次。每个脚本包含本轮实际best_MRE推理和原test评价，输出MSE、RMSE、MAE、MRE、PCK@2/5/10/20px。CUDA_VISIBLE_DEVICES选择物理卡，进程内使用cuda:0。

### TG3K

```bash
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  work_dirs/vitpose_base_tg3k_256x192_seed2/best_MRE_epoch_190.pth \
  --test-json datasets_analysis/coco_format/TG3K/annotations/test.json \
  --out work_dirs/tg3k_test_evaluation_seed2/best_MRE_epoch_190_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/tg3k_test_evaluation_seed2/best_MRE_epoch_190_predictions.json \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/tg3k_test_evaluation_seed2/best_MRE_epoch_190_metrics.json
```

### acdc

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  work_dirs/vitpose_base_acdc_256x192_seed2/best_MRE_epoch_110.pth \
  --test-json datasets_analysis/coco_format/acdc/annotations/test.json \
  --out work_dirs/acdc_test_evaluation_seed2/best_MRE_epoch_110_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/acdc_test_evaluation_seed2/best_MRE_epoch_110_predictions.json \
  --gt datasets_analysis/coco_format/acdc/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/acdc_test_evaluation_seed2/best_MRE_epoch_110_metrics.json
```

### hc18

```bash
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  work_dirs/vitpose_base_hc18_256x192_seed2/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/hc18/annotations/test.json \
  --out work_dirs/hc18_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/hc18_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/hc18/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/hc18_test_evaluation_seed2/best_MRE_epoch_210_metrics.json
```

### oasis

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  work_dirs/vitpose_base_oasis_256x192_seed2/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/oasis/annotations/test.json \
  --out work_dirs/oasis_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/oasis_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/oasis/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/oasis_test_evaluation_seed2/best_MRE_epoch_210_metrics.json
```

### Synapse-CT

```bash
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
```

### SKI10

```bash
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  work_dirs/vitpose_base_ski10_256x192_seed2/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/SKI10/annotations/test.json \
  --out work_dirs/ski10_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/ski10_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/SKI10/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/ski10_test_evaluation_seed2/best_MRE_epoch_210_metrics.json
```

### ChestX-ray

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  work_dirs/vitpose_base_chestxray_256x192_seed2/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/ChestX-ray/annotations/test.json \
  --out work_dirs/chestxray_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/chestxray_test_evaluation_seed2/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/ChestX-ray/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/chestxray_test_evaluation_seed2/best_MRE_epoch_210_metrics.json
```

### LUNA16_26

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  work_dirs/vitpose_base_luna16_26_256x192_seed2/best_MRE_epoch_190.pth \
  --test-json datasets_analysis/coco_format/LUNA16_26/annotations/test.json \
  --out work_dirs/luna16_26_test_evaluation_seed2/best_MRE_epoch_190_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/luna16_26_test_evaluation_seed2/best_MRE_epoch_190_predictions.json \
  --gt datasets_analysis/coco_format/LUNA16_26/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/luna16_26_test_evaluation_seed2/best_MRE_epoch_190_metrics.json
```

顺序运行八个测试任务：

```bash
bash datasets_analysis/commands/test_seed2/all.sh
```

这条命令及上述单任务脚本均未由本次整理执行。输出文件已存在时，原推理/评价脚本会拒绝覆盖。

文件清单：training_results.json、validation_best.csv、first_run_comparison.csv、validation_history.csv、validation_two_seed_mean_std.csv、validation_mre_comparison.png、report.md；命令目录包含八份脚本和all.sh。
