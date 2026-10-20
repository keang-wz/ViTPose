# Seed=1 第二轮 test 推理与评价

八个任务已完成210epoch，日志和checkpoint核验seed=1。训练方法及模型选择标准与第一轮一致；当前整理的是验证集结果，下面是尚未执行的test命令。

验证结果、第一轮对照和临时验证集mean±std见 [第二轮报告](../results/seed1_training_summary/report.md)。主展示指标为MAE，checkpoint仍按验证MRE最小选择。

| 数据集 | 本轮最佳epoch |
|---|---:|
| TG3K | 210 |
| acdc | 200 |
| hc18 | 210 |
| oasis | 210 |
| Synapse-CT | 210 |
| SKI10 | 210 |
| ChestX-ray | 200 |
| LUNA16_26 | 210 |

acdc、ChestX-ray使用epoch200；其余使用epoch210。全部test输出带_seed1标识，第一轮预测、指标和可视化不会覆盖。

可分别运行下面的脚本（每个脚本先推理、后评价）：

```bash
bash datasets_analysis/commands/test_seed1/tg3k.sh
bash datasets_analysis/commands/test_seed1/acdc.sh
bash datasets_analysis/commands/test_seed1/hc18.sh
bash datasets_analysis/commands/test_seed1/oasis.sh
bash datasets_analysis/commands/test_seed1/synapse_ct.sh
bash datasets_analysis/commands/test_seed1/ski10.sh
bash datasets_analysis/commands/test_seed1/chestxray.sh
bash datasets_analysis/commands/test_seed1/luna16_26.sh
```

完整原始命令如下：
## Test推理与医学评价命令

在仓库根目录运行：

```bash
cd /home/wzk/WorkSpace/ViTPose
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

所有输出位于独立的 *_test_evaluation_seed1 目录，不覆盖第一轮。每个命令脚本包含本轮实际best_MRE推理和原test评价，输出MSE、RMSE、MAE、MRE、PCK@2/5/10/20px。CUDA_VISIBLE_DEVICES选择物理卡，进程内使用cuda:0。

### TG3K

```bash
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  work_dirs/vitpose_base_tg3k_256x192_seed1/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/TG3K/annotations/test.json \
  --out work_dirs/tg3k_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/tg3k_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/tg3k_test_evaluation_seed1/best_MRE_epoch_210_metrics.json
```

### acdc

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  work_dirs/vitpose_base_acdc_256x192_seed1/best_MRE_epoch_200.pth \
  --test-json datasets_analysis/coco_format/acdc/annotations/test.json \
  --out work_dirs/acdc_test_evaluation_seed1/best_MRE_epoch_200_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/acdc_test_evaluation_seed1/best_MRE_epoch_200_predictions.json \
  --gt datasets_analysis/coco_format/acdc/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/acdc_test_evaluation_seed1/best_MRE_epoch_200_metrics.json
```

### hc18

```bash
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  work_dirs/vitpose_base_hc18_256x192_seed1/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/hc18/annotations/test.json \
  --out work_dirs/hc18_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/hc18_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/hc18/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/hc18_test_evaluation_seed1/best_MRE_epoch_210_metrics.json
```

### oasis

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  work_dirs/vitpose_base_oasis_256x192_seed1/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/oasis/annotations/test.json \
  --out work_dirs/oasis_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/oasis_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/oasis/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/oasis_test_evaluation_seed1/best_MRE_epoch_210_metrics.json
```

### Synapse-CT

```bash
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  work_dirs/vitpose_base_synapse_ct_256x192_seed1/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/Synapse-CT/annotations/test.json \
  --out work_dirs/synapse_ct_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/synapse_ct_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/Synapse-CT/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/synapse_ct_test_evaluation_seed1/best_MRE_epoch_210_metrics.json
```

### SKI10

```bash
CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  work_dirs/vitpose_base_ski10_256x192_seed1/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/SKI10/annotations/test.json \
  --out work_dirs/ski10_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/ski10_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/SKI10/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/ski10_test_evaluation_seed1/best_MRE_epoch_210_metrics.json
```

### ChestX-ray

```bash
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
```

### LUNA16_26

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  work_dirs/vitpose_base_luna16_26_256x192_seed1/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/LUNA16_26/annotations/test.json \
  --out work_dirs/luna16_26_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/luna16_26_test_evaluation_seed1/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/LUNA16_26/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/luna16_26_test_evaluation_seed1/best_MRE_epoch_210_metrics.json
```

顺序运行八个测试任务：

```bash
bash datasets_analysis/commands/test_seed1/all.sh
```

这条命令及上述单任务脚本均未由本次整理执行。输出文件已存在时，原推理/评价脚本会拒绝覆盖。

文件清单：training_results.json、validation_best.csv、first_run_comparison.csv、validation_history.csv、validation_two_seed_mean_std.csv、validation_mre_comparison.png、report.md；命令目录包含八份脚本和all.sh。

