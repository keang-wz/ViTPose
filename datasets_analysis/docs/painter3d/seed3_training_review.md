# Painter-3D seed=3 训练核查与手动推理评价

本文保留训练完成时的核查记录。用户后续已完成测试集推理和评价，
本轮测试结果见 [seed=3 测试结果](seed3_test_results.md)，正式汇总见
[三 seed 最终测试结果](three_seed_test_results.md)。

四项训练均已完成 210 epochs，每项有完整的 21 次验证记录（epoch 10、20、…、210）。本次仅读取日志、配置及 checkpoint，没有启动训练、模型推理或测试集评价。

## 核查结论

- 日志及 best/final checkpoint 的 seed 均为 3，epoch 与 iteration 匹配；latest.pth 均指向 epoch_210.pth。
- 四个验证最优 checkpoint 均在 epoch 210，best_score 与日志一致。注意 TG3K 本轮最佳为 epoch 210，不能复用前两轮的 epoch 200 文件名。
- 最佳权重可通过 CPU weights_only 加载，所有浮点模型参数为有限值，输出头通道数分别为 14、13、5、32。
- 完整日志没有 Traceback、CUDA OOM、ERROR 或非有限验证指标。加载预训练权重时只有预期的 17 点 final_layer 形状不匹配提示。
- 日志中的模型、数据路径、增强、优化器、学习率策略、轮数与评价配置均与当前配置一致；独立加载共同预训练权重，resume_from=None。
- 仅评价 MRE、MSE、RMSE、MAE；ACDC=1.5625、HC18=0.108、ChestX-ray=1.4 mm/原图像素，TG3K 为像素单位，无 PCK。
- 小数据集每个 epoch 的训练 batch 数小于日志 interval=50，缺少逐 epoch loss 行不代表未训练；完成状态已由完整验证序列及 checkpoint 元数据验证。

## 最佳验证集结果

以下不是测试集成绩。距离指标使用单位列，MSE 为对应平方单位。MRE 来自 checkpoint 完整精度 best_score，其余指标来自日志。

| 数据集 | 最佳 epoch | 单位 | MRE | MSE | RMSE | MAE |
|---|---:|---|---:|---:|---:|---:|
| acdc | 210 | mm | 4.767227 | 83.439160 | 9.134500 | 2.944580 |
| hc18 | 210 | mm | 0.634806 | 0.273590 | 0.523060 | 0.406750 |
| TG3K | 210 | px | 1.056940 | 0.907600 | 0.952680 | 0.655710 |
| ChestX-ray | 210 | mm | 3.603529 | 12.458640 | 3.529680 | 2.250740 |

## 三个 seed 的验证 MRE 对比

| 数据集 | seed=1 | seed=2 | seed=3 |
|---|---:|---:|---:|
| acdc | 4.708065 | 4.645658 | 4.767227 |
| hc18 | 0.630001 | 0.645602 | 0.634806 |
| TG3K | 0.965550 | 1.132451 | 1.056940 |
| ChestX-ray | 3.597475 | 3.622189 | 3.603529 |

- HC18、ChestX-ray 三次验证 MRE 的变化较小；TG3K 的三次结果分别为 0.965550、1.132451、1.056940 px，本轮位于前两次之间。仅凭三次验证结果不对统计显著性作判断。
- ACDC 本轮验证 MRE 为 4.767227 mm，略高于前两次；最低 MSE 在 epoch 20，但仍按预先确定的最小 MRE 使用 epoch 210，不在训练完成后更改选择规则。
- 四个数据集的任务和单位不同，不能将其 MRE 数值直接混合平均或排名。seed=3 测试尚未执行，最终三个 seed 测试汇总应在本轮测试完成后生成。

[完整核查 JSON](seed3_training_audit.json) · [本轮验证结果 CSV](seed3_validation_summary.csv) · [三 seed 验证对比 CSV](three_seed_validation_comparison.csv)

## 用户手动推理和评价

在普通终端用下述 screen -r 进入对应已有会话；已在窗口内时无需再次连接。每个窗口先执行公共环境设置：

```bash
cd /home/wzk/WorkSpace/ViTPose
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### acdc

进入：`screen -r painter3d_acdc_seed3`。

推理：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  work_dirs/painter3d_seed3/acdc/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/acdc/annotations/test.json \
  --out work_dirs/painter3d_seed3/acdc/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed3/acdc/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/acdc/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed3/acdc/test_evaluation/metrics.json
```

### hc18

进入：`screen -r painter3d_hc18_seed3`。

推理：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  work_dirs/painter3d_seed3/hc18/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/hc18/annotations/test.json \
  --out work_dirs/painter3d_seed3/hc18/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed3/hc18/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/hc18/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed3/hc18/test_evaluation/metrics.json
```

### TG3K

进入：`screen -r painter3d_TG3K_seed3`。

推理：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  work_dirs/painter3d_seed3/tg3k/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/TG3K/annotations/test.json \
  --out work_dirs/painter3d_seed3/tg3k/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed3/tg3k/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/TG3K/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed3/tg3k/test_evaluation/metrics.json
```

### ChestX-ray

进入：`screen -r painter3d_ChestX-ray_seed3`。

推理：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  work_dirs/painter3d_seed3/chestxray/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/ChestX-ray/annotations/test.json \
  --out work_dirs/painter3d_seed3/chestxray/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed3/chestxray/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/ChestX-ray/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed3/chestxray/test_evaluation/metrics.json
```

输出目录自动创建；已存在的预测和指标文件不会被覆盖。上述命令均未执行，所有输出独立保存于 seed=3 目录。
