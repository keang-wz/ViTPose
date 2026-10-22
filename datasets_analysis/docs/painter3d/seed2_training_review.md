# Painter-3D seed=2 训练核查与手动推理评价

本文记录训练完成时的检查。用户后续已完成测试集推理和评价，
最终成绩见 [seed=2 测试结果](seed2_test_results.md)。

四个数据集均已完成 210 epochs，各有 21 次验证记录（epoch 10 到 210，间隔 10）。本次仅读取训练日志、配置及 checkpoint，没有启动训练、模型推理或测试集评价。

## 核查结论

- 四个运行的日志及 best/epoch_210 checkpoint 元数据均为 seed=2，迭代数与样本数及 batch size=8 一致，latest.pth 均指向 epoch_210.pth。
- 完整运行日志没有 Traceback、CUDA OOM、ERROR 或非有限验证指标。最佳权重的所有浮点模型参数均为有限值，输出头分别为 14、13、5、32 点。
- 日志记录的模型、数据路径、预处理、优化器、学习率策略、训练轮数及评价配置与当前新版配置一致；resume_from=None，使用共同预训练权重独立初始化。
- 只评价 MRE、MSE、RMSE、MAE；ACDC=1.5625、HC18=0.108、ChestX-ray=1.4 mm/原图像素，TG3K 保留像素单位，无 PCK。
- 每项仅有一份时间戳训练日志；最佳 checkpoint 的 epoch 和 best_score 与验证日志一致。加载初始 COCO 17 点权重时的 final_layer 形状提示属于预期输出头适配。
- HC18/TG3K/ChestX-ray 每个 epoch 的 batch 数小于日志 interval=50，缺少训练 loss 行不表示没有训练；完成状态已由完整验证序列及 checkpoint 验证。

## 最佳验证集结果

以下不是测试集成绩。距离指标按单位列，MSE 为该单位平方。MRE 使用 checkpoint 中的完整精度 best_score，其余指标来自日志。

| 数据集 | 最佳 epoch | 单位 | MRE | MSE | RMSE | MAE |
|---|---:|---|---:|---:|---:|---:|
| acdc | 200 | mm | 4.645658 | 78.614570 | 8.866490 | 2.881520 |
| hc18 | 210 | mm | 0.645602 | 0.279160 | 0.528360 | 0.412710 |
| TG3K | 200 | px | 1.132451 | 1.133000 | 1.064420 | 0.702800 |
| ChestX-ray | 210 | mm | 3.622189 | 12.564060 | 3.544580 | 2.259760 |

## 与 seed=1 的验证 MRE 比较

| 数据集 | seed=1 | seed=2 | 相对变化 |
|---|---:|---:|---:|
| acdc | 4.708065 | 4.645658 | -1.33% |
| hc18 | 0.630001 | 0.645602 | +2.48% |
| TG3K | 0.965550 | 1.132451 | +17.29% |
| ChestX-ray | 3.597475 | 3.622189 | +0.69% |

- ACDC 本轮验证 MRE 略低；HC18 和 ChestX-ray 略高。TG3K 本轮高约 17.29%，是两次运行中差异最大的一项；尚不能仅凭两次验证成绩判断总体稳定性。
- ACDC/TG3K 的 epoch 210 验证 MRE 分别为 4.67394 mm、1.14220 px，均略高于 epoch 200，故测试应使用 epoch 200 的最佳权重。
- ACDC 最低 MSE 出现在 epoch 40，但本实验预先确定以 MRE 最小选择 checkpoint，继续使用 epoch 200，不按测试表现更换权重。
- seed=2 测试尚未执行，本报告没有将 seed=1 的测试结果与 seed=2 的验证结果混合比较。

[完整核查 JSON](seed2_training_audit.json) · [验证结果 CSV](seed2_validation_summary.csv)

## 手动推理和评价

在普通终端通过相应 screen -r 命令进入已有会话；如果已经在窗口内，无需再次连接。每个窗口先运行：

```bash
cd /home/wzk/WorkSpace/ViTPose
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### acdc

连接：`screen -r painter3d_acdc_seed2`。

推理：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  work_dirs/painter3d_seed2/acdc/best_MRE_mm_epoch_200.pth \
  --test-json datasets_analysis/coco_format/painter3d/acdc/annotations/test.json \
  --out work_dirs/painter3d_seed2/acdc/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed2/acdc/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/acdc/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed2/acdc/test_evaluation/metrics.json
```

### hc18

连接：`screen -r painter3d_hc18_seed2`。

推理：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  work_dirs/painter3d_seed2/hc18/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/hc18/annotations/test.json \
  --out work_dirs/painter3d_seed2/hc18/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed2/hc18/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/hc18/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed2/hc18/test_evaluation/metrics.json
```

### TG3K

连接：`screen -r painter3d_TG3K_seed2`。

推理：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  work_dirs/painter3d_seed2/tg3k/best_MRE_epoch_200.pth \
  --test-json datasets_analysis/coco_format/painter3d/TG3K/annotations/test.json \
  --out work_dirs/painter3d_seed2/tg3k/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed2/tg3k/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/TG3K/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed2/tg3k/test_evaluation/metrics.json
```

### ChestX-ray

连接：`screen -r painter3d_ChestX-ray_seed2`。

推理：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  work_dirs/painter3d_seed2/chestxray/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/ChestX-ray/annotations/test.json \
  --out work_dirs/painter3d_seed2/chestxray/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

推理成功后，再执行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed2/chestxray/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/ChestX-ray/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed2/chestxray/test_evaluation/metrics.json
```

输出目录会自动创建；已有预测和指标文件不会被覆盖。以上均为待用户手动运行的命令，没有执行推理和评价。
