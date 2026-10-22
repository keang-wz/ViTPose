# Painter-3D seed=1 训练核查与手动测试命令

本文保留训练完成时的核查记录。用户后续已完成测试集推理和评价，
当前结果见 [seed=1 测试结果汇总](seed1_test_results.md)。

四个数据集均已完成 210 epochs；本次只读取日志、配置和 checkpoint，未运行模型、推理或测试集评价。
每项均有完整的 21 次验证记录（10、20、…、210），最终 checkpoint 的 seed/epoch/iteration 与训练样本数相符，latest.pth 指向 epoch_210.pth。
最佳权重的 seed、epoch、best_score、输出头点数与日志一致，模型权重全部为有限值。实际训练日志中的展开配置与当前新版配置一致。

## 验证集结果

以下为按验证 MRE 选中的同一个 checkpoint 上的四项指标，不是测试集结果。除 TG3K 外距离单位为 mm、MSE 为 mm²；TG3K 为 px/px²。

| 数据集 | 最佳 epoch | MRE | MSE | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| acdc | 210 | 4.70807 | 80.25400 | 8.95846 | 2.91159 |
| hc18 | 210 | 0.63000 | 0.26758 | 0.51728 | 0.40160 |
| TG3K | 200 | 0.96555 | 0.73823 | 0.85920 | 0.59905 |
| ChestX-ray | 210 | 3.59748 | 12.20809 | 3.49401 | 2.24476 |

## 核查说明

- 各目录保留了此前中断运行的日志和旧 train_console.log。本报告仅使用后一次完整运行的时间戳日志，没有合并两次训练。
- acdc、hc18、ChestX-ray 的最佳 epoch 为 210；TG3K 为 200，不能统一使用 latest.pth。
- ACDC 在 epoch 60 的 MSE 为 69.05435 mm²，而 epoch 210 的 MSE 为 80.25400 mm²；但 MRE 从 5.04225 mm 改善至 4.70807 mm。MRE 与平方误差的最优 epoch 可以不同，仍遵循事先确定的最小 MRE 规则。
- HC18、TG3K、ChestX-ray 的最佳 MRE checkpoint 也对应本次验证记录中的最低 MSE。
- 最终完整运行日志没有 Traceback、CUDA OOM、ERROR 或非有限验证指标。加载预训练权重时只出现两个预期的 17 点 final_layer 形状不匹配提示，新输出头维度已在训练权重中核实。
- hc18、TG3K、ChestX-ray 每个 epoch 的训练 batch 数小于日志 interval=50，因此 JSON 中没有逐 epoch 训练 loss 行；完整验证序列及 checkpoint 元数据均正常。
- JSON 日志中的 lr=0.0 是低精度日志显示，不据此判断实际学习率为零；ACDC 文本日志最后仍记录 1.188e-07（分层学习率参数组）。
- 不同数据集的验证误差不能单凭数值排序任务难度；测试集成绩尚待用户手动推理和评价。

完整证据见 [seed1_training_audit.json](seed1_training_audit.json)。

## 手动推理与评价

进入对应 screen 后先运行公共环境设置。若当前已经位于窗口内，无需再次 screen -r。下列命令均未执行。

```bash
cd /home/wzk/WorkSpace/ViTPose
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### acdc

从普通终端进入已有会话：`screen -r painter3d_acdc`。在窗口内运行：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  work_dirs/painter3d_seed1/acdc/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/acdc/annotations/test.json \
  --out work_dirs/painter3d_seed1/acdc/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed1/acdc/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/acdc/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed1/acdc/test_evaluation/metrics.json
```

### hc18

从普通终端进入已有会话：`screen -r painter3d_hc18`。在窗口内运行：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  work_dirs/painter3d_seed1/hc18/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/hc18/annotations/test.json \
  --out work_dirs/painter3d_seed1/hc18/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed1/hc18/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/hc18/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed1/hc18/test_evaluation/metrics.json
```

### TG3K

从普通终端进入已有会话：`screen -r painter3d_TG3K`。在窗口内运行：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  work_dirs/painter3d_seed1/tg3k/best_MRE_epoch_200.pth \
  --test-json datasets_analysis/coco_format/painter3d/TG3K/annotations/test.json \
  --out work_dirs/painter3d_seed1/tg3k/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed1/tg3k/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/TG3K/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed1/tg3k/test_evaluation/metrics.json
```

### ChestX-ray

从普通终端进入已有会话：`screen -r painter3d_ChestX-ray`。在窗口内运行：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  work_dirs/painter3d_seed1/chestxray/best_MRE_mm_epoch_210.pth \
  --test-json datasets_analysis/coco_format/painter3d/ChestX-ray/annotations/test.json \
  --out work_dirs/painter3d_seed1/chestxray/test_evaluation/predictions.json \
  --device cuda:0 --batch-size 8 --workers 2

.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/painter3d_seed1/chestxray/test_evaluation/predictions.json \
  --gt datasets_analysis/coco_format/painter3d/ChestX-ray/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  --metrics MRE MSE RMSE MAE \
  --out work_dirs/painter3d_seed1/chestxray/test_evaluation/metrics.json
```

只在推理命令成功后执行评价命令。输出目录自动创建，已有预测和指标文件不会被覆盖。
三个已知 spacing 的数据集按当前 PNG 的 mm/px 换算，TG3K 保留像素单位；没有 PCK。
直接使用这些命令不依赖 training_complete.txt，也不会启动训练或其他 seed。
