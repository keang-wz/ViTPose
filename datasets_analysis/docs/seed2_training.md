# 第三轮训练：seed=2

第一轮(seed1115987209)和第二轮(seed1)八个任务的训练及test结果已核验完整。第三轮复用原八份配置，训练实验变量只改变seed。独立_seed2输出目录保留前两轮所有结果。未启动训练。

## 保持一致的设置

ViTPose-B、原COCO预训练、210epoch、batch8、2workers、JointsMSELoss、原优化器/学习率/增强、固定split、LUNA16_26子集、验证MRE选best全部保持第一轮配置。当前resume_from=None，初次启动从COCO预训练开始。命令不新增cfg-options、deterministic、autoscale-lr、no-validate或线程数设置。
GPU选择沿用既有命令安排；历史物理GPU映射没有记录，不宣称已认证第一轮物理GPU分配。

## 运行环境

```bash
cd /home/wzk/WorkSpace/ViTPose
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 八个训练命令

可在独立终端分别执行。

### TG3K

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  --work-dir work_dirs/vitpose_base_tg3k_256x192_seed2 \
  --seed 2
```

### acdc

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/vitpose_base_acdc_256x192_seed2 \
  --seed 2
```

### hc18

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/vitpose_base_hc18_256x192_seed2 \
  --seed 2
```

### oasis

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  --work-dir work_dirs/vitpose_base_oasis_256x192_seed2 \
  --seed 2
```

### Synapse-CT

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  --work-dir work_dirs/vitpose_base_synapse_ct_256x192_seed2 \
  --seed 2
```

### SKI10

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  --work-dir work_dirs/vitpose_base_ski10_256x192_seed2 \
  --seed 2
```

### ChestX-ray

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/vitpose_base_chestxray_256x192_seed2 \
  --seed 2
```

### LUNA16_26

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  --work-dir work_dirs/vitpose_base_luna16_26_256x192_seed2 \
  --seed 2
```

也可顺序运行整个命令文件：

```bash
bash datasets_analysis/commands/train_seed2_baselines.sh
```

seed2完成后使用各任务本轮实际best_MRE进行原test推理与评价，预测/指标另存_seed2目录。最终按三个seed test成绩汇总MAE mean±std，同时保留MRE/MSE/RMSE/PCK，ddof=1。当前两轮临时test汇总见 ../results/two_round_summary/report.md。
