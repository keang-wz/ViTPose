# Painter-3D seed=3：四个独立 screen，用户手动启动

只准备命令，未创建 screen 或启动训练。每个数据集独立运行一次 seed=3，
不会继续其他 seed，也不会自动推理或评价。
准备命令时四个 `work_dirs/painter3d_seed3/<dataset>/` 输出目录均不存在。
已有 seed=1、seed=2 会话和结果保留。

## ACDC

在普通终端建立并进入 screen：

```bash
screen -S painter3d_acdc_seed3 -t acdc
```

在该 screen 内运行：

```bash
cd /home/wzk/WorkSpace/ViTPose
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/painter3d_seed3/acdc \
  --seed 3
```

按 `Ctrl+a`，松开后按 `d` 脱离，再在普通终端建立下一个 screen。

## HC18

```bash
screen -S painter3d_hc18_seed3 -t hc18
```

在该 screen 内运行：

```bash
cd /home/wzk/WorkSpace/ViTPose
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/painter3d_seed3/hc18 \
  --seed 3
```

## TG3K

```bash
screen -S painter3d_TG3K_seed3 -t TG3K
```

在该 screen 内运行：

```bash
cd /home/wzk/WorkSpace/ViTPose
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  --work-dir work_dirs/painter3d_seed3/tg3k \
  --seed 3
```

## ChestX-ray

```bash
screen -S painter3d_ChestX-ray_seed3 -t ChestX-ray
```

在该 screen 内运行：

```bash
cd /home/wzk/WorkSpace/ViTPose
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/painter3d_seed3/chestxray \
  --seed 3
```

## 重新连接

```bash
screen -r painter3d_acdc_seed3
screen -r painter3d_hc18_seed3
screen -r painter3d_TG3K_seed3
screen -r painter3d_ChestX-ray_seed3
```

seed=3 与 seed=1 使用相同数据划分、模型、初始化权重和训练超参数，
只更改随机种子及输出目录。从预训练权重重新初始化，不从 seed=1 续训。
只统计 MRE、MSE、RMSE、MAE；ACDC/HC18/ChestX-ray 为 mm/mm²，TG3K 为 px/px²。
训练完成后再单独选择该次验证最佳权重，手动进行测试集推理和评价。
