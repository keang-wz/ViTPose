# 第二轮训练：seed=1

复用第一轮八份配置，训练实验变量只改变随机种子。为保留第一轮结果，第二轮使用独立的 _seed1 输出目录。没有修改配置、数据划分、COCO标注、训练/推理/评价代码，没有启动训练。

## 与第一轮保持一致

已将当前配置与各数据集第一轮初始 .log 内嵌配置逐字段比较：model、optimizer、optimizer_config、lr_config、total_epochs、channel_cfg、data_cfg、三个pipeline、evaluation、checkpoint_config、log_config、workflow、load_from、resume_from、data、dist_params、opencv_num_threads、mp_start_method 均一致。

| 项目 | 第二轮采用的第一轮设置 |
|---|---|
| 模型/输入 | ViTPose-B，192×256 |
| 初始权重 | checkpoints/vitpose-b.pth |
| 训练 | 210 epoch，batch size 8，2 workers |
| Loss | JointsMSELoss(use_target_weight=True) |
| 优化器/学习率 | 原配置 AdamW、原学习率和衰减计划 |
| 数据 | 原train/val/test；LUNA16_26仍为原26点子集 |
| 最佳模型选择 | 验证MRE最小：save_best=MRE，rule=less |
| 恢复训练 | 当前配置resume_from=None，从COCO预训练初始化 |
| 随机种子 | 第一轮实际1115987209 → 第二轮1 |

命令不使用 --cfg-options，不新增 --deterministic、--autoscale-lr 或 --no-validate，也不新增线程数设置。沿用你运行第一轮时的虚拟环境和线程环境。
TG3K第一轮日志的顶层train_cfg.max_epochs=5不参与本仓库runner轮数控制；实际210epoch来自total_epochs。第二轮复用当前已验证配置，不复制这个无效覆盖字段。
acdc第一轮中途恢复是一次实验的续训记录；第二轮初次启动从预训练重新开始，不从第一轮的latest或best checkpoint继续。

## 初始化运行环境

以下命令从仓库根目录执行。若使用多个终端，每个终端先初始化一次。

```bash
cd /home/wzk/WorkSpace/ViTPose
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 八个训练命令

可以在独立终端分别执行。CUDA_VISIBLE_DEVICES采用此前命令安排；本项目历史日志只记录逻辑cuda:0，没有物理GPU映射证据，不能把此安排称为已核实的第一轮物理GPU分配。

### TG3K

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  --work-dir work_dirs/vitpose_base_tg3k_256x192_seed1 \
  --seed 1
```

### acdc

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/vitpose_base_acdc_256x192_seed1 \
  --seed 1
```

### hc18

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/vitpose_base_hc18_256x192_seed1 \
  --seed 1
```

### oasis

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  --work-dir work_dirs/vitpose_base_oasis_256x192_seed1 \
  --seed 1
```

### Synapse-CT

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  --work-dir work_dirs/vitpose_base_synapse_ct_256x192_seed1 \
  --seed 1
```

### SKI10

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  --work-dir work_dirs/vitpose_base_ski10_256x192_seed1 \
  --seed 1
```

### ChestX-ray

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/vitpose_base_chestxray_256x192_seed1 \
  --seed 1
```

### LUNA16_26

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  --work-dir work_dirs/vitpose_base_luna16_26_256x192_seed1 \
  --seed 1
```

## 输出与后续汇总

每个任务输出到 work_dirs/vitpose_base_<dataset_slug>_256x192_seed1/，日志、epoch checkpoint、best_MRE checkpoint均与第一轮分开。生成文档时八个seed1输出目录均不存在。最佳轮次由本轮验证结果决定，不预先写死210。
也可运行下面的文件依次训练八个数据集（顺序执行，不自动并行）：

```bash
bash datasets_analysis/commands/train_seed1_baselines.sh
```

本次仅生成命令与文档，未执行这个脚本。已有第一轮结果保留。
seed1训练完成后，使用本轮best_MRE分别做原test推理和医学评价。后续seed2也采用相同模型选择标准；最终按数据集汇总三个seed的test指标，主要展示MAE mean±std，同时保留MRE、RMSE、MSE和PCK，样本标准差ddof=1。当前没有seed1测试成绩，不生成虚构汇总。
