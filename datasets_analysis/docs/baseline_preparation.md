# 剩余七个 ViTPose-B 医学 baseline 配置与命令

本次仅准备配置、LUNA16_26 筛选结果、检查脚本及运行命令。没有运行训练或 TG3K test 推理。原始数据、既有 COCO JSON 和 TG3K 配置/训练结果均未修改。

## TG3K 最佳权重核查

`work_dirs/tg3k_smoke/best_MRE_epoch_210.pth` 是实际找到的最佳权重，非模板中的 work_dir。checkpoint meta：epoch=210、iter=3990、best_score=5.697495702622939；验证日志最优值也在 epoch 210。final_layer.weight 为 [5,256,1,1]，安全 weights_only 读取通过。此分数为验证集 MRE，不是 test 成绩。

以下命令从仓库根目录执行：

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

TG3K test 推理（输出放入新目录，不覆盖训练结果）：

```bash
CUDA_VISIBLE_DEVICES=1 python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  work_dirs/tg3k_smoke/best_MRE_epoch_210.pth \
  --test-json datasets_analysis/coco_format/TG3K/annotations/test.json \
  --out work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

TG3K test 评价：

```bash
python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --pck-unit pixels --pck-thresholds 2 5 10 20 \
  --out work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_metrics.json
```

输出 coordinate_mse、rmse、mae、mean_euclidean_distance_error 及 PCK@2/5/10/20 px（0–1 比例），还包含逐点统计；MSE 单位 pixel²，其余坐标误差为 pixel。两步也可运行 `bash datasets_analysis/commands/tg3k_test.sh`。输出存在时脚本拒绝覆盖。CUDA_VISIBLE_DEVICES=1 后进程内 cuda:0 对应物理 GPU1。

## 七个独立配置

全部从已验证 TG3K 模板适配：ViTPose-B、210 epochs、batch size 8、2 workers、checkpoints/vitpose-b.pth、JointsMSELoss、MedicalCocoDataset；以验证 MRE 最小保存最佳 checkpoint。无翻转、无半身增强，GT 全图 bbox。没有修改训练源码或医学评价实现。

| 数据集 | K | train/val/test | 配置 | dataset_info | GPU |
|---|---:|---|---|---|---:|
| acdc | 14 | 516/147/75 | `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py` | `configs/_base_/datasets/acdc.py` | 1 |
| hc18 | 13 | 210/60/30 | `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py` | `configs/_base_/datasets/hc18.py` | 0 |
| oasis | 8 | 942/269/135 | `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py` | `configs/_base_/datasets/oasis.py` | 1 |
| Synapse-CT | 13 | 2617/748/374 | `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py` | `configs/_base_/datasets/synapse_ct.py` | 0 |
| SKI10 | 8 | 2628/751/376 | `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py` | `configs/_base_/datasets/ski10.py` | 0 |
| ChestX-ray | 32 | 164/47/24 | `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py` | `configs/_base_/datasets/chestxray.py` | 1 |
| LUNA16_26 | 26 | 3749/1059/540 | `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py` | `configs/_base_/datasets/luna16_26.py` | 1 |

所有 ann_root 都是 `datasets_analysis/coco_format/<数据集>/annotations/`，分别读取既有 train/val/test。img_prefix 为共同原始数据根目录；LUNA16_26 的 file_name 仍为 LUNA16/data/...。acdc 的原名称是字符串 1..14，按原样保留；其他占位名 point_1..K 不意味着已确认解剖语义。

## LUNA16_26 筛选

| split | 原始记录 | 26 点保留 | 32 点排除 |
|---|---:|---:|---:|
| train | 4023 | 3749 | 274 |
| val | 1149 | 1059 | 90 |
| test | 575 | 540 | 35 |

逐条检查 keypoints 长度、annotation.source_point_ids 和 category.source_point_ids：26 点组均为 1..26，32 点组为 1..32。现有 COCO 没有解剖名称；不能证明两种布局相同编号表示同一部位。只保留完整的 26 点样本，绝不截断 32 点样本。
输出位于 `datasets_analysis/coco_format/LUNA16_26/annotations/{train,val,test}.json`，另有 selection_report.json。总计 5348 张图。保留原始 split、image/annotation/category id、记录相对顺序、坐标和 file_name；只删除不符合条件的记录，未重新编号或移动图片。source_sha256 仍描述原始 raw split，新增 info.selection 专门记录本次子集筛选来源及哈希。
筛选脚本：`datasets_analysis/tools/prepare_luna16_26.py`。已执行；输出目录存在时拒绝覆盖，无需再次运行。

## GPU 分配依据

读取到两卡均为 NVIDIA H100 PCIe，约 80 GiB。检查时 GPU0 已用约 15.8 GiB、GPU1 已用约 5.0 GiB；这是瞬时状态，非七任务训练峰值。
- GPU0：Synapse-CT + SKI10 + hc18，共 5455 张训练图，约 684 batch/epoch。
- GPU1：LUNA16_26 + oasis + acdc + ChestX-ray，共 5371 张训练图，约 673 batch/epoch。
按训练 batch 数安排，不按任务个数平均分配：把两个约 2600 样本的任务放在一张卡，最大 LUNA16_26 放另一张卡并搭配较小任务。GPU1 当前空闲显存更多，承载四个进程。LUNA16_26/ChestX-ray 点数较多，但同为 256×192 ViT-B，主要计算来自 backbone；点数对最终热图头的增量较小。数据集越小不意味着单进程显存越低。
此方案满足七任务并发启动需求，不保证并发比每卡串行更快。没有启动训练测量峰值，故不能保证当前其他进程占用下绝无 OOM；需要在执行时观察 nvidia-smi。每任务 batch 保持 8，不为并行擅自改变实验设置。

## 七个训练命令

在独立终端分别执行以下命令；需要真正同时启动全部任务时，使用文末并发脚本。

### acdc

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/acdc/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/vitpose_base_acdc_256x192 --seed 42
```

### hc18

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/hc18/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/vitpose_base_hc18_256x192 --seed 42
```

### oasis

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/oasis/ViTPose_base_oasis_256x192.py \
  --work-dir work_dirs/vitpose_base_oasis_256x192 --seed 42
```

### Synapse-CT

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/synapse_ct/ViTPose_base_synapse_ct_256x192.py \
  --work-dir work_dirs/vitpose_base_synapse_ct_256x192 --seed 42
```

### SKI10

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ski10/ViTPose_base_ski10_256x192.py \
  --work-dir work_dirs/vitpose_base_ski10_256x192 --seed 42
```

### ChestX-ray

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/chestxray/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/vitpose_base_chestxray_256x192 --seed 42
```

### LUNA16_26

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/luna16_26/ViTPose_base_luna16_26_256x192.py \
  --work-dir work_dirs/vitpose_base_luna16_26_256x192 --seed 42
```

并发启动脚本已生成但未执行：

```bash
bash datasets_analysis/commands/train_seven_baselines.sh
```

该脚本启用虚拟环境、设置仓库 PYTHONPATH、每进程 OMP/MKL 线程数为 2，后台启动七个训练进程并等待退出。每任务独立 work_dir；启动日志写入带时间戳的 work_dirs/medical_baseline_launch_*。随机种子统一为 42。

## 配置验证与产物

```bash
.venv/bin/python datasets_analysis/tools/check_baseline_configs.py
```

验证包括所有记录的点数/顺序和图片路径、21 个 split 各一个实际 batch、训练热图 [B,K,64,48]、GT 医学指标自检、LUNA16_26 与来源逐条一致性、输入文件哈希保持不变。检查只读取数据，不构建模型。结果见 baseline_config_validation.json。

新增产物：七份 dataset_info、七份配置；LUNA16_26 三份 JSON 和筛选报告；prepare_luna16_26.py、check_baseline_configs.py；commands/ 两份运行脚本；docs/ 下 baseline_configs_manifest.json、baseline_config_validation.json 和本文。
