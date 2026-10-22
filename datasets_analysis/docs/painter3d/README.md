# Painter-3D 新版四数据集格式适配

三个 seed 的训练、推理和评价均已由用户手动完成，全部 12 项结果通过文件及数值复核。
正式 baseline 采用三个 seed 的测试指标均值 ± 样本标准差（ddof=1），详见
[三 seed 最终测试结果与分析](three_seed_test_results.md)、
[最终汇总 CSV](three_seed_test_summary.csv) 和
[12 轮逐次结果 CSV](three_seed_test_individual.csv)。

用户已手动完成 seed=1 的四项训练。训练完成状态、验证集结果和经核查的
最佳权重推理命令见 [seed=1 训练核查](seed1_training_review.md)；
用户已手动完成测试集推理和评价，结果核查、分析和 CSV 见
[seed=1 测试结果汇总](seed1_test_results.md)。
用户已手动完成 seed=2 的训练和测试，验证集结果及训练记录见
[seed=2 训练核查](seed2_training_review.md)，最终测试结果和两 seed 对比见
[seed=2 测试结果](seed2_test_results.md)。
训练命令保留在 [seed=2 手动训练](../../commands/painter3d/seed2_manual_training.md)。
用户已手动完成 seed=3 的训练和测试，训练记录与命令见
[seed=3 训练核查](seed3_training_review.md)，本轮测试结果见
[seed=3 测试结果](seed3_test_results.md)。
训练命令保留在 [seed=3 手动训练](../../commands/painter3d/seed3_manual_training.md)。

## 新旧数据区分

新版与已有八个数据集中的四个名称相同，但图像和坐标有变化，不能混用。
新版通过独立的 `painter3d` 目录区分，旧版转换结果和配置保持原样。

| 内容 | 旧版 | 新版 |
|---|---|---|
| 原始数据根目录 | `/data/users/mqf/bushu/Painter-ct/data` | `/data/users/mqf/Painter-3D/data` |
| COCO 标注 | `datasets_analysis/coco_format/<dataset>/annotations/` | `datasets_analysis/coco_format/painter3d/<dataset>/annotations/` |
| 配置目录（相对 topdown_heatmap） | `acdc/`、`hc18/`、`tg3k/`、`chestxray/` | `painter3d/` |
| 数据检查报告 | `datasets_analysis/docs/` 中原有报告 | `datasets_analysis/docs/painter3d/` |
| 标注预览 | `datasets_analysis/samples/<dataset>/` | `datasets_analysis/samples/painter3d/<dataset>/` |
| 配置默认训练输出 | 原有 work_dir | `work_dirs/painter3d_seed42/<slug>/` |

新版仅继承旧版模型与预处理设置；三个 split 的 `ann_file`、`img_prefix`
均明确覆盖为新版路径。保留同样的关键点编号和名称，
不将同名视为同一份数据，也不复用旧版标注或训练权重作为新版结果。

## 数据检查

直接使用每个数据集 `data/train.json`、`data/val.json`、`data/test.json`
中的记录及顺序，不创建或重新检查划分方案。实际目录名为 `TG3K`。
已完整解码引用的 1,481 张输入图像，均为 RGB PNG，
没有越界关键点，关键点编号均为连续的 1..K。

| 数据集 | train | val | test | K | 原图尺寸 W×H |
|---|---:|---:|---:|---:|---|
| acdc | 516 | 147 | 75 | 14 | 37 种，最常见 256×216（217 张） |
| hc18 | 210 | 60 | 30 | 13 | 8 种，800×540 占 291 张 |
| TG3K | 146 | 42 | 20 | 5 | 278×191、214×209、325×193、240×204、241×174 |
| ChestX-ray | 164 | 47 | 24 | 32 | 全部 256×256 |

输入字段包括 `image_path`、`points`、`point_ids`；acdc 额外提供数字字符串
`point_names`。坐标为当前 PNG 的像素坐标。历史 `source_dir`
不用于加载输入，辅助 target/label 图片不作为模型输入。

完整分 split 统计及源 JSON SHA256 见
[dataset_analysis.json](dataset_analysis.json)。
各 split 第一条记录的关键点预览位于
`datasets_analysis/samples/painter3d/<dataset>/{train,val,test}.png`。

## 格式适配和验证

复用现有 `datasets_analysis/converters/common.py`，生成 12 份独立 COCO JSON。
每条原始记录对应一张 image 和一条 annotation，保持点序和坐标，
全图 bbox 为 `[0,0,W,H]`，visibility=2 表示提供了标注。
不修改、复制或缩放原始图像；保留源 split 哈希和原始行号。

新配置位于 `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/`，
分别继承旧版 acdc/hc18/tg3k/chestxray 配置，覆盖数据、输出路径和新版评价设置。
沿用 MedicalCocoDataset、ViTPose-B、输入 W×H=192×256、
热图 W×H=48×64，以及原有增强、损失和初始化设置。
评价单位及取消 PCK 的改动见下节。

在仓库根目录运行以下命令只进行转换和检查，不构建或训练模型：

```bash
# 已完成转换。该命令拒绝覆盖现有非空输出目录，无需重复执行。
.venv/bin/python datasets_analysis/tools/prepare_painter3d.py
# 可独立重复执行，无训练或模型推理。
.venv/bin/python datasets_analysis/tools/check_painter3d.py
```

[validation.json](validation.json) 记录了 12 个 split 的实际数据加载、
变换后 batch、训练目标热图维度、逐条坐标保真检查和 GT 指标自检。
报告记录新版配置、标注、图像根目录和独立 work_dir，并检查新旧路径隔离。
GT 自检不是模型预测成绩。另以所有点偏移 (3,4) 像素的合成预测检查非零误差：
ACDC MRE=7.8125mm、ChestX-ray MRE=7mm、HC18 MRE=0.54mm、TG3K MRE=5px。
这些仅验证公式和配置接入，不构建模型、不执行推理或训练。
所有源 JSON 哈希保持不变。

## 新版评价：仅 MRE、MSE、RMSE、MAE

四个新版数据集均不计算或输出 PCK。旧版八个数据集配置及默认像素评价行为保持不变。
按用户提供的数值，将 spacing 视为**当前 JSON 对应原始 PNG 的各向同性 mm/像素**：

| 数据集 | spacing（mm/px） | MRE / RMSE / MAE | MSE | 最佳权重依据 |
|---|---:|---|---|---|
| acdc | 1.5625 | mm | mm² | `MRE_mm` 最小 |
| ChestX-ray | 1.4 | mm | mm² | `MRE_mm` 最小 |
| hc18 | 0.108 | mm | mm² | `MRE_mm` 最小 |
| TG3K | 未知 | px | px² | `MRE` 最小 |

先将预测还原为原始 PNG 坐标，再计算 `Δx_mm = Δx_px × spacing`、
`Δy_mm = Δy_px × spacing`。绝不把 spacing 乘在 192×256 网络输入或 48×64 热图的坐标误差上。
如果给定 spacing 对应的是 PNG 导出前的另一分辨率，应先按导出缩放比例修正 spacing。

沿用已有指标定义，N 为所有有效标注点数量：

- `MRE = Σ sqrt(Δx² + Δy²) / N`：每个点欧氏距离的平均值。
- `MSE = Σ(Δx² + Δy²) / (2N)`：x、y 坐标误差平方的平均值。
- `RMSE = sqrt(MSE)`。
- `MAE = Σ(|Δx| + |Δy|) / (2N)`。

因此 MRE、RMSE、MAE 的物理值等于像素值乘 spacing，
MSE 的物理值等于像素值乘 spacing²。这里没有把原有“坐标 MSE”
改为“每点欧氏距离平方的平均”，两者相差系数 2。

训练日志使用 `MRE_mm`、`RMSE_mm`、`MAE_mm`、`MSE_mm2`；
TG3K 继续使用 `MRE`、`RMSE`、`MAE`、`MSE`，单位在报告中说明。
三个已知 spacing 的配置将生成 `best_MRE_mm_epoch_*.pth`，
TG3K 仍为 `best_MRE_epoch_*.pth`。
每个数据集内 spacing 是固定正数，因此不会改变最优 epoch 的排序。

ViTPose 训练本身仍优化原来的热图 `JointsMSELoss`，训练日志中的 loss/acc_pose
是优化过程诊断量，不是这里的毫米定位误差。模型、预训练权重、
AdamW、lr=5e-4、batch size=8、210 epochs、每 10 epochs 验证均与旧版一致。

独立推理工具会在预测 JSON 中保存评价配置，坐标仍以原图像素保存，
随后 `evaluate_medical_keypoints.py` 自动读取相同的 spacing 和四项指标。
也可以显式使用 `--config`，例如对用户自己生成的 hc18 预测进行评价：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions /path/to/hc18_predictions.json \
  --gt datasets_analysis/coco_format/painter3d/hc18/annotations/test.json \
  --config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  --out /path/to/hc18_metrics_mm.json
```

TG3K 后续确定 spacing 后，可以直接在同一份原图像素预测上添加
`--pixel-spacing-mm <实际数值> --metrics MRE MSE RMSE MAE` 重新评价，无需重新训练。
上述命令仅为使用说明，未执行模型训练或推理。

## 用户手动训练入口

seed=1、2、3 的四窗口 screen 训练命令，以及训练结束后单独手动开启的推理和评价命令，
见 [三种 seed 的 screen 命令](../../commands/painter3d/README.md)。

以下命令仅供用户手动执行，不包含自动启动或任务调度脚本。
默认沿用 batch size=8、210 epochs，以验证集最小 MRE 保存最佳权重；
seed=42 可由用户自行修改，并应配套指定独立的 `--work-dir`。

在仓库根目录执行公共环境设置：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

根据 GPU 使用情况调整 `CUDA_VISIBLE_DEVICES`，四个独立入口如下：

```bash
# 新版 acdc
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_acdc_256x192.py \
  --work-dir work_dirs/painter3d_seed42/acdc --seed 42

# 新版 hc18
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_hc18_256x192.py \
  --work-dir work_dirs/painter3d_seed42/hc18 --seed 42

# 新版 TG3K
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_tg3k_256x192.py \
  --work-dir work_dirs/painter3d_seed42/tg3k --seed 42

# 新版 ChestX-ray
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/painter3d/ViTPose_base_chestxray_256x192.py \
  --work-dir work_dirs/painter3d_seed42/chestxray --seed 42
```

新版尺寸不同于旧版的 448×896。新版毫米误差与旧版像素误差
不能直接比较数值，须先统一坐标尺度和物理标定。
