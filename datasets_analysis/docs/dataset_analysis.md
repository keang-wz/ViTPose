# 医学 2D Keypoint 数据分析

数据根目录：`/data/users/mqf/bushu/Painter-ct/data`。所有统计来自本次实际文件读取；完整解码全部 input 图像，辅助 target/label 图像只统计文件，不作为训练输入。
使用每个数据集 `data/{train,val,test}.json`，不创建新划分，不修改 raw 数据。根目录同名共享 JSON 的八个指定数据集记录也已对照；结果见 `shared_split_comparison.json`。

| 数据集 | train | val | test | 原始点数 | 输入尺寸 W×H | 模式 |
|---|---:|---:|---:|---|---|---|
| TG3K | 146 | 42 | 20 | {'5': 208} | {'448x896': 208} | {'RGB': 208} |
| acdc | 516 | 147 | 75 | {'14': 738} | {'448x896': 738} | {'RGB': 738} |
| hc18 | 210 | 60 | 30 | {'13': 300} | {'448x896': 300} | {'RGB': 300} |
| oasis | 942 | 269 | 135 | {'8': 1346} | {'448x896': 1346} | {'RGB': 1346} |
| Synapse-CT | 2617 | 748 | 374 | {'13': 3739} | {'448x896': 3739} | {'RGB': 3739} |
| SKI10 | 2628 | 751 | 376 | {'8': 3755} | {'448x896': 3755} | {'RGB': 3755} |
| ChestX-ray | 164 | 47 | 24 | {'32': 235} | {'448x896': 235} | {'RGB': 235} |
| LUNA16 | 4023 | 1149 | 575 | {'26': 5348, '32': 399} | {'488x896': 5747} | {'RGB': 5747} |

## 共同格式与转换规则

- 原格式为 JSON 数组，每条记录包含一个 input 图像和一个有序点列表。点为 `{x,y}`，单位为当前 PNG 像素；没有原始 image id、visibility 或独立目标实例列表。多器官可能合在一个点列表，不能据此推断一个解剖目标。
- 原始 point_ids 为 1..K，顺序直接保留；acdc 的 point_names 是字符串数字，其余没有解剖名称。占位名称 point_1 等仅对应原 id，不推断解剖含义。
- 有限坐标转换为 `[x,y,2]`；2 表示已提供标注的训练约定，并非验证过无遮挡。未知点格式或非有限坐标直接报错；不裁剪越界坐标，不删除记录。
- `file_name` 如 `TG3K/data/train/sample_00000/input.png`，相对于共同 data_root。SKI10 的 E:/data/pre/ 前缀按明确 dataset/data/ 后缀映射；不按 basename 模糊匹配。其他 source_dir 中 Windows 路径仅为历史来源，不用于读取图像。
- 每条记录生成一张 image 和一条 annotation，数据集内跨 split 的 id 唯一且确定；原 JSON 行号和 point_ids 保留。没有目标框，使用全图 bbox=[0,0,W,H]、area=W*H、iscrowd=0，属于全图 landmark baseline。
- 图像已经是 PNG 二维导出数据，使用 Pillow/OpenCV 常规读取；不需要读取 DICOM/NIfTI 或再次窗宽窗位处理。转换不改变像素、不缩放坐标。
- LUNA16 的 26 点和 32 点分别保留为两个 category，各自保持原始数组和顺序。未将 26 点填充成 32 点，因为相同 id 的解剖语义尚未确认。COCO 可以存储，当前固定 K 的 TopDownCocoDataset 不能直接混合训练。

## 各数据集实际结构与检查

### TG3K

```text
TG3K/data/
├── train.json
├── val.json
├── test.json
├── train/ (146 个 sample 目录)
│   └── sample_00000/
│       └── input.png, label.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png
├── val/ (42 个 sample 目录)
│   └── sample_00146/
│       └── input.png, label.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png
├── test/ (20 个 sample 目录)
│   └── sample_00188/
│       └── input.png, label.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png
```

- split 文件：`TG3K/data/train.json`, `TG3K/data/val.json`, `TG3K/data/test.json`
- 字段：`image_path, target_path, label_path, type, task, modality, dataset, points, point_ids, target_point_paths, source_dir`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:relative': 208, 'target_path:relative': 208, 'label_path:relative': 208, 'target_point_paths:relative': 1040, 'source_dir:windows_absolute': 208}`。
- 输入 PNG：208；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 1664}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 208}`；RGB 三通道像素完全相同/灰度图像数：208。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'5': 208}`；名称：`无；使用 point_id 占位名称`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 146 | 146 | 730 | 0 | 0 | True |
| val | 42 | 42 | 210 | 0 | 0 | True |
| test | 20 | 20 | 100 | 0 | 0 | True |

### acdc

```text
acdc/data/
├── train.json
├── val.json
├── test.json
├── train/ (516 个 sample 目录)
│   └── sample_00000/
│       └── input.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png, vis.png
├── val/ (147 个 sample 目录)
│   └── sample_00516/
│       └── input.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png, vis.png
├── test/ (75 个 sample 目录)
│   └── sample_00663/
│       └── input.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png, vis.png
```

- split 文件：`acdc/data/train.json`, `acdc/data/val.json`, `acdc/data/test.json`
- 字段：`image_path, target_path, type, task, modality, dataset, points, point_ids, point_names, target_point_paths, source_dir`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:relative': 738, 'target_path:relative': 738, 'target_point_paths:relative': 10332, 'source_dir:windows_absolute': 738}`。
- 输入 PNG：738；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 12546}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 738}`；RGB 三通道像素完全相同/灰度图像数：738。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'14': 738}`；名称：`[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']]`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 516 | 516 | 7224 | 0 | 0 | True |
| val | 147 | 147 | 2058 | 0 | 0 | True |
| test | 75 | 75 | 1050 | 0 | 0 | True |

### hc18

```text
hc18/data/
├── train.json
├── val.json
├── test.json
├── train/ (210 个 sample 目录)
│   └── sample_00000/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
├── val/ (60 个 sample 目录)
│   └── sample_00210/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
├── test/ (30 个 sample 目录)
│   └── sample_00270/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
```

- split 文件：`hc18/data/train.json`, `hc18/data/val.json`, `hc18/data/test.json`
- 字段：`image_path, target_path, label_path, type, task, modality, dataset, points, point_ids, target_point_paths, source_dir`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:relative': 300, 'target_path:relative': 300, 'label_path:relative': 300, 'target_point_paths:relative': 3900, 'source_dir:windows_absolute': 300}`。
- 输入 PNG：300；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 4800}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 300}`；RGB 三通道像素完全相同/灰度图像数：300。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'13': 300}`；名称：`无；使用 point_id 占位名称`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 210 | 210 | 2730 | 0 | 0 | True |
| val | 60 | 60 | 780 | 0 | 0 | True |
| test | 30 | 30 | 390 | 0 | 0 | True |

### oasis

```text
oasis/data/
├── train.json
├── val.json
├── test.json
├── train/ (942 个 sample 目录)
│   └── sample_00000/
│       └── input.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png
├── val/ (269 个 sample 目录)
│   └── sample_00942/
│       └── input.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png
├── test/ (135 个 sample 目录)
│   └── sample_01211/
│       └── input.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png
```

- split 文件：`oasis/data/train.json`, `oasis/data/val.json`, `oasis/data/test.json`
- 字段：`image_path, target_path, type, task, modality, dataset, points, point_ids, target_point_paths, source_dir`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:relative': 1346, 'target_path:relative': 1346, 'target_point_paths:relative': 10768, 'source_dir:windows_absolute': 1346}`。
- 输入 PNG：1346；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 13460}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 1346}`；RGB 三通道像素完全相同/灰度图像数：1346。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'8': 1346}`；名称：`无；使用 point_id 占位名称`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 942 | 942 | 7536 | 0 | 0 | True |
| val | 269 | 269 | 2152 | 0 | 0 | True |
| test | 135 | 135 | 1080 | 0 | 0 | True |

### Synapse-CT

```text
Synapse-CT/data/
├── train.json
├── val.json
├── test.json
├── train/ (2617 个 sample 目录)
│   └── sample_00000/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
├── val/ (748 个 sample 目录)
│   └── sample_02617/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
├── test/ (374 个 sample 目录)
│   └── sample_03365/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
```

- split 文件：`Synapse-CT/data/train.json`, `Synapse-CT/data/val.json`, `Synapse-CT/data/test.json`
- 字段：`image_path, target_path, label_path, type, task, modality, dataset, points, point_ids, target_point_paths, source_dir`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:relative': 3739, 'target_path:relative': 3739, 'label_path:relative': 3739, 'target_point_paths:relative': 48607, 'source_dir:windows_absolute': 3739}`。
- 输入 PNG：3739；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 59824}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 3739}`；RGB 三通道像素完全相同/灰度图像数：3739。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'13': 3739}`；名称：`无；使用 point_id 占位名称`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 2617 | 2617 | 34021 | 0 | 0 | True |
| val | 748 | 748 | 9724 | 0 | 0 | True |
| test | 374 | 374 | 4862 | 0 | 0 | True |

### SKI10

```text
SKI10/data/
├── train.json
├── val.json
├── test.json
├── train/ (2628 个 sample 目录)
│   └── sample_00000/
│       └── input.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png
├── val/ (751 个 sample 目录)
│   └── sample_00000/
│       └── input.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png
├── test/ (376 个 sample 目录)
│   └── sample_00000/
│       └── input.png, target.png, target_1.png, target_2.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png
```

- split 文件：`SKI10/data/train.json`, `SKI10/data/val.json`, `SKI10/data/test.json`
- 字段：`image_path, target_path, type, task, dataset, modality, points, point_ids, target_point_paths`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:windows_absolute': 3379, 'target_path:windows_absolute': 3379, 'target_point_paths:windows_absolute': 27032, 'image_path:relative': 376, 'target_path:relative': 376, 'target_point_paths:relative': 3008}`。
- 输入 PNG：3755；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 37550}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 3755}`；RGB 三通道像素完全相同/灰度图像数：3755。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'8': 3755}`；名称：`无；使用 point_id 占位名称`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 2628 | 2628 | 21024 | 0 | 0 | True |
| val | 751 | 751 | 6008 | 0 | 0 | True |
| test | 376 | 376 | 3008 | 0 | 0 | True |

### ChestX-ray

```text
ChestX-ray/data/
├── train.json
├── val.json
├── test.json
├── train/ (164 个 sample 目录)
│   └── sample_00000/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_15.png, target_16.png, target_17.png, target_18.png, target_19.png, target_2.png, target_20.png, target_21.png, target_22.png, target_23.png, target_24.png, target_25.png, target_26.png, target_27.png, target_28.png, target_29.png, target_3.png, target_30.png, target_31.png, target_32.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png, vis.png
├── val/ (47 个 sample 目录)
│   └── sample_00164/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_15.png, target_16.png, target_17.png, target_18.png, target_19.png, target_2.png, target_20.png, target_21.png, target_22.png, target_23.png, target_24.png, target_25.png, target_26.png, target_27.png, target_28.png, target_29.png, target_3.png, target_30.png, target_31.png, target_32.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png, vis.png
├── test/ (24 个 sample 目录)
│   └── sample_00211/
│       └── input.png, label.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_15.png, target_16.png, target_17.png, target_18.png, target_19.png, target_2.png, target_20.png, target_21.png, target_22.png, target_23.png, target_24.png, target_25.png, target_26.png, target_27.png, target_28.png, target_29.png, target_3.png, target_30.png, target_31.png, target_32.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png, vis.png
```

- split 文件：`ChestX-ray/data/train.json`, `ChestX-ray/data/val.json`, `ChestX-ray/data/test.json`
- 字段：`image_path, target_path, label_path, type, task, modality, dataset, points, point_ids, target_point_paths, source_dir`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:relative': 235, 'target_path:relative': 235, 'label_path:relative': 235, 'target_point_paths:relative': 7520, 'source_dir:windows_absolute': 235}`。
- 输入 PNG：235；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 8460}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 235}`；RGB 三通道像素完全相同/灰度图像数：235。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'32': 235}`；名称：`无；使用 point_id 占位名称`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 164 | 164 | 5248 | 0 | 0 | True |
| val | 47 | 47 | 1504 | 0 | 0 | True |
| test | 24 | 24 | 768 | 0 | 0 | True |

### LUNA16

```text
LUNA16/data/
├── train.json
├── val.json
├── test.json
├── train/ (4023 个 sample 目录)
│   └── sample_00000/
│       └── input.png, points_vis.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_15.png, target_16.png, target_17.png, target_18.png, target_19.png, target_2.png, target_20.png, target_21.png, target_22.png, target_23.png, target_24.png, target_25.png, target_26.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
├── val/ (1149 个 sample 目录)
│   └── sample_00000/
│       └── input.png, points_vis.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_15.png, target_16.png, target_17.png, target_18.png, target_19.png, target_2.png, target_20.png, target_21.png, target_22.png, target_23.png, target_24.png, target_25.png, target_26.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
├── test/ (575 个 sample 目录)
│   └── sample_00000/
│       └── input.png, points_vis.png, target.png, target_1.png, target_10.png, target_11.png, target_12.png, target_13.png, target_14.png, target_15.png, target_16.png, target_17.png, target_18.png, target_19.png, target_2.png, target_20.png, target_21.png, target_22.png, target_23.png, target_24.png, target_25.png, target_26.png, target_3.png, target_4.png, target_5.png, target_6.png, target_7.png, target_8.png, target_9.png
```

- split 文件：`LUNA16/data/train.json`, `LUNA16/data/val.json`, `LUNA16/data/test.json`
- 字段：`image_path, target_path, type, task, modality, dataset, points, point_ids, target_point_paths, points_vis_path`；点字段：`['x', 'y']`。
- 路径字段统计：`{'image_path:relative': 5747, 'target_path:relative': 5747, 'target_point_paths:relative': 151816, 'points_vis_path:relative': 5747}`。
- 输入 PNG：5747；所有文件扩展名计数（含辅助图像）：`{'.json': 3, '.png': 169057}`；未被 split 引用的 input.png：0。
- 输入格式：`{'PNG': 5747}`；RGB 三通道像素完全相同/灰度图像数：5747。磁盘模式见总表，视觉上灰度不等于单通道文件。
- 点数分布：`{'26': 5348, '32': 399}`；名称：`无；使用 point_id 占位名称`。
- 完全相同 source_dir 跨 split 数：0。这不是患者级去重证明；未改动或重建患者划分。

| split | images | annotations | points | empty | 越界点 | 验证通过 |
|---|---:|---:|---:|---:|---:|---|
| train | 4023 | 4023 | 106242 | 0 | 0 | True |
| val | 1149 | 1149 | 30414 | 0 | 0 | True |
| test | 575 | 575 | 15160 | 0 | 0 | True |

## 运行方式

在项目根目录使用已安装依赖的虚拟环境（Pillow、NumPy）：

```bash
.venv/bin/python datasets_analysis/converters/common.py --dataset all
.venv/bin/python datasets_analysis/converters/convert_TG3K.py --data-root /path/to/data --output-root /path/to/coco_format
.venv/bin/python datasets_analysis/tools/check_dataset.py
.venv/bin/python datasets_analysis/tools/visualize_keypoints.py --count 3 --seed 42
.venv/bin/python datasets_analysis/tools/write_report.py
.venv/bin/python datasets_analysis/tools/smoke_mmpose.py
```

输出：`datasets_analysis/coco_format/<dataset>/annotations/{train,val,test}.json` 和 `conversion_log.json`；验证报告位于 docs。可视化每个数据集默认 3 张，参数硬限制最多 5 张，只影响预览采样，不影响划分。

## ViTPose 接入结论

七个固定点数数据集已满足 COCO 数据结构要求；不能原样使用 COCO 人体 17 点配置。需要将 keypoint_head.out_channels、data_cfg.num_joints、num_output_channels、dataset_channel、inference_channel 全部改为各自 K，并提供对应 dataset_info。
实际验证：24 个 JSON 均由当前环境 xtcocotools.COCO 成功读取；七个固定 K 数据集的全部 21 个 split 经 TopDownCocoDataset 成功加载，样本数和 K 维度一致。详见 mmpose_smoke_results.json；该检查不运行模型、变换 pipeline 或评估。
此仓库使用 MMPose 旧版 img_prefix；设为 data_root（含末尾 /），ann_file 指向工作区对应 COCO JSON。use_gt_bbox=True、bbox_file=""，使用本次全图 ROI。原始图像不搬移、不复制。
尚无可靠左右翻转配对、upper/lower body 定义和医学 OKS sigmas；关闭人体 RandomFlip、HalfBodyTransform 和 flip_test，避免照搬人体评估参数。评估指标和医学归一化尺度需在训练配置阶段确定。checkpoint 可复用 backbone，输出头 K 不同需重新初始化。
LUNA16 需先确认两种点布局的对应语义，再决定统一 schema 或分别建模；当前输出是保真 COCO 存储，不宣称可直接供单一固定输出头训练。所有数据均未训练模型。
患者身份不是独立字段，source_dir 也不覆盖所有数据集。本次保持原始划分，未认证患者级无泄漏。
