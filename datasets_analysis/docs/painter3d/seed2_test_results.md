# Painter-3D seed=2 测试集结果

四个数据集的预测及评价文件均通过核查。共 149 张测试图像、2,308 个有效标注点，无缺失、重复或被忽略的标注点。
本次只读取已有预测并复核数值，没有重新运行模型、训练或修改原始结果文件。

## 总体结果

MRE、RMSE、MAE 使用“单位”列；MSE 使用其平方单位。各项均为全部测试标注点的平均，不是多 seed 均值。

| 数据集 | 图像/点数 | 最佳 epoch | 单位 | MRE | MSE | RMSE | MAE |
|---|---:|---:|---|---:|---:|---:|---:|
| acdc | 75/1050 | 200 | mm | 4.781267 | 79.951129 | 8.941540 | 2.995706 |
| hc18 | 30/390 | 210 | mm | 0.623849 | 0.257606 | 0.507549 | 0.400313 |
| TG3K | 20/100 | 200 | px | 1.004514 | 0.761087 | 0.872403 | 0.624074 |
| ChestX-ray | 24/768 | 210 | mm | 2.976058 | 7.436687 | 2.727029 | 1.863546 |

## 与验证集比较

验证指标来自同一最佳 checkpoint。不同划分的难度和样本数不同，此比较不代表统计显著性，也不用于重新选择权重。

| 数据集 | 验证 MRE | 测试 MRE | 测试减验证 | 相对变化 |
|---|---:|---:|---:|---:|
| acdc | 4.645658 | 4.781267 | +0.135609 mm | +2.92% |
| hc18 | 0.645602 | 0.623849 | -0.021753 mm | -3.37% |
| TG3K | 1.132451 | 1.004514 | -0.127936 px | -11.30% |
| ChestX-ray | 3.622189 | 2.976058 | -0.646131 mm | -17.84% |

## 误差集中位置

以下编号对应原始 point_ids，不推断解剖名称。平方误差占比用于定位总体 MSE 的来源，没有删除任何点或重定义正式指标。

### acdc

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 12 | 21.203396 mm | 611.780154 mm^2 | 54.66% |
| 11 | 17.638165 mm | 451.472655 mm^2 | 40.33% |
| 14 | 5.184899 mm | 18.657990 mm^2 | 1.67% |

按单张图像 MRE 排序，前三个待查看样本：

- `acdc/data/test/sample_00705/input.png`：MRE=11.295671 mm。
- `acdc/data/test/sample_00717/input.png`：MRE=10.522346 mm。
- `acdc/data/test/sample_00684/input.png`：MRE=9.678140 mm。

### hc18

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 13 | 0.754570 mm | 0.342179 mm^2 | 10.22% |
| 6 | 0.686840 mm | 0.319987 mm^2 | 9.56% |
| 8 | 0.681174 mm | 0.293965 mm^2 | 8.78% |

按单张图像 MRE 排序，前三个待查看样本：

- `hc18/data/test/sample_00290/input.png`：MRE=1.232062 mm。
- `hc18/data/test/sample_00281/input.png`：MRE=1.160344 mm。
- `hc18/data/test/sample_00272/input.png`：MRE=0.896651 mm。

### TG3K

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 2 | 1.292156 px | 1.136183 px^2 | 29.86% |
| 3 | 1.098854 px | 1.024031 px^2 | 26.91% |
| 4 | 1.024200 px | 0.768178 px^2 | 20.19% |

按单张图像 MRE 排序，前三个待查看样本：

- `TG3K/data/test/sample_00203/input.png`：MRE=1.644983 px。
- `TG3K/data/test/sample_00188/input.png`：MRE=1.570747 px。
- `TG3K/data/test/sample_00195/input.png`：MRE=1.304732 px。

### ChestX-ray

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 7 | 5.866771 mm | 26.445709 mm^2 | 11.11% |
| 9 | 5.335796 mm | 20.974803 mm^2 | 8.81% |
| 24 | 4.051614 mm | 18.291075 mm^2 | 7.69% |

按单张图像 MRE 排序，前三个待查看样本：

- `ChestX-ray/data/test/sample_00211/input.png`：MRE=3.643854 mm。
- `ChestX-ray/data/test/sample_00231/input.png`：MRE=3.571459 mm。
- `ChestX-ray/data/test/sample_00215/input.png`：MRE=3.547334 mm。

## 分析结论

- ACDC 的第 12、11 点合计贡献 94.99% 的平方误差，整体 MSE/RMSE 偏高主要由这两类点造成。应优先查看其预测与标注的一致性；仅凭数值不能断定是标注问题、点序问题或模型定位失败。
- 测试与验证 MRE 的具体差异见上表。两个划分的难度和样本量不同，单次 seed 的比较不能证明泛化稳定性。
- 这些数据集的任务、关键点定义和尺度不同；TG3K 仍为像素单位，不能将四个 MRE 混合平均或据此直接排名。
- MSE 沿用 x/y 坐标平方误差平均，RMSE=sqrt(MSE)；它与 MRE 的欧氏距离平均定义不同，RMSE 小于 MRE 并非计算错误。
- 物理单位继续基于用户提供的当前 PNG 各向同性 spacing：ACDC=1.5625、HC18=0.108、ChestX-ray=1.4 mm/px。TG3K 后续可用现有像素预测转换。

## 核查范围及产物

- 预测、标注、配置、checkpoint 的 SHA256 与结果元数据匹配；使用此前核实的验证最佳权重：acdc epoch 200、hc18 epoch 210、TG3K epoch 200、ChestX-ray epoch 210。
- 源 test JSON 哈希与转换记录匹配。严格按 annotation_id/image_id 匹配，无遗漏或重复，置信度未过滤任何点。
- 重新计算的整体及逐点指标与原 metrics.json 完全一致，并通过独立 NumPy 公式复核。
- 仅包含 MRE、MSE、RMSE、MAE，无 PCK。原始 predictions.json、metrics.json 和权重均未改动。

[汇总 CSV](seed2_test_summary.csv) · [逐点 CSV](seed2_test_per_keypoint.csv) · [逐图 CSV](seed2_test_per_image.csv) · [完整核查 JSON](seed2_test_audit.json)
