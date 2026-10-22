# Painter-3D seed=3 测试集结果

四个数据集的预测及评价文件均通过核查。共 149 张测试图像、2,308 个有效标注点，无缺失、重复或被忽略的标注点。
本次只读取已有预测并复核数值，没有重新运行模型、训练或修改原始结果文件。

## 总体结果

MRE、RMSE、MAE 使用“单位”列；MSE 使用其平方单位。各项均为全部测试标注点的平均，不是多 seed 均值。

| 数据集 | 图像/点数 | 最佳 epoch | 单位 | MRE | MSE | RMSE | MAE |
|---|---:|---:|---|---:|---:|---:|---:|
| acdc | 75/1050 | 210 | mm | 4.706140 | 76.248749 | 8.732053 | 2.964704 |
| hc18 | 30/390 | 210 | mm | 0.617238 | 0.255434 | 0.505405 | 0.393360 |
| TG3K | 20/100 | 210 | px | 1.096174 | 0.927641 | 0.963141 | 0.671782 |
| ChestX-ray | 24/768 | 210 | mm | 3.049527 | 7.514951 | 2.741341 | 1.897863 |

## 与验证集比较

验证指标来自同一最佳 checkpoint。不同划分的难度和样本数不同，此比较不代表统计显著性，也不用于重新选择权重。

| 数据集 | 验证 MRE | 测试 MRE | 测试减验证 | 相对变化 |
|---|---:|---:|---:|---:|
| acdc | 4.767227 | 4.706140 | -0.061087 mm | -1.28% |
| hc18 | 0.634806 | 0.617238 | -0.017569 mm | -2.77% |
| TG3K | 1.056940 | 1.096174 | +0.039234 px | +3.71% |
| ChestX-ray | 3.603529 | 3.049527 | -0.554002 mm | -15.37% |

## 误差集中位置

以下编号对应原始 point_ids，不推断解剖名称。平方误差占比用于定位总体 MSE 的来源，没有删除任何点或重定义正式指标。

### acdc

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 12 | 20.940310 mm | 588.937439 mm^2 | 55.17% |
| 11 | 17.066142 mm | 424.459881 mm^2 | 39.76% |
| 14 | 4.884961 mm | 17.056617 mm^2 | 1.60% |

按单张图像 MRE 排序，前三个待查看样本：

- `acdc/data/test/sample_00717/input.png`：MRE=10.084324 mm。
- `acdc/data/test/sample_00684/input.png`：MRE=9.903076 mm。
- `acdc/data/test/sample_00697/input.png`：MRE=8.416924 mm。

### hc18

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 13 | 0.741922 mm | 0.333567 mm^2 | 10.05% |
| 8 | 0.710566 mm | 0.314401 mm^2 | 9.47% |
| 6 | 0.652468 mm | 0.291618 mm^2 | 8.78% |

按单张图像 MRE 排序，前三个待查看样本：

- `hc18/data/test/sample_00290/input.png`：MRE=1.197402 mm。
- `hc18/data/test/sample_00281/input.png`：MRE=1.170072 mm。
- `hc18/data/test/sample_00272/input.png`：MRE=1.042927 mm。

### TG3K

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 2 | 1.362461 px | 1.402148 px^2 | 30.23% |
| 3 | 1.230394 px | 1.210912 px^2 | 26.11% |
| 4 | 1.250167 px | 1.104678 px^2 | 23.82% |

按单张图像 MRE 排序，前三个待查看样本：

- `TG3K/data/test/sample_00203/input.png`：MRE=1.714592 px。
- `TG3K/data/test/sample_00205/input.png`：MRE=1.596111 px。
- `TG3K/data/test/sample_00188/input.png`：MRE=1.490275 px。

### ChestX-ray

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 7 | 6.176923 mm | 26.974056 mm^2 | 11.22% |
| 9 | 5.483357 mm | 19.655967 mm^2 | 8.17% |
| 23 | 4.591891 mm | 18.456699 mm^2 | 7.67% |

按单张图像 MRE 排序，前三个待查看样本：

- `ChestX-ray/data/test/sample_00233/input.png`：MRE=3.798320 mm。
- `ChestX-ray/data/test/sample_00230/input.png`：MRE=3.758901 mm。
- `ChestX-ray/data/test/sample_00219/input.png`：MRE=3.731970 mm。

## 分析结论

- ACDC 的第 12、11 点合计贡献 94.93% 的平方误差，整体 MSE/RMSE 偏高主要由这两类点造成。应优先查看其预测与标注的一致性；仅凭数值不能断定是标注问题、点序问题或模型定位失败。
- 测试与验证 MRE 的具体差异见上表。两个划分的难度和样本量不同，单次 seed 的比较不能证明泛化稳定性。
- 这些数据集的任务、关键点定义和尺度不同；TG3K 仍为像素单位，不能将四个 MRE 混合平均或据此直接排名。
- MSE 沿用 x/y 坐标平方误差平均，RMSE=sqrt(MSE)；它与 MRE 的欧氏距离平均定义不同，RMSE 小于 MRE 并非计算错误。
- 物理单位继续基于用户提供的当前 PNG 各向同性 spacing：ACDC=1.5625、HC18=0.108、ChestX-ray=1.4 mm/px。TG3K 后续可用现有像素预测转换。

## 核查范围及产物

- 预测、标注、配置、checkpoint 的 SHA256 与结果元数据匹配；使用此前核实的验证最佳权重：acdc epoch 210、hc18 epoch 210、TG3K epoch 210、ChestX-ray epoch 210。
- 源 test JSON 哈希与转换记录匹配。严格按 annotation_id/image_id 匹配，无遗漏或重复，置信度未过滤任何点。
- 重新计算的整体及逐点指标与原 metrics.json 完全一致，并通过独立 NumPy 公式复核。
- 仅包含 MRE、MSE、RMSE、MAE，无 PCK。原始 predictions.json、metrics.json 和权重均未改动。

[汇总 CSV](seed3_test_summary.csv) · [逐点 CSV](seed3_test_per_keypoint.csv) · [逐图 CSV](seed3_test_per_image.csv) · [完整核查 JSON](seed3_test_audit.json)
