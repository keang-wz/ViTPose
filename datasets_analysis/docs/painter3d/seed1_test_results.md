# Painter-3D seed=1 测试集结果

四个数据集的预测及评价文件均通过核查。共 149 张测试图像、2,308 个有效标注点，无缺失、重复或被忽略的标注点。
本次只读取已有预测并复核数值，没有重新运行模型、训练或修改原始结果文件。

## 总体结果

MRE、RMSE、MAE 使用“单位”列；MSE 使用其平方单位。各项均为全部测试标注点的平均，不是多 seed 均值。

| 数据集 | 图像/点数 | 最佳 epoch | 单位 | MRE | MSE | RMSE | MAE |
|---|---:|---:|---|---:|---:|---:|---:|
| acdc | 75/1050 | 210 | mm | 4.783656 | 79.463872 | 8.914251 | 2.990531 |
| hc18 | 30/390 | 210 | mm | 0.617866 | 0.253445 | 0.503434 | 0.394182 |
| TG3K | 20/100 | 200 | px | 0.916774 | 0.631528 | 0.794687 | 0.560677 |
| ChestX-ray | 24/768 | 210 | mm | 2.966224 | 7.087031 | 2.662148 | 1.855502 |

## 与验证集比较

验证指标来自同一最佳 checkpoint。不同划分的难度和样本数不同，此比较不代表统计显著性，也不用于重新选择权重。

| 数据集 | 验证 MRE | 测试 MRE | 测试减验证 | 相对变化 |
|---|---:|---:|---:|---:|
| acdc | 4.708065 | 4.783656 | +0.075590 mm | +1.61% |
| hc18 | 0.630001 | 0.617866 | -0.012134 mm | -1.93% |
| TG3K | 0.965550 | 0.916774 | -0.048777 px | -5.05% |
| ChestX-ray | 3.597475 | 2.966224 | -0.631251 mm | -17.55% |

## 误差集中位置

以下编号对应原始 point_ids，不推断解剖名称。平方误差占比用于定位总体 MSE 的来源，没有删除任何点或重定义正式指标。

### acdc

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 12 | 20.486232 mm | 579.428433 mm^2 | 52.08% |
| 11 | 18.507271 mm | 479.105820 mm^2 | 43.07% |
| 14 | 5.058098 mm | 17.682010 mm^2 | 1.59% |

按单张图像 MRE 排序，前三个待查看样本：

- `acdc/data/test/sample_00705/input.png`：MRE=11.140510 mm。
- `acdc/data/test/sample_00717/input.png`：MRE=10.252767 mm。
- `acdc/data/test/sample_00684/input.png`：MRE=9.907822 mm。

### hc18

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 13 | 0.727625 mm | 0.336194 mm^2 | 10.20% |
| 6 | 0.674029 mm | 0.302517 mm^2 | 9.18% |
| 2 | 0.661607 mm | 0.287332 mm^2 | 8.72% |

按单张图像 MRE 排序，前三个待查看样本：

- `hc18/data/test/sample_00290/input.png`：MRE=1.241062 mm。
- `hc18/data/test/sample_00281/input.png`：MRE=1.175540 mm。
- `hc18/data/test/sample_00272/input.png`：MRE=0.947031 mm。

### TG3K

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 2 | 1.273976 px | 1.148239 px^2 | 36.36% |
| 4 | 1.105561 px | 0.749930 px^2 | 23.75% |
| 5 | 0.886496 px | 0.541056 px^2 | 17.13% |

按单张图像 MRE 排序，前三个待查看样本：

- `TG3K/data/test/sample_00205/input.png`：MRE=1.534377 px。
- `TG3K/data/test/sample_00193/input.png`：MRE=1.413091 px。
- `TG3K/data/test/sample_00203/input.png`：MRE=1.241514 px。

### ChestX-ray

| point_id | MRE | MSE | 占总平方误差 |
|---|---:|---:|---:|
| 7 | 5.347873 mm | 21.501923 mm^2 | 9.48% |
| 9 | 5.083054 mm | 18.810135 mm^2 | 8.29% |
| 24 | 4.336188 mm | 17.574497 mm^2 | 7.75% |

按单张图像 MRE 排序，前三个待查看样本：

- `ChestX-ray/data/test/sample_00220/input.png`：MRE=3.614240 mm。
- `ChestX-ray/data/test/sample_00222/input.png`：MRE=3.544700 mm。
- `ChestX-ray/data/test/sample_00230/input.png`：MRE=3.507624 mm。

## 分析结论

- ACDC 的第 12、11 点合计贡献 95.15% 的平方误差，整体 MSE/RMSE 偏高主要由这两类点造成。应优先查看其预测与标注的一致性；仅凭数值不能断定是标注问题、点序问题或模型定位失败。
- 测试与验证 MRE 的具体差异见上表。两个划分的难度和样本量不同，单次 seed 的比较不能证明泛化稳定性。
- 这些数据集的任务、关键点定义和尺度不同；TG3K 仍为像素单位，不能将四个 MRE 混合平均或据此直接排名。
- MSE 沿用 x/y 坐标平方误差平均，RMSE=sqrt(MSE)；它与 MRE 的欧氏距离平均定义不同，RMSE 小于 MRE 并非计算错误。
- 物理单位继续基于用户提供的当前 PNG 各向同性 spacing：ACDC=1.5625、HC18=0.108、ChestX-ray=1.4 mm/px。TG3K 后续可用现有像素预测转换。

## 核查范围及产物

- 预测、标注、配置、checkpoint 的 SHA256 与结果元数据匹配；使用此前核实的验证最佳权重：acdc epoch 210、hc18 epoch 210、TG3K epoch 200、ChestX-ray epoch 210。
- 源 test JSON 哈希与转换记录匹配。严格按 annotation_id/image_id 匹配，无遗漏或重复，置信度未过滤任何点。
- 重新计算的整体及逐点指标与原 metrics.json 完全一致，并通过独立 NumPy 公式复核。
- 仅包含 MRE、MSE、RMSE、MAE，无 PCK。原始 predictions.json、metrics.json 和权重均未改动。

[汇总 CSV](seed1_test_summary.csv) · [逐点 CSV](seed1_test_per_keypoint.csv) · [逐图 CSV](seed1_test_per_image.csv) · [完整核查 JSON](seed1_test_audit.json)
