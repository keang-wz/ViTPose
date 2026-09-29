# 医学关键点数据处理

真实数据分析见 [docs/dataset_analysis.md](docs/dataset_analysis.md)，完整验证见 [docs/validation_results.json](docs/validation_results.json)。

在 ViTPose 项目根目录执行：

```bash
.venv/bin/python datasets_analysis/converters/common.py --dataset all
.venv/bin/python datasets_analysis/tools/check_dataset.py
.venv/bin/python datasets_analysis/tools/visualize_keypoints.py --count 3
.venv/bin/python datasets_analysis/tools/write_report.py
.venv/bin/python datasets_analysis/tools/smoke_mmpose.py
```

八个独立入口位于 converters/convert_<dataset>.py，统一支持 `--data-root`、`--output-root`。转换仅读已有 split，无随机划分操作。输出禁止位于 raw 数据根目录内；每份 JSON 附原始 split SHA256，conversion_log.json 记录转换数量。

coco_format/ 内含 24 份已生成 COCO JSON；samples/ 每个数据集 3 张预览。LUNA16 两种 schema 分别存储为 category，尚不适配单个固定 K 的输出头；其他七个数据集通过 MMPose loader 检查。没有训练模型。
