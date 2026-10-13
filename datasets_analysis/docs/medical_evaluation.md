# ViTPose 医学 2D 关键点评估

此流程保持 `JointsMSELoss(use_target_weight=True)`、热图生成及模型训练逻辑不变。**训练 heatmap MSE 和测试 coordinate MSE 是不同指标**。没有启动训练。

## 推理

从仓库根目录运行；`epoch_210.pth` 只是路径示例，必须替换为以后完成训练得到的 TG3K 五点 checkpoint：

```bash
.venv/bin/python datasets_analysis/tools/infer_medical_keypoints.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py \
  work_dirs/vitpose_base_tg3k_256x192/epoch_210.pth \
  --test-json datasets_analysis/coco_format/TG3K/annotations/test.json \
  --out work_dirs/tg3k_evaluation/test_predictions.json \
  --device cuda:0 --batch-size 8 --workers 2
```

CPU 可用 `--device cpu --batch-size 1 --workers 0`。图片根目录默认读取 config 的 test.img_prefix；搬迁数据后使用 `--data-root /path/to/data`，与 COCO 相对 file_name 拼接。必须显式传 test JSON，脚本不会误用 config 中的 val。既有划分和原始文件不修改。

直接调用 MMPose 测试 pipeline、`model.eval()`、`torch.inference_mode()` 和原始热图解码，坐标已从热图/模型输入映射回**原始 PNG 像素坐标**，不再进行第二次缩放。使用 GT bbox；当前转换数据为全图 ROI，不运行目标检测器。推理不把 GT 关键点坐标传入模型。适配器目前仅支持每张图一条目标标注、单个固定 K；可变点数的 LUNA16 不能直接混用。

checkpoint 严格匹配全部模型权重；17 点人体 COCO checkpoint 会报错，不允许跳过输出头后使用随机五点头生成“正式结果”。现有 `checkpoints/vitpose-b.pth` 是训练初始化来源，不是 TG3K 测试权重。匹配的五点权重仍需由使用者确保对应正确点序和训练任务。脚本不训练或自动生成任何正式 checkpoint。

预测 JSON 包含所有图像/目标，记录 config、checkpoint 和 annotation 的 SHA256。confidence 是解码器返回的逐点热图峰值分数，**不是校准概率**，可以为负数或超过 1；不做裁剪。

下例仅展示一条记录，数值为说明格式而构造，非模型实测结果。完整输出中的 `predictions` 必须覆盖全部测试标注：

```json
{
  "schema_version": 1,
  "coordinate_space": "original_image_pixels",
  "confidence_type": "decoded heatmap peak score; not a calibrated probability",
  "predictions": [
    {
      "image_id": 189,
      "annotation_id": 189,
      "category_id": 1,
      "keypoint_names": ["point_1", "point_2", "point_3", "point_4", "point_5"],
      "keypoints": [[270.2, 541.7], [280.1, 275.0], [292.8, 892.1], [198.3, 668.6], [340.4, 891.5]],
      "confidence": [0.93, 0.88, 0.85, 0.91, 0.87]
    }
  ]
}
```

## 离线医学评价

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/tg3k_evaluation/test_predictions.json \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --pck-thresholds 2 5 10 20 --pck-unit pixels \
  --out work_dirs/tg3k_evaluation/test_metrics.json
```

不依赖模型、GPU 或 COCO evaluator，只使用 Python/NumPy。JSON 输出包含整体和逐关键点指标、有效点数量、忽略的未标注点数量，以及输入文件哈希。已有输出文件不会覆盖，请使用新文件名。

设参与评价的 GT 点数为 N，dx、dy 为预测与 GT 在原图上的坐标差：

| 指标 | 定义 | 单位 |
|---|---|---|
| coordinate MSE | sum(dx²+dy²)/(2N) | pixel² |
| RMSE | sqrt(coordinate MSE) | pixel |
| MAE | sum(abs(dx)+abs(dy))/(2N) | pixel |
| mean Euclidean distance error（MRE） | sum(sqrt(dx²+dy²))/N | pixel |
| PCK@t px | count(sqrt(dx²+dy²) <= t)/N | 0–1 比例 |

RMSE 是逐坐标 RMSE，**不是** sqrt(mean(dx²+dy²))。PCK 边界包含等号。默认阈值为 2、5、10、20 像素，属于实验阈值，不是医学合格标准。

可选尺寸归一化 PCK：

```bash
.venv/bin/python datasets_analysis/tools/evaluate_medical_keypoints.py \
  --predictions work_dirs/tg3k_evaluation/test_predictions.json \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --pck-unit image_diagonal --pck-thresholds 0.01 0.02 0.05 0.1 \
  --out work_dirs/tg3k_evaluation/test_metrics_diagonal.json
```

此时逐点距离除以所在图片的 sqrt(W²+H²)；仅 PCK 归一化，其余四项保持像素单位。没有像素间距信息，不能将结果解释成毫米误差。

按 GT visibility>0（包含 v=1、2）评价；v=0 排除。所有有效点等权微平均，不先平均每张图片。置信度不用于过滤或加权。预测越界不截断，按真实距离评价。无有效 GT 点时报告错误；某个关键点全未标注时，其逐点指标为 null。缺失预测、重复 ID、未知图像、维度/点序错误、NaN/Inf、GT 哈希不一致均报错，不输出看似完整的指标。多目标离线评价要求 annotation_id，不用最近邻匹配掩盖错误。

下面是**数学示例，非真实模型成绩**：若 100 个点全部具有 dx=3、dy=4，overall 为：

```json
{
  "evaluated_keypoints": 100,
  "coordinate_mse": 12.5,
  "rmse": 3.5355339059327378,
  "mae": 3.5,
  "mean_euclidean_distance_error": 5.0,
  "pck": {"2": 0.0, "5": 1.0, "10": 1.0, "20": 1.0}
}
```

## 与之后训练验证的衔接

新增 `MedicalCocoDataset` 继承原 `TopDownCocoDataset`，只覆盖 `evaluate`。TG3K config 用 custom_imports 注册该类，验证报告 MSE、RMSE、MAE、MRE、PCK；`save_best='MRE', rule='less'`，不再按 AP 选模型。不使用 COCO AP/AR/OKS，dataset_info 中无虚构医学 sigmas。原人体官方配置及所有模型/训练源码均未改动。

训练加载方式、训练 loss 和数据增强保持当前 TG3K 配置的实现。test 指向既有 test.json，val 仍指向 val.json；将来只用 val 做模型选择，test 用于最终评估。

## 验证代码

```bash
.venv/bin/python -m unittest discover -s datasets_analysis/tests -p 'test_medical_evaluation.py' -v
.venv/bin/python datasets_analysis/tools/check_tg3k_dataloader.py
```

已知答案测试覆盖指标定义、PCK 边界、归一化、visibility、低置信度、乱序匹配、缺失/重复预测、非法值、多目标匹配及文件保护。集成检查使用临时随机小型 ViT，遍历真实 TG3K test 的 20 张图、生成预测并评价，再检查原始 GT 在 MedicalCocoDataset.evaluate 中的近零误差以及错误输出头拒绝逻辑。临时权重/预测自动删除；不把随机模型结果作为 baseline 成绩。
