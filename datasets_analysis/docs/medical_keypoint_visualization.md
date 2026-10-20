# 医学关键点预测可视化

`datasets_analysis/tools/visualize_medical_keypoints.py` 读取已有 COCO GT、`infer_medical_keypoints.py` 输出的预测 JSON 和原始图像。仅使用 NumPy/Pillow，不读取 checkpoint，不执行模型推理或训练，也不修改已有预测、评价或原始文件。

## 叠加内容

- 保持原始图像宽高，不缩放图像、坐标或误差，不按 EXIF 自动旋转。
- GT 为绿色圆点，预测为红色圆点；两点重合时保留绿色外圈。
- 黄色线连接对应 GT 和预测点，表达真实位移；红色细线仅用于连接预测点与避让后的文字标签。
- 预测点附近显示 `point_1` 等原 point_id 编号，以及 `error=xx.xx px`。
- 顶部在原画布内叠加 dataset、image_id、图像 MRE、单点最大误差及颜色图例。
- 文字尽量避让标题、点和其他标签；只移动标签位置，不移动关键点。密集点布局可能仍有文字重叠，可用 `--font-size 10` 调整。

单点误差为原图像素欧氏距离 `sqrt((pred_x-gt_x)^2+(pred_y-gt_y)^2)`。图像 MRE 为所有有效 GT 点的误差均值；多个目标时按所有有效点计算。GT visibility=0 的点不画、不计入误差，与医学评价一致。confidence 不用于过滤预测。预测越界时不裁剪坐标/误差，标签标注 `outside image`。

## 运行示例

在仓库根目录执行。`--images-root` 是与 COCO `file_name` 拼接的根目录：当前 JSON 已包含 `TG3K/data/...`，因此应传共同 data 根目录。

随机生成10张（默认 seed=42，只影响可视化取样，不改变数据划分）：

```bash
.venv/bin/python datasets_analysis/tools/visualize_medical_keypoints.py \
  --images-root /data/users/mqf/bushu/Painter-ct/data \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --predictions work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --out-dir results/visualizations/TG3K \
  --num-samples 10 --seed 42
```

生成图像 MRE 最大的5张：

```bash
.venv/bin/python datasets_analysis/tools/visualize_medical_keypoints.py \
  --images-root /data/users/mqf/bushu/Painter-ct/data \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --predictions work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --out-dir results/visualizations/TG3K \
  --largest-errors --num-samples 5
```

若要找单点最大误差样本，追加 `--error-ranking max`。默认 `--error-ranking mre`。不重新计算任何预测，只基于已有坐标计算绘图误差。

指定 image_id（不受 num-samples 限制）：

```bash
.venv/bin/python datasets_analysis/tools/visualize_medical_keypoints.py \
  --images-root /data/users/mqf/bushu/Painter-ct/data \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --predictions work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --out-dir results/visualizations/TG3K \
  --image-ids 189 197
```

`--image-ids` 和 `--largest-errors` 互斥；不带这两个参数时默认随机取样。请求数量超过有效图像数量时只生成全部有效图像。图片标题默认从 annotation 路径推断数据集名称，也可用 `--dataset-name TG3K` 显式指定。其余数据集替换 GT、prediction 和输出路径即可；LUNA16_26 保持原图路径 `LUNA16/data/...`。

每张图有多个目标时，必须提供 annotation_id，标签附带目标编号。脚本复用现有 evaluator 校验完整预测覆盖、点序、坐标空间和有限值；预测与 GT 的 annotation SHA256 不一致时拒绝绘图。尺寸不一致、缺图、未知 image_id 都报错，避免错误叠加。

## 输出与 Git

```text
results/visualizations/TG3K/
├── random/
│   ├── image_189.png
│   └── manifest.json
├── largest_errors/
│   ├── image_189.png
│   ├── image_197.png
│   └── manifest.json
└── selected/
    ├── image_189.png
    └── manifest.json
```

文件名按 image_id 命名，数字最少3位。manifest 记录实际选择的图像、每图 MRE/最大误差、GT/预测坐标、逐点误差及输入文件哈希，便于定位具体预测。图像文件和 manifest 都属于可重复生成的实验产物。

`/results/visualizations/` 已加入根 `.gitignore`。已有 `datasets_analysis/visualizations/`、`work_dirs/` 也被忽略。使用其他自定义输出位置时，应确保对应目录被忽略。

已有同名图片或 manifest 时脚本拒绝覆盖，不自动删除旧图片。不同 seed 建议使用独立输出根，例如 `results/visualizations/seed1/TG3K`。输出禁止放入 images-root 内，原始图像目录保持只读使用。

## 小规模验证

可在独立目录生成两张最大误差样例：

```bash
.venv/bin/python datasets_analysis/tools/visualize_medical_keypoints.py \
  --images-root /data/users/mqf/bushu/Painter-ct/data \
  --gt datasets_analysis/coco_format/TG3K/annotations/test.json \
  --predictions work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json \
  --out-dir results/visualizations/smoke/TG3K \
  --largest-errors --error-ranking max --num-samples 2
```

测试覆盖已知3/4坐标差得到5px、按ID匹配、随机可复现、MRE/最大误差排名、缺失点、多目标、原图尺寸/点位置/颜色、原输入不变及拒绝覆盖：

```bash
.venv/bin/python -m unittest discover -s datasets_analysis/tests \
  -p 'test_medical_keypoint_visualization.py' -v
```

## 第一轮八个数据集：全部、大误差、小误差

第一轮现有 prediction JSON 覆盖八个数据集的 test，共1,574张图像。新增批量入口使用最终结果报告中的实际 prediction/GT 来源，校验预测哈希和完整覆盖；没有重新推理，不读取 checkpoint。

```bash
.venv/bin/python datasets_analysis/tools/visualize_first_run.py \
  --out-root results/visaulization \
  --examples 5 --error-ranking max
```

这里按用户指定保留目录拼写 `visaulization`。每个数据集有三个文件夹：

```text
results/visaulization/
├── index.html
├── manifest.json
├── TG3K/
│   ├── index.html
│   ├── summary.json
│   ├── all/                  # 全部 test 图片
│   ├── largest_errors/       # 5张最大误差样例
│   └── smallest_errors/      # 5张最小误差样例
├── acdc/
├── hc18/
├── oasis/
├── Synapse-CT/
├── SKI10/
├── ChestX-ray/
└── LUNA16_26/
```

每个图片文件夹含 PNG、manifest.json 和 index.html。打开根 index.html 可选择数据集及分组，分组画廊支持按 image_id 或原图路径筛选，点击图片查看原尺寸PNG。画廊里的缩略显示只影响浏览，不修改PNG尺寸或坐标。

全部图只渲染一次；大小误差样例从 all/ 复制相同PNG。默认按图内最大单点误差排序：最大误差组用于观察严重失败点，最小误差组用于观察全部点都较准确的样本。`--error-ranking mre` 可改为图像平均误差排序。此处“好/差”是本数据集内的相对排序，没有医学合格阈值含义。

预期数量：

| 数据集 | all | largest_errors | smallest_errors |
|---|---:|---:|---:|
| TG3K | 20 | 5 | 5 |
| acdc | 75 | 5 | 5 |
| hc18 | 30 | 5 | 5 |
| oasis | 135 | 5 | 5 |
| Synapse-CT | 374 | 5 | 5 |
| SKI10 | 376 | 5 | 5 |
| ChestX-ray | 24 | 5 | 5 |
| LUNA16_26 | 540 | 5 | 5 |

总计1,574张不同测试图，含80份样例副本时共1,654个PNG。当前没有第一轮 train/val prediction JSON，因此这些画廊对应完整 test 预测，不包含未推理的 train/val 图像。

单数据集入口也支持：

- `--all-images`：生成该输入中全部有效图像到 all/，忽略 num-samples 上限。
- `--smallest-errors`：生成最小误差样例到 smallest_errors/。
- `--complete-gallery --num-samples 5 --error-ranking max`：一次生成该数据集三个分组。

`/results/visaulization/` 已加入 Git 忽略规则。重复生成时请使用新的 --out-root，不覆盖第一轮或其他 seed 的结果。
