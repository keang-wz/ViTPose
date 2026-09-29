"""Generate the human-readable report from measured validation results."""
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'converters'))
from common import BASE, DATASETS, SPLITS, load_splits, resolve_image, write_json

def main():
    report=json.loads((BASE/'docs/validation_results.json').read_text())
    root=Path(report['data_root']); comparison={}
    # Inspect only selected datasets in shared manifests.
    for s in SPLITS:
        shared=json.loads((root/f'{s}.json').read_text())
        comparison[s]={}
        for d in DATASETS:
            chosen=[r for r in shared if str(r.get('dataset','')).lower()==d.lower()]
            local=load_splits(root,d)[s]
            def signature(r):
                return (resolve_image(root,d,r['image_path'])[1],json.dumps(r['points'],sort_keys=True),tuple(r['point_ids']))
            comparison[s][d]=dict(shared_records=len(chosen),local_records=len(local),
                same_images_points_ids=Counter(signature(r) for r in chosen)==Counter(signature(r) for r in local))
    write_json(BASE/'docs/shared_split_comparison.json',comparison)
    lines=['# 医学 2D Keypoint 数据分析', '',
        f'数据根目录：`{root}`。所有统计来自本次实际文件读取；完整解码全部 input 图像，辅助 target/label 图像只统计文件，不作为训练输入。',
        '使用每个数据集 `data/{train,val,test}.json`，不创建新划分，不修改 raw 数据。根目录同名共享 JSON 的八个指定数据集记录也已对照；结果见 `shared_split_comparison.json`。',
        '', '| 数据集 | train | val | test | 原始点数 | 输入尺寸 W×H | 模式 |', '|---|---:|---:|---:|---|---|---|']
    for d,a in report['datasets'].items():
        v=a['validation'];lines.append(f'| {d} | {v["train"]["images"]} | {v["val"]["images"]} | {v["test"]["images"]} | {a["keypoint_counts"]} | {a["image_sizes"]} | {a["image_modes"]} |')
    lines+=['','## 共同格式与转换规则','',
        '- 原格式为 JSON 数组，每条记录包含一个 input 图像和一个有序点列表。点为 `{x,y}`，单位为当前 PNG 像素；没有原始 image id、visibility 或独立目标实例列表。多器官可能合在一个点列表，不能据此推断一个解剖目标。',
        '- 原始 point_ids 为 1..K，顺序直接保留；acdc 的 point_names 是字符串数字，其余没有解剖名称。占位名称 point_1 等仅对应原 id，不推断解剖含义。',
        '- 有限坐标转换为 `[x,y,2]`；2 表示已提供标注的训练约定，并非验证过无遮挡。未知点格式或非有限坐标直接报错；不裁剪越界坐标，不删除记录。',
        '- `file_name` 如 `TG3K/data/train/sample_00000/input.png`，相对于共同 data_root。SKI10 的 E:/data/pre/ 前缀按明确 dataset/data/ 后缀映射；不按 basename 模糊匹配。其他 source_dir 中 Windows 路径仅为历史来源，不用于读取图像。',
        '- 每条记录生成一张 image 和一条 annotation，数据集内跨 split 的 id 唯一且确定；原 JSON 行号和 point_ids 保留。没有目标框，使用全图 bbox=[0,0,W,H]、area=W*H、iscrowd=0，属于全图 landmark baseline。',
        '- 图像已经是 PNG 二维导出数据，使用 Pillow/OpenCV 常规读取；不需要读取 DICOM/NIfTI 或再次窗宽窗位处理。转换不改变像素、不缩放坐标。',
        '- LUNA16 的 26 点和 32 点分别保留为两个 category，各自保持原始数组和顺序。未将 26 点填充成 32 点，因为相同 id 的解剖语义尚未确认。COCO 可以存储，当前固定 K 的 TopDownCocoDataset 不能直接混合训练。',
        '', '## 各数据集实际结构与检查','']
    for d,a in report['datasets'].items():
        lines += [f'### {d}', '', '```text',f'{d}/data/', '├── train.json','├── val.json','├── test.json']
        for s,t in a['tree'].items():
            lines += [f'├── {s}/ ({t["sample_directories"]} 个 sample 目录)',f'│   └── {t["example_directory"]}/',f'│       └── '+', '.join(t['example_files'])]
        lines += ['```','',f'- split 文件：'+', '.join('`'+p+'`' for p in a['split_files']),
            f'- 字段：`{", ".join(a["fields"])}`；点字段：`{list(a["point_fields"])}`。',
            f'- 路径字段统计：`{a["path_types"]}`。',
            f'- 输入 PNG：{a["input_images"]}；所有文件扩展名计数（含辅助图像）：`{a["all_file_extensions"]}`；未被 split 引用的 input.png：{a["unreferenced_input_images"]}。',
            f'- 输入格式：`{a["image_formats"]}`；RGB 三通道像素完全相同/灰度图像数：{a["grayscale_pixel_images"]}。磁盘模式见总表，视觉上灰度不等于单通道文件。',
            f'- 点数分布：`{a["keypoint_counts"]}`；名称：`{a["point_names"] or "无；使用 point_id 占位名称"}`。',
            f'- 完全相同 source_dir 跨 split 数：{a["exact_source_dir_cross_split"]}。这不是患者级去重证明；未改动或重建患者划分。',
            '', '| split | images | annotations | points | empty | 越界点 | 验证通过 |', '|---|---:|---:|---:|---:|---:|---|']
        for s,v in a['validation'].items():
            lines.append(f'| {s} | {v["images"]} | {v["annotations"]} | {v["keypoints"]} | {v["empty_annotations"]} | {v["out_of_bounds_points"]} | {v["passed"]} |')
        lines+=['']
    lines+=['## 运行方式','', '在项目根目录使用已安装依赖的虚拟环境（Pillow、NumPy）：','', '```bash',
        '.venv/bin/python datasets_analysis/converters/common.py --dataset all',
        '.venv/bin/python datasets_analysis/converters/convert_TG3K.py --data-root /path/to/data --output-root /path/to/coco_format',
        '.venv/bin/python datasets_analysis/tools/check_dataset.py',
        '.venv/bin/python datasets_analysis/tools/visualize_keypoints.py --count 3 --seed 42',
        '.venv/bin/python datasets_analysis/tools/write_report.py',
        '.venv/bin/python datasets_analysis/tools/smoke_mmpose.py', '```','',
        '输出：`datasets_analysis/coco_format/<dataset>/annotations/{train,val,test}.json` 和 `conversion_log.json`；验证报告位于 docs。可视化每个数据集默认 3 张，参数硬限制最多 5 张，只影响预览采样，不影响划分。',
        '', '## ViTPose 接入结论','',
        '七个固定点数数据集已满足 COCO 数据结构要求；不能原样使用 COCO 人体 17 点配置。需要将 keypoint_head.out_channels、data_cfg.num_joints、num_output_channels、dataset_channel、inference_channel 全部改为各自 K，并提供对应 dataset_info。',
        '实际验证：24 个 JSON 均由当前环境 xtcocotools.COCO 成功读取；七个固定 K 数据集的全部 21 个 split 经 TopDownCocoDataset 成功加载，样本数和 K 维度一致。详见 mmpose_smoke_results.json；该检查不运行模型、变换 pipeline 或评估。',
        '此仓库使用 MMPose 旧版 img_prefix；设为 data_root（含末尾 /），ann_file 指向工作区对应 COCO JSON。use_gt_bbox=True、bbox_file=""，使用本次全图 ROI。原始图像不搬移、不复制。',
        '尚无可靠左右翻转配对、upper/lower body 定义和医学 OKS sigmas；关闭人体 RandomFlip、HalfBodyTransform 和 flip_test，避免照搬人体评估参数。评估指标和医学归一化尺度需在训练配置阶段确定。checkpoint 可复用 backbone，输出头 K 不同需重新初始化。',
        'LUNA16 需先确认两种点布局的对应语义，再决定统一 schema 或分别建模；当前输出是保真 COCO 存储，不宣称可直接供单一固定输出头训练。所有数据均未训练模型。',
        '患者身份不是独立字段，source_dir 也不覆盖所有数据集。本次保持原始划分，未认证患者级无泄漏。', '']
    (BASE/'docs/dataset_analysis.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
