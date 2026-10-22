"""Audit existing Painter-3D test predictions (default seed=1); never run a model."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets_analysis.medical_evaluation.metrics import evaluate, index_gt, read_json, sha256

DOCS = ROOT / 'datasets_analysis/docs/painter3d'
DATASETS = [('acdc', 'acdc', 14, 75, 1.5625),
            ('hc18', 'hc18', 13, 30, .108),
            ('TG3K', 'tg3k', 5, 20, None),
            ('ChestX-ray', 'chestxray', 32, 24, 1.4)]
FIELDS = dict(MRE='mean_euclidean_distance_error', MSE='coordinate_mse',
              RMSE='rmse', MAE='mae')


def four_metrics(delta):
    """Independent formula check; coordinate MSE averages both x and y."""
    dx, dy = delta[:, 0], delta[:, 1]
    mse = float((np.sum(dx*dx) + np.sum(dy*dy)) / (2 * len(delta)))
    return dict(MRE=float(np.mean(np.sqrt(dx*dx + dy*dy))), MSE=mse,
                RMSE=float(np.sqrt(mse)),
                MAE=float((np.sum(np.abs(dx)) + np.sum(np.abs(dy))) / (2 * len(delta))))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(seed=1):
    training = read_json(DOCS / f'seed{seed}_training_audit.json')['datasets']
    audit = dict(seed=seed, scope='Audit existing test predictions; no model execution; no PCK',
                 datasets={})
    summary, image_rows, point_rows = [], [], []
    for name, slug, k, count, spacing in DATASETS:
        folder = ROOT / f'work_dirs/painter3d_seed{seed}' / slug / 'test_evaluation'
        pred_path, metric_path = folder / 'predictions.json', folder / 'metrics.json'
        gt_path = ROOT / 'datasets_analysis/coco_format/painter3d' / name / 'annotations/test.json'
        pred, saved, gt = read_json(pred_path), read_json(metric_path), read_json(gt_path)
        meta = pred['metadata']
        trained = training[name]
        assert trained['checkpoint_checks']['best']['seed'] == seed
        checkpoint = ROOT / trained['checkpoint_checks']['best']['file']
        config = ROOT / trained['config']
        hashes = dict(predictions=sha256(pred_path), metrics=sha256(metric_path),
                      annotations=sha256(gt_path), config=sha256(config),
                      checkpoint=sha256(checkpoint))
        assert hashes['predictions'] == saved['inputs']['prediction_sha256']
        assert hashes['annotations'] == meta['annotation_sha256'] == saved['inputs']['annotation_sha256']
        assert hashes['config'] == meta['config_sha256'] == saved['inputs']['config_sha256']
        assert hashes['checkpoint'] == meta['checkpoint_sha256']
        assert (ROOT / meta['checkpoint']).resolve() == checkpoint.resolve()
        assert (ROOT / meta['config']).resolve() == (ROOT / saved['inputs']['config']).resolve() == config.resolve()
        source = Path('/data/users/mqf/Painter-3D/data') / gt['info']['source_split']
        assert sha256(source) == gt['info']['source_sha256']
        assert meta['images'] == count and meta['model_head_channels'] == k
        assert meta['evaluation']['metric'] == saved['metrics'] == ['MRE', 'MSE', 'RMSE', 'MAE']
        assert meta['evaluation'].get('pixel_spacing_mm') == saved.get('pixel_spacing_mm') == spacing
        # Checks coverage, point ordering, IDs, visibility and finite predictions.
        recalculated = evaluate(pred, gt, thresholds=None, pixel_spacing_mm=spacing)
        for field in ('units', 'images', 'annotations', 'annotated_images',
                      'ignored_unannotated_keypoints', 'overall', 'per_keypoint'):
            assert saved[field] == recalculated[field], (name, field)
        assert not any(key.lower().startswith('pck') for key in saved)
        assert set(saved['overall']) == {'evaluated_keypoints', *FIELDS.values()}
        assert len(pred['predictions']) == saved['images'] == count
        assert saved['overall']['evaluated_keypoints'] == count*k
        assert saved['ignored_unannotated_keypoints'] == 0
        images, _, annotations = index_gt(gt)
        unit = 'mm' if spacing is not None else 'px'
        scale = spacing if spacing is not None else 1
        deltas, local_images = [], []
        for prediction in pred['predictions']:
            ann = annotations[prediction['annotation_id']]
            points = np.asarray(ann['keypoints'], dtype=np.float64).reshape(-1, 3)
            valid = points[:, 2] > 0
            delta = (np.asarray(prediction['keypoints'])[valid] - points[valid, :2]) * scale
            deltas.append(delta)
            local_images.append(dict(dataset=name, seed=seed, unit=unit, mse_unit=unit+'^2',
                image_id=ann['image_id'], annotation_id=ann['id'],
                file_name=images[ann['image_id']]['file_name'],
                evaluated_keypoints=len(delta), **four_metrics(delta)))
        independent = four_metrics(np.concatenate(deltas))
        for metric, value in independent.items():
            assert np.isclose(value, saved['overall'][FIELDS[metric]], rtol=1e-12, atol=1e-12)
        total_squared_error = saved['overall']['coordinate_mse'] * count*k*2
        local_points = []
        for point in saved['per_keypoint']:
            assert 'pck' not in point and point['evaluated_keypoints'] == count
            local_points.append(dict(dataset=name, seed=seed, unit=unit, mse_unit=unit+'^2',
                point_id=point['index']+1, name=point['name'],
                evaluated_keypoints=point['evaluated_keypoints'],
                **{label: point[key] for label, key in FIELDS.items()},
                squared_error_share_percent=100*point['coordinate_mse']*count*2/total_squared_error))
        point_rows.extend(local_points)
        image_rows.extend(local_images)
        val_mre = trained['checkpoint_checks']['best']['best_score']
        row = dict(dataset=name, seed=seed, test_images=count, evaluated_keypoints=count*k,
                   best_epoch=trained['best_validation']['epoch'], unit=unit, mse_unit=unit+'^2',
                   pixel_spacing_mm=spacing, **independent, validation_MRE=val_mre,
                   test_minus_validation_MRE=independent['MRE']-val_mre,
                   test_vs_validation_MRE_percent=100*(independent['MRE']/val_mre-1))
        summary.append(row)
        audit['datasets'][name] = dict(passed=True, summary=row, hashes=hashes,
            source_json_sha256=gt['info']['source_sha256'],
            checkpoint=str(checkpoint.relative_to(ROOT)),
            metric_recalculation_exact_match=True, independent_formula_check_passed=True,
            top_keypoints_by_squared_error=sorted(local_points, key=lambda x:x['MSE'], reverse=True)[:3],
            top_images_by_MRE=sorted(local_images, key=lambda x:x['MRE'], reverse=True)[:5])
        print(f'PASS {name}: {count} images, {count*k} points, MRE={independent["MRE"]:.6f} {unit}', flush=True)

    write_csv(DOCS / f'seed{seed}_test_summary.csv', summary)
    write_csv(DOCS / f'seed{seed}_test_per_image.csv', image_rows)
    write_csv(DOCS / f'seed{seed}_test_per_keypoint.csv', point_rows)
    (DOCS / f'seed{seed}_test_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    lines = [f'# Painter-3D seed={seed} 测试集结果', '',
        '四个数据集的预测及评价文件均通过核查。共 149 张测试图像、2,308 个有效标注点，无缺失、重复或被忽略的标注点。',
        '本次只读取已有预测并复核数值，没有重新运行模型、训练或修改原始结果文件。', '',
        '## 总体结果', '',
        'MRE、RMSE、MAE 使用“单位”列；MSE 使用其平方单位。各项均为全部测试标注点的平均，不是多 seed 均值。', '',
        '| 数据集 | 图像/点数 | 最佳 epoch | 单位 | MRE | MSE | RMSE | MAE |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["dataset"]} | {row["test_images"]}/{row["evaluated_keypoints"]} | {row["best_epoch"]} | {row["unit"]} | '
                     + ' | '.join(f'{row[m]:.6f}' for m in FIELDS) + ' |')
    lines += ['', '## 与验证集比较', '',
        '验证指标来自同一最佳 checkpoint。不同划分的难度和样本数不同，此比较不代表统计显著性，也不用于重新选择权重。', '',
        '| 数据集 | 验证 MRE | 测试 MRE | 测试减验证 | 相对变化 |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["dataset"]} | {row["validation_MRE"]:.6f} | {row["MRE"]:.6f} | '
                     f'{row["test_minus_validation_MRE"]:+.6f} {row["unit"]} | {row["test_vs_validation_MRE_percent"]:+.2f}% |')
    lines += ['', '## 误差集中位置', '',
        '以下编号对应原始 point_ids，不推断解剖名称。平方误差占比用于定位总体 MSE 的来源，没有删除任何点或重定义正式指标。']
    for name, _, _, _, _ in DATASETS:
        detail = audit['datasets'][name]
        lines += ['', f'### {name}', '',
                  '| point_id | MRE | MSE | 占总平方误差 |',
                  '|---|---:|---:|---:|']
        for point in detail['top_keypoints_by_squared_error']:
            lines.append(f'| {point["point_id"]} | {point["MRE"]:.6f} {point["unit"]} | '
                         f'{point["MSE"]:.6f} {point["mse_unit"]} | {point["squared_error_share_percent"]:.2f}% |')
        lines += ['', '按单张图像 MRE 排序，前三个待查看样本：', '']
        for im in detail['top_images_by_MRE'][:3]:
            lines.append(f'- `{im["file_name"]}`：MRE={im["MRE"]:.6f} {im["unit"]}。')
    ac = audit['datasets']['acdc']['top_keypoints_by_squared_error']
    lines += ['', '## 分析结论', '',
        f'- ACDC 的第 {ac[0]["point_id"]}、{ac[1]["point_id"]} 点合计贡献 '
        f'{ac[0]["squared_error_share_percent"]+ac[1]["squared_error_share_percent"]:.2f}% 的平方误差，'
        '整体 MSE/RMSE 偏高主要由这两类点造成。应优先查看其预测与标注的一致性；仅凭数值不能断定是标注问题、点序问题或模型定位失败。',
        '- 测试与验证 MRE 的具体差异见上表。两个划分的难度和样本量不同，单次 seed 的比较不能证明泛化稳定性。',
        '- 这些数据集的任务、关键点定义和尺度不同；TG3K 仍为像素单位，不能将四个 MRE 混合平均或据此直接排名。',
        '- MSE 沿用 x/y 坐标平方误差平均，RMSE=sqrt(MSE)；它与 MRE 的欧氏距离平均定义不同，RMSE 小于 MRE 并非计算错误。',
        '- 物理单位继续基于用户提供的当前 PNG 各向同性 spacing：ACDC=1.5625、HC18=0.108、ChestX-ray=1.4 mm/px。TG3K 后续可用现有像素预测转换。', '',
        '## 核查范围及产物', '',
        '- 预测、标注、配置、checkpoint 的 SHA256 与结果元数据匹配；使用此前核实的验证最佳权重：'
        + '、'.join(f'{row["dataset"]} epoch {row["best_epoch"]}' for row in summary) + '。',
        '- 源 test JSON 哈希与转换记录匹配。严格按 annotation_id/image_id 匹配，无遗漏或重复，置信度未过滤任何点。',
        '- 重新计算的整体及逐点指标与原 metrics.json 完全一致，并通过独立 NumPy 公式复核。',
        '- 仅包含 MRE、MSE、RMSE、MAE，无 PCK。原始 predictions.json、metrics.json 和权重均未改动。', '',
        f'[汇总 CSV](seed{seed}_test_summary.csv) · [逐点 CSV](seed{seed}_test_per_keypoint.csv) · '
        f'[逐图 CSV](seed{seed}_test_per_image.csv) · [完整核查 JSON](seed{seed}_test_audit.json)', '']
    (DOCS / f'seed{seed}_test_results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[1, 2, 3], default=1)
    main(parser.parse_args().seed)
