"""Audit and consolidate Painter-3D seeds 1/2/3 without executing any model."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets_analysis.medical_evaluation.metrics import read_json, sha256
from datasets_analysis.tools.summarize_painter3d_seed1_results import (
    DATASETS, DOCS, FIELDS, main as audit_seed, write_csv)

SEEDS = (1, 2, 3)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (3,) and np.isfinite(values).all()
    return float(values.mean()), float(values.std(ddof=1))


def pair(mean, std):
    return f'{mean:.6f} ± {std:.6f}'


def plot_results(individual, summary, points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (name, _, _, _, _) in zip(axes.flat, DATASETS):
        rows = [row for row in individual if row['dataset'] == name]
        stat = next(row for row in summary if row['dataset'] == name)
        ax.plot(SEEDS, [row['MRE'] for row in rows], 'o-', color='#246b9c')
        ax.axhline(stat['MRE_mean'], color='#b44b36', linestyle='--', label='Mean')
        ax.axhspan(stat['MRE_mean']-stat['MRE_std'],
                   stat['MRE_mean']+stat['MRE_std'], color='#b44b36',
                   alpha=.12, label='Mean ± sample SD')
        ax.set(title=name, xlabel='Training seed',
               ylabel=f'Test MRE ({stat["unit"]})', xticks=SEEDS)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    for extension in ('png', 'pdf'):
        fig.savefig(DOCS / f'three_seed_test_mre.{extension}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for ax, (name, _, _, _, _) in zip(axes.flat, DATASETS):
        rows = [row for row in points if row['dataset'] == name]
        ids = [row['point_id'] for row in rows]
        ax.errorbar(ids, [row['MRE_mean'] for row in rows],
                    yerr=[row['MRE_std'] for row in rows], fmt='o-',
                    capsize=3, markersize=4, color='#246b9c')
        ax.set(title=name, xlabel='Original point ID',
               ylabel=f'Test MRE ({rows[0]["unit"]})', ylim=(0, None))
        ticks = ids if len(ids) <= 14 else ids[::2]
        ax.set_xticks(ticks)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Per-keypoint MRE: mean ± sample SD over three training seeds')
    for extension in ('png', 'pdf'):
        fig.savefig(DOCS / f'three_seed_per_keypoint_mre.{extension}', dpi=180)
    plt.close(fig)


def main():
    original_jsons = [ROOT / f'work_dirs/painter3d_seed{seed}/{slug}/test_evaluation/{name}.json'
                      for seed in SEEDS for _, slug, _, _, _ in DATASETS
                      for name in ('predictions', 'metrics')]
    original_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in original_jsons}
    # Revalidate all 12 runs against the actual files, including checkpoint hashes.
    for seed in SEEDS:
        audit_seed(seed)
    audits = {seed: read_json(DOCS / f'seed{seed}_test_audit.json') for seed in SEEDS}
    individual, summary, points, images = [], [], [], []
    for name, slug, k, count, spacing in DATASETS:
        runs = [audits[seed]['datasets'][name] for seed in SEEDS]
        assert all(run['passed'] for run in runs)
        for key in ('annotations', 'config'):
            assert len({run['hashes'][key] for run in runs}) == 1
        assert len({run['source_json_sha256'] for run in runs}) == 1
        assert len({run['checkpoint'] for run in runs}) == 3
        assert all(run['summary']['test_images'] == count and
                   run['summary']['evaluated_keypoints'] == count*k and
                   run['summary']['pixel_spacing_mm'] == spacing for run in runs)
        scores = [run['summary'] for run in runs]
        individual.extend(scores)
        stat = dict(dataset=name, seeds='1,2,3', n_seeds=3,
                    test_images_per_seed=count, evaluated_keypoints_per_seed=count*k,
                    unit=scores[0]['unit'], mse_unit=scores[0]['mse_unit'],
                    pixel_spacing_mm=spacing)
        for metric in FIELDS:
            stat[metric+'_mean'], stat[metric+'_std'] = moments([row[metric] for row in scores])
        stat['MRE_std_over_mean_percent'] = 100*stat['MRE_std']/stat['MRE_mean']
        summary.append(stat)
        keypoint_csvs = {seed: [row for row in read_csv(DOCS / f'seed{seed}_test_per_keypoint.csv')
                               if row['dataset'] == name] for seed in SEEDS}
        for j in range(k):
            group = [keypoint_csvs[seed][j] for seed in SEEDS]
            assert all(int(row['point_id']) == j+1 for row in group)
            assert len({row['name'] for row in group}) == 1
            row = dict(dataset=name, point_id=j+1, name=group[0]['name'],
                       unit=stat['unit'], mse_unit=stat['mse_unit'],
                       evaluated_keypoints_per_seed=count)
            for metric in FIELDS:
                row[metric+'_mean'], row[metric+'_std'] = moments([float(x[metric]) for x in group])
            row['squared_error_share_percent'] = 100*row['MSE_mean']/(k*stat['MSE_mean'])
            points.append(row)
        image_csvs = {
            seed: {int(row['image_id']): row for row in read_csv(DOCS / f'seed{seed}_test_per_image.csv')
                   if row['dataset'] == name} for seed in SEEDS}
        assert len(image_csvs[1]) == count
        assert set(image_csvs[1]) == set(image_csvs[2]) == set(image_csvs[3])
        for iid in sorted(image_csvs[1]):
            group = [image_csvs[seed][iid] for seed in SEEDS]
            assert len({row['file_name'] for row in group}) == 1
            row = dict(dataset=name, image_id=iid, file_name=group[0]['file_name'],
                       unit=stat['unit'], mse_unit=stat['mse_unit'],
                       evaluated_keypoints_per_seed=k)
            for metric in FIELDS:
                row[metric+'_mean'], row[metric+'_std'] = moments([float(x[metric]) for x in group])
            images.append(row)
    assert len(individual) == 12 and len(summary) == 4 and len(points) == 64 and len(images) == 149
    for path, digest in original_hashes.items():
        assert sha256(ROOT / path) == digest, f'Original result changed: {path}'
    write_csv(DOCS / 'three_seed_test_individual.csv', individual)
    write_csv(DOCS / 'three_seed_test_summary.csv', summary)
    write_csv(DOCS / 'three_seed_test_per_keypoint.csv', points)
    write_csv(DOCS / 'three_seed_test_per_image.csv', images)
    audit = dict(all_checks_passed=True, seeds=list(SEEDS), runs=12,
                 unique_test_images=149, annotated_points_per_seed=2308,
                 inference_evaluation_image_instances=447,
                 inference_evaluation_point_instances=6924,
                 aggregation='arithmetic mean of three per-seed test scores',
                 std='sample standard deviation across seeds, ddof=1, n=3',
                 selection='minimum validation MRE within each seed; all three seeds retained',
                 source_result_hashes_unchanged=True, source_result_sha256=original_hashes,
                 scope='File hashes and numerical recomputation; no model execution',
                 summary=summary, datasets={})
    for name, _, _, _, _ in DATASETS:
        local = [row for row in points if row['dataset'] == name]
        assert np.isclose(sum(row['squared_error_share_percent'] for row in local), 100)
        audit['datasets'][name] = dict(
            run_hashes={str(seed): audits[seed]['datasets'][name]['hashes'] for seed in SEEDS},
            top_keypoints_by_MSE=sorted(local, key=lambda row:row['MSE_mean'], reverse=True)[:3],
            top_images_by_MRE=sorted([row for row in images if row['dataset'] == name],
                                   key=lambda row:row['MRE_mean'], reverse=True)[:5])
    (DOCS / 'three_seed_test_audit.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    plot_results(individual, summary, points)
    write_report(individual, summary, points, audit)
    print('PASS: 12 runs audited, three-seed summaries and figures saved.', flush=True)


def write_report(individual, summary, points, audit):
    lines = ['# Painter-3D 四数据集三 seed 最终测试结果', '',
        'seed=1、2、3 的 12 项训练与测试均完成。本次依据已有预测复核指标，没有重新运行模型。',
        '每个 seed 完整覆盖相同的 149 张测试图像、2,308 个标注点；三轮是对相同测试集的重复评价，不是 447 个独立图像样本。', '',
        '## 最终 baseline：均值 ± 样本标准差', '',
        '每个指标先按单个 seed 的全部有效标注点计算，再对三个 seed 的分数取算术平均。',
        '标准差使用样本标准差（ddof=1，n=3），描述训练随机性，不是标准误或置信区间。',
        'MRE、RMSE、MAE 使用单位列；MSE 为对应平方单位。', '',
        '| 数据集 | 单位 | MRE | MSE | RMSE | MAE |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["dataset"]} | {row["unit"]} | ' +
                     ' | '.join(pair(row[m+'_mean'], row[m+'_std']) for m in FIELDS)+' |')
    lines += ['', '## 各 seed 的测试 MRE', '',
        '| 数据集 | seed=1 | seed=2 | seed=3 | 标准差/均值 |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        scores = [x for x in individual if x['dataset'] == row['dataset']]
        lines.append(f'| {row["dataset"]} | ' +
                     ' | '.join(f'{x["MRE"]:.6f}' for x in scores)+
                     f' | {row["MRE_std_over_mean_percent"]:.2f}% |')
    lines += ['', '全部 12 轮的四项分数、验证 MRE 和所选 epoch 见 three_seed_test_individual.csv。', '',
        '## 逐数据集分析', '']
    for stat in summary:
        name = stat['dataset']
        scores = [row for row in individual if row['dataset'] == name]
        top = audit['datasets'][name]['top_keypoints_by_MSE']
        lines += [f'### {name}', '',
            f'- 测试 MRE 为 {pair(stat["MRE_mean"], stat["MRE_std"])} {stat["unit"]}，'
            f'标准差/均值为 {stat["MRE_std_over_mean_percent"]:.2f}%。',
            '- 验证集选出的最佳 epoch 依次为 ' +
            '、'.join(f'seed={row["seed"]}: {row["best_epoch"]}' for row in scores)+'。',
            '- 测试减验证 MRE 依次为 ' +
            '、'.join(f'{row["test_minus_validation_MRE"]:+.6f} {stat["unit"]}' for row in scores)+'。',
            '', '| 原始点编号 | 三 seed MRE | 三 seed MSE | 占合计平方误差 |',
            '|---|---:|---:|---:|']
        for row in top:
            lines.append(f'| {row["point_id"]} | {pair(row["MRE_mean"], row["MRE_std"])} | '
                         f'{pair(row["MSE_mean"], row["MSE_std"])} | {row["squared_error_share_percent"]:.2f}% |')
        if name == 'acdc':
            share = sum(row['squared_error_share_percent'] for row in top[:2])
            lines += ['', f'第 {top[0]["point_id"]}、{top[1]["point_id"]} 点在三轮均为主要误差来源，'
                f'合计贡献 {share:.2f}% 的平方误差。重复出现说明高误差位置在这三次训练中持续存在；'
                '应优先对照这两点的预测、原图及源标注检查一致性。数值本身不能区分标注、点序或模型定位问题，正式分数保留全部点。']
        lines += ['', '跨 seed 平均 MRE 最大的三张测试图像：', '']
        for row in audit['datasets'][name]['top_images_by_MRE'][:3]:
            lines.append(f'- {row["file_name"]}：{pair(row["MRE_mean"], row["MRE_std"])} {row["unit"]}。')
        lines.append('')
    lines += ['## 结论与使用范围', '',
        '- HC18、ACDC 的 MRE 相对波动低于 1%；ChestX-ray 约 1.5%；TG3K 约 8.9%，在四项实验中对训练 seed 的波动最大。',
        '- 不按 test 成绩挑选 seed 或替换 checkpoint；最终 baseline 保留全部三个 seed，以各自验证 MRE 最小的权重评价。',
        '- 各数据集任务、点数及物理尺度不同，不跨数据集混合平均或直接排名。TG3K spacing 未知，保留 px/px²。',
        '- 三个 seed 的 RMSE 先各自计算再取平均，因此最终 RMSE 均值不必等于最终 MSE 均值的平方根。',
        '- MSE=Σ(Δx²+Δy²)/(2N)，RMSE=sqrt(MSE)，MAE=Σ(|Δx|+|Δy|)/(2N)，MRE 为每点欧氏距离的平均。',
        '- 毫米换算基于用户提供的当前 PNG 各向同性 spacing：ACDC=1.5625、HC18=0.108、ChestX-ray=1.4 mm/px。未从原始医学元数据重新验证物理标定。',
        '- 三次固定划分训练的样本标准差反映这三次运行的差异；不代表不同患者划分的不确定性，也不构成统计显著性结论。', '',
        '## 完整性核查', '',
        '- 12 份预测与结果文件覆盖完整；无重复、遗漏、被置信度过滤或非有限坐标。',
        '- 逐轮整体及逐点指标复算与 metrics.json 完全相同，并通过独立坐标公式复核；没有 PCK。',
        '- 预测、结果、标注、配置、checkpoint 哈希匹配，源 test JSON 与转换记录匹配；各数据集三个 seed 使用相同标注、配置与 spacing。',
        '- 原始预测、评价 JSON 和模型权重未改动；新旧同名数据集通过 painter3d 路径保持独立。', '',
        '## 文件', '',
        '- [最终均值和标准差 CSV](three_seed_test_summary.csv)',
        '- [12 轮逐次结果 CSV](three_seed_test_individual.csv)',
        '- [逐点三 seed 统计 CSV](three_seed_test_per_keypoint.csv)',
        '- [逐图三 seed 统计 CSV](three_seed_test_per_image.csv)',
        '- [完整核查 JSON](three_seed_test_audit.json)',
        '- [各 seed MRE 图 PNG](three_seed_test_mre.png) / [PDF](three_seed_test_mre.pdf)',
        '- [逐点 MRE 图 PNG](three_seed_per_keypoint_mre.png) / [PDF](three_seed_per_keypoint_mre.pdf)', '',
        '复现本次整理（只做文件核查和坐标计算）：', '',
        '    .venv/bin/python datasets_analysis/tools/summarize_painter3d_three_seed_results.py', '']
    (DOCS / 'three_seed_test_results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
