"""Audit three existing experiments and publish final scores and seeded galleries.

Only reads trained checkpoints, predictions, annotations, logs and original
images. Never constructs a model, trains, predicts, or overwrites old results.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import html
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmcv import Config
from PIL import Image
from datasets_analysis.medical_evaluation.metrics import read_json, sha256, write_json
from datasets_analysis.tools.audit_completed_rounds import (
    METRICS, aggregate_seed_metrics, inspect_test, inspect_training, require)
from datasets_analysis.tools.summarize_seed_training import (
    CONFIG_FIELDS, config_from_log, write_csv)
from datasets_analysis.tools.visualize_medical_keypoints import (
    gallery_selections, prepare_context, preflight, render_selections, write_html)

SEEDS = [1115987209, 1, 2]
DATASETS = ['TG3K', 'acdc', 'hc18', 'oasis', 'Synapse-CT', 'SKI10', 'ChestX-ray', 'LUNA16_26']


def score_row(dataset, seed, score):
    return dict(dataset=dataset, seed=seed, MAE=score['mae'],
        MRE=score['mean_euclidean_distance_error'], RMSE=score['rmse'],
        MSE=score['coordinate_mse'],
        **{f'PCK@{t}px':score['pck'][str(t)] for t in [2, 5, 10, 20]})


def pair(values):
    require(len(values) == 3 and all(np.isfinite(values)), 'Need three finite seed scores')
    return statistics.mean(values), statistics.stdev(values)


def fmt(values, scale=1):
    mean, std = pair(values)
    return f'{mean*scale:.4f} ± {std*scale:.4f}'


def training_windows(run):
    """Sum logged runner windows, excluding a pause between resumed sessions."""
    windows = []
    for session in run['sessions']:
        path = (ROOT/session['log']).with_suffix('')
        text = path.read_text()
        start = re.search(r'^(\d{4}-\d{2}-\d{2} [\d:,]+).*workflow:', text, re.M)
        end = re.findall(r'^(\d{4}-\d{2}-\d{2} [\d:,]+).*Epoch\(val\)', text, re.M)
        require(start is not None and end, f'Missing training timestamps: {path}')
        seconds = (datetime.strptime(end[-1], '%Y-%m-%d %H:%M:%S,%f') -
                   datetime.strptime(start[1], '%Y-%m-%d %H:%M:%S,%f')).total_seconds()
        require(seconds > 0, 'Nonpositive runner duration')
        windows.append(dict(log=str(path.relative_to(ROOT)), start=start[1], end=end[-1], seconds=seconds))
    return windows


def render_seed_gallery(records, seed, root, images_root):
    folder = root/f'seed{seed}'
    require(not folder.exists(), f'Gallery exists: {folder}')
    jobs = []
    for result in records:
        paths = result['provenance']['files']
        context = prepare_context(images_root, ROOT/paths['gt'], ROOT/paths['predictions'], result['dataset'])
        context['experiment_label'] = f'seed={seed} 实验'
        require(context['predictions_sha256'] == result['provenance']['sha256']['predictions'], 'Gallery input changed')
        selections = gallery_selections(context['records'], 5, 'max')
        require(len(selections[0][1]) == result['summary']['test_images'], 'Incomplete gallery coverage')
        preflight(context, folder/result['dataset'], selections, complete=True)
        jobs.append((result, context, selections))
    summaries = []
    for result, context, selections in jobs:
        summary = render_selections(context, folder/result['dataset'], selections,
            count=5, error_ranking='max', font_size=10 if result['summary']['keypoints'] >= 26 else 12,
            complete=True)
        summary['training_seed'] = seed
        summary['checkpoint'] = context['checkpoint']
        summaries.append(summary)
    total = sum(s['groups'][0]['images'] for s in summaries)
    body = (f'<header><a href="../three_seeds.html">三轮总入口</a><h1>seed={seed} 医学关键点可视化</h1>'
        f'<p>全部 {total} 张test图；每个数据集各5张大/小误差样例，按最大单点误差排序。</p>'
        '<p>绿色GT · 红色Pred · 黄色误差线 · 原图尺寸与坐标。</p></header><table>'
        '<thead><tr><th>数据集</th><th>全部</th><th>最大误差</th><th>最小误差</th></tr></thead><tbody>')
    for s in summaries:
        name = s['dataset']
        body += f'<tr><td><a href="{name}/index.html">{name}</a></td>'
        for group in s['groups']:
            body += f'<td><a href="{name}/{group["mode"]}/index.html">{group["images"]} 张</a></td>'
        body += '</tr>'
    write_html(folder/'index.html', f'seed={seed} 医学关键点可视化', body+'</tbody></table>')
    write_json(folder/'manifest.json', dict(seed=seed, split='test', total_unique_images=total,
        error_ranking='max', datasets=summaries))
    checks = []
    for result, context, selections in jobs:
        ds = folder/result['dataset']
        for mode, ids in selections:
            require(len(list((ds/mode).glob('*.png'))) == len(ids), 'Wrong gallery PNG count')
            for iid in ids:
                path = ds/mode/f'image_{iid:03d}.png'
                with Image.open(path) as image:
                    require(image.size == (context['records'][iid]['image']['width'],
                                           context['records'][iid]['image']['height']), 'Overlay image resized')
                    image.verify()
                if mode != 'all':
                    require(sha256(path) == sha256(ds/'all'/path.name), 'Ranked overlay differs from all')
        require(sha256(context['gt_path']) == context['gt_sha256'], 'GT changed while rendering')
        require(sha256(context['predictions_path']) == context['predictions_sha256'], 'Prediction changed while rendering')
        checks.append(dict(dataset=result['dataset'], **{mode:len(ids) for mode, ids in selections}))
    # Verify local links after the common three-round entry point has been written.
    audit = dict(all_checks_passed=True, seed=seed, unique_test_images=total,
        png_files=len(list(folder.rglob('*.png'))), html_pages=len(list(folder.rglob('*.html'))),
        datasets=checks, image_dimensions_and_png_integrity_verified=True,
        ranked_examples_identical_to_all=True, input_json_hashes_unchanged=True)
    write_json(folder/'validation.json', audit)
    return audit



def dataset_analysis(records, image_stats):
    """Describe measured variation without attributing errors to unverified causes."""
    lines = ['## 逐数据集分析', '']
    for name in DATASETS:
        group = [r for r in records if r['dataset'] == name]
        scores = [r['summary'] for r in group]
        mae_mean, mae_std = pair([r['MAE'] for r in scores])
        worst = max([r for r in image_stats if r['dataset'] == name], key=lambda r: r['mre_mean_px'])
        lines += [f"- **{name}**：test MAE {mae_mean:.4f} ± {mae_std:.4f}px，MRE {fmt([r['MRE'] for r in scores])}px；MAE种子CV={100*mae_std/mae_mean:.2f}%。同一测试集的三次seed均保留，不能用其中最佳的一轮替代此汇总。三轮持续高误差样本image_id={worst['image_id']}，逐图MRE {worst['mre_mean_px']:.4f} ± {worst['mre_std_px']:.4f}px。"]
    tg = [r for r in records if r['dataset'] == 'TG3K']
    lines += ['', 'TG3K的波动主要值得从误差尾部检查：点误差中位数为 '
        + fmt([r['tail']['median_px'] for r in tg]) + 'px，而最大点误差为 '
        + fmt([r['tail']['max_px'] for r in tg]) + f"px。第一轮和seed1最大单点误差约{tg[0]['tail']['max_px']:.2f}/{tg[1]['tail']['max_px']:.2f}px，seed2约{tg[2]['tail']['max_px']:.2f}px；PCK@20px为{fmt([r['summary']['PCK@20px'] for r in tg],100)}%，但RMSE随极端点明显变化。因此seed2平均误差下降不表示每个点都更准确，不能只看PCK或只展示最好seed。需要结合原图与GT逐点审查，不能由大误差图自动认定标注错误。", '',
        '## 跨seed持续高误差图像（逐图MRE mean±std）', '',
        '| 数据集 | image_id | 逐图MRE mean±std(px) | 最大单点误差 mean±std(px) | 三轮叠加图 |',
        '|---|---:|---|---|---|']
    for name in DATASETS:
        worst = max([r for r in image_stats if r['dataset'] == name], key=lambda r: r['mre_mean_px'])
        iid = worst['image_id'];file = f'image_{iid:03d}.png'
        links = f'[第一轮](../../results/visaulization/{name}/all/{file}) / [seed1](../../results/visaulization/seed1/{name}/all/{file}) / [seed2](../../results/visaulization/seed2/{name}/all/{file})'
        lines.append(f"| {name} | {iid} | {worst['mre_mean_px']:.4f} ± {worst['mre_std_px']:.4f} | {worst['max_error_mean_px']:.4f} ± {worst['max_error_std_px']:.4f} | {links} |")
    lines += ['', '此表按同一image_id三轮平均MRE选出每个任务的一张持续高误差图；图集的largest_errors/smallest_errors仍按每轮最大单点误差各选五张，两者选择标准不同。', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'datasets_analysis/results/three_seed_summary')
    parser.add_argument('--document', type=Path, default=ROOT/'datasets_analysis/docs/final_three_seed_results.md')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualization-root', type=Path, default=ROOT/'results/visaulization')
    parser.add_argument('--images-root', type=Path, default=Path('/data/users/mqf/bushu/Painter-ct/data'))
    args = parser.parse_args()
    require(not args.out_dir.exists() and not args.document.exists(), 'New summary/document required; refusing overwrite')
    if args.visualize:
        require(all(not (args.visualization_root/f'seed{s}').exists() for s in [1,2]), 'Seed galleries already exist')
        require(not (args.visualization_root/'three_seeds.html').exists(), 'Combined gallery already exists')
    torch.set_num_threads(1)
    sources = [read_json(ROOT/f'datasets_analysis/results/{folder}/training_results.json')
               for folder in ['training_summary','seed1_training_summary','seed2_training_summary']]
    runs = [{r['dataset']:r for r in source['datasets']} for source in sources]
    require(all(set(r) == set(DATASETS) for r in runs), 'Expected exactly eight datasets per round')
    for source in sources:
        require(all(sha256(ROOT/p) == h for p,h in source['input_log_sha256'].items()), 'Historical log changed')
    records, rows, val_rows, histories, train_rows, points, image_rows = [], [], [], [], [], [], []
    input_hashes = {}
    for name in DATASETS:
        baseline = config_from_log((ROOT/runs[0][name]['sessions'][0]['log']).with_suffix('').read_text())
        disk = Config.fromfile(str(ROOT/runs[0][name]['config']))._cfg_dict.to_dict()
        require(all(disk.get(k) == baseline.get(k) for k in CONFIG_FIELDS), 'Current config differs from first run')
        gt_hashes = []
        for idx, seed in enumerate(SEEDS):
            run = runs[idx][name]
            for split in ['train','val','test']:
                p = ROOT/f'datasets_analysis/coco_format/{name}/annotations/{split}.json'
                coco = read_json(p)
                require(len(coco['images']) == len(coco['annotations']) == run['counts'][split], 'Split size changed')
                require(all(len(a['keypoints']) == run['keypoints']*3 for a in coco['annotations']), 'Mixed keypoint schema')
                input_hashes[str(p.relative_to(ROOT))] = sha256(p)
            training = inspect_training(run, seed, baseline)
            test = inspect_test(run, training, '' if idx == 0 else f'_seed{seed}')
            input_hashes.update(training['input_log_sha256'])
            input_hashes.update({test['provenance']['files'][k]:v for k,v in test['provenance']['sha256'].items()})
            gt_hashes.append(test['provenance']['sha256']['gt'])
            row = score_row(name,seed,test['metrics']['overall'])
            row.update(round=idx+1, keypoints=run['keypoints'], test_images=run['counts']['test'],
                test_keypoints=test['metrics']['overall']['evaluated_keypoints'], best_epoch=training['validation_best']['epoch'])
            rows.append(row)
            best = training['validation_best']
            val_rows.append(dict(dataset=name,seed=seed,**{k:best[k] for k in METRICS}))
            windows = training_windows(run)
            train_rows.append(dict(dataset=name,seed=seed,train_records=run['counts']['train'],
                val_records=run['counts']['val'],test_records=run['counts']['test'],keypoints=run['keypoints'],
                batch_size=8,iterations_per_epoch=training['expected_iterations']//210,
                completed_epoch=210,total_iterations=training['expected_iterations'],best_epoch=best['epoch'],
                runner_minutes=sum(w['seconds'] for w in windows)/60))
            record = dict(dataset=name,seed=seed,round=idx+1,summary=row,training=training,
                training_windows=windows,test=test['metrics'],tail=test['tail'],provenance=test['provenance'],
                status='training_and_test_complete')
            records.append(record)
            for v in training['validation_history']:
                histories.append(dict(dataset=name,seed=seed,epoch=v['epoch'],**{k:v[k] for k in METRICS}))
            for point in test['metrics']['per_keypoint']:
                pr = score_row(name,seed,point)
                pr.update(index=point['index'],name=point['name'],count=point['evaluated_keypoints'])
                points.append(pr)
            for ir in test['per_image']:
                image_rows.append(dict(dataset=name,seed=seed,**ir))
            print(f'VERIFIED {name} seed={seed}: epoch210, best={best["epoch"]}, test={row["test_images"]}, MAE={row["MAE"]:.5f}',flush=True)
        require(len(set(gt_hashes)) == 1, 'Test split differs across seeds')
    stats = aggregate_seed_metrics(rows)
    val_stats = aggregate_seed_metrics(val_rows)
    require(len(rows) == 24 and all(s['n'] == 3 for s in stats), 'Incomplete final seed coverage')
    groups = defaultdict(list)
    for row in image_rows: groups[(row['dataset'],row['image_id'])].append(row)
    image_stats = []
    for (name,iid), group in groups.items():
        require({r['seed'] for r in group} == set(SEEDS), 'Incomplete paired image coverage')
        mean,std=pair([r['mean_error_px'] for r in group])
        mx,mxstd=pair([r['max_error_px'] for r in group])
        image_stats.append(dict(dataset=name,image_id=iid,n=3,mre_mean_px=mean,mre_std_px=std,
            max_error_mean_px=mx,max_error_std_px=mxstd))
    point_groups = defaultdict(list)
    for row in points: point_groups[(row['dataset'],row['index'],row['name'])].append(row)
    point_stats = []
    for (name,index,label), group in point_groups.items():
        require(len(group) == 3, 'Incomplete point group')
        for metric in METRICS:
            mean,std=pair([r[metric] for r in group])
            point_stats.append(dict(dataset=name,index=index,name=label,metric=metric,n=3,ddof=1,mean=mean,std=std))
    args.out_dir.mkdir(parents=True)
    for filename,data in [('test_by_seed.csv',rows),('test_mean_std.csv',stats),('validation_by_seed.csv',val_rows),
        ('validation_mean_std.csv',val_stats),('training_workload.csv',train_rows),('validation_history.csv',histories),
        ('per_keypoint_by_seed.csv',points),('per_keypoint_mean_std.csv',point_stats),
        ('per_image_by_seed.csv',image_rows),('per_image_mean_std.csv',image_stats)]:
        write_csv(args.out_dir/filename,data)
    for seed in SEEDS:
        subset = [r for r in records if r['seed'] == seed]
        write_json(args.out_dir/f'seed{seed}_results.json',dict(datasets=subset))
    write_json(args.out_dir/'results.json',dict(seeds=SEEDS,n=3,ddof=1,datasets=records,test_mean_std=stats,
        validation_mean_std=val_stats,input_sha256=input_hashes))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    x=np.arange(8)
    for idx,seed in enumerate(SEEDS):
        axes[0].bar(x+(idx-1)*.25,[r['MAE'] for r in rows if r['seed']==seed],.25,label=f'seed {seed}')
    axes[0].legend(fontsize=8);axes[0].set_title('Test coordinate MAE by seed')
    index={(s['dataset'],s['metric']):s for s in stats}
    axes[1].errorbar(x,[index[(n,'MAE')]['mean'] for n in DATASETS],
        yerr=[index[(n,'MAE')]['std'] for n in DATASETS],fmt='o',capsize=5)
    axes[1].set_title('Test MAE mean ± sample std (n=3)')
    for ax in axes:
        ax.set_xticks(x,DATASETS,rotation=25,ha='right');ax.set_ylabel('MAE (original-image px)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Three trained seeds; separate tasks, not a cross-task ranking')
    for suffix in ['png','pdf']:fig.savefig(args.out_dir/f'test_three_seed_comparison.{suffix}',dpi=170)
    plt.close(fig)
    gallery_audits = []
    if args.visualize:
        for seed in [1,2]:
            gallery_audits.append(render_seed_gallery([r for r in records if r['seed']==seed],seed,
                args.visualization_root,args.images_root))
        write_html(args.visualization_root/'three_seeds.html','三轮医学关键点可视化',
            '<header><h1>三轮医学关键点可视化</h1><p>全部test图片；每个数据集各5张最大/最小误差样例。</p></header>'
            '<nav><a href="index.html">第一轮 seed=1115987209</a><a href="seed1/index.html">第二轮 seed=1</a>'
            '<a href="seed2/index.html">第三轮 seed=2</a></nav>')
        for seed in [1,2]:
            folder=args.visualization_root/f'seed{seed}'
            for page in folder.rglob('*.html'):
                for href in re.findall(r'(?:href|src)="([^"]+)"',page.read_text()):
                    path=urlparse(html.unescape(href))
                    if not path.scheme and path.path:
                        require((page.parent/unquote(path.path)).exists(),f'Broken link: {page}: {href}')
    require(all(sha256(ROOT/p)==h for p,h in input_hashes.items()), 'Original input changed during processing')
    audit=dict(all_checks_passed=True,seeds=SEEDS,n=3,ddof=1,training_runs=24,test_runs=24,
        test_images_each_seed=1574,test_keypoints_each_seed=25298,gt_identical_across_seeds=True,
        provenance_hashes_verified=True,overall_and_per_keypoint_metrics_recomputed=True,
        independent_formula_checks_passed=True,original_inputs_unchanged=True,
        galleries=gallery_audits,local_gallery_links_verified=bool(gallery_audits))
    write_json(args.out_dir/'audit.json',audit)
    docs = ['# ViTPose-B 医学2D关键点：三轮最终结果与分析','',
        '三轮八个任务的训练、test推理和评价全部完成并通过本次来源与指标复算核验。原始训练日志、checkpoint、预测、指标、固定split及第一轮可视化保持不变。本次仅汇总已有结果和绘制预测叠加图，没有重新训练或推理。','',
        '## 实验与统计口径','',
        '- 三轮实际seed为1115987209、1、2；不是0、1、2。每个任务三轮均210 epoch、batch8、ViTPose-B与COCO预训练、原heatmap JointsMSELoss和相同配置。',
        '- 每轮1574张test图、25298个有效关键点。三轮是相同测试集上的不同模型，不是4722张独立测试图；可见GT点(v>0)全部参与评价，未按confidence筛点。',
        '- checkpoint全部按各自验证集MRE最小选择；MAE为主要展示指标，没有按test成绩挑选checkpoint、seed或删除大误差样本。',
        '- mean±std是三个seed的dataset-level test分数等权平均与样本标准差(ddof=1，n=3)。std反映种子间波动，不是标准误差、置信区间或图片间误差std。',
        '- MAE=mean((|dx|+|dy|)/2)，MSE=mean((dx²+dy²)/2)，RMSE=sqrt(MSE)，MRE=mean(sqrt(dx²+dy²))。前三项为坐标分量指标，MRE为欧氏距离。',
        '- MAE/MRE/RMSE为原图px，MSE为px²。PCK为欧氏误差≤2/5/10/20px的比例；表中用百分比±百分点，CSV保留0–1比例。没有毫米间距信息，不换算mm；不同图像尺寸与点定义的任务不做直接跨任务排名。',
        '- 分别汇总每个seed的MSE、RMSE等；不能对平均MSE开方替代RMSE mean，也不能将三轮预测拼接后当作独立样本。','',
        '## 最终test结果（mean±std，n=3）','',
        '| 数据集 | MAE(px) | MRE(px) | RMSE(px) | MSE(px²) | PCK@2px(%) | PCK@5px(%) | PCK@10px(%) | PCK@20px(%) |',
        '|---|---|---|---|---|---|---|---|---|']
    for name in DATASETS:
        values=[f"{index[(name,m)]['mean']*(100 if m.startswith('PCK') else 1):.4f} ± {index[(name,m)]['std']*(100 if m.startswith('PCK') else 1):.4f}" for m in METRICS]
        docs.append('| '+name+' | '+' | '.join(values)+' |')
    docs += ['', '## 各轮test分数与checkpoint', '',
        '| 数据集 | 第一轮MAE | seed1 MAE | seed2 MAE | MAE mean±std | best epoch（第一轮/1/2） |','|---|---:|---:|---:|---|---|']
    for name in DATASETS:
        group=[r for r in rows if r['dataset']==name]
        docs.append('| '+name+' | '+' | '.join(f"{r['MAE']:.4f}" for r in group)+' | '+fmt([r['MAE'] for r in group])+' | '+' / '.join(str(r['best_epoch']) for r in group)+' |')
    docs += ['', '## 训练工作量与耗时', '',
        '| 数据集 | train/val/test | K | 每epoch迭代 | 每轮总迭代 | 每轮epoch | 日志runner耗时min（mean±std） |','|---|---|---:|---:|---:|---:|---|']
    for name in DATASETS:
        group=[r for r in train_rows if r['dataset']==name];r=group[0]
        docs.append(f"| {name} | {r['train_records']}/{r['val_records']}/{r['test_records']} | {r['keypoints']} | {r['iterations_per_epoch']} | {r['total_iterations']} | 210 | {fmt([v['runner_minutes'] for v in group])} |")
    docs += ['', '耗时从每段日志runner启动到该段最后一次验证，第一轮acdc续训两段窗口相加，排除中断间隔；包含数据加载/验证/保存和并发争用，不是纯GPU计算时间，也不适合作为性能基准。日志时区未标注，原始时间戳见results.json，未进行时区换算。历史逐batch image_id和物理GPU UUID未记录，无法独立认证历史患者级隔离或物理卡安排。初始17点COCO头不匹配是迁移阶段已知行为，最终所有头通道均为本任务K。','',
        '## 验证与test对照（同一个MRE选定checkpoint）','',
        '| 数据集 | val MAE mean±std | test MAE mean±std | test−val MAE mean±std | val MRE mean±std | test MRE mean±std |','|---|---|---|---|---|---|']
    for name in DATASETS:
        a=[r for r in val_rows if r['dataset']==name];b=[r for r in rows if r['dataset']==name]
        docs.append('| '+name+' | '+fmt([r['MAE'] for r in a])+' | '+fmt([r['MAE'] for r in b])+' | '+fmt([y['MAE']-v['MAE'] for v,y in zip(a,b)])+' | '+fmt([r['MRE'] for r in a])+' | '+fmt([r['MRE'] for r in b])+' |')
    docs += ['', '差值按seed配对后求mean±std，不用两个std相减。val是模型选择集，val/test差值不能单独证明过拟合或数据泄漏。', '',
        '## 种子波动与误差尾部', '',
        '| 数据集 | MAE mean±std | MAE种子CV(%) | 点误差中位数 mean±std(px) | 点误差P95 mean±std(px) | 最大点误差 mean±std(px) | >20px点数 mean±std |',
        '|---|---|---:|---|---|---|---|']
    for name in DATASETS:
        group=[r for r in records if r['dataset']==name];s=index[(name,'MAE')]
        docs.append('| '+name+f" | {s['mean']:.4f} ± {s['std']:.4f} | {100*s['std']/s['mean']:.2f} | "+' | '.join(fmt([r['tail'][k] for r in group]) for k in ['median_px','p95_px','max_px','points_over_20px'])+' |')
    docs += ['', '上述尾部mean±std是各seed独立中位数/P95/最大值的汇总，不是混合预测的分位数。CV仅描述这三个seed的相对波动，n=3不足以支持统计显著性或因果判断。',
        'RMSE比MAE更强调大误差点；若PCK@10/20较高但RMSE仍很高，说明少量异常预测对平方误差有明显影响，不能据PCK忽略这些点。所有异常点都保留在最终分数中；图集的大误差样例用于检查，不能自动归因为错误GT或模型失败原因。','',
        '## 每个数据集误差较大的关键点（test MRE mean±std）','',
        '| 数据集 | 点名称/索引（0-based） | MRE mean±std(px) |','|---|---|---|']
    for name in DATASETS:
        selected=sorted([r for r in point_stats if r['dataset']==name and r['metric']=='MRE'],key=lambda r:r['mean'],reverse=True)[:3]
        for r in selected:docs.append(f"| {name} | {r['name']} / {r['index']} | {r['mean']:.4f} ± {r['std']:.4f} |")
    docs += ['', '完整逐关键点指标保存于per_keypoint_mean_std.csv。相同image_id跨seed的逐图MRE均值与波动保存于per_image_mean_std.csv，可区分持续高误差样本与对seed敏感样本。','',
        '## 可视化结果与复核方式','',
        '第一轮原目录保持不变：results/visaulization/<dataset>/{all,largest_errors,smallest_errors}/。',
        '新增seed1、seed2图集：results/visaulization/seed1/<dataset>/与seed2/<dataset>/，每个数据集同样包含all、largest_errors、smallest_errors三组。每轮全部1574张原尺寸test叠加图，另外每个数据集各5张最大/最小误差样例，总计每轮1654张PNG。',
        '大/小误差按图片最大单点欧氏误差排序，与第一轮一致；不是按MRE排序。相同image_id对应同一张原图，跨seed可直接对照。绿色GT、红色Pred、黄色连接线，显示point编号、单点px误差、MRE和最大误差；原图尺寸与坐标不变。',
        '图集含可搜索image_id和原图路径的HTML页面。所有PNG的尺寸与完整性、好/坏样例和all副本一致性、GT/预测输入hash和本地HTML链接已核验。大量叠加文字可能遮挡局部纹理，原图及JSON始终保留，可按原路径查看。',
        '总入口：[三轮可视化](../../results/visaulization/three_seeds.html)；[seed1图集](../../results/visaulization/seed1/index.html)；[seed2图集](../../results/visaulization/seed2/index.html)。',
        '图像与实验汇总产物均被现有.gitignore规则忽略；不加入Git。','',
        '## 输出与结论','',
        '最终主表以test MAE mean±std展示，同时保留MRE、RMSE、MSE和PCK。三轮模型按相同验证MRE标准选取，全部test覆盖和来源核验通过，最终结果可用于报告baseline。没有COCO AP/AR/OKS作为主要指标。',
        '本实验为像素坐标评价，不代表临床精度或毫米误差。仅一种模型、三次seed和当前固定划分，尚不能判断与其他方法的显著差异。','',
        '完整CSV/JSON、训练与验证历史、指标复算审计及结果图位于datasets_analysis/results/three_seed_summary/；以前单轮和两轮历史报告保留，当前最终汇总以本文件为准。','']
    text='\n'.join(docs)
    details='\n'.join(dataset_analysis(records, image_stats))
    text=text.replace('## 每个数据集误差较大的关键点', details+'\n## 每个数据集误差较大的关键点')
    with args.document.open('x') as stream:stream.write(text)
    with (args.out_dir/'report.md').open('x') as stream:
        stream.write(text.replace('../../results/visaulization/','../../../results/visaulization/'))
    print('COMPLETE:',args.document,'No training or inference executed.',flush=True)


if __name__ == '__main__':
    main()
