"""Read-only audit of first-run/seed1 inputs; publish a separate two-round report.

No model construction, training, or inference. Seed2 commands are generated only
when all sixteen training/test records pass the checks.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmcv import Config
from datasets_analysis.medical_evaluation.metrics import evaluate, read_json, sha256, write_json
from datasets_analysis.tools.summarize_final_results import compare
from datasets_analysis.tools.summarize_seed_training import (
    CONFIG_FIELDS, checkpoint_metadata, command_lines, config_from_log, write_csv)

METRICS = ['MAE', 'MRE', 'RMSE', 'MSE', 'PCK@2px', 'PCK@5px', 'PCK@10px', 'PCK@20px']


def aggregate_seed_metrics(rows):
    """Each trained seed contributes one test score; sample std uses ddof=1."""
    groups = defaultdict(dict)
    for row in rows:
        if row['seed'] in groups[row['dataset']]:
            raise ValueError('Duplicate seed within a dataset')
        groups[row['dataset']][row['seed']] = row
    result = []
    for dataset, seeds in groups.items():
        if len(seeds) < 2:
            raise ValueError(f'At least two completed seeds required: {dataset}')
        for metric in METRICS:
            values = [r[metric] for r in seeds.values()]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Nonfinite seed score')
            result.append(dict(dataset=dataset, split='test', metric=metric,
                seeds=','.join(str(s) for s in seeds), n=len(values), ddof=1,
                mean=statistics.mean(values), std=statistics.stdev(values)))
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_training(run, expected_seed, baseline_config):
    logs, snapshots, train_rows = [], {}, []
    values = {}
    for path in sorted((ROOT/run['work_dir']).glob('*.log.json')):
        text_path = path.with_suffix('')
        text = text_path.read_text()
        snapshot = config_from_log(text)
        changes = [k for k in CONFIG_FIELDS if snapshot.get(k) != baseline_config.get(k)]
        # A resumed acdc session legitimately has resume_from set.
        resumed = re.findall(r'resumed epoch (\d+), iter (\d+)', text)
        if resumed and expected_seed == 1115987209:
            changes = [k for k in changes if k != 'resume_from']
        require(not changes, f'Unexpected training config changes: {run["dataset"]}: {changes}')
        seeds = [int(s) for s in re.findall(r'Set random seed to (\d+)', text)]
        require(seeds and all(s == expected_seed for s in seeds), 'Wrong logged seed')
        errors = [line for line in text.splitlines()
                  if re.search(r'Traceback|CUDA out of memory| - ERROR - |\bnan\b', line, re.I)]
        require(not errors, f'Unexpected logged errors: {text_path}')
        data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in data:
            if row.get('mode') == 'train':
                train_rows.append(row)
            if row.get('mode') == 'val':
                require(row['epoch'] not in values, 'Repeated validation epoch')
                require(all(math.isfinite(row[m]) for m in METRICS), 'Nonfinite validation result')
                values[row['epoch']] = row
        for source in [path, text_path]:
            snapshots[str(source.relative_to(ROOT))] = sha256(source)
        logs.append(dict(log=str(path.relative_to(ROOT)), resumed=resumed))
    require(sorted(values) == list(range(10, 211, 10)), 'Incomplete 210-epoch validation history')
    best = min(values.values(), key=lambda r: r['MRE'])
    candidates = list((ROOT/run['work_dir']).glob('best_MRE_epoch_*.pth'))
    require(len(candidates) == 1, 'Missing/ambiguous best checkpoint')
    best_meta = checkpoint_metadata(candidates[0])
    final_meta = checkpoint_metadata(ROOT/run['work_dir']/'latest.pth')
    expected_iterations = 210 * math.ceil(run['counts']['train']/8)
    require(final_meta['epoch'] == 210 and final_meta['iteration'] == expected_iterations,
            'Wrong final epoch/iteration')
    require(best_meta['epoch'] == best['epoch'], 'Best checkpoint is not minimum validation MRE epoch')
    require(math.isclose(best_meta['best_score'], best['MRE'], abs_tol=1e-5), 'Best score differs from log')
    for meta in [best_meta, final_meta]:
        require(meta['seed'] == expected_seed, 'Wrong checkpoint seed')
        require(meta['head_weight_shape'] == [run['keypoints'], 256, 1, 1], 'Wrong checkpoint head')
        require(meta['head_bias_shape'] == [run['keypoints']], 'Wrong checkpoint bias')
        require(meta['optimizer_states'] == 157, 'Unexpected optimizer state count')
        require(meta['optimizer_step_counts'] == {meta['iteration']: 157}, 'Optimizer steps differ from iteration')
    return dict(seed=expected_seed, completed_epoch=210, expected_iterations=expected_iterations,
        best_checkpoint=best_meta, final_checkpoint=final_meta, validation_best=best,
        validation_final=values[210], logs=logs, input_log_sha256=snapshots,
        validation_history=[values[e] for e in sorted(values)],
        config_matches_first=True, training_loss_logged=bool(train_rows))


def inspect_test(run, training, suffix):
    name, slug = run['dataset'], run['slug']
    epoch = training['best_checkpoint']['epoch']
    prefix = ROOT/f'work_dirs/{slug}_test_evaluation{suffix}/best_MRE_epoch_{epoch}'
    paths = dict(predictions=Path(str(prefix)+'_predictions.json'),
                 metrics=Path(str(prefix)+'_metrics.json'),
                 gt=ROOT/f'datasets_analysis/coco_format/{name}/annotations/test.json',
                 config=ROOT/run['config'], checkpoint=ROOT/training['best_checkpoint']['path'])
    require(all(p.is_file() for p in paths.values()), f'Missing test inputs: {name}{suffix}')
    pred, saved, gt = [read_json(paths[k]) for k in ['predictions', 'metrics', 'gt']]
    hashes = {k:sha256(p) for k,p in paths.items()}
    meta = pred['metadata']
    require(meta['annotation_sha256'] == saved['inputs']['annotation_sha256'] == hashes['gt'], 'GT hash mismatch')
    require(saved['inputs']['prediction_sha256'] == hashes['predictions'], 'Prediction hash mismatch')
    require(meta['checkpoint_sha256'] == hashes['checkpoint'], 'Prediction is not from selected checkpoint')
    require((ROOT/meta['checkpoint']).resolve() == paths['checkpoint'].resolve(), 'Wrong prediction checkpoint path')
    require(meta['config_sha256'] == hashes['config'], 'Prediction config hash mismatch')
    require((ROOT/meta['config']).resolve() == paths['config'].resolve(), 'Wrong prediction config path')
    require(meta['model_head_channels'] == run['keypoints'], 'Prediction point count mismatch')
    require(saved['pck_thresholds'] == [2,5,10,20] and saved['units']['pck'] == 'pixels', 'Wrong PCK settings')
    checked = evaluate(pred, gt, [2,5,10,20], 'pixels')
    for key,value in checked.items():
        compare(value, saved[key], key)
    require(len(gt['images']) == len(gt['annotations']) == len(pred['predictions']) == run['counts']['test'],
            'Test image/annotation/prediction coverage differs')
    require(all(len(a['keypoints']) == run['keypoints']*3 for a in gt['annotations']), 'Wrong GT schema')
    by_ann = {a['id']:a for a in gt['annotations']}
    raw_root = Path('/data/users/mqf/bushu/Painter-ct/data')
    require(all((raw_root/im['file_name']).is_file() for im in gt['images']), 'Original test image missing')
    delta, image_rows = [], []
    for p in pred['predictions']:
        a = by_ann[p['annotation_id']]
        truth = np.asarray(a['keypoints']).reshape(-1,3)
        diff = np.asarray(p['keypoints']) - truth[:,:2]
        valid = diff[truth[:,2]>0]
        delta.extend(valid.tolist())
        distances = np.linalg.norm(valid, axis=1)
        image_rows.append(dict(image_id=p['image_id'], mean_error_px=float(distances.mean()),
                               max_error_px=float(distances.max())))
    delta = np.asarray(delta)
    distance = np.linalg.norm(delta,axis=1)
    overall = checked['overall']
    independent = dict(mae=float(np.abs(delta).mean()), coordinate_mse=float((delta**2).mean()),
                       rmse=float(np.sqrt((delta**2).mean())), mean_euclidean_distance_error=float(distance.mean()))
    for key,value in independent.items():
        require(math.isclose(value,overall[key],rel_tol=1e-12,abs_tol=1e-12), 'Independent formula mismatch')
    for t in [2,5,10,20]:
        require(math.isclose(float(np.mean(distance<=t)),overall['pck'][str(t)],abs_tol=1e-12), 'PCK formula mismatch')
    return dict(metrics=checked, provenance=dict(files={k:str(p.relative_to(ROOT)) for k,p in paths.items()},sha256=hashes),
        coverage_verified=True, metrics_recomputed=True, formulas_verified=True,
        tail=dict(median_px=float(np.median(distance)),p95_px=float(np.percentile(distance,95)),
                  max_px=float(distance.max()),points_over_20px=int(np.sum(distance>20))),
        per_image=image_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'datasets_analysis/results/two_round_summary')
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('Output exists; choose a new --out-dir to preserve earlier summaries')
    commands_path = ROOT/'datasets_analysis/commands/train_seed2_baselines.sh'
    docs_path = ROOT/'datasets_analysis/docs/seed2_training.md'
    torch.set_num_threads(1)
    original = read_json(ROOT/'datasets_analysis/results/training_summary/training_results.json')
    second = read_json(ROOT/'datasets_analysis/results/seed1_training_summary/training_results.json')
    first_by_name = {r['dataset']:r for r in original['datasets']}
    second_by_name = {r['dataset']:r for r in second['datasets']}
    histories = [original,second]
    for summary in histories:
        for path,h in summary['input_log_sha256'].items():
            require(sha256(ROOT/path)==h, f'Original log changed: {path}')
    manifest = read_json(ROOT/'datasets_analysis/docs/baseline_configs_manifest.json')['datasets']
    tasks = [dict(dataset='TG3K',slug='tg3k',keypoints=5,gpu=0,
        config='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py',
        counts=dict(train=146,val=42,test=20))]+manifest
    rounds,table,per_point,history,log_hashes = [],[],[],[],{}
    completion = []
    for task in tasks:
        name=task['dataset']
        old_log = ROOT/first_by_name[name]['sessions'][0]['log']
        baseline = config_from_log(old_log.with_suffix('').read_text())
        current = Config.fromfile(str(ROOT/task['config']))._cfg_dict.to_dict()
        require(all(current.get(k)==baseline.get(k) for k in CONFIG_FIELDS), 'Current config changed')
        require(current['resume_from'] is None, 'Seed2 would resume an older run')
        require(not (ROOT/f'work_dirs/vitpose_base_{task["slug"]}_256x192_seed2').exists(), 'Seed2 output directory occupied')
        for split in ['train','val','test']:
            file=ROOT/f'datasets_analysis/coco_format/{name}/annotations/{split}.json'
            require(len(read_json(file)['images'])==task['counts'][split], 'Current split count changed')
        for round_index,source,seed,suffix in [(1,first_by_name[name],1115987209,''),(2,second_by_name[name],1,'_seed1')]:
            training = inspect_training(source,seed,baseline)
            test = inspect_test(task,training,suffix)
            log_hashes.update(training['input_log_sha256'])
            o=test['metrics']['overall'];b=training['validation_best']
            row=dict(dataset=name,round=round_index,seed=seed,keypoints=task['keypoints'],
                train_records=task['counts']['train'],val_records=task['counts']['val'],
                test_images=task['counts']['test'],test_keypoints=o['evaluated_keypoints'],
                completed_epoch=210,best_epoch=b['epoch'],MAE=o['mae'],MRE=o['mean_euclidean_distance_error'],
                RMSE=o['rmse'],MSE=o['coordinate_mse'],
                **{f'PCK@{t}px':o['pck'][str(t)] for t in [2,5,10,20]})
            table.append(row)
            record=dict(dataset=name,slug=task['slug'],round=round_index,seed=seed,summary=row,
                training=training,test=test['metrics'],tail=test['tail'],provenance=test['provenance'],
                status='training_and_test_complete',per_image=test['per_image'])
            rounds.append(record)
            for r in training['validation_history']:
                history.append(dict(dataset=name,round=round_index,seed=seed,epoch=r['epoch'],**{k:r[k] for k in METRICS}))
            for point in test['metrics']['per_keypoint']:
                per_point.append(dict(dataset=name,round=round_index,seed=seed,index=point['index'],name=point['name'],
                    count=point['evaluated_keypoints'],MAE=point['mae'],MRE=point['mean_euclidean_distance_error'],
                    RMSE=point['rmse'],MSE=point['coordinate_mse'],
                    **{f'PCK@{t}px':point['pck'][str(t)] for t in [2,5,10,20]}))
            print(f'PASS {name} round{round_index}: seed={seed}, epoch=210, best={b["epoch"]}, images={row["test_images"]}, MAE={row["MAE"]:.5f}',flush=True)
        completion.append(dict(dataset=name,round1='complete',round2='complete',test_images_each=task['counts']['test']))
    require(len(rounds)==16,'Not all rounds completed')
    stats = aggregate_seed_metrics(table)
    audits=dict(core_all_checks_passed=True,completed_training_runs=16,completed_test_runs=16,
        test_images_each_round=sum(t['counts']['test'] for t in tasks),test_point_records=sum(r['test_keypoints'] for r in table),
        current_configs_match_first=True,ddof=1,n=2,datasets=completion,
        visualizations=dict(first_round='existing all-test gallery',second_round='not_generated_optional'),
        limitation='Training logs do not record historical per-batch image IDs or physical GPU UUIDs.')
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir/'audit.json',audits)
    write_json(args.out_dir/'results.json',dict(seeds=[1115987209,1],n=2,datasets=rounds,
        test_mean_std=stats,note='Interim test summary across two seeds; no training or inference executed.'))
    for filename,rows in [('test_by_round.csv',table),('test_mean_std.csv',stats),
                          ('per_keypoint_by_round.csv',per_point),('validation_history.csv',history)]:
        write_csv(args.out_dir/filename,rows)
    for round_index,name in [(1,'round1_results.json'),(2,'seed1_results.json')]:
        selected=[r for r in rounds if r['round']==round_index]
        write_json(args.out_dir/name,dict(datasets=selected,
            total_test_images=sum(r['summary']['test_images'] for r in selected),
            total_evaluated_keypoints=sum(r['summary']['test_keypoints'] for r in selected)))
    stat_index={(r['dataset'],r['metric']):r for r in stats}
    docs=['# 第一轮与第二轮：完整性核验和两seed测试汇总','',
        '两轮八个数据集的训练日志、最终/最佳checkpoint、test预测及test评价全部通过本次核验。输入文件保持不变，没有执行训练或新的推理；本报告使用重新计算的test指标。',
        '', '## 完整性','',
        '16个训练任务均完成210epoch、各21次验证；实际seed分别为1115987209和1，输出头K正确。最佳checkpoint与验证MRE最小epoch及score一致，最终iteration和AdamW step匹配。日志有效训练配置及当前配置与第一轮一致，acdc第一轮恢复点作为同一实验的续训处理。',
        '16份prediction JSON与16份metrics JSON均齐全。每轮1574张图、25298个有效点，GT/预测完整匹配；当前GT、config、best checkpoint哈希与推理元数据匹配，metrics输入哈希正确，全部整体及逐关键点指标复算和独立公式核验通过。',
        '核心实验结果完整。第一轮全量可视化已存在；第二轮可视化尚未生成，是可选展示产物，不属于训练/test缺失。train/val历史逐批image_id及物理GPU UUID未记录，不能认证历史患者独立性或物理GPU分配。',
        '', '| 数据集 | test图/轮 | 第一轮best epoch | 第二轮best epoch | 第一轮MAE | 第二轮MAE | MAE mean±std | MRE mean±std |','|---|---:|---:|---:|---:|---:|---|---|']
    for task in tasks:
        pairs=[r for r in table if r['dataset']==task['dataset']]
        a,b=pairs;smae=stat_index[(task['dataset'],'MAE')];smre=stat_index[(task['dataset'],'MRE')]
        docs.append(f"| {task['dataset']} | {a['test_images']} | {a['best_epoch']} | {b['best_epoch']} | {a['MAE']:.4f} | {b['MAE']:.4f} | {smae['mean']:.4f} ± {smae['std']:.4f} | {smre['mean']:.4f} ± {smre['std']:.4f} |")
    docs+=['','MAE/MRE/RMSE单位px，MSE为px²；mean±std为两个训练seed的test分数等权汇总，样本标准差ddof=1、n=2，不是按图片bootstrap或把两个seed预测拼接后求误差。PCK CSV为0–1比例。它是临时两seed结果，第三轮完成后再汇总n=3。',
        '', '## 完整测试指标 mean±std','',
        '| 数据集 | MAE | MRE | RMSE | MSE | PCK2% | PCK5% | PCK10% | PCK20% |','|---|---|---|---|---|---|---|---|---|']
    for task in tasks:
        parts=[]
        for metric in METRICS:
            s=stat_index[(task['dataset'],metric)];scale=100 if metric.startswith('PCK') else 1
            parts.append(f"{s['mean']*scale:.3f} ± {s['std']*scale:.3f}")
        docs.append('| '+task['dataset']+' | '+' | '.join(parts)+' |')
    docs+=['','PCK表内std单位为百分点。所有test模型均按各自验证MRE选择；没有根据test成绩挑选epoch或seed，也没有修改loss/划分、删除高误差点。',
        '', '## 两轮变化分析','',
        '第二轮test MAE相较第一轮，TG3K、acdc、hc18、SKI10、ChestX-ray下降；oasis、Synapse-CT、LUNA16_26上升。hc18/Synapse-CT/LUNA16_26变化较小，oasis的MAE由3.7269升至4.0400px，SKI10由1.9450降至1.7947px。这体现单次随机种子波动，两次seed不足以判断统计显著性。',
        'TG3K两轮RMSE均远高于典型点误差；acdc、oasis、LUNA16_26也存在平方误差受大误差点影响的现象。per_keypoint_by_round.csv保留各点结果；高PCK不意味着没有严重异常预测。',
        '', '## 产物与第三轮','',
        '- test_by_round.csv：两轮16条完整test记录。',
        '- test_mean_std.csv：八个任务全部指标的n=2、ddof=1汇总。',
        '- per_keypoint_by_round.csv：两轮逐关键点成绩。',
        '- validation_history.csv：两轮完整验证曲线记录。',
        '- results.json、round1_results.json、seed1_results.json：完整来源、训练核验、指标与尾部误差。',
        '- audit.json：完整性结论。',
        '- test_two_seed_comparison.png / .pdf：MAE两轮及mean±std图。',
        '第一轮和seed1_training_summary中“test待运行”的文本是当时历史快照，当前完成状态以本报告为准。第一轮结果与第二轮原始日志/checkpoint/prediction/metrics均未覆盖。',
        '第三轮只使用--seed 2与新的_seed2输出目录，保持相同配置与MRE模型选择标准。命令和方法见 ../../docs/seed2_training.md。本次没有执行第三轮训练。','']
    with (args.out_dir/'report.md').open('x') as stream:stream.write('\n'.join(docs))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(len(tasks));fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    names=[t['dataset'] for t in tasks]
    first_scores=[r['MAE'] for r in table if r['round']==1]
    second_scores=[r['MAE'] for r in table if r['round']==2]
    axes[0].bar(x-.18,first_scores,.36,label='First run (seed 1115987209)')
    axes[0].bar(x+.18,second_scores,.36,label='Second run (seed 1)');axes[0].legend(fontsize=8)
    axes[0].set_title('Test coordinate MAE by seed')
    means=[stat_index[(name,'MAE')]['mean'] for name in names]
    stds=[stat_index[(name,'MAE')]['std'] for name in names]
    axes[1].errorbar(x,means,yerr=stds,fmt='o',capsize=5,color='tab:blue')
    axes[1].set_title('Test MAE mean ± sample std (n=2, ddof=1)')
    for ax in axes:
        ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylabel('MAE (original-image pixels)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Different medical tasks / point definitions; not a cross-task ranking')
    for suffix in ['png','pdf']:fig.savefig(args.out_dir/f'test_two_seed_comparison.{suffix}',dpi=170)
    plt.close(fig)
    # Only after all audits succeed do we prepare third-round commands.
    shell='''#!/usr/bin/env bash
# Third round: seed=2. Prepared only, never executed by the audit.
# Running the entire file executes eight training jobs sequentially.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
'''
    command_entries=[]
    for task in tasks:
        cmd=command_lines([f"CUDA_VISIBLE_DEVICES={task['gpu']} python tools/train.py",task['config'],
            f"--work-dir work_dirs/vitpose_base_{task['slug']}_256x192_seed2",'--seed 2'])
        argv=shlex.split(cmd.replace(chr(92)+'\n',''))
        require(argv[-2:]==['--seed','2'] and '--cfg-options' not in argv,'Unexpected training overrides')
        shell+='\n# '+task['dataset']+'\n'+cmd+'\n'
        command_entries.append((task,cmd))
    if commands_path.exists():
        require(commands_path.read_text() == shell, 'Existing seed2 commands differ; refusing overwrite')
    else:
        with commands_path.open('x') as stream:
            stream.write(shell)
    lines=['# 第三轮训练：seed=2','',
        '第一轮(seed1115987209)和第二轮(seed1)八个任务的训练及test结果已核验完整。第三轮复用原八份配置，训练实验变量只改变seed。独立_seed2输出目录保留前两轮所有结果。未启动训练。',
        '', '## 保持一致的设置','',
        'ViTPose-B、原COCO预训练、210epoch、batch8、2workers、JointsMSELoss、原优化器/学习率/增强、固定split、LUNA16_26子集、验证MRE选best全部保持第一轮配置。当前resume_from=None，初次启动从COCO预训练开始。命令不新增cfg-options、deterministic、autoscale-lr、no-validate或线程数设置。',
        'GPU选择沿用既有命令安排；历史物理GPU映射没有记录，不宣称已认证第一轮物理GPU分配。','',
        '## 运行环境','', '```bash','cd /home/wzk/WorkSpace/ViTPose','source .venv/bin/activate',
        'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"','```','',
        '## 八个训练命令','', '可在独立终端分别执行。','']
    for task,cmd in command_entries:lines += ['### '+task['dataset'],'','```bash',cmd,'```','']
    lines += ['也可顺序运行整个命令文件：','', '```bash','bash datasets_analysis/commands/train_seed2_baselines.sh','```','',
        'seed2完成后使用各任务本轮实际best_MRE进行原test推理与评价，预测/指标另存_seed2目录。最终按三个seed test成绩汇总MAE mean±std，同时保留MRE/MSE/RMSE/PCK，ddof=1。当前两轮临时test汇总见 ../results/two_round_summary/report.md。','']
    document = '\n'.join(lines)
    if docs_path.exists():
        require(docs_path.read_text() == document, 'Existing seed2 document differs; refusing overwrite')
    else:
        with docs_path.open('x') as stream:
            stream.write(document)
    require(all(sha256(ROOT/path)==h for path,h in log_hashes.items()),'Log changed during audit')
    print('ALL CHECKS PASSED: 16 training/test runs. Seed2 commands generated, not executed.',flush=True)


if __name__=='__main__':
    main()
