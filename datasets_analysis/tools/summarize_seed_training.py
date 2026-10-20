"""Audit a completed seeded run, summarize validation, and prepare test commands.

Never builds a model, starts training, or runs inference. Existing first-run
results and checkpoints are read-only; new outputs refuse to overwrite files.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from datasets_analysis.medical_evaluation.metrics import read_json, sha256, write_json

METRICS = ['MAE', 'MRE', 'RMSE', 'MSE', 'PCK@2px', 'PCK@5px', 'PCK@10px', 'PCK@20px']
CONFIG_FIELDS = ['model', 'optimizer', 'optimizer_config', 'lr_config', 'total_epochs',
                 'channel_cfg', 'data_cfg', 'train_pipeline', 'val_pipeline',
                 'test_pipeline', 'evaluation', 'checkpoint_config', 'log_config',
                 'workflow', 'load_from', 'resume_from', 'data', 'dist_params',
                 'opencv_num_threads', 'mp_start_method']


def ast_value(node):
    """Read the literal/dict-only config snapshot without executing log content."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [ast_value(n) for n in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(ast_value(n) for n in node.elts)
    if isinstance(node, ast.Dict):
        return {ast_value(k): ast_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -ast_value(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        result = dict(ast_value(node.args[0])) if node.args else {}
        result.update({k.arg: ast_value(k.value) for k in node.keywords})
        return result
    raise ValueError(f'Unexpected config expression: {ast.dump(node)[:100]}')


def config_from_log(text):
    match = re.search(r'INFO - Config:\n(.*?)(?=^\d{4}-\d{2}-\d{2} )', text, re.M | re.S)
    if match is None:
        raise ValueError('Missing config snapshot')
    config = {}
    for node in ast.parse(match.group(1)).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            raise ValueError('Unexpected config statement')
        config[node.targets[0].id] = ast_value(node.value)
    return config


def checkpoint_metadata(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    meta = checkpoint['meta'];state = checkpoint['state_dict']
    states = checkpoint.get('optimizer', {}).get('state', {})
    steps = [int(s['step'].item()) if torch.is_tensor(s['step']) else int(s['step'])
             for s in states.values() if 'step' in s]
    result = dict(path=str(path.relative_to(ROOT)), epoch=meta['epoch'], iteration=meta['iter'],
        seed=meta.get('seed'), best_score=meta.get('hook_msgs', {}).get('best_score'),
        head_weight_shape=list(state['keypoint_head.final_layer.weight'].shape),
        head_bias_shape=list(state['keypoint_head.final_layer.bias'].shape),
        optimizer_states=len(states), optimizer_step_counts=dict(Counter(steps)),
        time=meta.get('time'))
    del checkpoint,state,states
    gc.collect()
    return result


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def command_lines(parts):
    # Real newlines and one shell continuation character, no literal escapes.
    return (' ' + chr(92) + '\n  ').join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--commands-dir', type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    seed = args.seed
    output = args.out_dir or ROOT/f'datasets_analysis/results/seed{seed}_training_summary'
    command_dir = args.commands_dir or ROOT/f'datasets_analysis/commands/test_seed{seed}'
    if output.exists() or command_dir.exists():
        raise FileExistsError('Summary or commands directory exists; choose new output paths')
    torch.set_num_threads(1)
    round_label = '第二轮' if seed == 1 else ('第三轮' if seed == 2 else f'seed={seed}实验')
    first = read_json(ROOT/'datasets_analysis/results/training_summary/training_results.json')
    first_by_name = {r['dataset']: r for r in first['datasets']}
    manifest = read_json(ROOT/'datasets_analysis/docs/baseline_configs_manifest.json')['datasets']
    tasks = [dict(dataset='TG3K', slug='tg3k', keypoints=5, gpu=0,
        config='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py',
        counts=dict(train=146, val=42, test=20))] + manifest
    results, rows, comparison, history, stats, log_hashes = [], [], [], [], [], {}
    commands = []
    for task in tasks:
        name, slug, k = task['dataset'], task['slug'], task['keypoints']
        for split in ['train', 'val', 'test']:
            annotation = ROOT/f'datasets_analysis/coco_format/{name}/annotations/{split}.json'
            if len(read_json(annotation)['images']) != task['counts'][split]:
                raise ValueError(f'Unexpected split count: {name}/{split}')
        workdir = ROOT/f'work_dirs/vitpose_base_{slug}_256x192_seed{seed}'
        paths = sorted(workdir.glob('*.log.json'))
        if not paths:
            raise ValueError(f'No logs: {workdir}')
        old = first_by_name[name]
        first_text_path = ROOT/old['sessions'][0]['log']
        old_config = config_from_log(first_text_path.with_suffix('').read_text())
        validation, train, sessions = {}, [], []
        for path in paths:
            text_path = path.with_suffix('');text = text_path.read_text()
            log_hashes[str(path.relative_to(ROOT))] = sha256(path)
            log_hashes[str(text_path.relative_to(ROOT))] = sha256(text_path)
            log = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            current = config_from_log(text)
            from mmcv import Config
            disk_config = Config.fromfile(str(ROOT/task['config']))._cfg_dict.to_dict()
            if any(disk_config.get(k) != current.get(k) for k in CONFIG_FIELDS):
                raise ValueError(f'Current config differs from training snapshot: {name}')
            changes = [field for field in CONFIG_FIELDS if current.get(field) != old_config.get(field)]
            if changes:
                raise ValueError(f'Changed first-run training settings: {name}: {changes}')
            actual_seeds = [int(v) for v in re.findall(r'Set random seed to (\d+)', text)]
            if not actual_seeds or any(v != seed for v in actual_seeds):
                raise ValueError(f'Wrong logged seed: {name}')
            vs = [r for r in log if r.get('mode') == 'val']
            ts = [r for r in log if r.get('mode') == 'train'];train.extend(ts)
            for row in vs:
                if row['epoch'] in validation:
                    raise ValueError('Ambiguous repeated validation epoch')
                if not all(math.isfinite(row[key]) for key in METRICS):
                    raise ValueError('Nonfinite metric')
                validation[row['epoch']] = row
            start = re.search(r'^(\d{4}-\d{2}-\d{2} [\d:,]+).*workflow:', text, re.M)
            end = re.findall(r'^(\d{4}-\d{2}-\d{2} [\d:,]+).*Epoch\(val\)', text, re.M)
            duration = (datetime.strptime(end[-1], '%Y-%m-%d %H:%M:%S,%f') -
                        datetime.strptime(start[1], '%Y-%m-%d %H:%M:%S,%f')).total_seconds()
            sessions.append(dict(log=str(path.relative_to(ROOT)), seed=seed,
                training_config_matches_first=True, runner_start=start[1], last_validation=end[-1],
                runner_window_seconds=duration,
                resumed=re.findall(r'resumed epoch (\d+), iter (\d+)', text),
                unexpected_errors=[line for line in text.splitlines()
                    if re.search(r'Traceback|CUDA out of memory| - ERROR - |\bnan\b', line, re.I)]))
        if any(session['unexpected_errors'] for session in sessions):
            raise ValueError(f'Training errors logged: {name}')
        values = [validation[e] for e in sorted(validation)]
        if [r['epoch'] for r in values] != list(range(10, 211, 10)):
            raise ValueError('Incomplete validation history')
        best = min(values, key=lambda r: r['MRE']);final_val = values[-1]
        candidates = list(workdir.glob('best_MRE_epoch_*.pth'))
        if len(candidates) != 1:
            raise ValueError('Ambiguous/missing best checkpoint')
        best_meta = checkpoint_metadata(candidates[0])
        final_meta = checkpoint_metadata(workdir/'latest.pth')
        expected_steps = 210 * math.ceil(task['counts']['train']/8)
        assert best_meta['seed'] == final_meta['seed'] == seed
        assert best_meta['epoch'] == best['epoch']
        assert math.isclose(best_meta['best_score'], best['MRE'], abs_tol=1e-5)
        assert final_meta['epoch'] == 210 and final_meta['iteration'] == expected_steps
        for meta in [best_meta, final_meta]:
            assert meta['head_weight_shape'] == [k, 256, 1, 1] and meta['head_bias_shape'] == [k]
            assert meta['optimizer_states'] == 157
            assert len(meta['optimizer_step_counts']) == 1
            assert next(iter(meta['optimizer_step_counts'])) == meta['iteration']
        row = dict(dataset=name, seed=seed, keypoints=k, train_records=task['counts']['train'],
            val_records=task['counts']['val'], test_records=task['counts']['test'],
            completed_epoch=210, iterations=final_meta['iteration'], best_epoch=best['epoch'],
            **{key:best[key] for key in METRICS}, best_checkpoint=best_meta['path'])
        rows.append(row)
        delta = dict(dataset=name, first_seed=old['final_checkpoint']['seed'], new_seed=seed,
            first_best_epoch=old['validation_best']['epoch'], new_best_epoch=best['epoch'])
        for key in METRICS:
            previous = old['validation_best'][key]
            delta.update({f'first_{key}':previous, f'seed{seed}_{key}':best[key], f'delta_{key}':best[key]-previous})
            pair = np.asarray([previous, best[key]])
            stats.append(dict(dataset=name, metric=key, split='validation', seeds=f'{old["final_checkpoint"]["seed"]},{seed}',
                n=2, mean=float(pair.mean()), std=float(pair.std(ddof=1)), ddof=1))
        comparison.append(delta)
        history.extend(dict(dataset=name,seed=seed,epoch=r['epoch'],**{key:r[key] for key in METRICS}) for r in values)
        result = dict(task)
        result.update(seed=seed,work_dir=str(workdir.relative_to(ROOT)),sessions=sessions,
            best_checkpoint=best_meta,final_checkpoint=final_meta,validation_best=best,
            validation_final=final_val,validation_first=values[0],
            train_loss_first_logged=train[0].get('heatmap_loss') if train else None,
            train_loss_last_logged=train[-1].get('heatmap_loss') if train else None,
            first_run_validation_best=old['validation_best'],test_status='pending')
        results.append(result)
        epoch = best['epoch'];prefix=f'work_dirs/{slug}_test_evaluation_seed{seed}/best_MRE_epoch_{epoch}'
        gt=f'datasets_analysis/coco_format/{name}/annotations/test.json'
        infer = command_lines([f"CUDA_VISIBLE_DEVICES={task['gpu']} python datasets_analysis/tools/infer_medical_keypoints.py",
            task['config'],best_meta['path'],f'--test-json {gt}',f'--out {prefix}_predictions.json',
            '--device cuda:0 --batch-size 8 --workers 2'])
        evaluation = command_lines(['python datasets_analysis/tools/evaluate_medical_keypoints.py',
            f'--predictions {prefix}_predictions.json',f'--gt {gt}',
            '--pck-unit pixels --pck-thresholds 2 5 10 20',f'--out {prefix}_metrics.json'])
        # Validate static arguments and paths; never execute these commands here.
        for cmd in [infer,evaluation]:
            argv=shlex.split(cmd.replace(chr(92)+'\n',''))
            assert not any(a in (chr(92),'n') for a in argv)
        assert (ROOT/gt).is_file()
        commands.append((result,infer,evaluation,prefix))
        print(f'VERIFIED {name}: seed={seed}, epoch=210, steps={expected_steps}, best={epoch}, MAE={best["MAE"]}, MRE={best["MRE"]}',flush=True)
    output.mkdir(parents=True);command_dir.mkdir(parents=True)
    write_json(output/'training_results.json',dict(seed=seed,datasets=results,input_log_sha256=log_hashes,
        test_results_pending=True,note='Validation results only. First-run results unchanged.'))
    write_csv(output/'validation_best.csv',rows)
    write_csv(output/'first_run_comparison.csv',comparison)
    write_csv(output/'validation_history.csv',history)
    write_csv(output/'validation_two_seed_mean_std.csv',stats)
    docs=[f'# {round_label} seed={seed}：训练结果与测试命令','',
        '本次仅读取日志和checkpoint，整理验证集成绩并生成命令；没有执行训练、test推理或评价。第一轮结果和可视化保留。',
        '', '## 核验与设置','',
        f'八个任务均完成210epoch，日志和checkpoint seed={seed}，每个任务21次验证。最终iteration与既定训练记录数、batch8、210epoch一致；优化器step与checkpoint iteration相同，头通道匹配各自K。日志内嵌模型、数据、优化器、学习率、loss、评价及loader配置与第一轮初始日志逐字段相同。',
        '历史日志没有逐批image_id，iteration一致不能独立证明每张训练图的身份。没有改变数据划分或重新生成COCO标注。',
        '仍按验证MRE选择checkpoint。本报告把MAE放在主要位置；所有其他指标均取同一个MRE最优epoch，不改为按MAE选模型。','',
        '| 数据集 | epoch | MAE | MRE | RMSE | MSE | PCK2% | PCK5% | PCK10% | PCK20% |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        docs.append(f"| {row['dataset']} | {row['best_epoch']} | {row['MAE']:.4f} | {row['MRE']:.4f} | {row['RMSE']:.4f} | {row['MSE']:.4f} | "+' | '.join(f"{row[f'PCK@{t}px']*100:.2f}" for t in [2,5,10,20])+' |')
    docs+=['','以上是验证集成绩，不是test。MSE单位px²，其他误差为原图px；报告PCK为百分比，CSV为0–1比例。',
        '', '## 与第一轮验证结果比较','',
        f'| 数据集 | 第一轮MAE | seed{seed} MAE | MAE变化 | 第一轮MRE | seed{seed} MRE | MRE变化 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        docs.append(f"| {row['dataset']} | {row['first_MAE']:.4f} | {row[f'seed{seed}_MAE']:.4f} | {row['delta_MAE']:+.4f} | {row['first_MRE']:.4f} | {row[f'seed{seed}_MRE']:.4f} | {row['delta_MRE']:+.4f} |")
    improving = [r['dataset'] for r in comparison if r['delta_MAE'] < 0]
    worsening = [r['dataset'] for r in comparison if r['delta_MAE'] > 0]
    docs += ['',
        '本轮验证MAE相较第一轮下降：' + '、'.join(improving) + '；上升：' + '、'.join(worsening) + '。这些是验证集变化，不能代替test泛化结果。',
        '各数据集best epoch见上表。必须使用本轮实际best_MRE，不能统一用latest(epoch210)替代。',
        'TG3K、hc18和ChestX-ray每轮不足日志间隔50，缺少训练loss记录不表示没有训练；最终step和验证记录已确认完成。',
        '', '## 验证集比较统计', '',
        f'validation_two_seed_mean_std.csv仅比较第一轮实际seed1115987209与当前seed{seed}的验证成绩，n=2、ddof=1。不是全部实验轮次汇总，也不是最终test结果。最终结果须等三轮test评价齐全后，按各seed test指标计算mean±std。',
        '', '## Test推理与医学评价命令', '',
        '在仓库根目录运行：', '', '```bash', 'cd /home/wzk/WorkSpace/ViTPose', 'source .venv/bin/activate',
        'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"', '```', '',
        f'所有输出位于独立的 *_test_evaluation_seed{seed} 目录，不覆盖其他轮次。每个脚本包含本轮实际best_MRE推理和原test评价，输出MSE、RMSE、MAE、MRE、PCK@2/5/10/20px。CUDA_VISIBLE_DEVICES选择物理卡，进程内使用cuda:0。', '']
    for result,infer,evaluation,prefix in commands:
        shell='''#!/usr/bin/env bash
# Prepared test inference/evaluation commands only; not executed by the summarizer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
'''+infer+'\n\n'+evaluation+'\n'
        with (command_dir/f'{result["slug"]}.sh').open('x') as stream:stream.write(shell)
        docs += [f'### {result["dataset"]}','','```bash',infer,'',evaluation,'```','']
    wrapper='''#!/usr/bin/env bash
# Running this file performs all eight test inferences/evaluations sequentially.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
'''
    for result,_,_,_ in commands:
        wrapper+=f'bash {command_dir.relative_to(ROOT)}/{result["slug"]}.sh\n'
    with (command_dir/'all.sh').open('x') as stream:stream.write(wrapper)
    docs+=['顺序运行八个测试任务：','', '```bash',f'bash {command_dir.relative_to(ROOT)}/all.sh','```','',
        '这条命令及上述单任务脚本均未由本次整理执行。输出文件已存在时，原推理/评价脚本会拒绝覆盖。','',
        '文件清单：training_results.json、validation_best.csv、first_run_comparison.csv、validation_history.csv、validation_two_seed_mean_std.csv、validation_mre_comparison.png、report.md；命令目录包含八份脚本和all.sh。','']
    with (output/'report.md').open('x') as stream:stream.write('\n'.join(docs))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    # Read original history CSV to preserve merged acdc resume trajectory.
    with (ROOT/'datasets_analysis/results/training_summary/validation_history.csv').open() as stream:
        old_curve=list(csv.DictReader(stream))
    for ax,result in zip(axes.flat,results):
        name=result['dataset'];previous=[r for r in old_curve if r['dataset']==name]
        current=[r for r in history if r['dataset']==name]
        ax.plot([int(r['epoch']) for r in previous],[float(r['MRE']) for r in previous],label='First run',linewidth=1.5)
        ax.plot([r['epoch'] for r in current],[r['MRE'] for r in current],label=f'Seed {seed}',linewidth=1.5)
        best=result['validation_best'];ax.scatter([best['epoch']],[best['MRE']],color='red',s=20)
        ax.set(title=name,xlabel='Epoch',ylabel='Validation MRE (px)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Validation MRE: first run vs seed {seed}; independent y-axes; red = selected checkpoint')
    fig.savefig(output/'validation_mre_comparison.png',dpi=160);plt.close(fig)
    assert all(sha256(ROOT/path)==h for path,h in log_hashes.items()), 'Log changed during summary'
    print('Saved:',output,'Commands:',command_dir,'No training or inference executed.',flush=True)


if __name__=='__main__':
    main()
