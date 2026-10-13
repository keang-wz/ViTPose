"""Read eight completed training runs; write summaries/commands, never infer/train."""
import csv
import gc
import json
import math
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from datasets_analysis.medical_evaluation.metrics import read_json, evaluate, sha256

OUT = ROOT/'datasets_analysis/results/training_summary'
METRICS = ['MSE','RMSE','MAE','MRE','PCK@2px','PCK@5px','PCK@10px','PCK@20px']


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checkpoint_info(path):
    c = torch.load(path, map_location='cpu', weights_only=True)
    m = c['meta']; state = c['state_dict']
    result = dict(path=str(path.relative_to(ROOT)), size_bytes=path.stat().st_size,
        epoch=m.get('epoch'), iteration=m.get('iter'), seed=m.get('seed'),
        best_score=m.get('hook_msgs',{}).get('best_score'),
        head_weight_shape=list(state['keypoint_head.final_layer.weight'].shape),
        head_bias_shape=list(state['keypoint_head.final_layer.bias'].shape))
    del c,state
    gc.collect()
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = read_json(ROOT/'datasets_analysis/docs/baseline_configs_manifest.json')['datasets']
    tasks = [dict(dataset='TG3K',slug='tg3k',keypoints=5,work_dir='work_dirs/tg3k_smoke',
        config='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py',
        counts=dict(train=146,val=42,test=20))] + manifest
    rows=[]; flat=[]; curve=[]; snapshots={}
    for task in tasks:
        wd=ROOT/task['work_dir'];vals={};trains=[];sessions=[]
        for path in sorted(wd.glob('*.log.json')):
            snapshots[str(path.relative_to(ROOT))]=sha256(path)
            log=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            vs=[r for r in log if r.get('mode')=='val'];ts=[r for r in log if r.get('mode')=='train']
            text=path.with_suffix('').read_text()
            resume=re.findall(r'resumed epoch (\d+), iter (\d+)',text)
            sessions.append(dict(log=str(path.relative_to(ROOT)),validation_epochs=[r['epoch'] for r in vs],
                training_log_records=len(ts),resume=[dict(epoch=int(e),iteration=int(i)) for e,i in resume],
                seed_messages=re.findall(r'Set random seed to (\d+)',text),
                unexpected_error_lines=[line for line in text.splitlines() if re.search(r'Traceback|CUDA out of memory| - ERROR - |\bnan\b',line,re.I)],
                checkpoint_shape_mismatch_lines=[line for line in text.splitlines() if 'size mismatch for ' in line]))
            for r in vs:
                if r['epoch'] in vals:
                    raise ValueError(f'Duplicate validation epoch in {task["dataset"]}; ambiguous run history')
                if not all(math.isfinite(r[k]) for k in METRICS):raise ValueError('Nonfinite validation metric')
                vals[r['epoch']]=r
            trains+=ts
        vals=[vals[e] for e in sorted(vals)]
        if not vals:raise ValueError('No validation data')
        best=min(vals,key=lambda r:r['MRE']);last=vals[-1];first=vals[0]
        candidates=list(wd.glob('best_MRE_epoch_*.pth'))
        if len(candidates)!=1:raise ValueError('Ambiguous best checkpoint selection')
        best_ckpt=checkpoint_info(candidates[0]);final_ckpt=checkpoint_info(wd/'latest.pth')
        k=task['keypoints']
        assert best_ckpt['head_weight_shape']==[k,256,1,1] and best_ckpt['head_bias_shape']==[k]
        assert final_ckpt['epoch']==210 and last['epoch']==210
        assert best_ckpt['epoch']==best['epoch']
        assert abs(best_ckpt['best_score']-best['MRE'])<1e-5
        expected_iters=210*math.ceil(task['counts']['train']/8)
        assert final_ckpt['iteration']==expected_iters
        assert [r['epoch'] for r in vals]==list(range(10,211,10))
        result=dict(**task,status='completed_210_epochs',sessions=sessions,
            best_checkpoint=best_ckpt,final_checkpoint=final_ckpt,validation_first=first,
            validation_best=best,validation_final=last,
            first_to_best_mre_improvement_percent=(first['MRE']-best['MRE'])/first['MRE']*100,
            final_minus_best_mre=last['MRE']-best['MRE'],
            logged_training_loss_first=trains[0].get('heatmap_loss') if trains else None,
            logged_training_loss_last=trains[-1].get('heatmap_loss') if trains else None,
            logged_peak_memory_mib=max((r.get('memory',0) for r in trains),default=None),
            test_status='not_found',test=None)
        if task['dataset']=='TG3K':
            pred_path=ROOT/'work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_predictions.json'
            metrics_path=ROOT/'work_dirs/tg3k_test_evaluation/best_MRE_epoch_210_metrics.json'
            gt_path=ROOT/'datasets_analysis/coco_format/TG3K/annotations/test.json'
            pred,gt,stored=read_json(pred_path),read_json(gt_path),read_json(metrics_path)
            assert stored['inputs']['prediction_sha256']==sha256(pred_path)
            assert stored['inputs']['annotation_sha256']==sha256(gt_path)
            assert pred['metadata']['annotation_sha256']==sha256(gt_path)
            assert pred['metadata']['checkpoint_sha256']==sha256(candidates[0])
            recomputed=evaluate(pred,gt)
            assert recomputed['overall']==stored['overall']
            result['test_status']='verified_existing_results'
            result['test']=dict(metrics_path=str(metrics_path.relative_to(ROOT)),**stored)
            gt_by_id={a['id']:a for a in gt['annotations']};ims={im['id']:im for im in gt['images']}
            errors=[]
            for p in pred['predictions']:
                a=gt_by_id[p['annotation_id']]
                points=np.asarray(a['keypoints']).reshape(-1,3)
                for j,(xy,g) in enumerate(zip(p['keypoints'],points)):
                    if g[2]<=0:continue
                    delta=np.asarray(xy)-g[:2]
                    errors.append(dict(image_id=p['image_id'],annotation_id=p['annotation_id'],
                        file_name=ims[p['image_id']]['file_name'],point_id=j+1,
                        gt_x=float(g[0]),gt_y=float(g[1]),pred_x=xy[0],pred_y=xy[1],
                        confidence=p['confidence'][j],distance_px=float(np.linalg.norm(delta)),squared_error=float(delta@delta)))
            errors.sort(key=lambda x:x['distance_px'],reverse=True)
            sse=sum(x['squared_error'] for x in errors)
            result['test_error_analysis']=dict(median_distance_px=float(np.median([x['distance_px'] for x in errors])),
                p95_distance_px=float(np.percentile([x['distance_px'] for x in errors],95)),
                points_over_20px=sum(x['distance_px']>20 for x in errors),
                top_two_squared_error_fraction=sum(x['squared_error'] for x in errors[:2])/sse,
                largest_errors=errors[:10])
            with (OUT/'tg3k_test_largest_errors.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors[:10])
        rows.append(result)
        flat.append(dict(dataset=task['dataset'],keypoints=k,train=task['counts']['train'],val=task['counts']['val'],test=task['counts']['test'],
            completed_epoch=final_ckpt['epoch'],best_epoch=best['epoch'],**{key:best[key] for key in METRICS},
            best_checkpoint=best_ckpt['path'],test_status=result['test_status']))
        curve.extend(dict(dataset=task['dataset'],**{key:r[key] for key in ['epoch']+METRICS}) for r in vals)
        print('VERIFIED',task['dataset'],'best',best['epoch'],'MRE',best['MRE'],'final iterations',final_ckpt['iteration'],flush=True)
    save_json(OUT/'training_results.json',dict(datasets=rows,input_log_sha256=snapshots,
        note='Validation metrics are taken together from minimum-MRE epoch; no new training or inference. PCK values are fractions.'))
    for name,data in [('validation_best.csv',flat),('validation_history.csv',curve)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    # Standalone chart, no interpolation or invented training-loss points.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    for ax,row in zip(axes.flat,rows):
        values=[r for r in curve if r['dataset']==row['dataset']]
        ax.plot([r['epoch'] for r in values],[r['MRE'] for r in values],marker='.',linewidth=1.5)
        best=row['validation_best'];ax.scatter([best['epoch']],[best['MRE']],color='red',zorder=4)
        ax.set(title=f'{row["dataset"]}: best {best["MRE"]:.3f} px @ {best["epoch"]}',xlabel='Epoch',ylabel='Validation MRE (px)')
        ax.grid(alpha=.25)
    fig.suptitle('ViTPose-B validation MRE; independent y-axis per dataset; red = selected checkpoint')
    fig.savefig(OUT/'validation_mre_curves.png',dpi=160);plt.close(fig)
    # Generate one reusable command script per remaining dataset, no execution.
    command_dir=ROOT/'datasets_analysis/commands/test_remaining';command_dir.mkdir(exist_ok=True)
    command_docs=[]
    for row in rows[1:]:
        slug=row['slug'];epoch=row['validation_best']['epoch']
        prefix=f'work_dirs/{slug}_test_evaluation/best_MRE_epoch_{epoch}'
        infer=f'''CUDA_VISIBLE_DEVICES=0 python datasets_analysis/tools/infer_medical_keypoints.py \\
  {row['config']} \\
  {row['best_checkpoint']['path']} \\
  --test-json datasets_analysis/coco_format/{row['dataset']}/annotations/test.json \\
  --out {prefix}_predictions.json \\
  --device cuda:0 --batch-size 8 --workers 2'''
        evaluation=f'''python datasets_analysis/tools/evaluate_medical_keypoints.py \\
  --predictions {prefix}_predictions.json \\
  --gt datasets_analysis/coco_format/{row['dataset']}/annotations/test.json \\
  --pck-unit pixels --pck-thresholds 2 5 10 20 \\
  --out {prefix}_metrics.json'''
        shell='''#!/usr/bin/env bash
# Test inference/evaluation only. This file was generated, not executed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
'''+infer+'\n\n'+evaluation+'\n'
        (command_dir/f'{slug}.sh').write_text(shell)
        command_docs.extend([f'### {row["dataset"]}','','```bash',infer,'',evaluation,'```',''])
    docs=['# 八个数据集训练结果整理','',
        '来源：各 work_dir 的全部 .log.json/.log、best_MRE checkpoint 与 latest checkpoint；TG3K test 来自已有预测和指标，并重新核对哈希和计算结果。没有执行新训练或推理，没有改动原始数据和训练产物。',
        '', '## 完成情况与最佳验证结果','',
        '八个数据集均完成 210 epoch，最终 iteration 与 ceil(train/8)×210 一致；各有 21 次验证记录（epoch 10..210）。checkpoint 头通道均匹配各自 K，best_score 与日志一致。下表所有指标取自同一个最小 MRE epoch，未按不同指标拼接最优值。PCK 表内为百分比。',
        '', '| 数据集 | K | 最佳epoch | MSE | RMSE | MAE | MRE | PCK2% | PCK5% | PCK10% | PCK20% |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        b=row['validation_best'];docs.append(f"| {row['dataset']} | {row['keypoints']} | {b['epoch']} | {b['MSE']:.4f} | {b['RMSE']:.4f} | {b['MAE']:.4f} | {b['MRE']:.4f} | {b['PCK@2px']*100:.2f} | {b['PCK@5px']*100:.2f} | {b['PCK@10px']*100:.2f} | {b['PCK@20px']*100:.2f} |")
    docs+=['','MSE 为 coordinate MSE（pixel²），其他误差为原图 pixel；RMSE=sqrt(coordinate MSE)，不是径向 RMS。训练 heatmap MSE 不是此表中的 coordinate MSE。','',
        '![Validation MRE](validation_mre_curves.png)','', '## 训练过程核查','',
        '- acdc 两段日志分别覆盖 epoch 1–110 和 111–210；第二段明确记录 resumed epoch 110, iter 7150。已合并，没有当作两次独立完整实验。',
        '- oasis 最佳 epoch 200，LUNA16_26 最佳 epoch 180；它们最后一轮并非 MRE 最优，测试命令使用 best_MRE 而非 latest。其余六个最佳均在 epoch 210；只说明本次验证记录的最优点在末轮，不能据此认定继续训练必然改善。',
        '- TG3K、hc18、ChestX-ray 每轮不足 50 batch，而 TextLoggerHook interval=50，因此 JSON 没有训练 loss 记录。这不表示没有训练；最终 checkpoint iteration 和完整验证记录已确认完成。',
        '- JSON 日志 lr 保留精度有限，多处显示 0.0；文本日志仍记录非零科学计数值，不能把四舍五入的 0.0 解释成停止学习。',
        '- 首次预训练加载中的 final_layer 17→K shape mismatch 是预期迁移行为；训练产物均已具有正确 K 输出头。',
        '', '| 数据集 | checkpoint seed | epoch10 MRE | 最佳MRE | 相对下降% | 最后epoch MRE | 首条/末条记录的heatmap loss |','|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        loss='未记录' if row['logged_training_loss_first'] is None else f"{row['logged_training_loss_first']} / {row['logged_training_loss_last']}"
        docs.append(f"| {row['dataset']} | {row['final_checkpoint']['seed']} | {row['validation_first']['MRE']:.4f} | {row['validation_best']['MRE']:.4f} | {row['first_to_best_mre_improvement_percent']:.2f} | {row['validation_final']['MRE']:.4f} | {loss} |")
    docs+=['','训练 loss 仅为被日志采样的首末记录，不是完整 epoch 平均。实际 seed 以上表 checkpoint 为准，不以此前命令示例 seed=42 代替真实运行值。',
        '', '## 结果分析','',
        '在各自验证集原图像素尺度下，Synapse-CT 的 MRE 最低（2.07847 px），SKI10 次之（3.32797 px）；acdc 的 MRE 最高（8.50499 px）。数据集的点定义、图像导出尺度和难度不同，不能据此作为跨任务公平排名，也不能换算为毫米误差。',
        'LUNA16_26 的 PCK@5px 接近 90%，但 RMSE=17.31251 px；TG3K、oasis、acdc 也呈现 RMSE 明显高于典型误差水平的现象，提示大误差点对平方误差的影响。除 TG3K 已有 test 预测外，目前不能确定各数据集异常点位置，需运行 test 后逐点核查。',
        'LUNA16_26 最后一轮 PCK@2/5px 高于 epoch180，但 MRE 略差，说明评价目标不同会选出不同 checkpoint；本实验始终按事先设定的验证 MRE 选模型。没有用测试集选择模型。',
        '', '## TG3K 已有 test 结果','']
    tg=rows[0];o=tg['test']['overall'];a=tg['test_error_analysis']
    docs+=['| MSE | RMSE | MAE | MRE | PCK2% | PCK5% | PCK10% | PCK20% |','|---:|---:|---:|---:|---:|---:|---:|---:|',
        f"| {o['coordinate_mse']:.4f} | {o['rmse']:.4f} | {o['mae']:.4f} | {o['mean_euclidean_distance_error']:.4f} | {o['pck']['2']*100:.2f} | {o['pck']['5']*100:.2f} | {o['pck']['10']*100:.2f} | {o['pck']['20']*100:.2f} |",'',
        f"20 张图、100 个有效点，缺失点 0。距离中位数 {a['median_distance_px']:.4f} px，95 分位 {a['p95_distance_px']:.4f} px；超过 20 px 的点共 {a['points_over_20px']} 个。最大的两个点贡献总平方误差的 {a['top_two_squared_error_fraction']*100:.2f}%。",'',
        '| image_id | point_id | distance px | confidence | file_name |','|---:|---:|---:|---:|---|']
    for e in a['largest_errors'][:2]:docs.append(f"| {e['image_id']} | {e['point_id']} | {e['distance_px']:.4f} | {e['confidence']:.4f} | `{e['file_name']}` |")
    docs+=['','point_5 的 test MRE 为 27.6249 px，而 point_1..4 为 2.2673/2.7331/3.4131/3.4888 px。需复核这些样本的图像与标注，但本次不据此修改或删除测试样本。其他七个数据集尚未找到测试预测/指标，不能把验证值称为 test 成绩。',
        '', '## 最佳 checkpoint 与测试命令','', '在仓库根目录执行：','', '```bash','source .venv/bin/activate','export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"','```','',
        '命令默认 GPU0；可以把 CUDA_VISIBLE_DEVICES 改为 1，进程内部仍使用 cuda:0。每个数据集分别输出 prediction JSON 和 metrics JSON；不会覆盖已有同名输出。下面仅为生成的运行命令，本次没有执行。','']+command_docs
    docs+=['也可分别运行生成的脚本，例如：','', '```bash','bash datasets_analysis/commands/test_remaining/acdc.sh','```','',
        '相同目录内还有 hc18.sh、oasis.sh、synapse_ct.sh、ski10.sh、chestxray.sh、luna16_26.sh。每个脚本依次执行一次测试推理和评价，不训练。',
        '', '## 文件清单','',
        '- training_results.json：完整可追溯汇总、checkpoint 元数据与日志来源。',
        '- validation_best.csv：八数据集最佳 MRE epoch 的全部指标，PCK 为 0–1 比例。',
        '- validation_history.csv：每 10 epoch 的验证曲线数据。',
        '- validation_mre_curves.png：可导出的八图训练验证曲线。',
        '- tg3k_test_largest_errors.csv：已有 TG3K test 最大误差前十点。',
        '- report.md：本报告与全部推理/评价命令。','']
    (OUT/'report.md').write_text('\n'.join(docs))
    assert all(sha256(ROOT/path)==h for path,h in snapshots.items())
    print('Saved summary and seven command scripts; no training/inference performed.')


if __name__=='__main__':main()
