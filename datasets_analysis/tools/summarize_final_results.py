"""Audit existing eight-dataset test results and publish final tables; no inference."""
import csv
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from datasets_analysis.medical_evaluation.metrics import read_json, evaluate, sha256

OUT = ROOT/'datasets_analysis/results/final_summary'


def save_json(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def save_csv(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def compare(a,b,path=''):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError(f'Metric schema differs: {path}')
        for key in a:compare(a[key],b[key],path+'/'+key)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError(f'Length differs: {path}')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'/{i}')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f'Metric differs: {path}: {a} != {b}')
    elif a!=b:raise ValueError(f'Metric differs: {path}')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'datasets_analysis/results/training_summary/training_results.json'
    training=read_json(source)
    for path,h in training['input_log_sha256'].items():
        if sha256(ROOT/path)!=h:raise ValueError(f'Training log changed: {path}; refresh training summary')
    results=[];table=[];comparisons=[];points=[];images=[];keypoints=[];hist={};audits=[]
    for run in training['datasets']:
        name,slug=run['dataset'],run['slug'];epoch=run['best_checkpoint']['epoch'];k=run['keypoints']
        prefix=ROOT/f'work_dirs/{slug}_test_evaluation/best_MRE_epoch_{epoch}'
        pp=Path(str(prefix)+'_predictions.json');mp=Path(str(prefix)+'_metrics.json')
        gp=ROOT/f'datasets_analysis/coco_format/{name}/annotations/test.json'
        cp=ROOT/run['config'];bp=ROOT/run['best_checkpoint']['path']
        pred,stored,gt=read_json(pp),read_json(mp),read_json(gp)
        hashes={key:sha256(path) for key,path in [('predictions',pp),('metrics',mp),('gt',gp),('config',cp),('checkpoint',bp)]}
        meta=pred['metadata']
        assert hashes['predictions']==stored['inputs']['prediction_sha256']
        assert hashes['gt']==stored['inputs']['annotation_sha256']==meta['annotation_sha256']
        assert hashes['checkpoint']==meta['checkpoint_sha256']
        assert hashes['config']==meta['config_sha256']
        assert (ROOT/meta['checkpoint']).resolve()==bp.resolve()
        assert meta['model_head_channels']==k
        assert stored['pck_thresholds']==[2,5,10,20] and stored['units']['pck']=='pixels'
        recomputed=evaluate(pred,gt,[2,5,10,20],'pixels')
        for key in recomputed:compare(recomputed[key],stored[key],key)
        assert len(gt['images'])==run['counts']['test']==len(pred['predictions'])
        assert all(len(a['keypoints'])==k*3 for a in gt['annotations'])
        by_ann={a['id']:a for a in gt['annotations']};by_image={im['id']:im for im in gt['images']}
        local=[];deltas=[];pred_outside=gt_outside=0
        for p in pred['predictions']:
            a=by_ann[p['annotation_id']];im=by_image[p['image_id']]
            g=np.asarray(a['keypoints'],dtype=float).reshape(-1,3)
            image_errors=[]
            for j,(xy,truth,confidence) in enumerate(zip(p['keypoints'],g,p['confidence'])):
                if truth[2]<=0:continue
                delta=np.asarray(xy)-truth[:2];dist=float(np.linalg.norm(delta));sse=float(delta@delta)
                outside=not(0<=xy[0]<im['width'] and 0<=xy[1]<im['height'])
                gt_bad=not(0<=truth[0]<im['width'] and 0<=truth[1]<im['height'])
                pred_outside+=int(outside);gt_outside+=int(gt_bad)
                record=dict(dataset=name,image_id=p['image_id'],annotation_id=p['annotation_id'],
                    file_name=im['file_name'],point_id=a.get('source_point_ids',list(range(1,k+1)))[j],
                    point_name=p.get('keypoint_names',[str(n) for n in range(k)])[j],
                    gt_x=float(truth[0]),gt_y=float(truth[1]),pred_x=xy[0],pred_y=xy[1],
                    confidence=confidence,distance_px=dist,squared_distance_px2=sse,prediction_outside_image=outside)
                local.append(record);deltas.append(delta);image_errors.append(dist)
            if image_errors:
                images.append(dict(dataset=name,image_id=p['image_id'],file_name=im['file_name'],
                    evaluated_keypoints=len(image_errors),mean_distance_px=float(np.mean(image_errors)),
                    max_distance_px=max(image_errors),points_over_20px=sum(e>20 for e in image_errors)))
        points.extend(local)
        delta=np.asarray(deltas);distance=np.linalg.norm(delta,axis=1);o=recomputed['overall']
        # Independent direct-formula check in addition to reusing the evaluator.
        assert np.isclose(np.mean(delta**2),o['coordinate_mse'],rtol=1e-12)
        assert np.isclose(np.sqrt(np.mean(delta**2)),o['rmse'],rtol=1e-12)
        assert np.isclose(np.mean(np.abs(delta)),o['mae'],rtol=1e-12)
        assert np.isclose(np.mean(distance),o['mean_euclidean_distance_error'],rtol=1e-12)
        for t in [2,5,10,20]:assert np.isclose(np.mean(distance<=t),o['pck'][str(t)],rtol=1e-12)
        local.sort(key=lambda x:x['distance_px'],reverse=True)
        squared=distance**2;ordered=np.sort(squared)[::-1]
        tail_count=int(np.sum(distance>20));ntop=max(1,math.ceil(len(distance)*.01))
        tail=dict(median_distance_px=float(np.median(distance)),p90_distance_px=float(np.percentile(distance,90)),
            p95_distance_px=float(np.percentile(distance,95)),p99_distance_px=float(np.percentile(distance,99)),
            max_distance_px=float(distance.max()),points_over_20px=tail_count,
            fraction_over_20px=float(np.mean(distance>20)),
            over_20px_squared_error_fraction=float(squared[distance>20].sum()/squared.sum()) if squared.sum() else 0,
            top_1percent_point_count=ntop,top_1percent_squared_error_fraction=float(ordered[:ntop].sum()/ordered.sum()) if ordered.sum() else 0,
            predicted_points_outside_image=pred_outside,gt_points_outside_image=gt_outside)
        best=run['validation_best'];test_mre=o['mean_euclidean_distance_error']
        summary=dict(dataset=name,keypoints=k,train_images=run['counts']['train'],val_images=run['counts']['val'],
            test_images=run['counts']['test'],test_keypoints=len(distance),best_epoch=epoch,
            MSE=o['coordinate_mse'],RMSE=o['rmse'],MAE=o['mae'],MRE=test_mre,
            **{f'PCK@{t}px':o['pck'][str(t)] for t in [2,5,10,20]},
            median_distance_px=tail['median_distance_px'],p95_distance_px=tail['p95_distance_px'],
            points_over_20px=tail_count)
        table.append(summary)
        comparison=dict(dataset=name,best_epoch=epoch,val_MRE=best['MRE'],test_MRE=test_mre,
            test_minus_val_MRE=test_mre-best['MRE'],relative_change_percent=(test_mre/best['MRE']-1)*100,
            val_PCK5=best['PCK@5px'],test_PCK5=o['pck']['5'],val_PCK10=best['PCK@10px'],test_PCK10=o['pck']['10'])
        comparisons.append(comparison)
        for point in recomputed['per_keypoint']:
            keypoints.append(dict(dataset=name,category_id=point['category_id'],index=point['index'],name=point['name'],
                count=point['evaluated_keypoints'],MSE=point['coordinate_mse'],RMSE=point['rmse'],MAE=point['mae'],
                MRE=point['mean_euclidean_distance_error'],**{f'PCK@{t}px':point['pck'][str(t)] for t in [2,5,10,20]}))
        provenance=dict(dataset=name,files={key:str(path.relative_to(ROOT)) for key,path in [('predictions',pp),('metrics',mp),('gt',gp),('config',cp),('checkpoint',bp)]},
            sha256=hashes,prediction_coverage_verified=True,all_metric_fields_recomputed=True,
            independent_metric_formulas_verified=True,checkpoint_and_config_hash_match=True,
            input_metric_files_unchanged=True)
        audits.append(provenance)
        results.append(dict(dataset=name,summary=summary,validation_best=best,validation_final=run['validation_final'],
            completed_epoch=run['final_checkpoint']['epoch'],seed=run['final_checkpoint']['seed'],
            val_test_comparison=comparison,tail=tail,test=recomputed,largest_errors=local[:10],provenance=provenance))
        hist[name]=distance
        print('VERIFIED',name,'images',len(gt['images']),'points',len(distance),'MRE',round(test_mre,5),'over20',tail_count,flush=True)
    for name,data in [('test_results.csv',table),('val_test_comparison.csv',comparisons),('per_keypoint_results.csv',keypoints),
                      ('per_image_errors.csv',images),('per_point_errors.csv',points)]:save_csv(name,data)
    worst=[r for d in results for r in d['largest_errors']]
    save_csv('largest_errors_per_dataset.csv',worst)
    save_json('final_results.json',dict(datasets=results,total_test_images=sum(r['test_images'] for r in table),
        total_evaluated_keypoints=sum(r['test_keypoints'] for r in table),
        note='Per-dataset final test results, selected by validation MRE. No pooled cross-task score; no samples filtered by confidence/error.'))
    save_json('audit.json',dict(training_summary_sha256=sha256(source),datasets=audits,all_checks_passed=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[r['dataset'] for r in table];x=np.arange(len(names));colors=plt.get_cmap('tab10').colors
    fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    ax=axs[0,0];ax.bar(x-.18,[r['val_MRE'] for r in comparisons],.36,label='Validation (selected epoch)')
    ax.bar(x+.18,[r['test_MRE'] for r in comparisons],.36,label='Test');ax.set_ylabel('MRE (original-image pixels)');ax.legend()
    ax.set_xticks(x,names,rotation=25,ha='right');ax.set_title('Validation / test error')
    ax=axs[0,1]
    for j,t in enumerate([2,5,10,20]):ax.bar(x+(j-1.5)*.2,[r[f'PCK@{t}px']*100 for r in table],.2,label=f'PCK@{t}px')
    ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Correct keypoints (%)');ax.legend(ncol=2);ax.set_title('Test PCK')
    ax=axs[1,0]
    for i,name in enumerate(names):
        d=np.sort(hist[name]);ax.step(d,np.arange(1,len(d)+1)/len(d)*100,where='post',label=name,color=colors[i])
    ax.set_xlim(0,20);ax.set_ylim(0,100);ax.set_xlabel('Distance error (px), shown up to 20');ax.set_ylabel('Cumulative keypoints (%)');ax.legend(fontsize=8);ax.set_title('Test error CDF; points beyond 20px remain in denominator')
    ax=axs[1,1]
    ax.bar(x-.18,[r['tail']['fraction_over_20px']*100 for r in results],.36,label='Points >20px (%)')
    ax.bar(x+.18,[r['tail']['over_20px_squared_error_fraction']*100 for r in results],.36,label='Their share of squared error (%)')
    ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylabel('Percent');ax.set_ylim(0,105);ax.legend(fontsize=8);ax.set_title('Influence of large errors')
    for ax in axs.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle('ViTPose-B: final medical keypoint results (different tasks / pixel scales; not a model ranking)')
    for suffix in ['png','pdf']:fig.savefig(OUT/f'final_results_overview.{suffix}',dpi=180)
    plt.close(fig)
    docs=['# ViTPose-B 医学 2D 关键点：最终实验结果','',
        '本报告读取八个数据集现有训练日志汇总、test prediction JSON、test metrics JSON、GT 和最佳 checkpoint；逐文件核对 SHA256，重新计算所有整体及逐点指标，并以独立 NumPy 公式交叉核验。所有检查通过；未执行新训练或推理，未修改原始文件。',
        '',f'测试总计 {sum(r["test_images"] for r in table)} 张图、{sum(r["test_keypoints"] for r in table)} 个有效关键点。所有原有测试标注均有预测。各数据集单独统计，不把不同任务混成一个平均分。',
        '', '## 实验设置与结果来源','',
        '- ViTPose-B，输入 192×256，COCO 预训练迁移，5/14/13/8/13/8/32/26 点输出头；JointsMSELoss 保持原样。',
        '- 八个实验都完成 210 epoch；实际 checkpoint seed 为 1115987209。acdc 从 epoch110 断点续训至210，两个日志作为同一实验。',
        '- 只按验证集 MRE 选择模型：oasis epoch200、LUNA16_26 epoch180，其余 epoch210。测试集未用于选择 checkpoint。',
        '- 每图一个既有 GT 全图 bbox，非检测器端到端评估；所有预测坐标已还原至原始 PNG。',
        '- train/val/test 保持原划分。LUNA16_26 是已有 split 内完整26点样本子集（3749/1059/540），不含32点样本。',
        '- MSE=sum(dx²+dy²)/(2N)，RMSE=sqrt(MSE)，MAE=sum(|dx|+|dy|)/(2N)，MRE=mean(sqrt(dx²+dy²))。MSE单位px²，其余误差为px；没有像素间距，不能解释为毫米。',
        '- PCK@t 为 distance≤t 的有效点比例，表中转换为百分比。只排除 GT visibility=0；不按预测置信度过滤、不裁剪预测、不移除异常值。不使用 AP/AR/OKS 作为主指标。',
        '', '## 最终 test 主结果','',
        '| 数据集 | K | 测试图/点 | epoch | MSE | RMSE | MAE | MRE | PCK2% | PCK5% | PCK10% | PCK20% |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        docs.append(f"| {r['dataset']} | {r['keypoints']} | {r['test_images']}/{r['test_keypoints']} | {r['best_epoch']} | {r['MSE']:.4f} | {r['RMSE']:.4f} | {r['MAE']:.4f} | {r['MRE']:.4f} | "+' | '.join(f"{r[f'PCK@{t}px']*100:.2f}" for t in [2,5,10,20])+' |')
    docs+=['','![最终结果概览](final_results_overview.png)','', '## 验证集与测试集对照','',
        '| 数据集 | val MRE | test MRE | test−val (px) | 相对变化% |','|---|---:|---:|---:|---:|']
    for r in comparisons:docs.append(f"| {r['dataset']} | {r['val_MRE']:.4f} | {r['test_MRE']:.4f} | {r['test_minus_val_MRE']:+.4f} | {r['relative_change_percent']:+.2f} |")
    docs+=['','Synapse-CT、SKI10、hc18、ChestX-ray 的 test MRE 略低于各自 val；acdc 和 LUNA16_26 略高，oasis 与 TG3K 的上升更明显。一次固定划分不能证明统计显著差异或单独证明过拟合，尤其 TG3K test 仅20张、ChestX-ray仅24张。',
        '', '## 误差分布与异常点','',
        '| 数据集 | 中位数px | P95 px | 最大px | >20px点数 | >20px占比% | 这些点贡献平方误差% |','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        t=r['tail'];docs.append(f"| {r['dataset']} | {t['median_distance_px']:.4f} | {t['p95_distance_px']:.4f} | {t['max_distance_px']:.4f} | {t['points_over_20px']} | {t['fraction_over_20px']*100:.2f} | {t['over_20px_squared_error_fraction']*100:.2f} |")
    docs+=['','TG3K、acdc、oasis、LUNA16_26 的 RMSE 受到长尾大误差显著影响，不能仅看 PCK@10px 或平均数。以下逐点分析来自真实预测与标注的差值；误差本身不能区分模型失败、标注不一致或点位定义问题，需要后续检查图像，不能据此自动改标签。','']
    for r in results:
        worst=sorted(r['test']['per_keypoint'],key=lambda q:q['mean_euclidean_distance_error'] or 0,reverse=True)[:3]
        docs += [f"### {r['dataset']}",'',
            '平均误差最大的点：'+ '；'.join(f"{p['name']}={p['mean_euclidean_distance_error']:.4f}px" for p in worst)+'。',
            f"测试 {r['summary']['test_images']} 图，MRE={r['summary']['MRE']:.4f}px，PCK@5px={r['summary']['PCK@5px']*100:.2f}%。预测越界点 {r['tail']['predicted_points_outside_image']}，GT越界点 {r['tail']['gt_points_outside_image']}；越界预测也保留计分。",'',
            '| image_id | point_id/name | 距离px | confidence | 文件 |','|---:|---|---:|---:|---|']
        for e in r['largest_errors'][:2]:docs.append(f"| {e['image_id']} | {e['point_id']}/{e['point_name']} | {e['distance_px']:.4f} | {e['confidence']:.4f} | `{e['file_name']}` |")
        docs+=['']
    docs+=['## 可用于 baseline 总结的结论','',
        '1. 八个任务均完成训练、最佳模型选择及完整 test 评价，预测覆盖与数值复算通过。可将本报告的 test 表作为本次固定划分实验的最终 baseline 表。',
        '2. Synapse-CT 的 test MRE=2.0124px、PCK@5px=96.09%；SKI10 为3.2784px、87.77%。在各自任务上定位较精确，但这些是不同任务的绝对像素指标，不能当作跨数据集公平优劣排名。',
        '3. hc18/ChestX-ray 的 PCK@2px、@5px 相对较低，@10px、@20px 较高，说明大量点落在中等误差区间；不能仅用宽松阈值描述精确定位能力。',
        '4. acdc 的第11、12点、oasis 的point_8、TG3K 的point_5 是优先核查对象；LUNA16_26 应重点检查 point_23/6/4 及高误差样本。现有标注未提供可靠解剖名称，不将编号解释成器官结构。',
        '5. 不根据 test 结果重新选 epoch、删点或改 split。本次只有一个 seed/一次划分，没有多随机种子方差或患者级独立性审计，不作统计显著或临床性能结论。',
        '', '## 交付文件','',
        '- test_results.csv：最终 test 主表（CSV 内 PCK 为0–1比例）。',
        '- val_test_comparison.csv：最佳验证与测试结果差异。',
        '- per_keypoint_results.csv：各关键点全部指标。',
        '- per_image_errors.csv：每张图平均/最大误差。',
        '- per_point_errors.csv：全部有效点预测/GT坐标、confidence和误差。',
        '- largest_errors_per_dataset.csv：每个数据集误差最大的10个点。',
        '- final_results.json：完整结构化结果与指标、长尾统计、文件来源。',
        '- audit.json：预测/GT/config/checkpoint哈希及复算检查记录。',
        '- final_results_overview.png / .pdf：可导出的最终对比图。',
        '- report.md：本报告。此前训练曲线和日志汇总仍在 ../training_summary/，其中“测试待运行”描述的是当时历史状态；本报告为最终结果。',
        '', '复现整理（仅读取结果，不训练、不推理）：','', '```bash','.venv/bin/python datasets_analysis/tools/summarize_final_results.py','```','']
    (OUT/'report.md').write_text('\n'.join(docs))
    for a in audits:
        for key in ['predictions','metrics','gt','config']:
            assert sha256(ROOT/a['files'][key])==a['sha256'][key]
    print('Final report complete. Images:',sum(r['test_images'] for r in table),'points:',sum(r['test_keypoints'] for r in table))


if __name__=='__main__':main()
