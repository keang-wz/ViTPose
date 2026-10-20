"""Build all/bad/good galleries for the eight existing first-run test predictions."""
import argparse
import html
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets_analysis.converters.common import safe_output
from datasets_analysis.medical_evaluation.metrics import read_json, write_json
from datasets_analysis.tools.visualize_medical_keypoints import (
    gallery_selections, preflight, prepare_context, render_selections, write_html)

EXPECTED = {'TG3K', 'acdc', 'hc18', 'oasis', 'Synapse-CT', 'SKI10', 'ChestX-ray', 'LUNA16_26'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-root', type=Path, default=Path('/data/users/mqf/bushu/Painter-ct/data'))
    p.add_argument('--out-root', type=Path, default=ROOT/'results/visaulization')
    p.add_argument('--examples', type=int, default=5)
    p.add_argument('--error-ranking', choices=['mre', 'max'], default='max')
    p.add_argument('--results-json', type=Path,
                   default=ROOT/'datasets_analysis/results/final_summary/final_results.json')
    args = p.parse_args()
    if args.examples < 1:
        p.error('--examples must be positive')
    results = read_json(args.results_json)['datasets']
    if {r['dataset'] for r in results} != EXPECTED or len(results) != len(EXPECTED):
        raise ValueError('Expected exactly the eight first-run medical datasets')
    output = safe_output(args.out_root, args.images_root)
    if (output/'index.html').exists() or (output/'manifest.json').exists():
        raise FileExistsError('Root gallery exists; use a new --out-root')
    jobs = []
    for result in results:
        name = result['dataset'];files = result['provenance']['files']
        context = prepare_context(args.images_root, ROOT/files['gt'], ROOT/files['predictions'], name)
        if context['predictions_sha256'] != result['provenance']['sha256']['predictions']:
            raise ValueError(f'First-run prediction changed: {name}')
        if len(context['records']) != result['summary']['test_images']:
            raise ValueError(f'Wrong test coverage: {name}')
        selections = gallery_selections(context['records'], args.examples, args.error_ranking)
        if len(selections[0][1]) != len(context['records']):
            raise ValueError(f'Image without annotated GT points: {name}')
        preflight(context, output/name, selections, complete=True)
        jobs.append((result, context, selections))
    summaries = []
    for result, context, selections in jobs:
        font_size = 10 if result['summary']['keypoints'] >= 26 else 12
        summary = render_selections(context, output/result['dataset'], selections,
            args.examples, args.error_ranking, font_size=font_size, complete=True)
        summary['training_seed'] = result['seed']
        summary['checkpoint'] = context['checkpoint']
        summaries.append(summary)
    total = sum(s['groups'][0]['images'] for s in summaries)
    body = ('<header><h1>第一轮 ViTPose-B 医学关键点可视化</h1>'
            f'<p>八个数据集 · 全部 {total} 张 test 图像 · 各组均按真实已有预测生成</p>'
            f'<p>大/小误差样例按 {html.escape(args.error_ranking)} 排序，每组最多 {args.examples} 张。</p>'
            '<p>绿色 GT · 红色预测 · 黄色误差线。点击文件夹进入图片画廊。</p></header>'
            '<table><thead><tr><th>数据集</th><th>全部图片</th><th>大误差</th><th>小误差</th></tr></thead><tbody>')
    for summary in summaries:
        name = summary['dataset'];body += f'<tr><td><a href="{name}/index.html">{name}</a></td>'
        for group in summary['groups']:
            body += f'<td><a href="{name}/{group["mode"]}/index.html">{group["images"]} 张</a></td>'
        body += '</tr>'
    body += '</tbody></table>'
    write_html(output/'index.html', '第一轮医学关键点可视化', body)
    write_json(output/'manifest.json', dict(experiment='first_run', split='test',
        total_unique_images=total, error_ranking=args.error_ranking, datasets=summaries))
    print(f'COMPLETE: {total} unique test images; browse {output / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
