"""Overlay existing medical keypoint predictions and COCO GT at original size.

Uses only JSON, NumPy and Pillow. No checkpoint, model, inference or training.
"""
import argparse
import html
import math
from pathlib import Path
import random
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets_analysis.converters.common import safe_output
from datasets_analysis.medical_evaluation.metrics import (
    evaluate, index_gt, read_json, sha256, write_json)

GT_COLOR = (0, 255, 0, 255)
PRED_COLOR = (255, 0, 0, 255)
ERROR_COLOR = (255, 210, 0, 220)


def prepare_records(prediction, gt):
    """Reuse evaluator validation; resolve predictions by IDs, never list order."""
    evaluate(prediction, gt)
    images, categories, annotations = index_gt(gt)
    by_image = {}
    for ann in annotations.values():
        by_image.setdefault(ann['image_id'], []).append(ann['id'])
    records = {iid: dict(image=im, points=[]) for iid, im in images.items()}
    for pred in prediction['predictions']:
        iid = pred['image_id']
        aid = pred.get('annotation_id')
        if aid is None:
            # evaluate() already rejects ambiguous multi-target matching.
            aid = by_image[iid][0]
        ann = annotations[aid]
        truth = np.asarray(ann['keypoints'], dtype=float).reshape(-1, 3)
        point_ids = ann.get('source_point_ids', list(range(1, len(truth) + 1)))
        if len(point_ids) != len(truth) or len(set(point_ids)) != len(point_ids):
            raise ValueError(f'Invalid source_point_ids: annotation {aid}')
        for index, (xy, gt_point) in enumerate(zip(pred['keypoints'], truth)):
            if gt_point[2] == 0:
                continue  # Same visibility rule as medical evaluation.
            error = float(np.linalg.norm(np.asarray(xy) - gt_point[:2]))
            records[iid]['points'].append(dict(
                annotation_id=aid, point_id=point_ids[index],
                label=f'point_{point_ids[index]}', gt=gt_point[:2].tolist(),
                prediction=list(xy), confidence=pred['confidence'][index],
                error_px=error))
    for record in records.values():
        errors = [point['error_px'] for point in record['points']]
        record['mre_px'] = float(np.mean(errors)) if errors else None
        record['max_error_px'] = max(errors) if errors else None
    return records, categories


def select_images(records, count, image_ids=None, largest_errors=False,
                  error_ranking='mre', seed=42, all_images=False, smallest_errors=False):
    eligible = sorted(iid for iid, rec in records.items() if rec['points'])
    if image_ids is not None:
        if len(set(image_ids)) != len(image_ids):
            raise ValueError('Duplicate --image-ids')
        invalid = [iid for iid in image_ids if iid not in eligible]
        if invalid:
            raise ValueError(f'Unknown image IDs or no annotated GT points: {invalid}')
        return 'selected', image_ids
    if not eligible:
        raise ValueError('No images with annotated GT keypoints')
    if all_images:
        return 'all', eligible
    count = min(count, len(eligible))
    if largest_errors or smallest_errors:
        key = 'mre_px' if error_ranking == 'mre' else 'max_error_px'
        direction = -1 if largest_errors else 1
        mode = 'largest_errors' if largest_errors else 'smallest_errors'
        return mode, sorted(eligible, key=lambda iid: (direction * records[iid][key], iid))[:count]
    return 'random', random.Random(seed).sample(eligible, count)


def image_path(images_root, file_name):
    relative = Path(file_name)
    path = (images_root / relative).resolve()
    if (relative.is_absolute() or '\\' in file_name
            or re.match(r'^[A-Za-z]:', file_name) or '..' in relative.parts
            or images_root not in path.parents):
        raise ValueError(f'Expected safe COCO relative file_name: {file_name}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def font_at(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def intersection_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def label_position(anchor, size, canvas_size, occupied, header_bottom):
    """Prefer nearby non-overlapping labels; only text moves, never point coords."""
    x, y = anchor
    tw, th = size
    width, height = canvas_size
    xmax, ymax = max(2, width - tw - 2), max(2, height - th - 2)
    ymin = min(header_bottom + 3, ymax)
    candidates = []
    for gap in (8, 24, 48, 80, 112):
        candidates.extend([(x + gap, y + gap), (x + gap, y - th - gap),
                           (x - tw - gap, y + gap), (x - tw - gap, y - th - gap),
                           (x + gap, y - th / 2), (x - tw - gap, y - th / 2),
                           (x - tw / 2, y + gap), (x - tw / 2, y - th - gap)])
    scored = []
    for left, top in candidates:
        left, top = max(2, min(left, xmax)), max(ymin, min(top, ymax))
        box = (left, top, left + tw, top + th)
        overlap = sum(intersection_area(box, other) for other in occupied)
        distance = math.hypot(left + tw / 2 - x, top + th / 2 - y)
        scored.append((overlap, distance, box))
    return min(scored, key=lambda item: item[:2])[2]


def draw_overlay(source, record, dataset_name, font_size=12, marker_radius=3):
    """Draw on a same-size RGBA layer; do not resize or EXIF-rotate the source."""
    canvas = source.convert('RGB')
    layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = canvas.size
    iid = record['image']['id']
    title = (f'{dataset_name} | image_id={iid}\n'
             f'MRE={record["mre_px"]:.2f} px | max={record["max_error_px"]:.2f} px')
    title_font = font_at(font_size)
    for size in range(font_size - 1, 6, -1):
        if draw.multiline_textbbox((0, 0), title, font=title_font)[2] <= width - 16:
            break
        title_font = font_at(size)
    bbox = draw.multiline_textbbox((0, 0), title, font=title_font, spacing=3)
    text_height = bbox[3] - bbox[1]
    legend_font = font_at(max(9, font_size - 1))
    legend_y = 8 + text_height + 4
    header_bottom = min(height - 1, legend_y + font_size + 7)
    draw.rectangle((0, 0, width - 1, header_bottom), fill=(0, 0, 0, 175))
    draw.multiline_text((8, 5), title, font=title_font, fill='white', spacing=3)
    draw.text((8, legend_y), 'GT', font=legend_font, fill=GT_COLOR)
    draw.text((42, legend_y), 'Pred', font=legend_font, fill=PRED_COLOR)
    draw.text((89, legend_y), 'GT-Pred error', font=legend_font, fill=ERROR_COLOR)

    occupied = [(0, 0, width, header_bottom)]
    for point in record['points']:
        gx, gy = point['gt'];px, py = point['prediction']
        draw.line((gx, gy, px, py), fill=ERROR_COLOR, width=1)
        for x, y in [(gx, gy), (px, py)]:
            occupied.append((x - marker_radius - 3, y - marker_radius - 3,
                             x + marker_radius + 3, y + marker_radius + 3))
    font = font_at(font_size)
    multiple_targets = len({p['annotation_id'] for p in record['points']}) > 1
    for point in record['points']:
        px, py = point['prediction']
        outside = not (0 <= px < width and 0 <= py < height)
        label = point['label']
        if multiple_targets:
            label += f' (ann={point["annotation_id"]})'
        text = f'{label}\nerror={point["error_px"]:.2f} px'
        if outside:
            text += '\noutside image'
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, stroke_width=1)
        tw, th = bbox[2] - bbox[0] + 6, bbox[3] - bbox[1] + 6
        box = label_position((px, py), (tw, th), canvas.size, occupied, header_bottom)
        occupied.append(box)
        # A red callout points back to the prediction when labels need spacing.
        nearest = (max(box[0], min(px, box[2])), max(box[1], min(py, box[3])))
        draw.line((px, py, *nearest), fill=(255, 80, 80, 170), width=1)
        draw.rectangle(box, fill=(0, 0, 0, 155))
        draw.multiline_text((box[0] + 3 - bbox[0], box[1] + 3 - bbox[1]), text,
                            font=font, spacing=2, fill=PRED_COLOR,
                            stroke_width=1, stroke_fill=(0, 0, 0, 255))
    # Draw markers last so callouts cannot hide their original coordinate centers.
    for point in record['points']:
        gx, gy = point['gt'];px, py = point['prediction']
        radius = marker_radius + 1  # Green rim remains visible if GT==prediction.
        draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), fill=GT_COLOR)
        radius = marker_radius
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=PRED_COLOR)
    return Image.alpha_composite(canvas.convert('RGBA'), layer).convert('RGB')


def infer_dataset_name(gt_path, categories):
    if gt_path.parent.name == 'annotations':
        return gt_path.parent.parent.name
    names = {cat.get('supercategory', cat['name']) for cat in categories.values()}
    return next(iter(names)) if len(names) == 1 else gt_path.stem


HTML_STYLE = """
body{margin:0;padding:24px;background:#111827;color:#e5e7eb;font:15px system-ui,sans-serif}
a{color:#93c5fd}header{max-width:1200px;margin:auto auto 24px}input{padding:10px;width:min(400px,90%)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.card{background:#1f2937;border-radius:8px;padding:12px;overflow:hidden}
.card img{display:block;width:100%;height:auto}.meta{line-height:1.6;overflow-wrap:anywhere}
nav a{display:inline-block;padding:8px 12px;margin:0 8px 8px 0;background:#374151;border-radius:6px}
table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #374151}
"""


def write_html(path, title, body):
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(title)}</title><style>{HTML_STYLE}</style>'
                f'<body>{body}</body></html>')
    with path.open('x', encoding='utf-8') as stream:
        stream.write(document)


def write_gallery_index(folder, dataset, mode, entries):
    names = dict(all='全部测试图片', largest_errors='大误差样例', smallest_errors='小误差样例',
                 selected='指定图片', random='随机样例')
    title = f'{dataset} — {names[mode]} ({len(entries)} 张)'
    body = (f'<header><a href="../index.html">返回数据集目录</a><h1>{html.escape(title)}</h1>'
            '<p>绿色 GT · 红色预测 · 黄色误差线；PNG 保持原图尺寸。</p>'
            '<input id="search" placeholder="按 image_id 或原图路径筛选" aria-label="筛选图片"></header>'
            '<main class="grid">')
    for entry in entries:
        output = html.escape(entry['output'], quote=True)
        source = html.escape(entry['file_name'])
        search = html.escape(f"{entry['image_id']} {entry['file_name']}".lower(), quote=True)
        body += (f'<article class="card" data-search="{search}"><div class="meta">'
                 f"<b>image_id={entry['image_id']}</b><br>MRE={entry['mre_px']:.2f} px · "
                 f"max={entry['max_error_px']:.2f} px<br>{source}</div>"
                 f'<a href="{output}" target="_blank" rel="noopener">'
                 f'<img src="{output}" loading="lazy" width="{entry["width"]}" '
                 f'height="{entry["height"]}" alt="image {entry["image_id"]}"></a></article>')
    body += """</main><script>
document.getElementById('search').addEventListener('input', function(){
 const q=this.value.trim().toLowerCase();
 document.querySelectorAll('.card').forEach(c=>c.hidden=!c.dataset.search.includes(q));
});
</script>"""
    write_html(folder/'index.html', title, body)


def prepare_context(images_root, gt_path, predictions_path, dataset_name=None):
    images_root = Path(images_root).resolve()
    if not images_root.is_dir():
        raise ValueError('--images-root must be an existing directory')
    gt, prediction = read_json(gt_path), read_json(predictions_path)
    gt_hash, prediction_hash = sha256(gt_path), sha256(predictions_path)
    expected_hash = prediction.get('metadata', {}).get('annotation_sha256')
    if expected_hash is not None and expected_hash != gt_hash:
        raise ValueError('Prediction/GT SHA256 mismatch; check split')
    records, categories = prepare_records(prediction, gt)
    if any(not isinstance(iid, int) or isinstance(iid, bool) for iid in records):
        raise ValueError('COCO image IDs must be integers')
    return dict(images_root=images_root, records=records,
                dataset=dataset_name or infer_dataset_name(Path(gt_path), categories),
                gt_path=Path(gt_path), predictions_path=Path(predictions_path),
                gt_sha256=gt_hash, predictions_sha256=prediction_hash,
                checkpoint=prediction.get('metadata', {}).get('checkpoint'))


def gallery_selections(records, count=5, error_ranking='max'):
    return [select_images(records, count, all_images=True),
            select_images(records, count, largest_errors=True, error_ranking=error_ranking),
            select_images(records, count, smallest_errors=True, error_ranking=error_ranking)]


def preflight(context, output_root, selections, complete=False):
    """Check every selected source before writing; never overwrite an old run."""
    output_root = safe_output(output_root, context['images_root'])
    if complete and ((output_root/'index.html').exists() or (output_root/'summary.json').exists()):
        raise FileExistsError(f'Gallery exists: {output_root}; use another output directory')
    checked = {}
    for mode, ids in selections:
        folder = output_root/mode
        for name in ['manifest.json', 'index.html'] + [f'image_{iid:03d}.png' for iid in ids]:
            path = folder/name
            if path.exists() or path.is_symlink():
                raise FileExistsError(f'Output exists: {path}; use another --out-dir')
        for iid in ids:
            if iid in checked:
                continue
            record = context['records'][iid]
            source = image_path(context['images_root'], record['image']['file_name'])
            with Image.open(source) as im:
                if im.size != (record['image']['width'], record['image']['height']):
                    raise ValueError(f'Image/GT size mismatch: {source}')
            checked[iid] = source
    return output_root, checked


def render_selections(context, output_root, selections, count=5, error_ranking='mre',
                      seed=42, font_size=12, marker_radius=3, complete=False):
    output_root, sources = preflight(context, output_root, selections, complete)
    dataset, records = context['dataset'], context['records']
    summaries = []
    for mode, ids in selections:
        folder = output_root/mode
        folder.mkdir(parents=True, exist_ok=True)
        manifest = dict(dataset=dataset, mode=mode, seed=seed, error_ranking=error_ranking,
                        coordinate_space='original_image_pixels', checkpoint=context['checkpoint'],
                        gt_sha256=context['gt_sha256'], predictions_sha256=context['predictions_sha256'],
                        skipped_images_without_annotated_points=[iid for iid, r in records.items() if not r['points']],
                        images=[])
        for index, iid in enumerate(ids, 1):
            record = records[iid];target = folder/f'image_{iid:03d}.png'
            original_overlay = output_root/'all'/target.name
            if complete and mode != 'all':
                # Copy the already-rendered image; points/labels remain identical.
                with original_overlay.open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                width, height = record['image']['width'], record['image']['height']
            else:
                with Image.open(sources[iid]) as im:
                    canvas = draw_overlay(im, record, dataset, font_size, marker_radius)
                with target.open('xb') as stream:
                    canvas.save(stream, format='PNG')
                width, height = canvas.size
            manifest['images'].append(dict(image_id=iid, file_name=record['image']['file_name'],
                output=target.name, width=width, height=height, mre_px=record['mre_px'],
                max_error_px=record['max_error_px'], points=record['points']))
            if not complete or index % 25 == 0 or index == len(ids):
                print(f'{dataset}/{mode}: {index}/{len(ids)}', flush=True)
        write_json(folder/'manifest.json', manifest,
                   protected=[context['gt_path'], context['predictions_path']])
        write_gallery_index(folder, dataset, mode, manifest['images'])
        summaries.append(dict(mode=mode, images=len(ids), image_ids=ids,
                              mre_px=[records[iid]['mre_px'] for iid in ids],
                              max_error_px=[records[iid]['max_error_px'] for iid in ids]))
    # A complete gallery has exactly three groups and one browsing entry point.
    if complete:
        names = dict(all='全部图片', largest_errors='大误差样例', smallest_errors='小误差样例')
        body = (f'<header><a href="../index.html">全部数据集</a><h1>{html.escape(dataset)}</h1>'
                f'<p>{html.escape(context.get("experiment_label", "第一轮实验"))} · test · 样例按 {html.escape(error_ranking)} 排序</p><nav>')
        for group in summaries:
            body += f'<a href="{group["mode"]}/index.html">{names[group["mode"]]} ({group["images"]} 张)</a>'
        body += '</nav><p>点击进入文件夹查看所有图片，可按 image_id 或原图路径筛选。</p></header>'
        write_html(output_root/'index.html', dataset, body)
        write_json(output_root/'summary.json', dict(dataset=dataset, split='test',
            error_ranking=error_ranking, groups=summaries,
            gt_sha256=context['gt_sha256'], predictions_sha256=context['predictions_sha256']))
    if (sha256(context['gt_path']) != context['gt_sha256']
            or sha256(context['predictions_path']) != context['predictions_sha256']):
        raise ValueError('Input JSON changed while rendering')
    return dict(dataset=dataset, output=str(output_root), groups=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-root', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--num-samples', type=int, default=10)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--image-ids', type=int, nargs='+')
    selection.add_argument('--largest-errors', action='store_true')
    selection.add_argument('--smallest-errors', action='store_true')
    selection.add_argument('--all-images', action='store_true')
    selection.add_argument('--complete-gallery', action='store_true',
                           help='Render all images plus largest/smallest error examples')
    parser.add_argument('--error-ranking', choices=['mre', 'max'], default='mre')
    parser.add_argument('--seed', type=int, default=42, help='Visualization sampling only')
    parser.add_argument('--dataset-name')
    parser.add_argument('--font-size', type=int, default=12)
    parser.add_argument('--marker-radius', type=int, default=3)
    args = parser.parse_args()
    if args.num_samples < 1 or args.font_size < 7 or args.marker_radius < 1:
        parser.error('num-samples and marker-radius must be positive; font-size must be >=7')
    if args.error_ranking != 'mre' and not (args.largest_errors or args.smallest_errors or args.complete_gallery):
        parser.error('--error-ranking applies to error-ranked examples or complete galleries')
    context = prepare_context(args.images_root, args.gt, args.predictions, args.dataset_name)
    if args.complete_gallery:
        selections = gallery_selections(context['records'], args.num_samples, args.error_ranking)
    else:
        selections = [select_images(context['records'], args.num_samples, args.image_ids,
            args.largest_errors, args.error_ranking, args.seed, args.all_images, args.smallest_errors)]
    render_selections(context, args.out_dir, selections, args.num_samples, args.error_ranking,
                      args.seed, args.font_size, args.marker_radius, args.complete_gallery)


if __name__ == '__main__':
    main()
