"""Convert the four Painter-3D datasets using their supplied train/val/test JSON.

Reuses the existing converter without resizing images, editing coordinates,
sampling records, or constructing/reassessing split assignments.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets_analysis.converters.common import (
    convert, load_splits, resolve_image, safe_output, write_json)

DATASETS = {'acdc': 14, 'hc18': 13, 'TG3K': 5, 'ChestX-ray': 32}
DATA_ROOT = Path('/data/users/mqf/Painter-3D/data')
OUTPUT_ROOT = ROOT / 'datasets_analysis/coco_format/painter3d'
REPORT_ROOT = ROOT / 'datasets_analysis/docs/painter3d'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--report-root', type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    output = safe_output(args.output_root, args.data_root)
    reports = safe_output(args.report_root, args.data_root)
    preview_root = safe_output(ROOT / 'datasets_analysis/samples/painter3d', args.data_root)
    # Do not overwrite an earlier conversion accidentally.
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output already contains files: {output}')
    report = dict(data_root=str(args.data_root.resolve()),
                  split_policy='Use dataset-local data/{train,val,test}.json verbatim',
                  datasets={})
    for name, k in DATASETS.items():
        splits = load_splits(args.data_root, name)
        stats = {}
        for split, rows in splits.items():
            sizes, modes, formats = Counter(), Counter(), Counter()
            outside = 0
            source = args.data_root / name / 'data' / f'{split}.json'
            for index, row in enumerate(rows):
                if row['point_ids'] != list(range(1, k + 1)) or len(row['points']) != k:
                    raise ValueError(f'{name}/{split}/{index}: unexpected keypoint layout')
                path, _ = resolve_image(args.data_root, name, row['image_path'])
                with Image.open(path) as im:
                    im.load()
                    w, h = im.size
                    sizes[f'{w}x{h}'] += 1
                    modes[im.mode] += 1
                    formats[im.format] += 1
                    outside += sum(not (0 <= p['x'] < w and 0 <= p['y'] < h)
                                   for p in row['points'])
                    if index == 0:
                        preview = im.convert('RGB')
                        draw = ImageDraw.Draw(preview)
                        radius = max(2, round(min(w, h) / 160))
                        for point_id, point in zip(row['point_ids'], row['points']):
                            x, y = point['x'], point['y']
                            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                                         outline='red', width=2)
                            draw.text((x+radius+1, y), str(point_id), fill='yellow')
                        target = preview_root / name / f'{split}.png'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        preview.save(target)
            stats[split] = dict(images=len(rows), keypoints=k, sizes=dict(sizes),
                                modes=dict(modes), formats=dict(formats),
                                out_of_bounds_points=outside,
                                source_json=str(source),
                                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        converted = convert(name, args.data_root, output)
        for split, stat in stats.items():
            assert converted[split]['images'] == stat['images']
            assert hashlib.sha256(Path(stat['source_json']).read_bytes()).hexdigest() == stat['source_sha256']
        report['datasets'][name] = stats
    write_json(reports / 'dataset_analysis.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
