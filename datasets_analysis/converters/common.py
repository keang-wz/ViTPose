"""Lossless split-preserving conversion of the eight inspected medical datasets."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from PIL import Image

DATASETS = ('TG3K', 'acdc', 'hc18', 'oasis', 'Synapse-CT', 'SKI10', 'ChestX-ray', 'LUNA16')
SPLITS = ('train', 'val', 'test')
DEFAULT_ROOT = Path('/data/users/mqf/bushu/Painter-ct/data')
BASE = Path(__file__).resolve().parents[1]

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def safe_output(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root in path.parents:
        raise ValueError('Output must be outside raw data root')
    return path

def resolve_image(root, dataset, value):
    """Map an explicit dataset/data suffix, never search by basename."""
    value = value.replace('\\', '/')
    marker = dataset + '/data/'
    if marker in value:
        suffix = value.split(marker, 1)[1]
    elif re.match(r'^[A-Za-z]:', value) or value.startswith('/'):
        raise ValueError(f'Unrecognized absolute image path: {value}')
    else:
        suffix = value
    rel = Path(dataset) / 'data' / suffix
    path = (Path(root) / rel).resolve()
    base = (Path(root) / dataset / 'data').resolve()
    if base not in path.parents or '..' in rel.parts:
        raise ValueError(f'Unsafe image path: {value}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, rel.as_posix()

def load_splits(root, dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    result = {}
    for split in SPLITS:
        path = Path(root) / dataset / 'data' / (split + '.json')
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f'Expected JSON list: {path}')
        result[split] = rows
    return result

def convert(dataset, root=DEFAULT_ROOT, output=BASE / 'coco_format'):
    root = Path(root).resolve()
    output = safe_output(output, root)
    splits = load_splits(root, dataset)
    patterns = list(dict.fromkeys(tuple(r['point_ids']) for rows in splits.values() for r in rows))
    # Each schema remains a separate category; no anatomical equivalence is assumed.
    categories = []
    for cid, ids in enumerate(patterns, 1):
        example = next(r for rows in splits.values() for r in rows if tuple(r['point_ids']) == ids)
        names = example.get('point_names', [f'point_{i}' for i in ids])
        categories.append(dict(id=cid, name=f'{dataset}_{len(ids)}point', supercategory=dataset,
                               keypoints=names, skeleton=[], source_point_ids=list(ids)))
    result = {}
    seen = set()
    next_id = 1
    for split, rows in splits.items():
        images, annotations = [], []
        for row_index, row in enumerate(rows):
            path, relative = resolve_image(root, dataset, row['image_path'])
            if relative in seen:
                raise ValueError(f'Duplicate image within/across splits: {relative}')
            seen.add(relative)
            with Image.open(path) as im:
                width, height = im.size
            ids, pts = row['point_ids'], row['points']
            if len(ids) != len(pts) or len(ids) != len(set(ids)):
                raise ValueError(f'Invalid point_ids: {relative}')
            kp = []
            for p in pts:
                if set(p) != {'x', 'y'}:
                    raise ValueError(f'Unknown point schema: {relative}: {p}')
                x, y = p['x'], p['y']
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
                    raise ValueError(f'Invalid coordinates: {relative}')
                kp.extend([x, y, 2])
            cid = patterns.index(tuple(ids)) + 1
            if row.get('point_names', categories[cid-1]['keypoints']) != categories[cid-1]['keypoints']:
                raise ValueError('Inconsistent point names')
            images.append(dict(id=next_id, file_name=relative, width=width, height=height))
            annotations.append(dict(id=next_id, image_id=next_id, category_id=cid, keypoints=kp,
                num_keypoints=len(pts), bbox=[0, 0, width, height], area=width*height, iscrowd=0,
                source_row_index=row_index, source_point_ids=ids))
            next_id += 1
        source = root / dataset / 'data' / (split + '.json')
        coco = dict(info=dict(description='Existing split; full-image ROI; v=2 means annotated, not verified visibility',
                    source_split=f'{dataset}/data/{split}.json', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                    images=images, annotations=annotations, categories=categories)
        write_json(output / dataset / 'annotations' / (split + '.json'), coco)
        result[split] = dict(images=len(images), annotations=len(annotations), points=sum(a['num_keypoints'] for a in annotations))
        print(f'{dataset}/{split}: {result[split]}', flush=True)
    write_json(output / dataset / 'conversion_log.json', dict(dataset=dataset, splits=result,
        category_point_counts=[len(c['keypoints']) for c in categories],
        fixed_head_compatible=len(categories)==1, bbox_policy='full image; no original object boxes'))
    return result

def cli(dataset=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output-root', type=Path, default=BASE/'coco_format')
    if dataset is None:
        p.add_argument('--dataset', choices=(*DATASETS, 'all'), default='all')
    args = p.parse_args()
    for d in ([dataset] if dataset else DATASETS if args.dataset == 'all' else [args.dataset]):
        convert(d, args.data_root, args.output_root)

if __name__ == '__main__':
    cli()
