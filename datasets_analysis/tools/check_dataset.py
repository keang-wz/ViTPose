"""Read every input image; audit raw metadata and independently validate COCO."""
import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'converters'))
from common import BASE, DATASETS, SPLITS, DEFAULT_ROOT, load_splits, resolve_image, safe_output, write_json
from PIL import Image
import numpy as np

def path_kind(value):
    return 'windows_absolute' if re.match(r'^[A-Za-z]:[\\/]', value) else 'linux_absolute' if value.startswith('/') else 'relative'

def audit(root, dataset, output):
    splits = load_splits(root, dataset)
    fields, point_fields, paths, sizes, modes, formats, counts = (Counter() for _ in range(7))
    tree = {}; grayscale = 0; total = 0; names = set(); source_dirs = {}
    all_files = Counter(p.suffix.lower() for p in (root/dataset).rglob('*') if p.is_file())
    input_files = {p.resolve() for p in (root/dataset).rglob('input.png')}
    referenced = set(); errors = []; validation = {}; seen_paths = set(); seen_ids = set()
    for split, rows in splits.items():
        rawpath = root/dataset/'data'/f'{split}.json'
        coco = json.loads((output/dataset/'annotations'/f'{split}.json').read_text())
        cats = {c['id']: c for c in coco['categories']}
        stats = dict(images=len(coco['images']), annotations=len(coco['annotations']), keypoint_count_distribution={},
                     keypoints=0, empty_annotations=0, out_of_bounds_points=0, out_of_bounds_annotations=0,
                     missing_slots=0, issues=[])
        def issue(msg):
            stats['issues'].append(msg)
        if len(rows) != len(coco['images']) or len(rows) != len(coco['annotations']):
            issue('Source/output record count mismatch')
        if hashlib.sha256(rawpath.read_bytes()).hexdigest() != coco['info']['source_sha256']:
            issue('Source split hash changed')
        anns = {a['image_id']: a for a in coco['annotations']}
        if len(anns) != len(coco['annotations']) or len(set(a['id'] for a in coco['annotations'])) != len(coco['annotations']):
            issue('Duplicate annotation ids or multiple annotations per image')
        kc = Counter()
        for idx, row in enumerate(rows):
            fields.update(row.keys()); point_fields.update(k for p in row['points'] for k in p)
            counts[len(row['points'])] += 1
            if 'point_names' in row: names.add(tuple(row['point_names']))
            for key, value in row.items():
                if 'path' in key or key == 'source_dir':
                    for v in value if isinstance(value, list) else [value]:
                        paths[f'{key}:{path_kind(v)}'] += 1
            if row.get('source_dir'):
                source_dirs.setdefault(row['source_dir'], set()).add(split)
            path, rel = resolve_image(root, dataset, row['image_path']); referenced.add(path)
            with Image.open(path) as im:
                im.load()  # Full decode, not just reading the header.
                w,h = im.size; sizes[f'{w}x{h}'] += 1; modes[im.mode] += 1; formats[im.format] += 1
                a = np.asarray(im)
                grayscale += int(a.ndim == 2 or (a.ndim == 3 and a.shape[2] >= 3 and np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2])))
            total += 1
            if idx >= len(coco['images']):
                continue
            ci = coco['images'][idx]; ann = anns.get(ci['id'])
            if ci['file_name'] != rel or ci['width'] != w or ci['height'] != h:
                issue(f'Image/order mismatch: {rel}')
            if ci['id'] in seen_ids or rel in seen_paths:
                issue(f'Duplicate image id/path: {rel}')
            seen_ids.add(ci['id']);seen_paths.add(rel)
            if Path(ci['file_name']).is_absolute() or '\\' in ci['file_name'] or re.match(r'^[A-Za-z]:',ci['file_name']):
                issue(f'Nonportable path: {rel}')
            if ann is None:
                issue(f'Missing annotation: {rel}');continue
            expected = [v for p in row['points'] for v in (p['x'],p['y'],2)]
            if ann['keypoints'] != expected or ann['source_point_ids'] != row['point_ids'] or ann['source_row_index'] != idx:
                issue(f'Keypoint/order mismatch: {rel}')
            if ann['category_id'] not in cats or len(ann['keypoints']) != 3*len(cats[ann['category_id']]['keypoints']):
                issue(f'Category schema mismatch: {rel}')
            kp = ann['keypoints']; n = len(kp)//3; kc[n] += 1
            if ann['num_keypoints'] != sum(v>0 for v in kp[2::3]): issue(f'num_keypoints mismatch: {rel}')
            if ann['bbox'] != [0,0,w,h] or ann['area'] != w*h or ann['iscrowd'] != 0: issue(f'ROI mismatch: {rel}')
            stats['keypoints'] += n
            stats['empty_annotations'] += int(ann['num_keypoints']==0)
            oob = 0
            for x,y,v in zip(kp[::3],kp[1::3],kp[2::3]):
                if not all(math.isfinite(q) for q in (x,y,v)) or v not in (0,1,2): issue(f'Invalid point: {rel}')
                stats['missing_slots'] += int(v==0)
                oob += int(v>0 and not (0<=x<w and 0<=y<h))
            stats['out_of_bounds_points'] += oob
            stats['out_of_bounds_annotations'] += int(oob>0)
            if oob and len(errors)<20: errors.append(dict(split=split,file_name=rel,out_of_bounds=oob))
        stats['keypoint_count_distribution'] = dict(kc)
        stats['passed'] = not stats['issues'] and not stats['out_of_bounds_points'] and not stats['empty_annotations']
        validation[split] = stats
        folders = sorted((root/dataset/'data'/split).iterdir())
        tree[split] = dict(sample_directories=len(folders), example_directory=folders[0].name,
                           example_files=sorted(p.name for p in folders[0].iterdir()))
        print(f'Checked {dataset}/{split}: {len(rows)} images; OOB={stats["out_of_bounds_points"]}', flush=True)
    return dict(split_files=[f'{dataset}/data/{s}.json' for s in SPLITS], fields=dict(fields), point_fields=dict(point_fields),
        path_types=dict(paths), image_sizes=dict(sizes), image_modes=dict(modes), image_formats=dict(formats),
        grayscale_pixel_images=grayscale, input_images=total, all_file_extensions=dict(all_files),
        unreferenced_input_images=len(input_files-referenced), keypoint_counts=dict(counts), point_names=[list(x) for x in sorted(names)],
        exact_source_dir_cross_split=sum(len(s)>1 for s in source_dirs.values()),
        tree=tree, out_of_bounds_examples=errors, validation=validation,
        fixed_head_compatible=len(counts)==1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--output-root',type=Path,default=BASE/'coco_format')
    p.add_argument('--report',type=Path,default=BASE/'docs'/'validation_results.json')
    args=p.parse_args(); report=safe_output(args.report,args.data_root)
    results={d:audit(args.data_root,d,args.output_root) for d in DATASETS}
    write_json(report,dict(data_root=str(args.data_root),datasets=results,
        note='COCO structural validity is separate from fixed-head training readiness; no split changes.'))
    if any(not s['passed'] for d in results.values() for s in d['validation'].values()): sys.exit(1)
if __name__=='__main__': main()
