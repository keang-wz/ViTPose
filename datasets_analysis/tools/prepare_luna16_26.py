"""Select existing 26-point COCO records without changing split, IDs or pixels."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets_analysis.medical_evaluation.metrics import index_gt, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'datasets_analysis/coco_format/LUNA16')
    p.add_argument('--output', type=Path, default=ROOT/'datasets_analysis/coco_format/LUNA16_26')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Output exists; refusing to overwrite existing conversion/results')
    outputs, report, seen = {}, {}, set()
    for split in ('train', 'val', 'test'):
        path = args.source/'annotations'/f'{split}.json'
        c = read_json(path)
        images, categories, annotations = index_gt(c)
        counts = Counter()
        selected = []
        for a in annotations.values():
            k = len(a['keypoints']) // 3
            counts[k] += 1
            if k not in (26, 32):
                raise ValueError(f'Unexpected point count: {k}')
            if a['source_point_ids'] != list(range(1, k+1)):
                raise ValueError('Unexpected source_point_ids/order')
            if categories[a['category_id']]['source_point_ids'] != a['source_point_ids']:
                raise ValueError('Category/annotation point_ids disagree')
            if k == 26:
                selected.append(a)
        image_ids = {a['image_id'] for a in selected}
        category_ids = {a['category_id'] for a in selected}
        if len(image_ids) != len(selected) or not selected:
            raise ValueError('Expected one nonempty annotation per selected image')
        filtered_images = [im for im in c['images'] if im['id'] in image_ids]
        for im in filtered_images:
            if im['file_name'] in seen:
                raise ValueError('Image overlaps existing splits')
            seen.add(im['file_name'])
        out = copy.deepcopy(c)
        out['images'] = filtered_images
        out['annotations'] = selected
        out['categories'] = [cat for cat in c['categories'] if cat['id'] in category_ids]
        # Existing info/source_sha256 still refers to the original raw split;
        # selection provenance makes clear this output is a subset.
        out.setdefault('info', {})['selection'] = dict(
            rule='len(keypoints)==78 and source_point_ids==[1..26]',
            source_coco=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            source_coco_sha256=sha256(path), split=split, renumbered=False)
        outputs[split] = out
        report[split] = dict(source_sha256=sha256(path), original_images=len(images),
            original_point_counts=dict(counts), selected_images=len(filtered_images),
            selected_annotations=len(selected), excluded_32_point_annotations=counts[32],
            point_ids=list(range(1,27)), anatomical_names_available=False,
            ids_order_coordinates_file_names_preserved=True)
    for split, out in outputs.items():
        write_json(args.output/'annotations'/f'{split}.json', out)
    write_json(args.output/'selection_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
