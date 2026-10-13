"""Check TG3K config and all train/val batches on CPU; never builds a model.

Run from any directory with this repository's .venv/bin/python.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmcv import Config
from mmpose.datasets import build_dataloader, build_dataset

DEFAULT_CONFIG = ROOT / 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--workers', type=int, default=None,
                        help='Default: configured workers_per_gpu; use 0 for debugging')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config_path = args.config.resolve()
    os.chdir(ROOT)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    cfg = Config.fromfile(str(config_path))
    workers = cfg.data.workers_per_gpu if args.workers is None else args.workers
    require(workers >= 0, 'workers must be nonnegative')
    require(cfg.model.keypoint_head.out_channels == 5, 'Head must output 5 channels')
    require(not cfg.model.test_cfg.flip_test, 'Unknown flip pairs: disable flip_test')
    require(cfg.load_from is None or Path(cfg.load_from).is_file(), 'Missing configured checkpoint')
    require(cfg.channel_cfg.num_output_channels == cfg.channel_cfg.dataset_joints == 5,
            'Invalid global channel config')
    seen = set()
    hashes = {}
    for split in ('train', 'val'):
        dc = copy.deepcopy(cfg.data[split])
        expected_file = ROOT / f'datasets_analysis/coco_format/TG3K/annotations/{split}.json'
        require(Path(dc.ann_file).resolve() == expected_file, f'{split}: incorrect annotation path')
        hashes[expected_file] = digest(expected_file)
        coco = json.loads(expected_file.read_text())
        raw_split = Path(cfg.data_root) / 'TG3K' / 'data' / f'{split}.json'
        hashes[raw_split] = digest(raw_split)
        require(hashes[raw_split] == coco['info']['source_sha256'], 'Raw split changed since conversion')
        require(dc.type == 'MedicalCocoDataset', 'Expected COCO-format dataset reader')
        require(dc.data_cfg.num_joints == dc.data_cfg.num_output_channels == 5, 'Invalid data K')
        require(dc.data_cfg.dataset_channel == [list(range(5))] and
                dc.data_cfg.inference_channel == list(range(5)), 'Point order mismatch')
        require(dc.data_cfg.use_gt_bbox and not dc.data_cfg.bbox_file, 'Expected full-image GT ROI')
        require(dc.dataset_info.dataset_name == 'TG3K', 'Wrong dataset_info')
        names = [dc.dataset_info.keypoint_info[i]['name'] for i in range(5)]
        require(names == coco['categories'][0]['keypoints'], 'Keypoint names/order differ from COCO')
        require(cfg.evaluation.save_best == 'MRE' and cfg.evaluation.rule == 'less',
                'Expected medical distance-based model selection')
        require(not any(p['type'] in ('TopDownRandomFlip', 'TopDownHalfBodyTransform') for p in dc.pipeline),
                'Human-specific augmentation present')
        require(dc.get('test_mode', False) == (split == 'val'), 'Wrong train/val mode')
        dataset = build_dataset(dc)
        require(len(dataset) == len(coco['images']) == len(coco['annotations']), 'Dropped annotations')
        by_id = {im['id']: im for im in coco['images']}
        for record, ann in zip(dataset.db, coco['annotations']):
            im = by_id[ann['image_id']]
            image_path = Path(dc.img_prefix) / im['file_name']
            require(image_path.is_file(), f'Missing image: {image_path}')
            require(Path(record['image_file']) == image_path, 'Image path mismatch')
            require(str(image_path) not in seen, 'Duplicate image within/across train and val')
            seen.add(str(image_path))
            original = np.asarray(ann['keypoints']).reshape(5, 3)
            require(np.allclose(record['joints_3d'][:, :2], original[:, :2]), 'Point coordinate/order mismatch')
            require(np.all(record['joints_3d_visible'][:, :2] == 1), 'Unexpected missing points')
        loader_cfg = cfg.data.get(f'{split}_dataloader', {})
        batch_size = loader_cfg.get('samples_per_gpu', cfg.data.samples_per_gpu)
        loader = build_dataloader(dataset, samples_per_gpu=batch_size,
            workers_per_gpu=workers, num_gpus=1, dist=False, shuffle=(split == 'train'),
            seed=args.seed, drop_last=False, pin_memory=False)
        total = batches = supervised = 0
        for batch in loader:
            images = batch['img']
            b = images.shape[0]
            require(tuple(images.shape[1:]) == (3, 256, 192), 'Wrong image tensor shape')
            require(torch.isfinite(images).all().item(), 'Nonfinite image pixels')
            metadata = [item for group in batch['img_metas'].data for item in group]
            require(len(metadata) == b, 'Metadata collation mismatch')
            require(all(Path(m['image_file']).is_file() for m in metadata), 'Bad batched image path')
            if split == 'train':
                targets, weights = batch['target'], batch['target_weight']
                require(tuple(targets.shape) == (b, 5, 64, 48), 'Wrong heatmap tensor shape')
                require(tuple(weights.shape) == (b, 5, 1), 'Wrong target weight shape')
                require(torch.isfinite(targets).all().item() and torch.isfinite(weights).all().item(),
                        'Nonfinite heatmap/weight')
                require(((weights >= 0) & (weights <= 1)).all().item(), 'Invalid target weights')
                supervised += int((weights > 0).sum())
            total += b
            batches += 1
        require(total == len(dataset), 'Dataloader did not visit all samples')
        if split == 'train':
            require(supervised > 0, 'No supervised targets remain after transforms')
        print(f'PASS {split}: {total} samples, {batches} batches, image=[B,3,256,192]'
              + (' target=[B,5,64,48] weight=[B,5,1]' if split == 'train' else ''), flush=True)
    require(all(digest(p) == h for p, h in hashes.items()), 'Input annotation file changed during check')
    print(f'PASS: workers={workers}; original splits unchanged; no model, optimizer or training created.')
    print('Medical validation metrics enabled; heatmap MSE loss unchanged.')


if __name__ == '__main__':
    main()
