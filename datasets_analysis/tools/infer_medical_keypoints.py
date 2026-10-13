"""Strict-checkpoint inference using MMPose's original test pipeline/decoder."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from mmcv import Config
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from datasets_analysis.medical_evaluation.metrics import (
    evaluate, index_gt, prediction_from_outputs, read_json, sha256, write_json)


def load_weights(path):
    # Only local model checkpoints; no downloads or fallback to random weights.
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    return {k.removeprefix('module.'): v for k, v in state.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--test-json', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data-root', type=Path, help='Directory joined with COCO file_name')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        p.error('batch-size must be positive; workers must be nonnegative')
    if args.out.exists():
        p.error('Output already exists; choose a new filename')
    if not args.checkpoint.is_file():
        p.error('A local trained checkpoint is required')
    cfg = Config.fromfile(str(args.config))
    dc = copy.deepcopy(cfg.data.test)
    dc.ann_file = str(args.test_json.resolve())  # Mandatory override, never implicit val fallback.
    dc.test_mode = True
    dc.data_cfg.use_gt_bbox = True
    dc.data_cfg.bbox_file = ''
    root = (args.data_root or Path(dc.img_prefix)).resolve()
    dc.img_prefix = str(root) + '/'
    gt = read_json(args.test_json)
    images, cats, anns = index_gt(gt)
    k = cfg.model.keypoint_head.out_channels
    if any(len(c['keypoints']) != k for c in cats.values()):
        raise ValueError('GT keypoint counts differ from model head (mixed schemas unsupported)')
    if len({a['image_id'] for a in anns.values()}) != len(anns) or len(anns) != len(images):
        raise ValueError('This inference adapter requires exactly one annotation per image')
    names = [dc.dataset_info.keypoint_info[i]['name'] for i in range(k)]
    if any(c['keypoints'] != names for c in cats.values()):
        raise ValueError('Config/GT keypoint order or names differ')
    if dc.data_cfg.num_joints != k:
        raise ValueError('data_cfg.num_joints and model head differ')
    for im in images.values():
        relative = Path(im['file_name'])
        path = (root / relative).resolve()
        if relative.is_absolute() or root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid/missing relative image path: {relative}')
    # Fail before constructing a large model if e.g. a 17-point COCO head is used.
    state = load_weights(args.checkpoint)
    head = state.get('keypoint_head.final_layer.weight')
    if head is None or head.shape[0] != k:
        raise ValueError(f'Checkpoint must have a trained {k}-point head; COCO 17-point weights are not test weights')
    cfg.model.pretrained = None
    torch.set_num_threads(1)
    model = build_posenet(cfg.model)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    dataset = build_dataset(dc)
    if len(dataset) != len(anns):
        raise ValueError('Dataset loader skipped annotations; evaluation cannot silently omit records')
    loader = build_dataloader(dataset, samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers, num_gpus=1, dist=False,
        shuffle=False, drop_last=False, pin_memory=False)
    outputs = []
    with torch.inference_mode():
        for i, batch in enumerate(loader, 1):
            metas = [m for group in batch['img_metas'].data for m in group]
            result = model(img=batch['img'].to(args.device), img_metas=metas,
                           return_loss=False, return_heatmap=False)
            # Do not retain heatmaps or GPU tensors.
            outputs.append(dict(preds=np.asarray(result['preds']), image_paths=result['image_paths']))
            print(f'Inference batch {i}/{len(loader)}', flush=True)
    prediction = prediction_from_outputs(outputs, gt, dc.img_prefix)
    # Validate coverage/finite values before emitting any usable prediction file.
    evaluate(prediction, gt)
    prediction['metadata'] = dict(config=str(args.config), checkpoint=str(args.checkpoint),
        config_sha256=sha256(args.config), checkpoint_sha256=sha256(args.checkpoint),
        annotation_sha256=sha256(args.test_json), roi='GT bbox (full-image ROI for converted medical datasets)',
        device=args.device, images=len(images), model_head_channels=k)
    write_json(args.out, prediction, protected=[args.config, args.checkpoint, args.test_json])
    print(f'Saved {len(prediction["predictions"])} predictions: {args.out}')


if __name__ == '__main__':
    main()
