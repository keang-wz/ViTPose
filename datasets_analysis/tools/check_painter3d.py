"""Check conversion fidelity and actual MMPose pipelines for Painter-3D."""
import copy
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
from datasets_analysis.converters.common import load_splits, resolve_image, write_json
from datasets_analysis.medical_evaluation.metrics import sha256
from datasets_analysis.tools.prepare_painter3d import DATASETS, DATA_ROOT, OUTPUT_ROOT, REPORT_ROOT

CONFIG_ROOT = ROOT / 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap'
SLUGS = {'acdc': 'acdc', 'hc18': 'hc18', 'TG3K': 'tg3k', 'ChestX-ray': 'chestxray'}
SPACING_MM = {'acdc': 1.5625, 'hc18': 0.108, 'TG3K': None, 'ChestX-ray': 1.4}


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    results = {}
    for name, k in DATASETS.items():
        slug = SLUGS[name]
        filename = f'ViTPose_base_{slug}_256x192.py'
        cfg = Config.fromfile(str(CONFIG_ROOT / 'painter3d' / filename))
        old = Config.fromfile(str(CONFIG_ROOT / slug / filename))
        for key in ('model', 'optimizer', 'optimizer_config', 'lr_config',
                    'total_epochs', 'load_from', 'resume_from',
                    'train_pipeline', 'val_pipeline', 'test_pipeline', 'data_cfg'):
            assert cfg[key] == old[key], f'Baseline changed: {name}/{key}'
        assert cfg.total_epochs == 210 and cfg.data.samples_per_gpu == 8
        assert cfg.model.keypoint_head.out_channels == cfg.data_cfg.num_joints == k
        assert Path(cfg.load_from).is_file()
        assert Path(cfg.work_dir).parts[:2] == ('work_dirs', 'painter3d_seed42')
        spacing = SPACING_MM[name]
        metric_key = 'MRE_mm' if spacing is not None else 'MRE'
        assert cfg.evaluation.metric == ['MRE', 'MSE', 'RMSE', 'MAE']
        assert cfg.evaluation.get('pixel_spacing_mm') == spacing
        assert cfg.evaluation.save_best == metric_key
        assert cfg.evaluation.interval == old.evaluation.interval == 10
        assert cfg.evaluation.rule == old.evaluation.rule == 'less'
        assert 'pck_thresholds' not in cfg.evaluation and 'pck_unit' not in cfg.evaluation
        source = load_splits(DATA_ROOT, name)
        split_results = {}
        for split, rows in source.items():
            dc = copy.deepcopy(cfg.data[split])
            assert Path(dc.img_prefix) == DATA_ROOT
            assert Path(dc.ann_file).resolve() == OUTPUT_ROOT / name / 'annotations' / f'{split}.json'
            coco = json.loads(Path(dc.ann_file).read_text())
            assert coco['info']['source_sha256'] == sha256(DATA_ROOT / name / 'data' / f'{split}.json')
            assert len(rows) == len(coco['images']) == len(coco['annotations'])
            assert len(coco['categories']) == 1
            assert coco['categories'][0]['keypoints'] == [
                dc.dataset_info.keypoint_info[i]['name'] for i in range(k)]
            for index, (row, im, ann) in enumerate(zip(rows, coco['images'], coco['annotations'])):
                path, relative = resolve_image(DATA_ROOT, name, row['image_path'])
                assert im['file_name'] == relative and path.is_file()
                assert ann['source_row_index'] == index
                assert ann['image_id'] == im['id']
                assert ann['source_point_ids'] == row['point_ids']
                assert ann['keypoints'] == [v for p in row['points'] for v in (p['x'], p['y'], 2)]
                assert ann['bbox'] == [0, 0, im['width'], im['height']]
            dataset = build_dataset(dc)
            assert len(dataset) == len(rows), 'Loader omitted records'
            loader = build_dataloader(dataset, samples_per_gpu=8, workers_per_gpu=0,
                                      num_gpus=1, dist=False, shuffle=False,
                                      drop_last=False, pin_memory=False)
            batch = next(iter(loader))
            assert tuple(batch['img'].shape[1:]) == (3, 256, 192)
            assert torch.isfinite(batch['img']).all()
            if split == 'train':
                assert tuple(batch['target'].shape[1:]) == (k, 64, 48)
                assert tuple(batch['target_weight'].shape[1:]) == (k, 1)
                assert torch.isfinite(batch['target']).all()
            predictions = [
                dict(preds=np.column_stack((r['joints_3d'][:, :2], np.ones(k)))[None],
                     image_paths=[r['image_file']]) for r in dataset.db]
            eval_args = dict(metric=cfg.evaluation.metric, pixel_spacing_mm=spacing)
            metrics = dataset.evaluate(predictions, **eval_args)
            keys = ({'MRE_mm', 'MSE_mm2', 'RMSE_mm', 'MAE_mm'} if spacing is not None
                    else {'MRE', 'MSE', 'RMSE', 'MAE'})
            assert set(metrics) == keys and metrics[metric_key] < 1e-4
            # Known nonzero error exercises both conversion and the adapter's
            # log/save_best keys without building a model or running training.
            shifted = copy.deepcopy(predictions)
            for prediction in shifted:
                prediction['preds'][..., :2] += [3, 4]
            measured = dataset.evaluate(shifted, **eval_args)
            scale = spacing if spacing is not None else 1
            expected = dict(MRE=5*scale, MSE=12.5*scale**2,
                            RMSE=np.sqrt(12.5)*scale, MAE=3.5*scale)
            for metric, value in expected.items():
                label = metric if spacing is None else metric + ('_mm2' if metric == 'MSE' else '_mm')
                assert np.isclose(measured[label], value, rtol=1e-5, atol=1e-4), (label, measured[label], value)
            split_results[split] = dict(images=len(dataset),
                annotation_file=dc.ann_file, image_root=dc.img_prefix,
                batch_shape=list(batch['img'].shape),
                target_shape=list(batch['target'].shape) if split == 'train' else None,
                annotation_sha256=sha256(dc.ann_file), gt_self_check=dict(metrics),
                synthetic_3_4_pixel_offset_metrics=dict(measured))
        results[name] = dict(passed=True, training_settings_unchanged=True,
                             config=str(CONFIG_ROOT / 'painter3d' / filename),
                             work_dir=cfg.work_dir, evaluation=dict(cfg.evaluation),
                             metric_units='mm (MSE: mm^2)' if spacing is not None else 'px (MSE: px^2)',
                             splits=split_results)
        print(f'PASS {name}', flush=True)
    write_json(REPORT_ROOT / 'validation.json', results)


if __name__ == '__main__':
    main()
