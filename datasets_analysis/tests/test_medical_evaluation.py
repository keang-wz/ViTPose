"""Known-answer metric tests, failure modes, and real decoder integration."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets_analysis.medical_evaluation.metrics import evaluate, prediction_from_outputs, write_json


def fixture():
    gt = dict(images=[dict(id=1, width=30, height=40, file_name='a.png')],
              categories=[dict(id=1, keypoints=['p1', 'p2', 'p3'])],
              annotations=[dict(id=9, image_id=1, category_id=1,
                                keypoints=[10, 10, 2, 0, 0, 0, 20, 20, 1])])
    pred = dict(coordinate_space='original_image_pixels', predictions=[dict(
        image_id=1, annotation_id=9, category_id=1,
        keypoints=[[13, 14], [999, 999], [20, 20]], confidence=[-0.1, 0.99, 0.0])])
    return pred, gt


class MetricsTest(unittest.TestCase):
    def test_known_values_visibility_and_no_confidence_filter(self):
        pred, gt = fixture()
        r = evaluate(pred, gt, [0, 2, 5])
        o = r['overall']
        self.assertEqual(o['coordinate_mse'], 6.25)
        self.assertEqual(o['rmse'], 2.5)
        self.assertEqual(o['mae'], 1.75)
        self.assertEqual(o['mean_euclidean_distance_error'], 2.5)
        self.assertEqual(o['pck'], {'0': .5, '2': .5, '5': 1.0})
        self.assertEqual(o['evaluated_keypoints'], 2)
        self.assertEqual(r['ignored_unannotated_keypoints'], 1)
        self.assertIsNone(r['per_keypoint'][1]['coordinate_mse'])

    def test_diagonal_threshold(self):
        pred, gt = fixture()
        r = evaluate(pred, gt, [.09, .1], 'image_diagonal')
        self.assertEqual(r['overall']['pck'], {'0.09': .5, '0.1': 1.0})
        self.assertEqual(r['overall']['coordinate_mse'], 6.25)

    def test_missing_duplicate_unknown_rejected(self):
        for kind in ('missing', 'duplicate', 'unknown'):
            pred, gt = fixture()
            if kind == 'missing': pred['predictions'] = []
            if kind == 'duplicate': pred['predictions'] *= 2
            if kind == 'unknown': pred['predictions'][0]['image_id'] = 5
            with self.subTest(kind=kind), self.assertRaises(ValueError): evaluate(pred, gt)

    def test_bad_prediction_and_order_rejected(self):
        for kind in ('nan', 'shape', 'names', 'space', 'category'):
            pred, gt = fixture()
            p = pred['predictions'][0]
            if kind == 'nan': p['confidence'][0] = float('nan')
            if kind == 'shape': p['keypoints'].pop()
            if kind == 'names': p['keypoint_names'] = ['p3', 'p2', 'p1']
            if kind == 'space': pred['coordinate_space'] = 'heatmap_pixels'
            if kind == 'category': p['category_id'] = 2
            with self.subTest(kind=kind), self.assertRaises(ValueError): evaluate(pred, gt)

    def test_invalid_gt_and_thresholds(self):
        pred, gt = fixture()
        for thresholds in ([], [-1], [float('nan')], [2, 2]):
            with self.assertRaises(ValueError): evaluate(pred, gt, thresholds)
        gt['annotations'][0]['keypoints'][2] = 3
        with self.assertRaises(ValueError): evaluate(pred, gt)

    def test_no_valid_points(self):
        pred, gt = fixture()
        gt['annotations'][0]['keypoints'][2::3] = [0, 0, 0]
        with self.assertRaises(ValueError): evaluate(pred, gt)

    def test_multi_target_requires_id(self):
        pred, gt = fixture()
        second = copy.deepcopy(gt['annotations'][0]); second['id'] = 10
        gt['annotations'].append(second)
        pred['predictions'].append(copy.deepcopy(pred['predictions'][0]))
        pred['predictions'][1]['annotation_id'] = 10
        self.assertEqual(evaluate(pred, gt)['annotations'], 2)
        del pred['predictions'][0]['annotation_id']
        with self.assertRaises(ValueError): evaluate(pred, gt)

    def test_matching_not_list_order(self):
        pred, gt = fixture()
        gt['images'].append(dict(id=2, width=30, height=40, file_name='b.png'))
        a = copy.deepcopy(gt['annotations'][0]);a.update(id=10, image_id=2)
        gt['annotations'].append(a)
        p = copy.deepcopy(pred['predictions'][0]);p.update(image_id=2, annotation_id=10)
        pred['predictions'].insert(0,p)
        self.assertEqual(evaluate(pred, gt)['overall']['coordinate_mse'], 6.25)

    def test_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'raw.json';p.write_text('original')
            with self.assertRaises(FileExistsError): write_json(p, {})
            self.assertEqual(p.read_text(), 'original')


class IntegrationTest(unittest.TestCase):
    def test_real_pipeline_and_decoder_with_synthetic_weights(self):
        # Small ViT solely for executable CPU smoke testing. These random weights
        # are NOT a TG3K baseline, and all predictions/checkpoints are temporary.
        import torch
        from mmcv import Config
        from mmpose.models import build_posenet
        torch.set_num_threads(1)
        torch.manual_seed(42)
        base = ROOT/'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/tg3k/ViTPose_base_tg3k_256x192.py'
        gt_path = ROOT/'datasets_analysis/coco_format/TG3K/annotations/test.json'
        with tempfile.TemporaryDirectory(prefix='medical_eval_smoke_') as tmp:
            tmp = Path(tmp)
            config = tmp/'smoke.py'
            config.write_text(f'_base_ = [{str(base)!r}]\nmodel = dict(backbone=dict(embed_dim=32, depth=1, num_heads=4), keypoint_head=dict(in_channels=32, num_deconv_filters=(16,16)))\n')
            cfg = Config.fromfile(str(config))
            cfg.model.pretrained = None
            model = build_posenet(cfg.model)
            checkpoint = tmp/'synthetic_random.pth'
            torch.save(dict(state_dict=model.state_dict()), checkpoint)
            predictions = tmp/'predictions.json'
            cmd = [sys.executable, str(ROOT/'datasets_analysis/tools/infer_medical_keypoints.py'),
                   str(config), str(checkpoint), '--test-json', str(gt_path), '--out', str(predictions),
                   '--device', 'cpu', '--batch-size', '3']
            run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            pred = json.loads(predictions.read_text())
            gt = json.loads(gt_path.read_text())
            self.assertEqual(len(pred['predictions']), len(gt['images']))
            self.assertEqual({p['image_id'] for p in pred['predictions']}, {im['id'] for im in gt['images']})
            result = tmp/'metrics.json'
            run = subprocess.run([sys.executable, str(ROOT/'datasets_analysis/tools/evaluate_medical_keypoints.py'),
                '--predictions', str(predictions), '--gt', str(gt_path), '--out', str(result)],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(json.loads(result.read_text())['overall']['evaluated_keypoints'], 100)
            # Test the registered evaluation adapter with exact GT coordinates.
            from mmpose.datasets import build_dataset
            dc = copy.deepcopy(cfg.data.test);dc.pipeline = []
            dc.ann_file = str(gt_path)
            dataset = build_dataset(dc)
            outputs = []
            for rec in dataset.db:
                pts = np.column_stack([rec['joints_3d'][:, :2], np.ones(5)])
                outputs.append(dict(preds=pts[None], image_paths=[rec['image_file']]))
            measured = dataset.evaluate(outputs, metric=['MSE', 'RMSE', 'MAE', 'MRE', 'PCK'])
            # The dataset stores original decimal coordinates as float32.
            self.assertLess(measured['MRE'], 1e-4)
            self.assertEqual(measured['PCK@2px'], 1)
            # Shape mismatch must fail instead of emitting random-head predictions.
            invalid = model.state_dict()
            invalid['keypoint_head.final_layer.weight'] = torch.zeros(17,16,1,1)
            torch.save(dict(state_dict=invalid), checkpoint)
            predictions.unlink()
            run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('trained 5-point head', run.stderr)
            self.assertFalse(predictions.exists())
            self.assertEqual(cfg.model.keypoint_head.loss_keypoint.type, 'JointsMSELoss')


if __name__ == '__main__':
    unittest.main()
