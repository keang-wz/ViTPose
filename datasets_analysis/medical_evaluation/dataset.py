"""Evaluation-only extension; inherited loading, augmentation and loss unchanged."""
from collections import OrderedDict
import json
from mmpose.datasets import DATASETS, TopDownCocoDataset
from .metrics import evaluate, prediction_from_outputs


@DATASETS.register_module()
class MedicalCocoDataset(TopDownCocoDataset):
    def evaluate(self, results, res_folder=None, metric='MRE',
                 pck_thresholds=(2, 5, 10, 20), pck_unit='pixels',
                 pixel_spacing_mm=None, **kwargs):
        # res_folder deliberately unused: the runner records scalar metrics;
        # standalone inference/evaluation scripts persist detailed JSON.
        requested = [metric] if isinstance(metric, str) else list(metric)
        allowed = {'MSE', 'RMSE', 'MAE', 'MRE', 'PCK'}
        if not set(requested).issubset(allowed):
            raise ValueError(f'Medical metrics only: {sorted(allowed)}')
        with open(self.ann_file, encoding='utf-8') as stream:
            gt = json.load(stream)
        pred = prediction_from_outputs(results, gt, self.img_prefix)
        thresholds = pck_thresholds if 'PCK' in requested else None
        report = evaluate(pred, gt, thresholds, pck_unit, pixel_spacing_mm)['overall']
        mapping = dict(MSE='coordinate_mse', RMSE='rmse', MAE='mae',
                       MRE='mean_euclidean_distance_error')
        def label(name):
            if pixel_spacing_mm is None:
                return name
            return f'{name}_mm2' if name == 'MSE' else f'{name}_mm'
        out = OrderedDict((label(name), report[key]) for name, key in mapping.items() if name in requested)
        if 'PCK' in requested:
            suffix = {'pixels': 'px', 'image_diagonal': 'diag', 'mm': 'mm'}[pck_unit]
            out.update((f'PCK@{t}{suffix}', value) for t, value in report['pck'].items())
        return out
