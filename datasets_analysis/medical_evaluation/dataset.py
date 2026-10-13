"""Evaluation-only extension; inherited loading, augmentation and loss unchanged."""
from collections import OrderedDict
import json
from mmpose.datasets import DATASETS, TopDownCocoDataset
from .metrics import evaluate, prediction_from_outputs


@DATASETS.register_module()
class MedicalCocoDataset(TopDownCocoDataset):
    def evaluate(self, results, res_folder=None, metric='MRE',
                 pck_thresholds=(2, 5, 10, 20), pck_unit='pixels', **kwargs):
        # res_folder deliberately unused: the runner records scalar metrics;
        # standalone inference/evaluation scripts persist detailed JSON.
        requested = [metric] if isinstance(metric, str) else list(metric)
        allowed = {'MSE', 'RMSE', 'MAE', 'MRE', 'PCK'}
        if not set(requested).issubset(allowed):
            raise ValueError(f'Medical metrics only: {sorted(allowed)}')
        with open(self.ann_file, encoding='utf-8') as stream:
            gt = json.load(stream)
        pred = prediction_from_outputs(results, gt, self.img_prefix)
        report = evaluate(pred, gt, pck_thresholds, pck_unit)['overall']
        mapping = dict(MSE='coordinate_mse', RMSE='rmse', MAE='mae',
                       MRE='mean_euclidean_distance_error')
        out = OrderedDict((name, report[key]) for name, key in mapping.items() if name in requested)
        if 'PCK' in requested:
            suffix = 'px' if pck_unit == 'pixels' else 'diag'
            out.update((f'PCK@{t}{suffix}', value) for t, value in report['pck'].items())
        return out
