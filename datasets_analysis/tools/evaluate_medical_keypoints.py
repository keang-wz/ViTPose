"""Evaluate original-image coordinates against COCO keypoint GT; no COCO AP."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets_analysis.medical_evaluation.metrics import evaluate, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pck-thresholds', type=float, nargs='+')
    p.add_argument('--config', type=Path, help='Use the training config evaluation units and thresholds')
    p.add_argument('--pixel-spacing-mm', type=float,
                   help='Isotropic mm per original-image pixel')
    p.add_argument('--metrics', nargs='+', choices=['MRE', 'MSE', 'RMSE', 'MAE', 'PCK'],
                   help='Defaults to prediction/config metrics, or legacy metrics including PCK')
    p.add_argument('--pck-unit', choices=['pixels', 'image_diagonal', 'mm'])
    args = p.parse_args()
    pred = read_json(args.predictions)
    gt_hash = sha256(args.gt)
    expected_hash = pred.get('metadata', {}).get('annotation_sha256')
    if expected_hash is not None and expected_hash != gt_hash:
        raise ValueError('Prediction/GT annotation SHA256 mismatch; check split')
    settings = pred.get('metadata', {}).get('evaluation', {})
    if args.config:
        from mmcv import Config
        settings = Config.fromfile(str(args.config)).evaluation
    spacing = args.pixel_spacing_mm if args.pixel_spacing_mm is not None else settings.get('pixel_spacing_mm')
    unit = args.pck_unit or ('mm' if args.pixel_spacing_mm is not None else
                             settings.get('pck_unit', 'pixels'))
    configured_thresholds = (settings.get('pck_thresholds')
                             if unit == settings.get('pck_unit', 'pixels') else None)
    thresholds = args.pck_thresholds or configured_thresholds or (
        [.01, .02, .05, .1] if unit == 'image_diagonal' else [2, 5, 10, 20])
    requested = args.metrics or settings.get('metric', ['MRE', 'MSE', 'RMSE', 'MAE', 'PCK'])
    if isinstance(requested, str):
        requested = [requested]
    if 'PCK' not in requested:
        thresholds = None
    result = evaluate(pred, read_json(args.gt), thresholds, unit, spacing)
    metric_fields = dict(MRE='mean_euclidean_distance_error', MSE='coordinate_mse',
                         RMSE='rmse', MAE='mae', PCK='pck')
    for metric, field in metric_fields.items():
        if metric not in requested:
            result['units'].pop(field, None)
            for summary in [result['overall'], *result['per_keypoint']]:
                summary.pop(field, None)
    result['metrics'] = requested
    result['inputs'] = dict(prediction_sha256=sha256(args.predictions), annotation_sha256=gt_hash)
    if args.config:
        result['inputs'].update(config=str(args.config), config_sha256=sha256(args.config))
    write_json(args.out, result, protected=[args.predictions, args.gt] + ([args.config] if args.config else []))
    import json
    print(json.dumps(result['overall'], indent=2))
    print(json.dumps(result['units'], indent=2))
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
