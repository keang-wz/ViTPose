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
    p.add_argument('--pck-unit', choices=['pixels', 'image_diagonal'], default='pixels')
    args = p.parse_args()
    pred = read_json(args.predictions)
    gt_hash = sha256(args.gt)
    expected_hash = pred.get('metadata', {}).get('annotation_sha256')
    if expected_hash is not None and expected_hash != gt_hash:
        raise ValueError('Prediction/GT annotation SHA256 mismatch; check split')
    thresholds = args.pck_thresholds or ([2, 5, 10, 20] if args.pck_unit == 'pixels' else [.01, .02, .05, .1])
    result = evaluate(pred, read_json(args.gt), thresholds, args.pck_unit)
    result['inputs'] = dict(prediction_sha256=sha256(args.predictions), annotation_sha256=gt_hash)
    write_json(args.out, result, protected=[args.predictions, args.gt])
    import json
    print(json.dumps(result['overall'], indent=2))
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
