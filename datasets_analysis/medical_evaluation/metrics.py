"""Strict COCO-GT matching with pixel or calibrated physical-distance metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data, protected=()):
    path = Path(path).resolve()
    if path in {Path(p).resolve() for p in protected}:
        raise ValueError('Output must not overwrite an input file')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    # Exclusive creation protects existing raw files and previous results.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def index_gt(gt):
    images = {x['id']: x for x in gt['images']}
    cats = {x['id']: x for x in gt['categories']}
    anns = {x['id']: x for x in gt['annotations']}
    if (len(images) != len(gt['images']) or len(cats) != len(gt['categories'])
            or len(anns) != len(gt['annotations'])):
        raise ValueError('Duplicate GT image/category/annotation IDs')
    if not anns:
        raise ValueError('No GT annotations')
    for im in images.values():
        if not (np.isfinite([im['width'], im['height']]).all()
                and im['width'] > 0 and im['height'] > 0):
            raise ValueError('Invalid image dimensions')
    for a in anns.values():
        if a['image_id'] not in images or a['category_id'] not in cats:
            raise ValueError('Invalid GT foreign key')
        k = len(cats[a['category_id']]['keypoints'])
        points = np.asarray(a['keypoints'], dtype=np.float64)
        if k == 0 or points.shape != (k * 3,) or not np.isfinite(points).all():
            raise ValueError(f'Invalid GT keypoints: annotation {a["id"]}')
        if not np.isin(points[2::3], [0, 1, 2]).all():
            raise ValueError('GT visibility must be 0, 1 or 2')
        if a.get('iscrowd', 0):
            raise ValueError('Crowd annotations are unsupported; no silent exclusions')
    return images, cats, anns


def summarize(delta, scales, thresholds):
    if len(delta) == 0:
        result = dict(evaluated_keypoints=0, coordinate_mse=None, rmse=None,
                      mae=None, mean_euclidean_distance_error=None)
        if thresholds is not None:
            result['pck'] = {f'{t:g}': None for t in thresholds}
        return result
    delta = np.asarray(delta, dtype=np.float64)
    distance = np.linalg.norm(delta, axis=1)
    mse = float(np.mean(delta ** 2))
    result = dict(evaluated_keypoints=len(delta), coordinate_mse=mse,
                rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(delta))),
                mean_euclidean_distance_error=float(distance.mean()))
    if thresholds is not None:
        result['pck'] = {f'{t:g}': float(np.mean(distance / scales <= t)) for t in thresholds}
    return result


def evaluate(prediction, gt, thresholds=(2, 5, 10, 20), pck_unit='pixels',
             pixel_spacing_mm=None):
    # None explicitly disables PCK; preserve legacy defaults for existing runs.
    if thresholds is not None:
        thresholds = [float(t) for t in thresholds]
        if (not thresholds or not np.isfinite(thresholds).all()
                or any(t < 0 for t in thresholds) or len(set(thresholds)) != len(thresholds)):
            raise ValueError('PCK thresholds must be unique finite nonnegative numbers')
    if pck_unit not in ('pixels', 'image_diagonal', 'mm'):
        raise ValueError('Unknown PCK unit')
    spacing = 1.0
    if pixel_spacing_mm is not None:
        if (isinstance(pixel_spacing_mm, (bool, str))
                or np.ndim(pixel_spacing_mm) != 0):
            raise ValueError('pixel_spacing_mm must be a positive finite scalar')
        spacing = float(pixel_spacing_mm)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('pixel_spacing_mm must be a positive finite scalar')
        if thresholds is not None and pck_unit != 'mm':
            raise ValueError('Physical-distance evaluation requires PCK thresholds in mm')
    elif thresholds is not None and pck_unit == 'mm':
        raise ValueError('PCK in mm requires pixel_spacing_mm for original-image pixels')
    if prediction.get('coordinate_space') != 'original_image_pixels':
        raise ValueError('Prediction must declare original_image_pixels coordinate_space')
    images, cats, anns = index_gt(gt)
    per_image = {}
    for a in anns.values():
        per_image.setdefault(a['image_id'], []).append(a['id'])
    matched = {}
    for pred in prediction['predictions']:
        iid = pred['image_id']
        if iid not in per_image:
            raise ValueError(f'Prediction for unknown/unannotated image {iid}')
        aid = pred.get('annotation_id')
        if aid is None:
            if len(per_image[iid]) != 1:
                raise ValueError('Multiple targets require annotation_id; no proximity matching')
            aid = per_image[iid][0]
        if aid not in anns or anns[aid]['image_id'] != iid:
            raise ValueError('Prediction annotation_id/image_id mismatch')
        if aid in matched:
            raise ValueError(f'Duplicate prediction for annotation {aid}')
        matched[aid] = pred
    if set(matched) != set(anns):
        missing = sorted(set(anns) - set(matched))
        raise ValueError(f'Missing predictions for {len(missing)} annotations: {missing[:10]}')
    deltas, scales, grouped = [], [], {}
    ignored = 0
    for aid, a in anns.items():
        pred = matched[aid]
        cid = a['category_id']
        names = cats[cid]['keypoints']
        if pred.get('category_id', cid) != cid:
            raise ValueError('Prediction category mismatch')
        if pred.get('keypoint_names', names) != names:
            raise ValueError('Prediction keypoint order/names mismatch')
        p = np.asarray(pred['keypoints'], dtype=np.float64)
        conf = np.asarray(pred['confidence'], dtype=np.float64)
        if p.shape != (len(names), 2) or conf.shape != (len(names),):
            raise ValueError('Wrong prediction point/confidence shape')
        if not np.isfinite(p).all() or not np.isfinite(conf).all():
            raise ValueError('Nonfinite prediction; failed detections may not be silently dropped')
        g = np.asarray(a['keypoints'], dtype=np.float64).reshape(-1, 3)
        im = images[a['image_id']]
        scale = np.hypot(im['width'], im['height']) if pck_unit == 'image_diagonal' else 1.0
        for j, name in enumerate(names):
            group = grouped.setdefault((cid, j), dict(name=name, delta=[], scales=[]))
            if g[j, 2] == 0:
                ignored += 1
                continue
            # Predictions have already been decoded back to the original PNG.
            # Never apply source-image spacing to resized inputs or heatmaps.
            delta = (p[j] - g[j, :2]) * spacing
            deltas.append(delta)
            scales.append(scale)
            group['delta'].append(delta)
            group['scales'].append(scale)
    if not deltas:
        raise ValueError('No annotated GT keypoints (visibility > 0)')
    result = dict(schema_version=1, coordinate_space='original_image_pixels',
        units=dict(coordinate_mse='pixel^2', rmse='pixel', mae='pixel',
                   mean_euclidean_distance_error='pixel', pck=pck_unit),
        pck_thresholds=thresholds, pck_boundary='distance <= threshold',
        aggregation='micro average over GT points with visibility > 0; confidence never filters points',
        images=len(images), annotated_images=len(per_image), annotations=len(anns),
        ignored_unannotated_keypoints=ignored,
        overall=summarize(deltas, np.asarray(scales), thresholds), per_keypoint=[])
    if pixel_spacing_mm is not None:
        result.update(schema_version=2, pixel_spacing_mm=spacing,
                      spacing_coordinate_space='original_image_pixels',
                      spacing_assumption='dataset-wide isotropic spacing supplied by user',
                      pck_threshold_unit='mm')
        result['units'] = dict(coordinate_mse='mm^2', rmse='mm', mae='mm',
                               mean_euclidean_distance_error='mm', pck='fraction')
    if thresholds is None:
        for key in ('pck_thresholds', 'pck_boundary', 'pck_threshold_unit'):
            result.pop(key, None)
        result['units'].pop('pck', None)
    for (cid, j), group in sorted(grouped.items()):
        result['per_keypoint'].append(dict(category_id=cid, index=j, name=group['name'],
            **summarize(group['delta'], np.asarray(group['scales']), thresholds)))
    return result


def prediction_from_outputs(outputs, gt, img_prefix):
    """Adapt MMPose decoded batches for the current one-target-per-image data."""
    images, cats, anns = index_gt(gt)
    by_image = {}
    for ann in anns.values():
        if ann['image_id'] in by_image:
            raise ValueError('Inference adapter currently requires one target per image')
        by_image[ann['image_id']] = ann
    by_path = {}
    for iid, im in images.items():
        path = str((Path(img_prefix) / im['file_name']).resolve())
        if path in by_path:
            raise ValueError('Duplicate image file_name')
        by_path[path] = iid
    records = []
    for output in outputs:
        if len(output['preds']) != len(output['image_paths']):
            raise ValueError('Model output batch length mismatch')
        for points, path in zip(output['preds'], output['image_paths']):
            iid = by_path[str(Path(path).resolve())]
            a = by_image[iid]
            names = cats[a['category_id']]['keypoints']
            points = np.asarray(points)
            if points.shape != (len(names), 3):
                raise ValueError('Decoded model point count differs from GT schema')
            records.append(dict(image_id=iid, annotation_id=a['id'], category_id=a['category_id'],
                keypoint_names=names, keypoints=points[:, :2].tolist(), confidence=points[:, 2].tolist()))
    return dict(schema_version=1, coordinate_space='original_image_pixels',
                confidence_type='decoded heatmap peak score; not a calibrated probability', predictions=records)
