"""Original hc18 baseline, using Painter-3D images and its supplied JSON splits."""
_base_ = ['../hc18/ViTPose_base_hc18_256x192.py']

data_root = '/data/users/mqf/Painter-3D/data'
ann_root = 'datasets_analysis/coco_format/painter3d/hc18/annotations'
data = dict(
    train=dict(ann_file=f'{ann_root}/train.json', img_prefix=f'{data_root}/'),
    val=dict(ann_file=f'{ann_root}/val.json', img_prefix=f'{data_root}/'),
    test=dict(ann_file=f'{ann_root}/test.json', img_prefix=f'{data_root}/'),
)
work_dir = 'work_dirs/painter3d_seed42/hc18'

# User-supplied isotropic spacing for the original PNG, not network input pixels.
evaluation = dict(
    _delete_=True, interval=10, metric=['MRE', 'MSE', 'RMSE', 'MAE'],
    pixel_spacing_mm=0.108, save_best='MRE_mm', rule='less')
