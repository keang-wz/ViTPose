"""Original TG3K baseline, using Painter-3D images and its supplied JSON splits."""
_base_ = ['../tg3k/ViTPose_base_tg3k_256x192.py']

data_root = '/data/users/mqf/Painter-3D/data'
ann_root = 'datasets_analysis/coco_format/painter3d/TG3K/annotations'
data = dict(
    train=dict(ann_file=f'{ann_root}/train.json', img_prefix=f'{data_root}/'),
    val=dict(ann_file=f'{ann_root}/val.json', img_prefix=f'{data_root}/'),
    test=dict(ann_file=f'{ann_root}/test.json', img_prefix=f'{data_root}/'),
)
work_dir = 'work_dirs/painter3d_seed42/tg3k'

# Unknown physical spacing: keep pixel-distance errors and disable PCK.
evaluation = dict(
    _delete_=True, interval=10, metric=['MRE', 'MSE', 'RMSE', 'MAE'],
    save_best='MRE', rule='less')
