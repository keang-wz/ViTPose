# Standalone hc18 adaptation of coco/ViTPose_base_coco_256x192.py.
# No source data or split changes. Medical pixel-coordinate metrics are used
# for validation; heatmap JointsMSELoss and model training are unchanged.
_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/hc18.py'
]
custom_imports = dict(
    imports=['datasets_analysis.medical_evaluation.dataset'],
    allow_failed_imports=False)
evaluation = dict(
    interval=10, metric=['MSE', 'RMSE', 'MAE', 'MRE', 'PCK'],
    pck_thresholds=[2, 5, 10, 20], pck_unit='pixels',
    save_best='MRE', rule='less')

optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=12, 
                                    layer_decay_rate=0.75,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                    )
                )

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
target_type = 'GaussianHeatmap'
num_keypoints = 13
channel_cfg = dict(
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[list(range(13))],
    inference_channel=list(range(13)),
)

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    use_nms=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

# Run repository tools from the repository root. file_name in COCO JSON is
# relative to this common raw root, already including hc18/data/.
data_root = '/data/users/mqf/bushu/Painter-ct/data'
ann_root = 'datasets_analysis/coco_format/hc18/annotations'
# COCO JSON reader supports custom K through data_cfg and dataset_info.
dataset_type = 'MedicalCocoDataset'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    train=dict(
        type=dataset_type,
        ann_file=f'{ann_root}/train.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        test_mode=True,
        type=dataset_type,
        ann_file=f'{ann_root}/val.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    # The standalone inference script also requires --test-json explicitly.
    test=dict(
        test_mode=True,
        type=dataset_type,
        ann_file=f'{ann_root}/test.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)


# Prepared for later fine-tuning only; loading a config starts no training.
# MMCV non-strict loading retains compatible weights and skips the old 17-point
# final layer (weight/bias shapes differ), leaving the new 13-point layer random.
load_from = 'checkpoints/vitpose-b.pth'
resume_from = None
work_dir = 'work_dirs/vitpose_base_hc18_256x192'
