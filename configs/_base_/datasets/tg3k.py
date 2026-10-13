"""TG3K's original point_ids 1..5, in the COCO converter's unchanged order.

Anatomical names, left/right pairs and skeleton edges are not supplied by the
source. Neutral names are intentional. Do not enable flipping or half-body
augmentation without verified point semantics.
"""

dataset_info = dict(
    dataset_name='TG3K',
    paper_info={},
    keypoint_info={
        i: dict(id=i, name=f'point_{i + 1}', color=[255, 128, 0],
                type='', swap='')
        for i in range(5)
    },
    skeleton_info={},
    joint_weights=[1.0] * 5,
    # Medical evaluation does not use OKS/AP or anatomically uncalibrated sigmas.
    sigmas=[],
)
