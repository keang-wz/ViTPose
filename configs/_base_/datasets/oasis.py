"""oasis: preserve existing COCO keypoint order; no inferred anatomy/flip pairs."""
num_keypoints = 8
keypoint_names = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8']
dataset_info = dict(
    dataset_name='oasis',
    paper_info={},
    keypoint_info={
        i: dict(id=i, name=name, color=[255, 128, 0], type='', swap='')
        for i, name in enumerate(keypoint_names)
    },
    skeleton_info={},
    joint_weights=[1.0] * num_keypoints,
    sigmas=[],  # Medical pixel-coordinate metrics, no OKS.
)
