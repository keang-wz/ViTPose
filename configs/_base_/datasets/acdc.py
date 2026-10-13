"""acdc: preserve existing COCO keypoint order; no inferred anatomy/flip pairs."""
num_keypoints = 14
keypoint_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
dataset_info = dict(
    dataset_name='acdc',
    paper_info={},
    keypoint_info={
        i: dict(id=i, name=name, color=[255, 128, 0], type='', swap='')
        for i, name in enumerate(keypoint_names)
    },
    skeleton_info={},
    joint_weights=[1.0] * num_keypoints,
    sigmas=[],  # Medical pixel-coordinate metrics, no OKS.
)
