"""ChestX-ray: preserve existing COCO keypoint order; no inferred anatomy/flip pairs."""
num_keypoints = 32
keypoint_names = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10', 'point_11', 'point_12', 'point_13', 'point_14', 'point_15', 'point_16', 'point_17', 'point_18', 'point_19', 'point_20', 'point_21', 'point_22', 'point_23', 'point_24', 'point_25', 'point_26', 'point_27', 'point_28', 'point_29', 'point_30', 'point_31', 'point_32']
dataset_info = dict(
    dataset_name='ChestX-ray',
    paper_info={},
    keypoint_info={
        i: dict(id=i, name=name, color=[255, 128, 0], type='', swap='')
        for i, name in enumerate(keypoint_names)
    },
    skeleton_info={},
    joint_weights=[1.0] * num_keypoints,
    sigmas=[],  # Medical pixel-coordinate metrics, no OKS.
)
