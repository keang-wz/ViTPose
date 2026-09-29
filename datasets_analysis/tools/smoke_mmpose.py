"""Load fixed-K outputs in this checkout's MMPose without a model or training."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'converters'))
from common import BASE, DATASETS, DEFAULT_ROOT, SPLITS, write_json
from mmpose.datasets import TopDownCocoDataset

results={}
for d in DATASETS:
    results[d]={}
    for s in SPLITS:
        path=BASE/'coco_format'/d/'annotations'/f'{s}.json'
        coco=json.loads(path.read_text())
        if len(coco['categories'])!=1:
            results[d][s]={'status':'blocked','reason':'Multiple keypoint schemas; fixed-K loader is incompatible'}
            continue
        names=coco['categories'][0]['keypoints'];k=len(names)
        info=dict(dataset_name=d,paper_info={},keypoint_info={i:dict(id=i,name=name,type='',swap='') for i,name in enumerate(names)},
                  skeleton_info={},joint_weights=[1.0]*k,sigmas=[])
        cfg=dict(image_size=[192,256],heatmap_size=[48,64],num_joints=k,num_output_channels=k,
                 dataset_channel=[list(range(k))],inference_channel=list(range(k)),use_gt_bbox=True,
                 bbox_file='',soft_nms=False,nms_thr=1.0,oks_thr=0.9,vis_thr=0.2)
        ds=TopDownCocoDataset(str(path),str(DEFAULT_ROOT)+'/',cfg,[],info,test_mode=True)
        assert len(ds)==len(coco['images'])
        assert all(r['joints_3d'].shape==(k,3) and Path(r['image_file']).is_file() for r in ds.db)
        results[d][s]={'status':'passed','loaded_samples':len(ds),'keypoints':k}
write_json(BASE/'docs/mmpose_smoke_results.json',results)
print('Loader only: no model, no training, no evaluation; sigmas deliberately unspecified.')
