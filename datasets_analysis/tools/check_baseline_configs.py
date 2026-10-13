"""Validate seven prepared baselines without models, training or inference."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmcv import Config
from mmpose.datasets import build_dataloader, build_dataset
from datasets_analysis.medical_evaluation.metrics import read_json, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=ROOT/'datasets_analysis/docs/baseline_config_validation.json')
    args=p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    manifest=read_json(ROOT/'datasets_analysis/docs/baseline_configs_manifest.json')
    assert sha256(manifest['template'])==manifest['template_sha256'], 'TG3K template changed'
    original=Config.fromfile(manifest['template'])
    results={}
    protected={Path(manifest['template']):sha256(manifest['template'])}
    for item in manifest['datasets']:
        c=Config.fromfile(item['config']);k=item['keypoints'];name=item['dataset']
        assert c.model.backbone==original.model.backbone
        assert c.model.keypoint_head.out_channels==c.data_cfg.num_joints==c.num_keypoints==k
        assert c.model.keypoint_head.loss_keypoint==original.model.keypoint_head.loss_keypoint
        assert c.load_from=='checkpoints/vitpose-b.pth' and c.resume_from is None
        assert c.evaluation==original.evaluation and c.evaluation.save_best=='MRE'
        assert c.total_epochs==210 and c.data.samples_per_gpu==8
        split_results={};seen=set()
        for split in ('train','val','test'):
            dc=copy.deepcopy(c.data[split]);a=Path(dc.ann_file)
            assert a==Path(f'datasets_analysis/coco_format/{name}/annotations/{split}.json')
            protected[a]=sha256(a);gt=read_json(a)
            assert len(gt['categories'])==1
            names=gt['categories'][0]['keypoints']
            assert names==[dc.dataset_info.keypoint_info[j]['name'] for j in range(k)]
            assert dc.dataset_info.dataset_name==name
            assert len(dc.dataset_info.joint_weights)==k
            assert dc.data_cfg.num_joints==dc.data_cfg.num_output_channels==k
            assert dc.data_cfg.inference_channel==list(range(k))
            assert dc.data_cfg.dataset_channel==[list(range(k))]
            assert dc.data_cfg.use_gt_bbox and not dc.data_cfg.bbox_file
            assert not c.model.test_cfg.flip_test
            for im in gt['images']:
                path=Path(dc.img_prefix)/im['file_name']
                assert path.is_file() and str(path) not in seen
                seen.add(str(path))
            assert all(len(ann['keypoints'])==k*3 and ann['source_point_ids']==list(range(1,k+1)) for ann in gt['annotations'])
            if name=='LUNA16_26':
                source=Path(f'datasets_analysis/coco_format/LUNA16/annotations/{split}.json')
                protected[source]=sha256(source)
                old=read_json(source)
                keep=[ann for ann in old['annotations'] if len(ann['keypoints'])==78]
                ids={ann['image_id'] for ann in keep}
                assert gt['annotations']==keep
                assert gt['images']==[im for im in old['images'] if im['id'] in ids]
                assert gt['info']['selection']['source_coco_sha256']==sha256(source)
            dataset=build_dataset(dc)
            assert len(dataset)==item['counts'][split]==len(gt['annotations'])
            loader=build_dataloader(dataset,samples_per_gpu=8,workers_per_gpu=0,
                num_gpus=1,dist=False,shuffle=False,drop_last=False,pin_memory=False)
            batch=next(iter(loader))
            assert tuple(batch['img'].shape[1:])==(3,256,192)
            assert torch.isfinite(batch['img']).all()
            if split=='train':
                assert tuple(batch['target'].shape[1:])==(k,64,48)
                assert tuple(batch['target_weight'].shape[1:])==(k,1)
                assert torch.isfinite(batch['target']).all()
            # Validate medical metric integration using GT coordinates, no model.
            predictions=[]
            for rec in dataset.db:
                pts=np.column_stack((rec['joints_3d'][:,:2],np.ones(k)))
                predictions.append(dict(preds=pts[None],image_paths=[rec['image_file']]))
            metrics=dataset.evaluate(predictions,metric=['MSE','RMSE','MAE','MRE','PCK'])
            assert metrics['MRE']<1e-4 and metrics['PCK@2px']==1
            split_results[split]=dict(images=len(dataset),annotations=len(gt['annotations']),
                keypoints=k,all_image_paths_exist=True,first_batch_shape=list(batch['img'].shape),
                heatmap_shape=list(batch['target'].shape) if split=='train' else None,
                gt_self_check_mre=metrics['MRE'],annotation_sha256=sha256(a))
        results[name]=dict(config=item['config'],splits=split_results,passed=True)
        print('PASS',name,item['counts'],'K=',k,flush=True)
    assert all(sha256(path)==value for path,value in protected.items()), 'Input changed'
    report=dict(datasets=results,protected_input_hashes_unchanged=True,
        scope='All records/path checks + one actual transformed batch per split + GT metric self-check; no model/training/inference')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',args.report)


if __name__=='__main__':main()
