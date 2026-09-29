"""Render at most five samples per dataset; random sampling never changes splits."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'converters'))
from common import BASE, DATASETS, DEFAULT_ROOT, SPLITS, safe_output
from PIL import Image, ImageDraw

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--coco-root',type=Path,default=BASE/'coco_format')
    p.add_argument('--output-root',type=Path,default=BASE/'samples')
    p.add_argument('--count',type=int,choices=range(1,6),default=3)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();out=safe_output(args.output_root,args.data_root)
    rng=random.Random(args.seed)
    for d in DATASETS:
        pool=[]
        for split in SPLITS:
            c=json.loads((args.coco_root/d/'annotations'/f'{split}.json').read_text())
            anns={a['image_id']:a for a in c['annotations']}
            pool.extend((split,im,anns[im['id']]) for im in c['images'])
        folder=out/d;folder.mkdir(parents=True,exist_ok=True)
        # Fixed filenames ensure repeated runs do not accumulate sample images.
        for old in folder.glob('sample_*.png'): old.unlink()
        for i,(split,im,ann) in enumerate(rng.sample(pool,min(args.count,len(pool)))):
            with Image.open(args.data_root/im['file_name']) as src: canvas=src.convert('RGB')
            draw=ImageDraw.Draw(canvas)
            for pid,(x,y,v) in zip(ann['source_point_ids'],zip(ann['keypoints'][::3],ann['keypoints'][1::3],ann['keypoints'][2::3])):
                if v:
                    draw.ellipse((x-3,y-3,x+3,y+3),fill='red');draw.text((x+4,y-5),str(pid),fill='yellow')
            draw.text((5,5),f'{d} {split} id={im["id"]} K={ann["num_keypoints"]}',fill='red')
            canvas.save(folder/f'sample_{i+1}.png')
if __name__=='__main__':main()
