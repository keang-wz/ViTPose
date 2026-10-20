"""Check original-pixel overlays, ID matching, ranking, and input protection."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
from datasets_analysis.medical_evaluation.metrics import sha256
from datasets_analysis.tools.visualize_medical_keypoints import (
    draw_overlay, prepare_records, select_images)


def fixture():
    gt = dict(images=[dict(id=i, width=240, height=320, file_name=f'{i}.png') for i in [3, 1, 2]],
        categories=[dict(id=1, name='fixture', supercategory='fixture', keypoints=['a', 'b'])],
        annotations=[dict(id=i, image_id=i, category_id=1, source_point_ids=[1, 2],
                         keypoints=[40, 140, 2, 120, 200, 2]) for i in [3, 1, 2]])
    # Image 1: distances 5,0 => MRE2.5. Image2: 3,3 => MRE3.
    # MRE ranking chooses image2; maximum-error ranking chooses image1.
    coords = {1: [[43, 144], [120, 200]], 2: [[43, 140], [123, 200]],
              3: [[40, 140], [120, 200]]}
    pred = dict(coordinate_space='original_image_pixels', predictions=[dict(
        image_id=i, annotation_id=i, category_id=1, keypoint_names=['a', 'b'],
        keypoints=coords[i], confidence=[.8, .9]) for i in [2, 3, 1]])
    return pred, gt


class VisualizationTest(unittest.TestCase):
    def test_known_error_and_id_matching(self):
        pred, gt = fixture()
        records, _ = prepare_records(pred, gt)
        self.assertEqual(records[1]['points'][0]['error_px'], 5)
        self.assertEqual(records[1]['points'][0]['label'], 'point_1')
        self.assertEqual(records[1]['mre_px'], 2.5)
        self.assertEqual(records[1]['max_error_px'], 5)

    def test_ranking_and_selection(self):
        pred, gt = fixture();records, _ = prepare_records(pred, gt)
        self.assertEqual(select_images(records, 1, largest_errors=True)[1], [2])
        self.assertEqual(select_images(records, 1, largest_errors=True, error_ranking='max')[1], [1])
        self.assertEqual(select_images(records, 1, all_images=True), ('all', [1, 2, 3]))
        self.assertEqual(select_images(records, 2, smallest_errors=True, error_ranking='max'),
                         ('smallest_errors', [3, 2]))
        self.assertEqual(select_images(records, 10)[1].__len__(), 3)
        self.assertEqual(select_images(records, 2, seed=42), select_images(records, 2, seed=42))
        self.assertEqual(select_images(records, 1, image_ids=[3, 1]), ('selected', [3, 1]))
        for ids in ([99], [1, 1]):
            with self.assertRaises(ValueError): select_images(records, 1, image_ids=ids)

    def test_visibility_and_multi_target(self):
        pred, gt = fixture()
        gt['annotations'][1]['keypoints'][5] = 0
        record = prepare_records(pred, gt)[0][1]
        self.assertEqual(record['mre_px'], 5)
        self.assertEqual(len(record['points']), 1)
        second = copy.deepcopy(gt['annotations'][1]);second['id'] = 10
        gt['annotations'].append(second)
        p = copy.deepcopy(pred['predictions'][2]);p['annotation_id'] = 10
        pred['predictions'].append(p)
        record = prepare_records(pred, gt)[0][1]
        self.assertEqual({p['annotation_id'] for p in record['points']}, {1, 10})
        self.assertEqual(record['mre_px'], 5)

    def test_draw_preserves_size_coordinates_and_source(self):
        pred, gt = fixture();record = prepare_records(pred, gt)[0][1]
        source = Image.new('RGB', (240, 320), 'black');before = source.tobytes()
        out = draw_overlay(source, record, 'fixture')
        self.assertEqual(out.size, source.size)
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(out.getpixel((40, 140)), (0, 255, 0))
        self.assertEqual(out.getpixel((43, 144)), (255, 0, 0))
        self.assertEqual(out.getpixel((230, 310)), (0, 0, 0))

    def test_missing_predictions_rejected(self):
        pred, gt = fixture();pred['predictions'].pop()
        with self.assertRaises(ValueError): prepare_records(pred, gt)

    def test_cli_inputs_unchanged_and_outputs_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp);raw = base/'images';raw.mkdir()
            pred, gt = fixture()
            for im in gt['images']: Image.new('RGB', (240, 320), 'black').save(raw/im['file_name'])
            gp, pp = base/'gt.json', base/'predictions.json'
            gp.write_text(json.dumps(gt));pp.write_text(json.dumps(pred))
            inputs = [gp, pp, *raw.glob('*.png')];hashes = {p: sha256(p) for p in inputs}
            out = base/'visualizations'
            command = [sys.executable, str(ROOT/'datasets_analysis/tools/visualize_medical_keypoints.py'),
                '--images-root', str(raw), '--gt', str(gp), '--predictions', str(pp), '--out-dir', str(out),
                '--num-samples', '1', '--largest-errors', '--error-ranking', 'max']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            image = out/'largest_errors/image_001.png'
            self.assertTrue(image.is_file())
            manifest = json.loads((out/'largest_errors/manifest.json').read_text())
            self.assertEqual(manifest['images'][0]['image_id'], 1)
            self.assertEqual(manifest['images'][0]['mre_px'], 2.5)
            self.assertEqual(manifest['images'][0]['max_error_px'], 5)
            saved_hash = sha256(image)
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(sha256(image), saved_hash)
            self.assertTrue(all(sha256(p) == h for p, h in hashes.items()))
            command[command.index('--out-dir') + 1] = str(raw/'forbidden')
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse((raw/'forbidden').exists())


    def test_complete_gallery_renders_all_and_copies_ranked_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp);raw = base/'images';raw.mkdir()
            pred, gt = fixture()
            for im in gt['images']:
                Image.new('RGB', (240, 320), 'black').save(raw/im['file_name'])
            gp, pp = base/'gt.json', base/'predictions.json'
            gp.write_text(json.dumps(gt));pp.write_text(json.dumps(pred))
            inputs = [gp, pp, *raw.glob('*.png')];hashes = {p: sha256(p) for p in inputs}
            out = base/'gallery'
            command = [sys.executable, str(ROOT/'datasets_analysis/tools/visualize_medical_keypoints.py'),
                '--images-root', str(raw), '--gt', str(gp), '--predictions', str(pp),
                '--out-dir', str(out), '--complete-gallery', '--num-samples', '2', '--error-ranking', 'max']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual({p.name for p in out.iterdir() if p.is_dir()},
                             {'all', 'largest_errors', 'smallest_errors'})
            self.assertEqual(len(list((out/'all').glob('*.png'))), 3)
            summary = json.loads((out/'summary.json').read_text())
            groups = {g['mode']: g for g in summary['groups']}
            self.assertEqual(groups['largest_errors']['image_ids'], [1, 2])
            self.assertEqual(groups['smallest_errors']['image_ids'], [3, 2])
            for mode in ('largest_errors', 'smallest_errors'):
                for p in (out/mode).glob('*.png'):
                    self.assertEqual(sha256(p), sha256(out/'all'/p.name))
            self.assertTrue((out/'index.html').is_file())
            self.assertTrue(all((out/mode/'index.html').is_file() for mode in groups))
            self.assertTrue(all(sha256(p) == h for p, h in hashes.items()))


if __name__ == '__main__':
    unittest.main()
