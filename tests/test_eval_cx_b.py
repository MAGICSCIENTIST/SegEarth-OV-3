import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import eval_cx_b as evaluation
from cx_b.modules import A3PromptAdapter, build_concept_aggregator, save_variant
from tools import visual_utils as visual


class CXBEvaluationTests(unittest.TestCase):
    def make_run(self, root):
        checkpoint = root / 'checkpoints'
        checkpoint.mkdir()
        data = root / 'dataset'
        for name in ['img', 'gt']:
            (data / name).mkdir(parents=True)
        image_path = data / 'img/sample.png'
        Image.fromarray(np.full((12, 12, 3), 100, dtype=np.uint8)).save(image_path)
        gt = np.ones((12, 12), dtype=np.uint8)
        gt[:, 6:] = 2
        gt[0] = 0
        Image.fromarray(gt).save(data / 'gt/sample.png')
        cache = root / 'cache'
        config = dict(concept_path='concepts.txt', model=dict(
            classname_path='concepts.txt', confidence_threshold=0.1, prob_thd=0.1),
            cx_b=dict(prompt_method='A3-2', concept_method='dynamic',
                      data_mode='cache', metric_threshold=0.5),
            validation_dataloader=dict(dataset=dict(
                data_root=str(data), data_prefix=dict(img_path='img', seg_map_path='gt'))))
        run = dict(config=config, fingerprint='test-run', inputs=dict(
            concept_content='building,house\nforest', prompt_head_cache=str(cache),
            selected_concepts=None))
        (checkpoint / 'run_config.json').write_text(json.dumps(run), encoding='utf-8')
        identities = []
        bases = dict(semantic_weight=0.5, presence_gain=1.0, presence_intercept=0.0)
        for index, prompt in enumerate(['building', 'house', 'forest']):
            concept = int(index == 2)
            identity = dict(prompt_index=index, prompt=prompt,
                            concept_index=concept, concept_id=['building', 'forest'][concept],
                            prompt_position=index if concept == 0 else 0,
                            valid_labels=[concept + 1])
            identities.append(identity)
            save_variant(checkpoint / 'prompt_checkpoints' / f'prompt_{index:03d}' / 'best.pth',
                         A3PromptAdapter(**bases), dict(kind='prompt', method='A3-2',
                         base_parameters=bases, identity=identity, run_fingerprint='test-run'))
            probability = torch.full((12, 12), 0.2)
            probability[:, :6] = 0.8 if concept == 0 else 0.1
            probability[:, 6:] = 0.1 if concept == 0 else 0.8
            heads = dict(semantic_score_map=probability,
                         instance_score_map=probability, presence_score=torch.tensor(0.5))
            directory = cache / 'val' / f'prompt_{index:03d}'
            directory.mkdir(parents=True)
            torch.save(dict(image_path=str(image_path.resolve()), prompt_idx=index,
                            prompt=prompt, heads=heads, cache_signature=dict(
                            confidence_threshold=0.1, use_sem_seg=True,
                            use_transformer_decoder=True, instance_filter_use_presence=True)),
                       directory / '000000_sample.pth')
        for index, name in enumerate(['building', 'forest']):
            members = [r for r in identities if r['concept_index'] == index]
            save_variant(checkpoint / 'concept' / f'concept_{index:03d}_{name}' / 'best.pth',
                         build_concept_aggregator('dynamic', len(members)),
                         dict(kind='concept', method='dynamic', prompt_count=len(members),
                              members=members, run_fingerprint='test-run'))
        return checkpoint

    def test_cached_export_and_optional_visualizations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_run(root)
            for optional in [False, True]:
                output = root / ('all' if optional else 'default')
                flags = ['--vis-concept', '--vis-prompt'] if optional else []
                args = evaluation.parse_args(['--pth-dir', str(checkpoint), '--out',
                                              str(output), '--device', 'cpu'] + flags)
                with patch.object(evaluation, 'save_panel') as panel:
                    result = evaluation.run_evaluation(args)
                self.assertEqual(panel.call_count, 3)
                self.assertAlmostEqual(result['independent_binary']['mIoU'], 1.0)
                self.assertAlmostEqual(result['multiclass_argmax']['mIoU'], 1.0)
                self.assertEqual((output / 'vis/concept').exists(), optional)
                self.assertEqual((output / 'vis/prompt').exists(), optional)
                self.assertTrue((output / 'vis/out/sample.png.png').is_file())
                with np.load(output / 'data/sample.png.npz') as saved:
                    self.assertEqual(saved['concept_scores'].shape, (2, 12, 12))
                    self.assertEqual(saved['prompt_scores'].shape, (3, 12, 12))
                    self.assertTrue(np.all(saved['prediction'][:, :6] == 0))
                    self.assertTrue(np.all(saved['prediction'][:, 6:] == 1))
                    np.testing.assert_allclose(saved['concept_scores'][1], saved['prompt_scores'][2])
                    np.testing.assert_array_equal(np.asarray(Image.open(output / 'data/sample.png.png')),
                                                  saved['prediction'])
                    gt = np.asarray(Image.open(root / 'dataset/gt/sample.png'))
                    np.testing.assert_array_equal(saved['gt_raw'], gt)
                    self.assertEqual(saved['gt_raw'].dtype, gt.dtype)
                    np.testing.assert_array_equal(saved['gt_valid_mask'], gt != 0)
                    self.assertTrue(np.all(saved['gt_concept'][0] == evaluation.NODATA))
                    np.testing.assert_array_equal(saved['gt_concept_masks'][0], gt == 1)
                    np.testing.assert_array_equal(saved['gt_prompt_masks'][0], gt == 1)
                    np.testing.assert_array_equal(saved['gt_prompt_masks'][1], gt == 1)
                    np.testing.assert_array_equal(saved['gt_prompt_masks'][2], gt == 2)
                    np.testing.assert_array_equal(saved['gt_concept_indices'], [0, 1])
                    np.testing.assert_array_equal(
                        np.load(output / 'data/gt/raw/sample.png.npy'), gt)
                    np.testing.assert_array_equal(np.asarray(Image.open(
                        output / 'data/gt/concept_labels/sample.png.png')), saved['gt_concept'])
                binary_gt = np.asarray(Image.open(
                    output / 'data/gt/concept/sample.png/000_building.png'))
                self.assertTrue(np.all(binary_gt[0] == 255))
                self.assertTrue(np.all(binary_gt[1:, :6] == 1))
                self.assertTrue(np.all(binary_gt[1:, 6:] == 0))
                self.assertTrue((output / 'vis/gt/sample.png.png').is_file())
                manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
                self.assertTrue((output / manifest['files'][0]['gt_outputs']['raw']).is_file())

    def test_mixed_run_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_run(Path(directory))
            path = checkpoint / 'concept/concept_000_building/best.pth'
            payload = torch.load(path)
            payload['metadata']['run_fingerprint'] = 'another-run'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'run mismatch'):
                evaluation.load_run(checkpoint, 'cpu')

    def test_cache_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_run(root)
            run, _, _, records, _, _ = evaluation.load_run(checkpoint, 'cpu')
            provider = evaluation.SavedHeadCache(root / 'cache', 'val', run['config']['model'], 'cpu')
            record = dict(records[0], prompt='wrong prompt')
            with self.assertRaisesRegex(ValueError, 'identity/settings mismatch'):
                provider.load(record, 0, root / 'dataset/img/sample.png')

    def test_label_mapping_and_rejection(self):
        gt = np.array([[0, 1, 2]], dtype=np.uint8)
        mapped = evaluation.map_gt(gt, [{}, {}], False, 0)
        np.testing.assert_array_equal(mapped, [[evaluation.NODATA, 0, 1]])
        scores = np.array([[[0.1, 0.8]], [[0.2, 0.3]]])
        np.testing.assert_array_equal(evaluation.classify_scores(scores, [1, 5], 0.5),
                                      [[evaluation.NODATA, 1]])

    def test_json_gt_export_preserves_groups_and_unselected_concepts(self):
        concepts = [dict(concept_id='forest', tnls_dl3=[10, 11]),
                    dict(concept_id='water', tnls_dl3=[20])]
        gt = np.array([[10, 11, 20, 255, 99]], dtype=np.uint16)
        mapped = evaluation.map_gt(gt, concepts, True, 255)
        with tempfile.TemporaryDirectory() as directory:
            arrays, _ = evaluation.export_ground_truth(
                Path(directory), Path('sample.tif'), gt, mapped, concepts, True, 255,
                [dict(concept_index=1, prompt_index=2)])
            np.testing.assert_array_equal(arrays['gt_concept_masks'],
                                          [[[True, True, False, False, False]],
                                           [[False, False, True, False, False]]])
            np.testing.assert_array_equal(arrays['gt_prompt_masks'][0], gt == 20)
            np.testing.assert_array_equal(arrays['gt_valid_mask'], gt != 255)
            self.assertEqual(arrays['gt_concept'][0, 4], evaluation.NODATA)
            self.assertEqual(arrays['gt_raw'].dtype, np.uint16)

    def test_dynamic_inference_uses_saved_modules(self):
        torch.manual_seed(7)
        shape = (4, 5)
        heads = {i: dict(semantic_score_map=torch.rand(shape),
                         instance_score_map=torch.rand(shape),
                         presence_score=torch.tensor(0.8)) for i in range(2)}
        prompts = {i: A3PromptAdapter(semantic_weight=0.3 + i * 0.2) for i in range(2)}
        aggregator = build_concept_aggregator('dynamic', 2)
        records = [dict(prompt_index=i, concept_index=0) for i in range(2)]
        with torch.no_grad():
            actual, _ = evaluation.predict_image(records, [0], prompts, {0: aggregator},
                                                  lambda r: heads[r['prompt_index']])
            p = torch.stack([prompts[i](heads[i]) for i in range(2)])
            expected = aggregator(p, presence=torch.tensor([0.8, 0.8]))
        np.testing.assert_allclose(actual[0], expected.numpy(), rtol=1e-6)

    def test_visual_utils_has_no_entrypoint_dependencies_and_legacy_arrays_work(self):
        tree = ast.parse(Path(visual.__file__).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(
                    ('train', 'segearth', 'config', 'diagnose', 'mmseg', 'custom')))
        raster = np.full((8, 8, 3), 100, dtype=np.uint8)
        labels = torch.zeros((8, 8), dtype=torch.long)
        concepts = [dict(concept_id='forest')]
        prediction = visual.render_prediction(raster, labels, concepts, 0.5)
        binary = visual.render_concept_mask(raster, labels.numpy(), 0, concepts[0], 0.5)
        np.testing.assert_array_equal(prediction, binary)
        self.assertFalse(np.array_equal(prediction, raster))
        drawn = visual.draw_sem_seg(raster, labels[None], ['forest'], [[80, 200, 60]],
                                    lambda mask: (1, 1), with_labels=False)
        self.assertEqual(drawn.shape, raster.shape)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'input.png'
            Image.fromarray(raster).save(image)
            visual.save_training_comparisons(
                str(image), labels, labels, concepts, directory, 1)
            self.assertTrue((Path(directory) / 'vis_data/vis_image/input_epoch_0001.jpg').is_file())
            sample = dict(img_path=str(image), pred_sem_seg=dict(data=labels[None]),
                          gt_sem_seg=dict(data=labels[None]),
                          raw_prompt_scores=dict(data=torch.ones(1, 8, 8)))
            visual.save_tnls_concept_comparisons(
                sample, [dict(concept_id='forest', prompts=['forest'])], directory)
            self.assertTrue((Path(directory) / 'vis_data/promot/input__PROMPT__forest.jpg').is_file())


if __name__ == '__main__':
    unittest.main()
