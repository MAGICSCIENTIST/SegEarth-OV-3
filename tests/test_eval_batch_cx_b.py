import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import eval_batch_cx_b as batch
import test_eval_cx_b as fixtures
from cx_b.modules import SegOV3PromptAdapter, build_concept_aggregator, save_variant


class BatchEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir(parents=True, exist_ok=True)
        run = fixtures.CXBEvaluationTests().make_run(root)
        snapshot = batch.read_json(run/'run_config.json')
        snapshot['seed'] = 3407
        snapshot['config']['cx_b']['loss_method'] = 'A3-2'
        (run/'run_config.json').write_text(json.dumps(snapshot), encoding='utf-8')
        (run/'completed.json').write_text(json.dumps(dict(status='complete', fingerprint='test-run')))
        return run

    def test_confusion_rejection_and_missing_classes(self):
        confusion = np.zeros((3, 4), np.int64)
        gt = np.array([[0, 0, 1, 1, batch.NODATA]])
        pred = np.array([[0, 1, 1, batch.NODATA, 2]])
        batch.add_confusion(confusion, pred, gt, gt != batch.NODATA)
        result = batch.summarize_confusion(confusion, ['a', 'b', 'c'])
        np.testing.assert_array_equal(confusion, [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
        self.assertAlmostEqual(result['summary']['mIoU'], (50+100/3)/2)
        self.assertEqual(result['summary']['aAcc'], 50)
        self.assertEqual(result['unassigned_pixels'], 1)
        self.assertIsNone(result['per_class'][2]['IoU'])
        self.assertEqual(result['per_class'][1]['FN'], 1)

    def test_threshold_boundary_and_background_fallback(self):
        scores = np.array([[[.1, .1]], [[.5, .49999]]])
        np.testing.assert_array_equal(batch.classify_scores(scores, [0, 1], .5, 0), [[1, 0]])
        valid = np.ones((1, 2), bool)
        ge, gt = np.zeros((1, 3), np.int64), np.zeros((1, 3), np.int64)
        truth = np.array([[True, False]])
        batch.add_binary(ge, 0, scores[1] >= .5, truth, valid)
        batch.add_binary(gt, 0, scores[1] > .5, truth, valid)
        np.testing.assert_array_equal(ge, [[1, 0, 0]])
        np.testing.assert_array_equal(gt, [[0, 0, 1]])

    def test_multiclass_zero_overlap_fscore_matches_mmseg(self):
        result = batch.summarize_confusion(np.array([[0, 1, 0], [1, 0, 0]]), ['a', 'b'])
        self.assertEqual(result['summary']['mIoU'], 0)
        self.assertEqual(result['summary']['mPrecision'], 0)
        self.assertIsNone(result['summary']['mFscore'])
        self.assertTrue(all(row['Fscore'] is None for row in result['per_class']))

    def test_resume_skip_and_changed_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self.fixture(root)
            # A second cached sample lets us interrupt between images.
            for folder in ('img', 'gt'):
                source = root/'dataset'/folder/'sample.png'
                (source.parent/'sample2.png').write_bytes(source.read_bytes())
            for path in (root/'cache/val').glob('*/000000_sample.pth'):
                payload = torch.load(path)
                payload['image_path'] = str((root/'dataset/img/sample2.png').resolve())
                torch.save(payload, path.parent/'000001_sample2.pth')
            args = batch.parse_args(['--root', str(directory), '--device', 'cpu', '--save-every', '1'])
            from eval_cx_b import SavedHeadCache
            original = SavedHeadCache.load
            def interrupt(cache, record, index, image):
                if index == 1:
                    raise RuntimeError('test interruption')
                return original(cache, record, index, image)
            with patch.object(SavedHeadCache, 'load', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    batch.evaluate_run(directory, args)
            output = batch.output_directory(directory, args)
            self.assertEqual(batch.read_json(output/'progress.json')['processed_images'], 1)
            seen = []
            def track(cache, record, index, image):
                seen.append(index)
                return original(cache, record, index, image)
            with patch.object(SavedHeadCache, 'load', track):
                result = batch.evaluate_run(directory, args)['result']
            self.assertEqual(seen, [1, 1, 1])
            self.assertEqual(result['metrics']['multiclass_common']['summary']['mIoU'], 100)
            self.assertEqual(result['metrics']['multiclass_saved']['valid_pixels'], 264)
            for name in ('metrics.json', 'summary.csv', 'per_class.csv', 'per_prompt.csv', 'results.xlsx'):
                self.assertTrue((output/name).is_file())
            with patch.object(batch, 'build_head_source', side_effect=AssertionError('should skip')):
                self.assertEqual(batch.evaluate_run(directory, args)['status'], 'skipped_existing')
            args.common_threshold = .9
            with patch.object(SavedHeadCache, 'load', track):
                updated = batch.evaluate_run(directory, args)['result']
            self.assertNotEqual(updated['fingerprint'], result['fingerprint'])
            self.assertEqual(updated['metrics']['multiclass_common']['summary']['aAcc'], 50)
            # Full validation and smoke outputs never share their completion marker.
            args.max_images = 1
            self.assertNotEqual(batch.output_directory(directory, args), output)

    def test_batch_failure_isolation_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = self.fixture(root/'a_bad')
            good = self.fixture(root/'b_good')
            incomplete = self.fixture(root/'c_incomplete')
            (incomplete/'completed.json').unlink()
            (root/'a_bad/cache/val/prompt_000/000000_sample.pth').unlink()
            code = batch.main(['--root', str(root), '--device', 'cpu'])
            self.assertEqual(code, 1)
            rows = batch.read_json(root/'batch_multiclass_val.json')['runs']
            self.assertEqual([r['status'] for r in rows], ['failed', 'evaluated', 'skipped_incomplete'])
            self.assertTrue((good/'eval_multiclass/val/metrics.json').exists())
            self.assertTrue((bad/'eval_multiclass/val/last_error.json').exists())
            self.assertFalse((incomplete/'eval_multiclass').exists())

    def test_completed_fingerprint_mismatch_and_list_only(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = self.fixture(Path(temp))
            (directory/'completed.json').write_text(json.dumps(dict(status='complete', fingerprint='wrong')))
            args = batch.parse_args(['--root', str(directory)])
            self.assertEqual(batch.discover_runs(args)[0]['status'], 'failed')
            self.assertEqual(batch.main(['--root', str(directory), '--list-only']), 1)
            self.assertFalse((directory/'eval_multiclass').exists())

    def test_loveda_baseline_labels_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'seed_3407'
            directory.mkdir()
            for part in ('img', 'gt', 'base/result'):
                (root/part).mkdir(parents=True)
            gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], np.uint8)
            image_path = root/'img/example.png'
            Image.fromarray(np.zeros((1, 8, 3), np.uint8)).save(image_path)
            Image.fromarray(gt).save(root/'gt/example.png')
            Image.fromarray(np.maximum(gt, 1)).save(root/'base/result/example.png')
            cx = dict(prompt_method='segov3', concept_method='max', loss_method='A3-2',
                      data_mode='cache', prediction_background=0, metric_threshold=.5)
            model = dict(classname_path='classes.txt', confidence_threshold=.5, prob_thd=.5)
            run = dict(fingerprint='loveda', seed=3407, config=dict(model=model, cx_b=cx,
                       validation_dataloader=dict(dataset=dict(data_root=str(root),
                            data_prefix=dict(img_path='img', seg_map_path='gt')))),
                       inputs=dict(concept_content='\n'.join(batch.LOVEDA_CLASSES),
                                   prompt_head_cache=str(root/'cache')))
            (directory/'run_config.json').write_text(json.dumps(run))
            for index, name in enumerate(batch.LOVEDA_CLASSES):
                identity = dict(prompt_index=index, prompt=name, concept_index=index,
                                concept_id=name, prompt_position=0, valid_labels=[index+1])
                save_variant(directory/'prompt_checkpoints'/f'prompt_{index:03d}'/'best.pth',
                             SegOV3PromptAdapter(), dict(kind='prompt', method='segov3',
                             base_parameters={}, identity=identity, run_fingerprint='loveda'))
                save_variant(directory/'concept'/f'concept_{index:03d}_{name}'/'best.pth',
                             build_concept_aggregator('max', 1), dict(kind='concept', method='max',
                             prompt_count=1, members=[identity], run_fingerprint='loveda'))
                path = root/'cache/val'/f'prompt_{index:03d}'/'000000_example.pth'
                path.parent.mkdir(parents=True)
                probability = torch.from_numpy(np.where(gt==index+1, .8, .1).astype(np.float32))
                torch.save(dict(image_path=str(image_path), prompt_idx=index, prompt=name,
                           cache_signature=dict(confidence_threshold=.5, use_sem_seg=True,
                                use_transformer_decoder=True, instance_filter_use_presence=True),
                           heads=dict(semantic_score_map=probability, instance_score_map=probability,
                                      presence_score=torch.tensor(1.))), path)
            args = batch.parse_args(['--root', str(directory), '--device', 'cpu', '--base-dir', str(root/'base')])
            result = batch.evaluate_run(directory, args)['result']
            for policy in ('multiclass_saved', 'multiclass_common', 'baseline_multiclass'):
                self.assertEqual(result['metrics'][policy]['summary']['mIoU'], 100)
                self.assertEqual(result['metrics'][policy]['valid_pixels'], 7)
            self.assertEqual(result['baseline_comparison']['delta_percentage_points']['multiclass_common']['mIoU'], 0)
            (directory/'completed.json').write_text(json.dumps(
                dict(status='complete', fingerprint='loveda')))
            self.assertEqual(batch.main(['--root', str(directory), '--device', 'cpu',
                                         '--base-dir', str(root/'base')]), 0)
            summary = batch.read_json(directory/'batch_multiclass_val.json')['runs'][0]
            self.assertEqual(summary['status'], 'skipped_existing')
            self.assertEqual(summary['loveda_mIoU'], 100)
            self.assertEqual(summary['multiclass_common_delta_vs_loveda_mIoU'], 0)
            scan_args = batch.parse_args(['--root', str(directory), '--device', 'cpu',
                                         '--base-dir', str(root/'base'), '--use-best=false'])
            scanned = batch.scan_checkpoints(directory, scan_args)
            self.assertEqual(scanned['selected_delta_vs_loveda_mIoU'], 0)
            self.assertEqual(scanned['checkpoint_count'], 14)
            report = batch.read_json(Path(scanned['checkpoint_scan_output'])/'checkpoint_comparison.json')
            self.assertEqual(report['selected']['candidate_id'], 'original_best')
            self.assertTrue(all(r['delta_vs_loveda_mIoU'] == 0 for r in report['checkpoints']))

    def scan_fixture(self, root):
        directory = self.fixture(root)
        # Original forest best is deliberately under-confident. The saved epoch
        # restores its correct scores, so final competition must prefer it.
        forest = directory/'prompt_checkpoints/prompt_002'
        payload = torch.load(forest/'best.pth')
        payload['metadata'].update(epoch=10, checkpoint_type='periodic')
        torch.save(payload, forest/'epoch_0010.pth')
        payload['metadata'].update(epoch=12, checkpoint_type='last')
        torch.save(payload, forest/'last.pth')
        module, metadata = batch.load_run(directory, 'cpu')[4][2], payload['metadata']
        with torch.no_grad():
            module.presence_intercept.fill_(-8)
        save_variant(forest/'best.pth', module, dict(metadata, epoch=2, checkpoint_type='best'))
        concept = directory/'concept/concept_000_building'
        payload = torch.load(concept/'best.pth')
        payload['metadata'].update(epoch=20, checkpoint_type='periodic')
        torch.save(payload, concept/'epoch_0020.pth')
        return directory

    def test_scan_selects_final_miou_and_exports_every_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self.scan_fixture(root)
            before = {str(p): batch.sha256(p) for p in directory.rglob('*.pth')}
            args = batch.parse_args(['--root', str(directory), '--device', 'cpu', '--use-best=false'])
            self.assertFalse(args.use_best)
            from eval_cx_b import SavedHeadCache
            original_load, calls = SavedHeadCache.load, []
            def track(cache, record, index, image):
                calls.append((record['prompt_index'], index))
                return original_load(cache, record, index, image)
            with patch.object(SavedHeadCache, 'load', track):
                result = batch.scan_checkpoints(directory, args)
            # Three raw heads once for all single replacements, once for assembly.
            self.assertEqual(len(calls), 6)
            scan = batch.output_directory(directory, args)/'checkpoint_scan'
            report = batch.read_json(scan/'checkpoint_comparison.json')
            self.assertEqual(len(report['checkpoints']), 8)
            self.assertEqual(report['original_best']['mIoU'], 25)
            self.assertEqual(report['selected']['mIoU'], 100)
            self.assertEqual(report['selected']['delta_vs_original_best_mIoU'], 75)
            self.assertEqual(report['winners_assembly']['mIoU'], 100)
            forest = 'prompt_checkpoints/prompt_002/best.pth'
            self.assertEqual(report['module_winners'][forest], 'prompt_checkpoints/prompt_002/epoch_0010.pth')
            self.assertEqual(report['module_winners']['concept/concept_000_building/best.pth'],
                             'concept/concept_000_building/best.pth')
            for row in report['checkpoints']:
                self.assertTrue((Path(row['output'])/'metrics.json').exists())
            for name in ('summary.md', 'checkpoints.csv', 'assemblies.csv', 'checkpoint_comparison.xlsx', 'selected_checkpoints.json'):
                self.assertTrue((scan/name).exists())
            self.assertEqual(before, {str(p): batch.sha256(p) for p in directory.rglob('*.pth')})
            with patch.object(batch, 'build_head_source', side_effect=AssertionError('must reuse completed results')):
                self.assertEqual(batch.scan_checkpoints(directory, args)['status'], 'skipped_existing')
            self.assertEqual(result['selected_mIoU'], 100)
            # CLI writes separate batch summaries and includes newly selected metrics.
            self.assertEqual(batch.main(['--root', str(directory), '--device', 'cpu', '--no-use-best']), 0)
            row = batch.read_json(directory/'batch_multiclass_val_all_checkpoints.json')['runs'][0]
            self.assertEqual(row['selected_delta_vs_original_best_mIoU'], 75)
            # Changing one candidate invalidates it, but not unrelated candidates.
            candidate_path = directory/'prompt_checkpoints/prompt_002/epoch_0010.pth'
            payload = torch.load(directory/forest)
            payload['metadata'].update(epoch=10, checkpoint_type='periodic')
            torch.save(payload, candidate_path)
            calls.clear()
            with patch.object(SavedHeadCache, 'load', track):
                rescanned = batch.scan_checkpoints(directory, args)
            self.assertEqual(len(calls), 6)
            self.assertEqual(rescanned['selected_candidate'], 'prompt_checkpoints/prompt_002/last.pth')

    def test_grouped_replacements_match_independent_forward_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self.scan_fixture(root)
            for folder in ('img', 'gt'):
                source = root/'dataset'/folder/'sample.png'
                (source.parent/'sample2.png').write_bytes(source.read_bytes())
            for path in (root/'cache/val').glob('*/000000_sample.pth'):
                payload = torch.load(path)
                payload['image_path'] = str((root/'dataset/img/sample2.png').resolve())
                torch.save(payload, path.parent/'000001_sample2.pth')
            args = batch.parse_args(['--root', str(directory), '--device', 'cpu', '--save-every=1'])
            candidates, _ = batch.checkpoint_candidates(directory, args)
            from eval_cx_b import SavedHeadCache, load_run
            original_load = SavedHeadCache.load
            def interrupt(cache, record, index, image):
                if index == 1:
                    raise RuntimeError('scan interruption')
                return original_load(cache, record, index, image)
            with patch.object(SavedHeadCache, 'load', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'scan interruption'):
                    batch.evaluate_run(directory, args, candidates)
            for candidate in candidates:
                self.assertEqual(batch.read_json(candidate['output']/'progress.json')['processed_images'], 1)
            seen = []
            def track(cache, record, index, image):
                seen.append(index)
                return original_load(cache, record, index, image)
            with patch.object(SavedHeadCache, 'load', track):
                grouped = batch.evaluate_run(directory, args, candidates)
            self.assertEqual(seen, [1, 1, 1])
            # Independent complete-model forward, without the changed-concept shortcut.
            for candidate in candidates[1:]:
                args.out_root = root/'independent'/Path(candidate['id']).with_suffix('')
                overrides = candidate['overrides']
                def overridden(directory, device, replacements=None):
                    return load_run(directory, device, overrides)
                with patch.object(batch, 'load_run', overridden):
                    reference = batch.evaluate_run(directory, args)['result']
                self.assertEqual(grouped[candidate['id']]['result']['metrics'], reference['metrics'])

    def test_checkpoint_override_rejects_wrong_identity_and_unknown_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = self.scan_fixture(Path(temp))
            slot = 'prompt_checkpoints/prompt_002/best.pth'
            path = directory/'prompt_checkpoints/prompt_002/epoch_0010.pth'
            payload = torch.load(path)
            payload['metadata']['identity']['prompt_index'] = 99
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                batch.load_run(directory, 'cpu', {slot: path.relative_to(directory).as_posix()})
            with self.assertRaisesRegex(ValueError, 'Unknown checkpoint override'):
                batch.load_run(directory, 'cpu', {'unknown/best.pth': 'unknown/epoch.pth'})


if __name__ == '__main__':
    unittest.main()
