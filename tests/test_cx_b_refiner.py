import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import eval_cx_b as evaluation
import eval_batch_cx_b as batch
import train_batch_prompt_concept_variants as entry
from cx_b.configuration import effective_joint_train, resolve_combination_config
from cx_b.joint import JointPipeline, JointLoss, evaluate_joint, final_prediction
from cx_b.refiners import (OVClassSpatialRefiner, build_refiner, load_refiner,
                          refiner_payload, predict_open_vocabulary)
from cx_b.losses import IndependentBinaryLoss
from cx_b.reporting import LayerCounts
import test_cx_b_joint as joint_fixtures
import test_cx_b_joint_initialization as init_fixtures


class RefinerTests(unittest.TestCase):
    def test_zero_initialization_exact_and_gradient_connected(self):
        torch.manual_seed(5)
        r = OVClassSpatialRefiner()
        s = torch.rand(7, 9, 10)
        s[0, 0, :3] = torch.tensor([0., .5, 1.])
        out = r(s)
        self.assertTrue(torch.equal(out, s))
        self.assertTrue(torch.equal(final_prediction(out), final_prediction(s)))
        out.square().sum().backward()
        self.assertGreater(r.network[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(sum(p.numel() for p in r.parameters()), 321)

    def test_variable_vocabulary_permutation_and_endpoints_after_training(self):
        r = OVClassSpatialRefiner().double()
        with torch.no_grad():
            r.network[-1].weight.normal_()
            r.network[-1].bias.fill_(.2)
        for count in (1, 3, 7, 11):
            s = torch.rand(2, count, 6, 5, dtype=torch.float64)
            s[:, :, 0, 0] = 0
            s[:, :, 0, 1] = 1
            p = torch.randperm(count)
            bg = count-1
            actual = r(s[:, p], background=int((p == bg).nonzero()[0]))
            expected = r(s, background=bg)[:, p]
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            self.assertTrue(torch.equal(actual[:, :, 0, 0], s[:, :, 0, 0]))
            self.assertTrue(torch.equal(actual[:, :, 0, 1], s[:, :, 0, 1]))
        with self.assertRaises(ValueError):
            r(torch.empty(0, 4, 4, dtype=torch.float64))

    def test_off_is_no_construction_rng_or_loss_change(self):
        cfg, _ = resolve_combination_config('A3-2', 'region_dynamic', 'ce_lovasz')
        original = copy.deepcopy(cfg['cx_b']['joint_train'])
        for method, strength in [('none', 1), ('ov_class_spatial', 0)]:
            cx = dict(cfg['cx_b'], joint_refiner=method, joint_refiner_strength=strength,
                      refiner_train=dict(binary_weight=0, epochs=123, lr=.1))
            before = torch.get_rng_state().clone()
            with patch('cx_b.refiners.OVClassSpatialRefiner', side_effect=AssertionError('must not construct')):
                self.assertIsNone(build_refiner(cx))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(effective_joint_train(cx), original)

    def test_frozen_base_and_post_training_off_restore(self):
        model, heads, target = joint_fixtures.JointTests().pipeline('multiclass_joint')
        refiner = OVClassSpatialRefiner()
        pipe = JointPipeline(copy.deepcopy(model.prompts), copy.deepcopy(model.concepts), model.records,
                             model.mode, refiner=refiner)
        original = {k:v.clone() for k,v in pipe.state_dict().items() if not k.startswith('refiner.')}
        baseline = model(heads).detach()
        loss = JointLoss(IndependentBinaryLoss(), model.mode,
                         dict(loss_method='ce_lovasz', lovasz_weight=.25, binary_weight=0))
        opt = torch.optim.Adam(refiner.parameters(), lr=.01)
        for _ in range(3):
            pipe.train()
            self.assertFalse(pipe.prompts.training)
            self.assertFalse(pipe.concepts.training)
            opt.zero_grad()
            loss(pipe(heads), target, torch.ones_like(target, dtype=torch.bool)).backward()
            opt.step()
        self.assertFalse(torch.equal(pipe(heads), baseline))
        for k,v in original.items():
            self.assertTrue(torch.equal(v, pipe.state_dict()[k]))
        self.assertTrue(all(p.grad is None for p in pipe.prompts.parameters()))
        pipe.refiner = None
        self.assertTrue(torch.equal(pipe(heads), baseline))
        self.assertIsNone(pipe.raw_scores)

    def test_standalone_shared_weights_and_query_independent_prior(self):
        r = OVClassSpatialRefiner()
        with torch.no_grad():
            r.network[-1].bias.fill_(.2)
        payload = refiner_payload(r, class_names=['old_a', 'old_b'])
        rng = torch.get_rng_state().clone()
        loaded = load_refiner(payload)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        heads = [[dict(semantic_score_map=torch.rand(3, 4), instance_score_map=torch.rand(3, 4),
                       presence_score=torch.tensor(.8))] for _ in range(5)]
        base = predict_open_vocabulary(heads)
        new = predict_open_vocabulary(heads, loaded)
        torch.testing.assert_close(new, r(base, background=None))
        self.assertEqual(new.shape, (5,3,4))
        payload['feature_schema'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'feature_schema'):
            load_refiner(payload)

    def test_cli_routing_and_separate_train_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'best.pth'
            source.write_bytes(b'CLI dry run only')
            args = ['--prompt_methd=A3-2','--concept_method=region_dynamic','--loss_method=ce_lovasz',
                    '--joint-mode=multiclass_joint','--joint-init=checkpoint',
                    '--joint-init-checkpoint='+str(source),'--dry-run']
            def resolve(extra):
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture):
                    entry.main(args+extra)
                return json.loads(capture.getvalue())
            base = resolve([])
            for flags in [['--joint-refiner=none'], ['--joint-refiner=ov_class_spatial','--joint-refiner-strength=0']]:
                off = resolve(flags)
                self.assertEqual(base['combination'], off['combination'])
                self.assertEqual(base['config']['cx_b']['joint_train'], off['config']['cx_b']['joint_train'])
            on = resolve(['--joint-refiner=ov_class_spatial','--refiner-epochs=2'])
            self.assertIn('__r_ov_class_spatial', on['combination'])
            self.assertEqual(on['effective_joint_train']['epochs'], 2)
            self.assertEqual(on['effective_joint_train']['binary_weight'], 0)
            self.assertEqual(on['config']['cx_b']['joint_train']['binary_weight'], .25)

    def test_corrections_and_damage_preserve_raw_binary_diagnostics(self):
        raw = torch.tensor([[[.9, .9, .1, .1]], [[.1, .1, .9, .9]]])
        scores = raw.flip(-1)
        target = torch.tensor([[0, 1, 1, 0]])
        valid = torch.tensor([[True, True, True, False]])
        counts = LayerCounts([], 2)
        counts.update(scores, [], target, valid, raw_scores=raw)
        result = counts.result()
        self.assertEqual(result['refiner_changes'], dict(changed_pixels=3, corrected_pixels=1, damaged_pixels=2))
        old = LayerCounts([], 2)
        old.update(raw, [], target, valid)
        self.assertEqual(result['concept_binary'], old.result()['concept_binary'])
        self.assertNotEqual(result['refined_concept_binary'], result['concept_binary'])


class RefinerIntegrationTests(unittest.TestCase):
    def test_complete_run_reload_eval_ablation_export_and_resume(self):
        fixture = init_fixtures.JointInitializationTests()
        with contextlib.redirect_stdout(io.StringIO()):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config['cx_b'].update(joint_refiner='ov_class_spatial', joint_refiner_strength=1.,
                                     refiner_train=dict(epochs=2, lr=.01), export_probabilities=True)
        trainer = fixture.trainer('r1')
        inputs = json.loads((fixture.source/'run_config.json').read_text())['inputs']
        runtime = fixture.runtime()
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(trainer, '_resolved_paths', return_value=(str(fixture.root/'concepts.txt'), str(fixture.root/'cache'))), \
             patch.object(trainer, 'run_fingerprint', return_value=('r1-fixture', inputs)), \
             patch.object(trainer, '_build_runtime', return_value=runtime):
            trainer.run()
        destination = Path(trainer.output_dir)
        final = json.loads((destination/'final_metrics.json').read_text())['metrics']
        best = torch.load(destination/'joint/best.pth')
        last = torch.load(destination/'joint/last.pth')
        source = torch.load(fixture.checkpoint)
        self.assertEqual(best['format'], 'cx_b_joint_v2')
        self.assertEqual(best['refiner']['provenance']['source_checkpoint_sha256'],
                         init_fixtures.sha256_file(fixture.checkpoint))
        self.assertEqual(final, best['metrics'])
        self.assertTrue((destination/'refiner/best.pth').is_file())
        for slot, payload in source['modules'].items():
            for key, tensor in payload['state_dict'].items():
                self.assertTrue(torch.equal(tensor, last['modules'][slot]['state_dict'][key]))
        self.assertGreater(last['refiner']['state_dict']['network.4.weight'].abs().sum(), 0)
        for name in ('best', 'last', 'epoch_0000'):
            loaded = evaluation.load_run(destination, 'cpu', joint_checkpoint=f'joint/{name}.pth')
            self.assertIsNotNone(loaded[-1].refiner)
        loaded = evaluation.load_run(destination, 'cpu')
        records, prompts, aggregators = loaded[3:]
        get_heads = lambda r: runtime[3].load('val', r['prompt_index'], 0, '')
        trained = evaluation.load_run(destination, 'cpu', joint_checkpoint='joint/last.pth')
        trained_scores, _ = evaluation.predict_image(trained[3], trained[2], trained[4], trained[5], get_heads)
        self.assertFalse(np.array_equal(trained_scores, trained[5].raw_scores))
        with torch.no_grad():
            expected_scores = trained[5].refiner(torch.from_numpy(trained[5].raw_scores), background=0).numpy()
        np.testing.assert_array_equal(trained_scores, expected_scores)
        scores, _ = evaluation.predict_image(records, loaded[2], prompts, aggregators, get_heads)
        manifest = json.loads((destination/'probability_manifest.json').read_text())
        for row in manifest:
            self.assertEqual(row['score_scope'], 'refined')
            np.testing.assert_array_equal(np.load(destination/row['probability_path']), scores[row['concept_index']])
        off = evaluation.load_run(destination, 'cpu', joint_checkpoint='joint/last.pth', joint_refiner='none')
        old = evaluation.load_run(fixture.source, 'cpu')
        disabled, _ = evaluation.predict_image(off[3], off[2], off[4], off[5], get_heads)
        original, _ = evaluation.predict_image(old[3], old[2], old[4], old[5], get_heads)
        np.testing.assert_array_equal(disabled, original)
        args = batch.parse_args(['--root', str(destination), '--device=cpu'])
        with contextlib.redirect_stdout(io.StringIO()):
            evaluated = batch.evaluate_run(destination, args)['result']
            scan = batch.scan_joint_checkpoints(destination, args)
        self.assertAlmostEqual(evaluated['metrics']['multiclass_common']['summary']['mIoU'], final['mIoU']*100)
        self.assertIn('refined_binary', evaluated['metrics'])
        self.assertIn('refiner_changes', evaluated)
        self.assertEqual(scan['checkpoint_count'], 5)
        off_args = batch.parse_args(['--root', str(destination), '--device=cpu', '--joint-refiner-strength=0'])
        self.assertNotEqual(batch.output_directory(destination, args), batch.output_directory(destination, off_args))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(['--root', str(destination), '--device=cpu', '--joint-refiner=none']), 0)
        self.assertTrue((destination/'batch_multiclass_val__refiner_none__strength_saved.json').is_file())
        self.assertFalse((destination/'batch_multiclass_val.json').exists())
        export_dir = fixture.root/'eval_export'
        export_args = evaluation.parse_args(['--pth-dir', str(destination), '--out', str(export_dir), '--device=cpu'])
        with contextlib.redirect_stdout(io.StringIO()), patch.object(evaluation, 'save_panel'):
            exported = evaluation.run_evaluation(export_args)
        self.assertAlmostEqual(exported['multiclass_argmax']['mIoU'], final['mIoU'])
        self.assertIn('refined_binary', exported)
        self.assertEqual(exported['refiner_changes'], final['refiner_changes'])
        with np.load(next((export_dir/'data').glob('*.npz'))) as data:
            np.testing.assert_array_equal(data['concept_scores'], scores)
            self.assertIn('raw_concept_scores', data)
        fixture.config['cx_b']['joint_init_checkpoint'] = str(destination/'joint/best.pth')
        restored = fixture.trainer('restored')
        paths = restored._joint_paths(fixture.concepts, fixture.records)
        from cx_b.joint_initialization import load_joint_initialization
        modules, _ = load_joint_initialization(destination/'joint/best.pth', fixture.config, fixture.records,
                        str(fixture.root/'cache'), [p.relative_to(restored.output_dir).as_posix() for group in paths for p in group])
        self.assertIsNotNone(modules.refiner_payload)
        corrupted = copy.deepcopy(best)
        corrupted.pop('refiner')
        torch.save(corrupted, destination/'joint/broken.pth')
        with self.assertRaisesRegex(ValueError, 'missing refiner'):
            evaluation.load_run(destination, 'cpu', joint_checkpoint='joint/broken.pth')


if __name__ == '__main__':
    unittest.main()
