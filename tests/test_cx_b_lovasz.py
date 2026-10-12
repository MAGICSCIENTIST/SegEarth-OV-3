import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import train_batch_prompt_concept_variants as entry
from cx_b.configuration import (resolve_combination_config, validate_config,
                                ConfigurationError, effective_joint_train)
from cx_b.joint import JointLoss, competition_logits
from cx_b.losses import build_binary_loss
from cx_b.multiclass_lovasz import lovasz_softmax


class JointLovaszTests(unittest.TestCase):
    def test_pure_lovasz_matches_value_and_gradient_without_ce_or_binary(self):
        cfg, _ = resolve_combination_config('A3-2', 'region_dynamic', 'lovasz')
        scores = torch.tensor([[[.4, .7, .8]], [[.8, .6, .3]]], requires_grad=True)
        target = torch.tensor([[1, 0, -1]])
        valid = target >= 0
        expected = lovasz_softmax(competition_logits(scores), target, -1)
        expected_grad = torch.autograd.grad(expected, scores)[0]
        binary = Mock(side_effect=AssertionError('Pure Lovasz must not call binary loss'))
        objective = JointLoss(binary, 'multiclass_joint', cfg['cx_b']['joint_train'])
        with patch('cx_b.joint.F.cross_entropy', side_effect=AssertionError('Must not compute CE')):
            actual = objective(scores, target, valid)
            actual_grad = torch.autograd.grad(actual, scores)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        binary.assert_not_called()
        self.assertGreater(actual_grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(actual_grad).all())
        self.assertEqual(actual_grad[..., 2].abs().sum().item(), 0)
        self.assertIsNone(objective(scores, target, torch.zeros_like(valid)))
        # The new option matches the previously documented weight-override objective.
        old_settings = dict(cfg['cx_b']['joint_train'], loss_method='ce_lovasz')
        old = JointLoss(binary, 'multiclass_joint', old_settings)(scores, target, valid)
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, torch.autograd.grad(old, scores)[0])

    def test_pure_lovasz_routing_preserves_warmup_and_refiner_off_semantics(self):
        cfg, _ = resolve_combination_config('A3-2', 'region_dynamic', 'lovasz')
        original = copy.deepcopy(cfg['cx_b']['joint_train'])
        for stage in ('prompt', 'concept'):
            self.assertEqual(build_binary_loss(cfg, stage).method, 'A3-2')
        for mode in ('multiclass_joint', 'multiclass_prompt', 'multiclass_concept'):
            for method, strength in [('none', 1), ('ov_class_spatial', 0), ('ov_class_spatial', 1)]:
                with self.subTest(mode=mode, method=method, strength=strength):
                    current = copy.deepcopy(cfg)
                    cx = current['cx_b']
                    cx.update(joint_mode=mode, joint_refiner=method, joint_refiner_strength=strength,
                              refiner_train=dict(epochs=2, lr=.0001))
                    validate_config(current)
                    train = effective_joint_train(cx)
                    self.assertEqual(train['loss_method'], 'lovasz')
                    self.assertEqual([train[k] for k in ('competition_weight', 'binary_weight', 'lovasz_weight')], [0., 0., 1.])
                    if method == 'none' or strength == 0:
                        self.assertEqual(train, original)
                    else:
                        self.assertEqual(train['epochs'], 2)

    def test_pure_lovasz_rejects_conflicting_objectives(self):
        cfg, _ = resolve_combination_config('A3-2', 'region_dynamic', 'lovasz')
        for section in ('joint_train', 'refiner_train'):
            for field, value in [('competition_weight', 1.), ('binary_weight', .25),
                                 ('lovasz_weight', .25), ('loss_method', 'ce_lovasz')]:
                with self.subTest(section=section, field=field):
                    current = copy.deepcopy(cfg)
                    current['cx_b']['joint_refiner'] = 'ov_class_spatial'
                    current['cx_b'].setdefault(section, {})[field] = value
                    with self.assertRaises(ConfigurationError):
                        validate_config(current)
        for mode in ('independent', 'concept_prompt'):
            current = copy.deepcopy(cfg)
            current['cx_b']['joint_mode'] = mode
            with self.assertRaisesRegex(ConfigurationError, 'multiclass'):
                validate_config(current)
        with self.assertRaisesRegex(ConfigurationError, 'competition_weight'):
            JointLoss(Mock(), 'multiclass_joint', dict(loss_method='lovasz', competition_weight=1.))

    def test_pure_lovasz_cli_names_output_and_records_actual_refiner_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'best.pth'
            source.write_bytes(b'dry-run only')
            args = ['--prompt_methd=A3-2', '--concept_method=region_dynamic', '--loss_method=lovasz',
                    '--joint-init=checkpoint', '--joint-init-checkpoint='+str(source),
                    '--output-root='+temp, '--dry-run']
            for extra in ([], ['--joint-refiner=ov_class_spatial', '--refiner-epochs=5',
                              '--refiner-patience=3', '--refiner-lr=0.0001']):
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture):
                    self.assertEqual(entry.main(args+extra), 0)
                result = json.loads(capture.getvalue())
                self.assertIn('__l_lovasz__j_multiclass_joint', result['output_dir'])
                self.assertIn('__init_ckpt_', result['output_dir'])
                self.assertIn('seed_3407_', result['output_dir'])
                self.assertFalse(Path(result['output_dir']).exists())
                if extra:
                    self.assertEqual(result['train_scope'], 'refiner_only')
                    actual = result['effective_joint_train']
                    self.assertEqual(actual['loss_method'], 'lovasz')
                    self.assertEqual([actual[k] for k in ('competition_weight', 'binary_weight', 'lovasz_weight')], [0., 0., 1.])

    def test_new_config_retains_a3_warmup_and_selects_new_output(self):
        cfg, path = resolve_combination_config('A3-2', 'region_dynamic', 'ce_lovasz')
        self.assertTrue(path.endswith('p_a3_2__c_region_dynamic__l_ce_lovasz.py'))
        for stage in ('prompt', 'concept'):
            self.assertEqual(build_binary_loss(cfg, stage).method, 'A3-2')
        self.assertEqual(cfg['cx_b']['joint_mode'], 'multiclass_joint')
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            entry.main(['--prompt_methd=A3-2', '--concept_method=region_dynamic',
                        '--loss_method=ce_lovasz', '--dry-run'])
        self.assertIn('__l_ce_lovasz__j_multiclass_joint', json.loads(capture.getvalue())['output_dir'])
        cfg['cx_b']['joint_mode'] = 'independent'
        with self.assertRaisesRegex(ConfigurationError, 'multiclass'):
            validate_config(cfg)

    def test_pure_lovasz_checkpoint_training_and_evaluation_reload(self):
        import eval_cx_b as evaluation
        from test_cx_b_joint_initialization import JointInitializationTests
        fixture = JointInitializationTests()
        with contextlib.redirect_stdout(io.StringIO()):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config['cx_b']['loss_method'] = 'lovasz'
        fixture.config['cx_b']['joint_train'].update(
            loss_method='lovasz', competition_weight=0., binary_weight=0., lovasz_weight=1.)
        inputs = json.loads((fixture.source/'run_config.json').read_text())['inputs']
        original_bytes = fixture.checkpoint.read_bytes()
        source = torch.load(fixture.checkpoint, map_location='cpu')
        for method in ('none', 'ov_class_spatial'):
            with self.subTest(refiner=method):
                fixture.config['cx_b'].update(joint_refiner=method,
                    refiner_train=dict(epochs=1, lr=.01))
                trainer = fixture.trainer('pure_lovasz_'+method)
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(trainer, '_resolved_paths', return_value=(str(fixture.root/'concepts.txt'), str(fixture.root/'cache'))), \
                     patch.object(trainer, 'run_fingerprint', return_value=('pure-'+method, inputs)), \
                     patch.object(trainer, '_build_runtime', return_value=fixture.runtime()), \
                     patch.object(trainer.report, 'binary', side_effect=AssertionError('Must skip independent stages')):
                    trainer.run()
                destination = Path(trainer.output_dir)
                initial = torch.load(destination/'joint/epoch_0000.pth', map_location='cpu')
                self.assertEqual(initial['metrics']['mIoU'], source['metrics']['mIoU'])
                self.assertEqual(fixture.checkpoint.read_bytes(), original_bytes)
                last = torch.load(destination/'joint/last.pth', map_location='cpu')
                if method != 'none':
                    for slot, payload in source['modules'].items():
                        for key, value in payload['state_dict'].items():
                            self.assertTrue(torch.equal(value, last['modules'][slot]['state_dict'][key]))
                    self.assertGreater(last['refiner']['state_dict']['network.4.weight'].abs().sum().item(), 0)
                for checkpoint in ('epoch_0000', 'last', 'best'):
                    loaded = evaluation.load_run(destination, 'cpu', joint_checkpoint=f'joint/{checkpoint}.pth')
                    self.assertEqual(loaded[0]['config']['cx_b']['loss_method'], 'lovasz')

    def test_hand_calculation_and_ignore(self):
        logits = torch.tensor([[[.8, .4, .9]], [[.2, .6, .1]]]).log().requires_grad_()
        target = torch.tensor([[0, 1, -1]])
        loss = lovasz_softmax(logits, target, -1)
        self.assertAlmostEqual(loss.item(), .35, places=6)
        loss.backward()
        self.assertEqual(logits.grad[..., 2].abs().sum().item(), 0)
        self.assertEqual(lovasz_softmax(logits, torch.full_like(target, -1), -1).item(), 0)

    def test_joint_uses_competition_logits_and_changes_gradients(self):
        cfg, _ = resolve_combination_config('A3-2', 'region_dynamic', 'ce_lovasz')
        settings = cfg['cx_b']['joint_train']
        scores = torch.tensor([[[.4, .7, .8]], [[.8, .6, .3]]], requires_grad=True)
        target = torch.tensor([[1, 0, -1]])
        valid = target >= 0
        binary = build_binary_loss(cfg, 'concept')
        old = JointLoss(binary, 'multiclass_joint', dict(settings, loss_method='ce'))(scores, target, valid)
        new = JointLoss(binary, 'multiclass_joint', settings)(scores, target, valid)
        expected = lovasz_softmax(competition_logits(scores), target, -1)
        torch.testing.assert_close(new, old + .25*expected)
        old_grad = torch.autograd.grad(old, scores, retain_graph=True)[0]
        new_grad = torch.autograd.grad(new, scores, retain_graph=True)[0]
        self.assertGreater((new_grad-old_grad).abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(new_grad).all())
        zero = JointLoss(binary, 'multiclass_joint', dict(settings, lovasz_weight=0))(scores, target, valid)
        torch.testing.assert_close(zero, old, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
