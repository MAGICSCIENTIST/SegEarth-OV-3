import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import eval_batch_cx_b as batch
import eval_cx_b as evaluation
import train_batch_prompt_concept_variants as entry
from cx_b.configuration import (JOINT_MODES, COMBINATION_CONFIGS,
                                resolve_combination_config, validate_config, ConfigurationError)
from cx_b.joint import (JointPipeline, JointLoss, evaluate_joint, final_prediction,
                        map_joint_target, optimize_joint, confusion_metrics)
from cx_b.losses import IndependentBinaryLoss, build_binary_loss
from cx_b.modules import A3PromptAdapter, SegOV3PromptAdapter, build_concept_aggregator, build_prompt_module
from cx_b.trainer import CXBTrainer
import test_eval_batch_cx_b as fixtures


class JointTests(unittest.TestCase):
    def pipeline(self, mode, fixed=False):
        torch.manual_seed(12)
        records = [dict(prompt_index=i, concept_index=i//2) for i in range(4)]
        prompts = [SegOV3PromptAdapter() if fixed else A3PromptAdapter() for _ in records]
        concepts = [build_concept_aggregator('max' if fixed else 'dynamic', 2) for _ in range(2)]
        model = JointPipeline(prompts, concepts, records, mode)
        heads = {i: dict(semantic_score_map=torch.rand(5, 6)*.8+.1,
                         instance_score_map=torch.rand(5, 6)*.8+.1,
                         presence_score=torch.tensor(.8)) for i in range(4)}
        target = torch.zeros((5, 6), dtype=torch.long)
        target[:, 3:] = 1
        return model, heads, target

    def test_cli_modes_preserve_method_registry_and_isolate_outputs(self):
        for mode in JOINT_MODES:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                self.assertEqual(entry.main(['--prompt_methd=A3-2', '--concept_method=dynamic',
                    '--loss_method=soft_iou_bce', '--joint-mode='+mode,
                    '--joint-epochs=3', '--joint-patience=2', '--joint-lr=.002', '--dry-run']), 0)
            result = json.loads(capture.getvalue())
            self.assertEqual(result['joint_mode'], mode)
            self.assertEqual(result['config']['cx_b']['joint_train']['epochs'], 3)
            self.assertEqual('__j_' in result['combination'], mode != 'independent')
            if mode != 'independent':
                self.assertEqual(result['config']['model']['prob_thd'], .5)
                self.assertEqual(result['config']['cx_b']['prediction_background'], 0)
        cfg, _ = resolve_combination_config('A3-2', 'dynamic', 'A3-2')
        self.assertEqual(cfg['cx_b']['joint_mode'], 'independent')
        cfg['cx_b']['joint_mode'] = 'multiclass_joint'
        cfg['cx_b']['joint_train']['epochs'] = -1
        with self.assertRaises(ConfigurationError):
            validate_config(cfg)

    def test_gradient_updates_obey_each_mode(self):
        for mode in JOINT_MODES[1:]:
            model, heads, target = self.pipeline(mode)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
            for method in ('A3-2', 'soft_iou_bce', 'bce_tversky'):
                optimizer.zero_grad()
                loss = JointLoss(IndependentBinaryLoss(method), mode, {})(model(heads), target, target >= 0)
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            changed = {group: any(not torch.equal(before[k], v) for k, v in model.state_dict().items()
                                  if k.startswith(group+'.')) for group in ('prompts', 'concepts')}
            self.assertEqual(changed['prompts'], mode != 'multiclass_concept', mode)
            self.assertEqual(changed['concepts'], mode != 'multiclass_prompt', mode)

    def test_competition_changes_gradient_when_other_class_changes(self):
        target = torch.tensor([[0]])
        valid = torch.ones_like(target, dtype=torch.bool)
        gradients = {}
        for mode in ('concept_prompt', 'multiclass_joint'):
            gradients[mode] = []
            for competitor in (.6, .9):
                scores = torch.tensor([[[.7]], [[competitor]]], requires_grad=True)
                JointLoss(IndependentBinaryLoss(), mode, {})(scores, target, valid).backward()
                gradients[mode].append(scores.grad[0].item())
        self.assertEqual(*gradients['concept_prompt'])
        self.assertNotEqual(*gradients['multiclass_joint'])

    def test_all_existing_method_combinations_support_joint_forward_and_backward(self):
        _, heads, target = self.pipeline('multiclass_joint')
        records = [dict(prompt_index=i, concept_index=i//2) for i in range(4)]
        regions = {i: (torch.ones(1, 5, 6, dtype=torch.bool), torch.tensor([.8])) for i in range(4)}
        for prompt, concept, loss in COMBINATION_CONFIGS:
            # Frozen A3/dynamic bases exercise promotion of reusable buffers too.
            prompts = [build_prompt_module(prompt, {}, trainable=False, base_trainable=False) for _ in records]
            model = JointPipeline(prompts, [build_concept_aggregator(concept, 2) for _ in range(2)],
                                  records, 'multiclass_joint')
            config, _ = resolve_combination_config(prompt, concept, loss)
            objective = JointLoss(build_binary_loss(config, 'concept'), 'multiclass_joint', config['cx_b']['joint_train'])(
                model(heads, regions), target, target >= 0)
            self.assertTrue(torch.isfinite(objective), (prompt, concept, loss))
            parameters = list(model.parameters())
            if parameters:
                objective.backward()
                self.assertTrue(any(p.grad is not None for p in parameters), (prompt, concept, loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None))

    def test_hard_validation_matches_batch_eval_including_threshold_ties(self):
        scores = torch.tensor([[[.1, .5, .2, .7]], [[.49, .5, .5, .8]], [[.2, .1, .1, .1]]])
        pred = final_prediction(scores)
        np.testing.assert_array_equal(pred.numpy(), evaluation.classify_scores(scores.numpy(), [0, 1, 2], .5, 0))
        gt = np.array([[0, 1, 1, 0]])
        confusion = np.zeros((3, 4), dtype=np.int64)
        batch.add_confusion(confusion, pred.numpy(), gt, np.ones_like(gt, bool))
        reference = batch.summarize_confusion(confusion, ['bg', 'a', 'b'])['summary']
        metrics = confusion_metrics(confusion[:, :3])
        for key, value in reference.items():
            if value is None:
                self.assertIsNone(metrics[key])
            else:
                self.assertAlmostEqual(100*metrics[key], value)
        mapped, valid = map_joint_target(torch.tensor([[0, 1, 2]]), [[1], [2]], 0)
        self.assertEqual(mapped.tolist(), [[-1, 0, 1]])
        self.assertEqual(valid.tolist(), [[False, True, True]])
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            map_joint_target(torch.tensor([[1]]), [[1], [1]], 0)
        with self.assertRaisesRegex(ValueError, 'not covered'):
            map_joint_target(torch.tensor([[3]]), [[1], [2]], 0)

    def test_fixed_segov3_max_keeps_epoch_zero_and_does_not_train(self):
        model, heads, target = self.pipeline('multiclass_joint', fixed=True)
        samples = lambda: iter([(heads, None, target, target >= 0)])
        cfg = dict(epochs=20, lr=.001, patience=7)
        expected = model(heads).detach().clone()
        def forbidden():
            raise AssertionError('Fixed pipeline must never request training data')
        result = optimize_joint(model, forbidden, samples, JointLoss(IndependentBinaryLoss(), model.mode, cfg), cfg)
        self.assertEqual(result['status'], 'fixed')
        self.assertEqual(result['best_epoch'], 0)
        self.assertTrue(torch.equal(model(heads), expected))

    def test_best_restores_whole_pipeline_and_last_is_actual_last(self):
        model, heads, target = self.pipeline('multiclass_joint')
        samples = lambda: iter([(heads, None, target, target >= 0)])
        cfg = dict(epochs=3, lr=.02, patience=0, checkpoint_interval_epochs=1)
        snapshots = {}
        def save(name, epoch, pipeline, optimizer, metrics):
            snapshots[name] = {k: v.detach().clone() for k, v in pipeline.state_dict().items()}
        metrics = [dict(mIoU=x) for x in (.2, .4, .8, .3)]
        with patch('cx_b.joint.evaluate_joint', side_effect=metrics):
            result = optimize_joint(model, samples, samples, JointLoss(IndependentBinaryLoss(), model.mode, cfg), cfg, save)
        self.assertEqual(result['best_epoch'], 2)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, snapshots['epoch_0002'][key]))
            self.assertTrue(torch.equal(snapshots['last'][key], snapshots['epoch_0003'][key]))
        self.assertTrue(any(not torch.equal(snapshots['best'][k], snapshots['last'][k]) for k in snapshots['best']))

    def test_joint_stage_exports_loadable_coherent_checkpoints_and_batch_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = fixtures.BatchEvaluationTests().fixture(root)
            snapshot = batch.read_json(directory/'run_config.json')
            config = snapshot['config']
            cx = config['cx_b']
            cx.update(joint_mode='multiclass_joint', metric_threshold=.5,
                      joint_train=dict(epochs=2, lr=.01, patience=0, threshold=.5, background=0,
                                       checkpoint_interval_epochs=1))
            config['model']['prob_thd'] = .5
            cx['prediction_background'] = 0
            (directory/'run_config.json').write_text(json.dumps(snapshot))
            run, concepts, _, records, _, _ = evaluation.load_run(directory, 'cpu')
            for name in ('prompt_metrics.json', 'concept_metrics.json', 'summary_metrics.json'):
                (directory/name).write_text('{}')
            for name in ('prompt_metrics.csv', 'concept_metrics.csv', 'results.xlsx', 'results.txt'):
                (directory/name).write_text('warmup')
            torch.save(dict(records=[dict(r) for r in records]), directory/'resolved_prompt_params.pth')
            trainer = CXBTrainer(config, __file__, directory)
            args = batch.parse_args(['--root', str(directory), '--device=cpu'])
            cache = evaluation.SavedHeadCache(root/'cache', 'val', config['model'], 'cpu')
            image = root/'dataset/img/sample.png'
            heads = {r['prompt_index']: cache.load(r, 0, image) for r in records}
            raw = torch.tensor(np.asarray(evaluation.Image.open(root/'dataset/gt/sample.png')).copy()).long()
            target, valid = map_joint_target(raw, [[1], [2]], 0)
            samples = lambda: iter([(heads, None, target, valid)])
            with patch.object(trainer, '_joint_samples', return_value=samples):
                result = trainer._run_joint_stage(concepts, records, SimpleNamespace(device=torch.device('cpu')),
                                                  None, None, None, snapshot['fingerprint'])
            self.assertEqual(result['summary']['mIoU'], 100.)
            self.assertTrue((directory/'warmup/concept_metrics.json').is_file())
            self.assertEqual(batch.read_json(directory/'summary_metrics.json')['metric_scope'], 'final_multiclass')
            for name in ('best', 'last', 'epoch_0000', 'epoch_0001', 'epoch_0002'):
                loaded = evaluation.load_run(directory, 'cpu', joint_checkpoint=f'joint/{name}.pth')
                self.assertEqual(len(loaded[4]), 3)
            direct = batch.evaluate_run(directory, args)['result']
            self.assertAlmostEqual(direct['metrics']['multiclass_common']['summary']['mIoU'], result['summary']['mIoU'])
            scanned = batch.scan_checkpoints(directory, args)
            self.assertEqual(scanned['checkpoint_count'], 5)
            self.assertEqual(scanned['selected_mIoU'], 100.)
            self.assertEqual(scanned['selected_delta_vs_original_best_mIoU'], 0.)

    def test_full_training_entry_keeps_warmup_and_finishes_joint_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = fixtures.BatchEvaluationTests().fixture(root)
            snapshot = batch.read_json(directory/'run_config.json')
            config = snapshot['config']
            common, _ = resolve_combination_config('A3-2', 'dynamic', 'A3-2')
            config['cx_b'] = common['cx_b']
            cx = config['cx_b']
            cx.update(joint_mode='multiclass_joint', prediction_background=0)
            for stage in ('prompt_train', 'concept_train', 'joint_train'):
                cx[stage].update(epochs=1, validate_every_epochs=1, checkpoint_interval_epochs=1)
            config['model']['prob_thd'] = .5
            concepts_file = root/'concepts.txt'
            concepts_file.write_text(snapshot['inputs']['concept_content'])
            raw = torch.tensor(np.asarray(evaluation.Image.open(root/'dataset/gt/sample.png')).copy()).long()
            sample = dict(data_samples=SimpleNamespace(
                metainfo=dict(img_path=str(root/'dataset/img/sample.png')),
                gt_sem_seg=SimpleNamespace(data=raw.unsqueeze(0))))
            heads = {i: torch.load(next((root/'cache/val'/f'prompt_{i:03d}').glob('*.pth')))['heads'] for i in range(3)}
            provider = SimpleNamespace(load=lambda split, prompt, index, image: heads[prompt])
            model = SimpleNamespace(device=torch.device('cpu'))
            trainer = CXBTrainer(config, concepts_file, directory, resume=False)
            with patch.object(trainer, '_resolved_paths', return_value=(str(concepts_file), str(root/'cache'))), \
                 patch.object(trainer, 'run_fingerprint', return_value=('tiny_joint_run', snapshot['inputs'])), \
                 patch.object(trainer, '_build_runtime', return_value=(model, [sample], [sample], provider)):
                result = trainer.run()
            self.assertEqual(result['joint_mode'], 'multiclass_joint')
            self.assertEqual(batch.read_json(directory/'completed.json')['status'], 'complete')
            self.assertTrue((directory/'joint/best.pth').exists())
            self.assertTrue((directory/'warmup/summary_metrics.json').exists())
            self.assertTrue((directory/'log_fix.txt').exists())
            self.assertTrue((directory/'val_fix.txt').exists())
            self.assertEqual([p.name for p in directory.rglob('*.xlsx')], ['result_fix.xlsx'])
            summary = batch.read_json(directory/'summary_metrics.json')
            self.assertEqual(summary['metric_scope'], 'final_multiclass')
            args = batch.parse_args(['--root', str(directory), '--device=cpu'])
            evaluated = batch.evaluate_run(directory, args)['result']
            self.assertAlmostEqual(evaluated['metrics']['multiclass_common']['summary']['mIoU'],
                                   summary['metrics']['mIoU'])


if __name__ == '__main__':
    unittest.main()
