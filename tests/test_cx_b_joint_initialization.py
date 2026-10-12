import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import train_batch_prompt_concept_variants as entry
import eval_cx_b as evaluation
from concept_utils import load_concepts
from cx_b.checkpoints import sha256_file
from cx_b.joint import JointPipeline, JointLoss, optimize_joint
from cx_b.joint_initialization import load_joint_initialization
from cx_b.losses import IndependentBinaryLoss
from cx_b.trainer import CXBTrainer, expected_prompt_records
from test_cx_b_reporting import run_training_fixture


class JointInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, config, _, _ = run_training_fixture(self.root, 'multiclass_joint')
        self.config = copy.deepcopy(config)
        self.checkpoint = self.source/'joint/best.pth'
        self.config['cx_b'].update(joint_init='checkpoint', joint_init_checkpoint=str(self.checkpoint))
        self.concepts = load_concepts(str(self.root/'concepts.txt'))
        self.records = expected_prompt_records(self.concepts, str(self.root/'concepts.txt'))

    def trainer(self, name='finetune'):
        return CXBTrainer(self.config, self.root/'concepts.txt', self.root/name, resume=False)

    def load(self, trainer=None):
        trainer = trainer or self.trainer()
        prompts, concepts = trainer._joint_paths(self.concepts, self.records)
        slots = [p.relative_to(trainer.output_dir).as_posix() for p in prompts+concepts]
        return load_joint_initialization(self.checkpoint, self.config, self.records, str(self.root/'cache'), slots)

    def runtime(self):
        raw = torch.tensor(np.asarray(Image.open(self.root/'dataset/gt/sample.png')).copy()).long()
        sample = dict(data_samples=SimpleNamespace(metainfo=dict(img_path=str(self.root/'dataset/img/sample.png')),
                                                  gt_sem_seg=SimpleNamespace(data=raw.unsqueeze(0))))
        heads = {i: torch.load(next((self.root/'cache/val'/f'prompt_{i:03d}').glob('*.pth')))['heads'] for i in range(3)}
        return (SimpleNamespace(device=torch.device('cpu')), [sample], [sample],
                SimpleNamespace(load=lambda split, prompt, index, image: heads[prompt]))

    def test_full_entry_skips_independent_stages_and_exports_evaluable_bundles(self):
        before = sha256_file(self.checkpoint)
        self.config['cx_b']['loss_method'] = 'ce_lovasz'
        self.config['cx_b']['joint_train'].update(loss_method='ce_lovasz', lovasz_weight=.25)
        trainer = self.trainer()
        inputs = json.loads((self.source/'run_config.json').read_text())['inputs']
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
             patch.object(trainer, '_resolved_paths', return_value=(str(self.root/'concepts.txt'), str(self.root/'cache'))), \
             patch.object(trainer, 'run_fingerprint', return_value=('new-joint-run', inputs)), \
             patch.object(trainer, '_build_runtime', return_value=self.runtime()), \
             patch.object(trainer, '_load_base_records', side_effect=AssertionError('must skip scratch initialization')), \
             patch.object(trainer.report, 'binary', side_effect=AssertionError('must skip independent training')):
            result = trainer.run()
        self.assertEqual(result['joint_init'], 'checkpoint')
        self.assertEqual(sha256_file(self.checkpoint), before)
        destination = Path(trainer.output_dir)
        self.assertFalse((destination/'warmup').exists())
        self.assertTrue((destination/'result_fix.xlsx').is_file())
        self.assertEqual(set(trainer.report.plan), {'joint', 'final'})
        self.assertIn('[joint-init-check]', output.getvalue())
        self.assertIn('timestamp=', output.getvalue())
        provenance = json.loads((destination/'joint_initialization.json').read_text())
        self.assertEqual(provenance['sha256'], before)
        self.assertEqual(provenance['optimizer'], 'new')
        source_bundle = torch.load(self.checkpoint)
        initial = torch.load(destination/'joint/epoch_0000.pth')
        self.assertEqual(initial['metrics'], source_bundle['metrics'])
        for slot, payload in source_bundle['modules'].items():
            for key, tensor in payload['state_dict'].items():
                self.assertTrue(torch.equal(tensor, initial['modules'][slot]['state_dict'][key]))
            self.assertEqual(initial['modules'][slot]['metadata']['run_fingerprint'], 'new-joint-run')
        for checkpoint in ('best', 'last', 'epoch_0000'):
            evaluation.load_run(destination, 'cpu', joint_checkpoint=f'joint/{checkpoint}.pth')
        evaluation.load_run(destination, 'cpu')

    def test_rejects_wrong_identity_cache_decision_and_mixed_epoch(self):
        self.load()
        self.config['cx_b']['joint_train']['threshold'] = .3
        with self.assertRaisesRegex(ValueError, 'decision mismatch'):
            self.load()
        self.config['cx_b']['joint_train']['threshold'] = .5
        self.records[0]['prompt'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity/membership'):
            self.load()
        self.records[0]['prompt'] = 'building'
        snapshot_path = self.source/'run_config.json'
        snapshot = json.loads(snapshot_path.read_text())
        snapshot['inputs']['prompt_head_cache'] = str(self.root/'wrong_cache')
        snapshot_path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, 'cache mismatch'):
            self.load()
        snapshot['inputs']['prompt_head_cache'] = str(self.root/'cache')
        snapshot_path.write_text(json.dumps(snapshot))
        bundle = torch.load(self.checkpoint)
        next(iter(bundle['modules'].values()))['metadata']['joint_epoch'] = -1
        torch.save(bundle, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'mixed method/run/epoch'):
            self.load()

    def test_epoch_zero_mismatch_stops_before_training(self):
        loaded, _ = self.load()
        pipeline = JointPipeline([m for m, _ in loaded[:3]], [m for m, _ in loaded[3:]],
                                 self.records, 'multiclass_joint')
        cfg = dict(epochs=1, lr=.001, expected_initial_metrics=dict(mIoU=.9, valid_pixels=10))
        forbidden = lambda: self.fail('No training should occur')
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('cx_b.joint.evaluate_joint', return_value=dict(mIoU=.7, valid_pixels=10)), \
             self.assertRaisesRegex(ValueError, 'epoch 0'):
            optimize_joint(pipeline, forbidden, forbidden, JointLoss(IndependentBinaryLoss(), pipeline.mode, cfg), cfg)

    def test_cli_isolates_finetuning_and_keeps_scratch_path(self):
        args = ['--prompt_methd=A3-2', '--concept_method=dynamic', '--loss_method=ce_lovasz',
                '--joint-mode=multiclass_joint', '--output-root='+str(self.root/'outputs'), '--dry-run']
        def run(extra):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(entry.main(args+extra), 0)
            return json.loads(captured.getvalue())
        scratch = run([])
        self.assertNotIn('__init_ckpt_', scratch['output_dir'])
        self.assertTrue(scratch['output_dir'].endswith('seed_3407'))
        flags = ['--joint-init=checkpoint', '--joint-init-checkpoint='+str(self.checkpoint)]
        first = run(flags)
        self.assertIn('__init_ckpt_', first['output_dir'])
        self.assertEqual(first['config']['cx_b']['joint_init_sha256'], sha256_file(self.checkpoint))
        Path(first['output_dir']).mkdir(parents=True)
        self.assertNotEqual(run(flags)['output_dir'], first['output_dir'])
        with self.assertRaises(SystemExit):
            run(['--joint-init=checkpoint'])
        with self.assertRaises(SystemExit):
            run(['--joint-init-checkpoint='+str(self.checkpoint)])
        with self.assertRaisesRegex(ValueError, 'outside the source run'):
            CXBTrainer(self.config, self.root/'concepts.txt', self.source)


if __name__ == '__main__':
    unittest.main()
