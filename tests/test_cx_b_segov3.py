"""Parity against the original source methods, without loading SAM weights."""
import ast
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
import torch.nn.functional as F
from PIL import Image

import eval_cx_b as evaluation
from cx_b.configuration import ConfigurationError, resolve_combination_config
from cx_b.modules import (
    SegOV3PromptAdapter, build_concept_aggregator, load_variant, save_variant,
)
from cx_b.trainer import CXBTrainer, expected_prompt_records
from concept_utils import load_concepts


ROOT = Path(__file__).resolve().parents[1]


def source_methods(filename, class_name, names):
    """Execute the actual source methods; skip heavyweight model constructors."""
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    module = ast.Module(body=methods, type_ignores=[])
    namespace = dict(torch=torch, F=F, Image=Image, PixelData=SimpleNamespace)
    exec(compile(module, filename, 'exec'), namespace)
    return {name: namespace[name] for name in names}


class SegOV3Tests(unittest.TestCase):
    def test_config_contract_and_no_a3_warmstart(self):
        cfg, _ = resolve_combination_config('segov3', 'max', 'A3-2')
        self.assertEqual(cfg['model']['confidence_threshold'], .5)
        self.assertEqual(cfg['model']['prob_thd'], .5)
        self.assertEqual(cfg['model']['bg_idx'], 0)
        self.assertEqual(cfg['model']['slide_crop'], 0)
        self.assertEqual(cfg['cx_b']['cache_dtype'], 'float32')
        self.assertEqual(cfg['cx_b']['prediction_background'], 0)
        for mode in ['a3_warmstart', 'a3_reuse']:
            with self.assertRaises(ConfigurationError):
                resolve_combination_config('segov3', 'max', 'A3-2',
                                           {'cx_b.prompt_init': mode})

    def test_exact_endpoints_batch_and_checkpoint(self):
        module = SegOV3PromptAdapter()
        heads = dict(semantic_score_map=torch.tensor([[[0., 1.]], [[.2, .8]]]),
                     instance_score_map=torch.tensor([[[1., 0.]], [[.9, .1]]]),
                     presence_score=torch.tensor([0., 1.]))
        expected = torch.tensor([[[0., 0.]], [[.9, .8]]])
        self.assertTrue(torch.equal(module(heads), expected))
        self.assertEqual(list(module.parameters()), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pth'
            save_variant(path, module, dict(kind='prompt', method='segov3', base_parameters={}))
            restored, _ = load_variant(path)
            self.assertTrue(torch.equal(restored(heads), expected))

    def test_segov3_panel_uses_original_max_and_presence(self):
        module = SegOV3PromptAdapter()
        heads = dict(semantic_score_map=torch.tensor([[.2, .8]]),
                     instance_score_map=torch.tensor([[.9, .1]]),
                     presence_score=torch.tensor(.5))
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluation, 'save_panel') as panel:
            evaluation.diagnostic_prompt_panel(
                dict(prompt='building', valid_labels=[2]), module, heads, module(heads),
                np.zeros((1, 2, 3), dtype=np.uint8), np.array([[2, 1]]),
                np.ones((1, 2), dtype=bool), Path(directory) / 'panel.png', .5)
            np.testing.assert_allclose(panel.call_args.args[3], [[.45, .05]])
            np.testing.assert_allclose(panel.call_args.args[4], [[.9, .8]])
            np.testing.assert_allclose(panel.call_args.args[5], [[.45, .4]])

    def test_original_scoring_aggregation_and_predict_match(self):
        original = source_methods('segearthov3_segmentor.py', 'SegEarthOV3Segmentation',
                                  ['_inference_single_view', 'aggregate_prompt_scores', 'predict'])
        extract = source_methods('segearthov3_learnable_segmentor.py',
                                 'SegEarthOV3LearnableSegmentation',
                                 ['_extract_prompt_heads_from_state', '_instance_score'])
        torch.manual_seed(19)
        words = ['background', 'building', 'house', 'water']
        states = {}
        for index, word in enumerate(words):
            # Include empty instances, resized masks, zero presence and ties.
            states[word] = dict(masks_logits=torch.rand(index, 1, 3, 4),
                                object_score=torch.rand(index),
                                semantic_mask_logits=torch.rand(1, 1, 3, 4),
                                presence_score=torch.tensor([0., .5, 1., .8][index]))
        processor = SimpleNamespace(
            set_image=lambda image: {}, reset_all_prompts=lambda state: None,
            set_text_prompt=lambda state, prompt: states[prompt])
        runtime = SimpleNamespace(processor=processor, device=torch.device('cpu'),
                                  query_words=words, query_idx=torch.tensor([0, 1, 1, 2]),
                                  num_queries=4, num_cls=3, use_sem_seg=True,
                                  use_transformer_decoder=True, use_presence_score=True,
                                  instance_filter_use_presence=True, slide_crop=0,
                                  prob_thd=.5, bg_idx=0, prompt_aggregation='max')
        for name, method in {**original, **extract}.items():
            setattr(runtime, name, method.__get__(runtime))
        with tempfile.TemporaryDirectory() as directory, patch(
                'torch.autocast', side_effect=lambda **kw: contextlib.nullcontext()):
            path = Path(directory) / 'sample.png'
            Image.new('RGB', (8, 7)).save(path)
            sample = SimpleNamespace(metainfo=dict(img_path=str(path), ori_shape=(7, 8)))
            sample.set_data = lambda values: sample.__dict__.update(values)
            original['predict'](runtime, None, [sample])
            heads = {i: runtime._extract_prompt_heads_from_state({}, i, 8, 7)[1]
                     for i in range(4)}
            records = [dict(prompt_index=i, concept_index=int(runtime.query_idx[i]))
                       for i in range(4)]
            scores, prompts = evaluation.predict_image(
                records, [0, 1, 2], {i: SegOV3PromptAdapter() for i in range(4)},
                {i: build_concept_aggregator('max', int((runtime.query_idx == i).sum()))
                 for i in range(3)}, lambda record: heads[record['prompt_index']])
            np.testing.assert_array_equal(np.stack(list(prompts.values())),
                                          sample.raw_prompt_scores.data.numpy())
            np.testing.assert_array_equal(scores, sample.seg_logits.data.numpy())
            np.testing.assert_array_equal(evaluation.classify_scores(scores, [0, 1, 2], .5, 0),
                                          sample.pred_sem_seg.data[0].numpy())
        # Exact threshold stays foreground; low scores fall back to background.
        np.testing.assert_array_equal(evaluation.classify_scores(
            np.array([[[.1, .1]], [[.5, .49999]]]), [0, 1], .5, 0), [[1, 0]])

    def test_existing_trainer_exports_fixed_prompt_and_composable_concept(self):
        concepts = load_concepts(str(ROOT / 'configs/cls_loveda.txt'))
        records = expected_prompt_records(concepts, 'configs/cls_loveda.txt', 'segov3')
        gt = torch.arange(64).reshape(8, 8) % 8
        sample = dict(data_samples=SimpleNamespace(
            metainfo={'img_path': 'synthetic.png'}, gt_sem_seg=SimpleNamespace(data=gt[None])))
        heads = dict(semantic_score_map=torch.full((8, 8), .8),
                     instance_score_map=torch.full((8, 8), .3), presence_score=torch.tensor(.9))
        provider = SimpleNamespace(load=lambda *args: heads)
        for concept in ['max', 'dynamic']:
            cfg, path = resolve_combination_config('segov3', concept, 'soft_iou_bce')
            cfg['cx_b']['concept_train'].update(epochs=1, validate_every_epochs=1)
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                trainer = CXBTrainer(cfg, path, directory, selected_concepts=[1])
                inputs = dict(concept_content=(ROOT / 'configs/cls_loveda.txt').read_text(),
                              selected_concepts=[1])
                with patch.object(trainer, 'run_fingerprint', return_value=('unit', inputs)), \
                     patch.object(trainer, '_build_runtime', return_value=(
                         SimpleNamespace(device=torch.device('cpu')),
                         [sample] if concept == 'dynamic' else None, [sample], provider)) as runtime:
                    trainer.run()
                self.assertEqual(runtime.call_args.kwargs['include_train'], concept == 'dynamic')
                metrics = json.loads((Path(directory) / 'prompt_metrics.json').read_text())
                self.assertEqual(len(metrics), 11)
                self.assertTrue(all(m['status'] == 'fixed' and m['best_epoch'] == 0 for m in metrics))
                payload = torch.load(Path(directory) / 'resolved_prompt_params.pth')
                self.assertTrue(all(r['parameters'] == {} for r in payload['records']))
                self.assertEqual(payload['records'][0]['prompt_score_version'], 'segov3_max_presence_v1')
                _, _, _, _, loaded, aggregators = evaluation.load_run(directory, 'cpu')
                self.assertEqual(len(loaded), 2)
                self.assertEqual(list(aggregators), [1])
                for index in [1, 2]:
                    self.assertTrue(torch.equal(loaded[index](heads), SegOV3PromptAdapter()(heads)))
                concept_results = json.loads((Path(directory) / 'concept_metrics.json').read_text())
                if concept == 'dynamic':
                    self.assertEqual(len(concept_results[0]['history']), 1)


if __name__ == '__main__':
    unittest.main()
