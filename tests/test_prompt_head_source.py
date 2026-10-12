"""Exercise online Stage A training and the existing explicit-cache path."""

from pathlib import Path
from io import StringIO
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

import train_batch_prompt_params_independent as batch
import train_prompt_params_independent as single


class TinyModel:
    device = torch.device('cpu')
    TRAINABLE_PROMPT_PARAMETER_NAMES = ('offset',)
    query_words = ['building', 'road']
    confidence_threshold = 0.1
    use_sem_seg = True
    use_transformer_decoder = True
    instance_filter_use_presence = True

    def __init__(self):
        self.offsets = [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(2)]
        self.backbone = torch.nn.Parameter(torch.tensor(1.0))
        self.extractions = []
        self.extraction_grad_modes = []

    def extract_prompt_heads_batch(self, image, indices, ori_shape):
        self.extraction_grad_modes.append(torch.is_grad_enabled())
        assert image.mode == 'RGB'
        assert ori_shape == (2, 2)
        self.extractions.append(list(indices))
        return [dict(
            instance_score_map=self.backbone * torch.tensor([[.2, .8], [.7, .1]]),
            semantic_score_map=self.backbone * torch.tensor([[.8, .2], [.6, .4]]),
            presence_score=self.backbone * torch.tensor(.7)) for _ in indices]

    def calibrate_prompt_head_logits(self, index, heads):
        return torch.logit(heads['semantic_score_map']) + self.offsets[index]

    def calibrate_prompt_head_logits_batch(self, indices, heads):
        return torch.stack([
            self.calibrate_prompt_head_logits(
                index, {name: value[i] for name, value in heads.items()})
            for i, index in enumerate(indices)])

    def calibrate_prompt_heads(self, index, heads):
        return self.calibrate_prompt_head_logits(index, heads).sigmoid()

    def initial_prompt_parameter_group(self, index):
        return {'offset': torch.tensor(0.0)}

    def prompt_parameter_group_items(self, index):
        return [('offset', self.offsets[index])]

    def prompt_parameter_group(self, index):
        return [self.offsets[index]]

    def constrain_prompt_parameter_group(self, index, **kwargs):
        pass


def make_sample(tmp_path):
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (2, 2)).save(image_path)
    return SimpleNamespace(
        metainfo={'img_path': str(image_path)},
        gt_sem_seg=SimpleNamespace(data=torch.tensor([[[1, 0], [1, 0]]])))


def arguments(cache=None):
    args = SimpleNamespace(**single.DEFAULTS)
    args.prompt_head_cache = cache
    args.rebuild_cache = False
    args.cache_dtype = 'float32'
    args.ignore_index = 255
    args.progress_interval = 0
    args.val_visualize_every_epochs = 0
    args.prompt_stratified_sampling = False
    args.cache_prefetch_workers = 2
    args.cache_prefetch_depth = 4
    args.pin_memory = True
    args.batch_loss_reduction = 'mean'
    return args


def check_cli_cache_is_optional(entry, test_case):
    with patch.object(sys, 'argv', ['train.py', 'config.py']):
        assert entry.parse_args().prompt_head_cache is None
    with (patch.object(sys, 'argv', ['train.py', 'config.py', '--rebuild-cache']),
          patch.object(sys, 'stderr', StringIO())):
        with test_case.assertRaises(SystemExit):
            entry.parse_args()
    with patch.object(sys, 'argv', [
            'train.py', 'config.py', '--prompt-head-cache=explicit',
            '--rebuild-cache']):
        assert entry.parse_args().prompt_head_cache == 'explicit'


def check_online_never_builds_reads_or_saves_cache(sample, tmp_path):
    model = TinyModel()
    dataset = [{'data_samples': sample}]
    args = arguments()
    before = set(tmp_path.rglob('*'))
    with (patch.object(single, 'build_head_cache', side_effect=AssertionError),
          patch.object(single, 'load_heads_cpu', side_effect=AssertionError),
          patch.object(single.torch, 'save', side_effect=AssertionError),
          patch.object(batch, 'ThreadPoolExecutor', side_effect=AssertionError)):
        assert not single.prepare_head_cache(model, dataset, dataset, [1, 0], args)
        for split in ('train', 'val'):
            loaded = list(batch.prefetched_cached_samples(
                dataset, [0], split, [1, 0], args, None, model))
            heads = loaded[0][3]
            assert list(heads) == [1, 0]
            assert all(not value.requires_grad
                       for head in heads.values() for value in head.values())
        assert model.extractions == [[1, 0], [1, 0]]
        assert model.extraction_grad_modes == [False, False]
    assert set(tmp_path.rglob('*')) == before


def check_online_train_and_val_match_fp32_cache(sample, tmp_path, batched):
    dataset = [{'data_samples': sample}]
    cached_model, online_model = TinyModel(), TinyModel()
    cached_args = arguments(str(tmp_path / 'explicit_cache'))
    assert single.prepare_head_cache(
        cached_model, dataset, dataset, [0, 1], cached_args)
    assert (Path(cached_args.prompt_head_cache) / 'train_manifest.json').exists()
    with patch.object(single, 'build_head_cache', side_effect=AssertionError):
        assert not single.prepare_head_cache(
            cached_model, dataset, dataset, [0, 1], cached_args)
    cached_model.extractions.clear()
    results = []
    for model, args in ((cached_model, cached_args), (online_model, arguments())):
        optimizers = {i: torch.optim.SGD(model.prompt_parameter_group(i), lr=.1)
                      for i in range(2)}
        if batched:
            train = batch.run_prompt_batch_epoch(
                model, dataset, 'train', [0, 1], {0: [1], 1: [0]}, args,
                optimizers=optimizers, epoch=1)
            val = batch.run_prompt_batch_epoch(
                model, dataset, 'val', [0, 1], {0: [1], 1: [0]}, args)
        else:
            train = single.run_prompt_epoch(
                model, dataset, 'train', 0, [1], args,
                optimizer=optimizers[0], epoch=1)
            val = single.run_prompt_epoch(model, dataset, 'val', 0, [1], args)
        assert model.offsets[0].item() != 0
        assert model.backbone.grad is None
        results.append((train, val))
    assert results[0] == results[1]
    assert cached_model.extractions == []
    assert online_model.extractions == ([[0, 1]] * 2 if batched else [[0]] * 2)
    assert online_model.extraction_grad_modes == [False, False]
    for cached, online in zip(cached_model.offsets, online_model.offsets):
        torch.testing.assert_close(cached, online, rtol=0, atol=0)


def check_stratified_sampling_can_extract_online(sample):
    args = arguments()
    args.min_gt_positive_pixels = 1
    args.min_response_positive_pixels = 1
    model = TinyModel()
    groups = single.build_prompt_strata(
        model, [{'data_samples': sample}], 0, [1], args)
    assert groups == {'A': [0], 'B': [], 'C': [], 'D': []}
    assert model.extractions == [[0]]


class PromptHeadSourceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sample = make_sample(self.root)

    def test_single_cli_optional_cache(self):
        check_cli_cache_is_optional(single, self)

    def test_batch_cli_optional_cache(self):
        check_cli_cache_is_optional(batch, self)

    def test_online_without_disk_or_worker_pool(self):
        check_online_never_builds_reads_or_saves_cache(self.sample, self.root)

    def test_single_train_val_and_gradients_match_cache(self):
        check_online_train_and_val_match_fp32_cache(self.sample, self.root, False)

    def test_batch_train_val_and_gradients_match_cache(self):
        check_online_train_and_val_match_fp32_cache(self.sample, self.root, True)

    def test_online_stratified_sampling(self):
        check_stratified_sampling_can_extract_online(self.sample)


if __name__ == '__main__':
    unittest.main()
