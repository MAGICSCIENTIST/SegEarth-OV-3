import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import train_learnable_params as shared
from diagnose_sam3_heads import render_binary_overlay, save_normal_panel
from segearthov3_learnable_segmentor import (
    IDENTIFIABLE_PARAMETER_NAMES,
    PromptCalibrationBank,
    legacy_to_identifiable,
)
from train_prompt_params_independent import (
    DEFAULTS,
    adjusted_rand_index,
    concept_ignore_index,
    concept_valid_labels,
    parse_args,
    resolve_identity_fallback,
    select_best_validation,
    stratified_epoch_indices,
)


def validation_row(epoch, hard_iou, recall=0.5):
    return dict(
        epoch=epoch,
        validation=dict(hard_iou=hard_iou, recall=recall, loss=1.0),
        optimization_parameters={'value': float(epoch)})


def test_hard_iou_checkpoint_selection():
    history = [
        validation_row(0, 0.20),
        validation_row(1, 0.19),
        validation_row(2, 0.21),
        validation_row(3, 0.21),
    ]
    selected, score = select_best_validation(history, 'hard_iou')
    assert selected['epoch'] == 2
    assert score == 0.21


def test_identity_fallback_at_min_delta():
    baseline = validation_row(0, 0.20)
    raw_best = validation_row(4, 0.205)
    selected, applied, reason = resolve_identity_fallback(
        baseline, raw_best, True, 0.005)
    assert selected['epoch'] == 0
    assert applied
    assert reason == 'gain_below_or_equal_min_delta'


def test_default_reparameterization_equivalence():
    converted = legacy_to_identifiable(0.5, 1.0, 0.5, 0.0)
    assert torch.allclose(torch.sigmoid(converted['raw_semantic_weight']),
                          torch.tensor(0.5), atol=1e-7)
    assert torch.allclose(torch.nn.functional.softplus(
        converted['raw_presence_gain']), torch.tensor(1.0), atol=1e-7)
    assert torch.allclose(converted['presence_intercept'], torch.tensor(0.0))


def test_legacy_checkpoint_migration_equivalence():
    semantic = torch.tensor(0.37)
    gain = torch.tensor(1.7)
    bias = torch.tensor(0.23)
    log_weight = torch.tensor(-0.2)
    converted = legacy_to_identifiable(
        semantic, gain, bias, log_weight)
    presence = torch.linspace(0.0, 1.0, 17)
    old_adjustment = gain * (presence - bias) + log_weight
    new_gain = torch.nn.functional.softplus(converted['raw_presence_gain'])
    new_adjustment = (
        new_gain * (presence - 0.5) + converted['presence_intercept'])
    assert torch.allclose(old_adjustment, new_adjustment, atol=1e-6)


def test_optimizer_contains_only_three_prompt_parameters():
    vectors = {
        'raw_semantic_weight': torch.zeros(2),
        'raw_presence_gain': torch.full((2,), 0.54132485),
        'presence_intercept': torch.zeros(2),
    }
    bank = PromptCalibrationBank(vectors, IDENTIFIABLE_PARAMETER_NAMES)
    names = [name for name, _ in bank.parameter_group_items(0)]
    assert tuple(names) == IDENTIFIABLE_PARAMETER_NAMES
    assert 'presence_bias' not in names
    assert 'log_prompt_weight' not in names
    prompt = bank.prompts[0]
    semantic = torch.sigmoid(prompt.raw_semantic_weight)
    gain = torch.nn.functional.softplus(prompt.raw_presence_gain)
    logits = (
        semantic * torch.tensor(0.8)
        + (1.0 - semantic) * torch.tensor(0.2)
        + gain * (torch.tensor(0.7) - 0.5)
        + prompt.presence_intercept)
    logits.square().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad)
        for parameter in bank.parameter_group(0))


def test_prompt_sampler_seed_reproducibility_and_train_only():
    groups = {'A': [0, 1], 'B': [2], 'C': [3, 4], 'D': [5]}
    first, counts = stratified_epoch_indices(
        groups, (4, 2, 3, 1), 100, seed=12)
    second, second_counts = stratified_epoch_indices(
        groups, (4, 2, 3, 1), 100, seed=12)
    different, _ = stratified_epoch_indices(
        groups, (4, 2, 3, 1), 100, seed=13)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert counts == second_counts == {'A': 40, 'B': 20, 'C': 30, 'D': 10}
    assert set(first).issubset({0, 1, 2, 3, 4, 5})


def test_prompt_sampler_renormalizes_empty_groups():
    groups = {'A': [0], 'B': [], 'C': [1], 'D': []}
    indices, counts = stratified_epoch_indices(
        groups, (4, 2, 3, 1), 70, seed=1)
    assert len(indices) == 70
    assert counts == {'A': 40, 'B': 0, 'C': 30, 'D': 0}


def test_adjusted_rand_index_is_permutation_invariant():
    assert adjusted_rand_index([0, 0, 1, 1], [7, 7, 3, 3]) == 1.0


def test_json_concept_uses_tnls_labels_and_configured_ignore_index():
    concept = {'tnls_dl3': [9, '10', 11]}
    path = 'configs/concepts.json'
    assert concept_valid_labels(concept, 3, path) == [9, 10, 11]
    assert concept_ignore_index(path, 255) == 255


def test_txt_concept_uses_one_based_line_label_and_zero_nodata():
    concept = {'tnls_dl3': [], 'prompts': ['building', 'house']}
    path = 'configs/cls_loveda.txt'
    assert concept_valid_labels(concept, 0, path) == [1]
    assert concept_valid_labels(concept, 1, path) == [2]
    assert concept_ignore_index(path, 255) == 0


def test_visual_type_defaults_to_full_panel_and_accepts_normal():
    assert DEFAULTS['visual_type'] == 'full_panel'
    argv = [
        'train_prompt_params_independent.py',
        'configs/demo.py',
        '--prompt-head-cache=cache',
        '--visual-typ=normal',
    ]
    with patch.object(sys, 'argv', argv):
        assert parse_args().visual_type == 'normal'


def test_normal_panel_uses_existing_overlay_colors(tmp_path):
    image = Image.new('RGB', (4, 4), color=(100, 100, 100))
    gt_mask = np.zeros((4, 4), dtype=bool)
    pred_mask = np.zeros((4, 4), dtype=bool)
    gt_mask[0, 0] = True
    pred_mask[0, 1] = True

    gt_overlay = render_binary_overlay(image, gt_mask, (255, 64, 64))
    pred_overlay = render_binary_overlay(image, pred_mask, (64, 255, 96))
    assert tuple(gt_overlay[0, 0]) == (185, 80, 80)
    assert tuple(pred_overlay[0, 1]) == (80, 185, 97)

    output_path = tmp_path / 'normal.png'
    save_normal_panel(
        image, 'demo | epoch 0', output_path, gt_mask, pred_mask)
    assert output_path.is_file()
    with Image.open(output_path) as panel:
        assert panel.width > panel.height


class FakeClusterModel:
    TRAINABLE_PROMPT_PARAMETER_NAMES = IDENTIFIABLE_PARAMETER_NAMES
    prompt_parameterization = 'identifiable_three'
    num_queries = 4
    concepts = [dict(concept_name='demo')]
    query_idx = torch.tensor([0, 0, 0, 0])
    semantic_weight = torch.tensor([0.4, 0.5, 0.6, 0.7])
    presence_gain = torch.tensor([0.8, 1.0, 1.2, 1.4])
    presence_bias = torch.tensor([0.5, 0.5, 0.5, 0.5])
    log_prompt_weight = torch.tensor([-0.2, 0.0, 0.2, 0.4])

    def prompt_parameter_items(self):
        return [
            ('raw_semantic_weight', torch.zeros(4)),
            ('raw_presence_gain', torch.ones(4)),
            ('presence_intercept', self.log_prompt_weight),
        ]

    def _prompt_keys(self):
        return [(f'p{index}', 'demo', index) for index in range(4)]


def test_cluster_accepts_only_prototype_prompts(tmp_path):
    records = []
    for index, exit_name in enumerate(
            ['prototype', 'prototype', 'prototype', 'identity_fallback']):
        records.append(dict(
            prompt_index=index,
            prompt_exit=exit_name,
            hard_iou_gain_vs_epoch0=0.02 if index < 3 else 0.0,
            baseline_metrics=dict(
                precision=0.4, recall=0.4, pred_gt_area_ratio=1.0),
            selected_metrics=dict(
                precision=0.5, recall=0.5, pred_gt_area_ratio=1.1)))
    path = tmp_path / 'clusters.json'
    shared.analyze_prompt_parameter_clusters(
        FakeClusterModel(),
        path,
        records=records,
        min_hard_iou_gain=0.005,
        min_cluster_size=3,
        max_clusters=8)
    analysis = json.loads(path.read_text(encoding='utf-8'))
    assert [item['prompt_index'] for item in analysis['prompts']] == [0, 1, 2]
    assert analysis['excluded_prompts']['identity_fallback'] == [3]


class StageAPromptTrainingTests(unittest.TestCase):
    def test_checkpoint_selection(self):
        test_hard_iou_checkpoint_selection()

    def test_fallback_boundary(self):
        test_identity_fallback_at_min_delta()

    def test_default_equivalence(self):
        test_default_reparameterization_equivalence()

    def test_migration_equivalence(self):
        test_legacy_checkpoint_migration_equivalence()

    def test_three_parameter_optimizer(self):
        test_optimizer_contains_only_three_prompt_parameters()

    def test_sampler_reproducibility(self):
        test_prompt_sampler_seed_reproducibility_and_train_only()

    def test_sampler_empty_group(self):
        test_prompt_sampler_renormalizes_empty_groups()

    def test_ari(self):
        test_adjusted_rand_index_is_permutation_invariant()

    def test_json_label_mapping(self):
        test_json_concept_uses_tnls_labels_and_configured_ignore_index()

    def test_txt_label_mapping(self):
        test_txt_concept_uses_one_based_line_label_and_zero_nodata()

    def test_visual_type(self):
        test_visual_type_defaults_to_full_panel_and_accepts_normal()

    def test_normal_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            test_normal_panel_uses_existing_overlay_colors(Path(directory))

    def test_prototype_only_clustering(self):
        with tempfile.TemporaryDirectory() as directory:
            test_cluster_accepts_only_prototype_prompts(Path(directory))


if __name__ == '__main__':
    unittest.main()
