from types import SimpleNamespace
import unittest

import torch

from train_batch_prompt_params_independent import (
    apply_prompt_batch_update,
    empty_device_metric_state,
    finalize_device_metric_states,
    prompt_batch_outputs,
    prompt_batches,
    reduce_prompt_losses,
    stack_prompt_heads,
    update_device_metric_state,
)
from segearthov3_learnable_segmentor import SegEarthOV3LearnableSegmentation


class FakeCalibrationModel:
    device = torch.device('cpu')
    fusion_mode = 'weighted_sum'
    use_presence_score = True
    prompt_parameterization = 'identifiable_three'
    probability_to_logits = staticmethod(
        SegEarthOV3LearnableSegmentation.probability_to_logits)
    _fuse_heads = SegEarthOV3LearnableSegmentation._fuse_heads
    calibrate_prompt_head_logits = (
        SegEarthOV3LearnableSegmentation.calibrate_prompt_head_logits)
    calibrate_prompt_head_logits_batch = (
        SegEarthOV3LearnableSegmentation.calibrate_prompt_head_logits_batch)

    def _bounded_prompt_values(self):
        return {
            'semantic_weight': torch.tensor([0.2, 0.7, 0.4]),
            'presence_gain': torch.tensor([0.8, 1.2, 0.5]),
            'presence_bias': torch.tensor([0.5, 0.5, 0.5]),
            'log_prompt_weight': torch.tensor([-0.1, 0.2, 0.0]),
        }

    def initial_prompt_parameter_group(self, prompt_idx):
        return {'value': torch.tensor(0.0)}

    def prompt_parameter_group_items(self, prompt_idx):
        return [('value', torch.tensor(0.0))]


class FakeBatchModel:
    def __init__(self):
        self.parameters_by_prompt = [
            torch.nn.Parameter(torch.tensor(1.0)),
            torch.nn.Parameter(torch.tensor(1.0)),
        ]

    def prompt_parameter_group(self, prompt_idx):
        return [self.parameters_by_prompt[prompt_idx]]

    def prompt_parameter_group_items(self, prompt_idx):
        return [('value', self.parameters_by_prompt[prompt_idx])]

    def constrain_prompt_parameter_group(self, prompt_idx, **kwargs):
        return None


class BatchPromptTrainingTests(unittest.TestCase):
    def test_prompt_batches(self):
        self.assertEqual(
            prompt_batches([0, 1, 2, 3, 4], 2),
            [[0, 1], [2, 3], [4]])

    def test_mean_is_default_reduction_behavior(self):
        losses = [torch.tensor(2.0), torch.tensor(4.0)]
        self.assertEqual(float(reduce_prompt_losses(losses, 'mean')), 3.0)

    def test_gradient_clipping_is_per_prompt(self):
        model = FakeBatchModel()
        optimizers = {
            index: torch.optim.SGD(model.prompt_parameter_group(index), lr=0.0)
            for index in range(2)
        }
        args = SimpleNamespace(
            batch_loss_reduction='mean',
            grad_clip_norm=1.0,
            presence_gain_max=8.0,
            prompt_weight_max=3.0)
        first, second = model.parameters_by_prompt
        outputs = {
            0: (100.0 * first, None, None, None),
            1: (second, None, None, None),
        }

        apply_prompt_batch_update(model, outputs, optimizers, args)

        self.assertAlmostEqual(float(first.grad), 1.0, places=5)
        self.assertAlmostEqual(float(second.grad), 0.5, places=5)

    def test_prompt_heads_are_stacked_in_requested_order(self):
        heads = {
            3: {
                'instance_score_map': torch.full((2, 2), 3.0),
                'semantic_score_map': torch.full((2, 2), 30.0),
                'presence_score': torch.tensor(0.3),
            },
            1: {
                'instance_score_map': torch.full((2, 2), 1.0),
                'semantic_score_map': torch.full((2, 2), 10.0),
                'presence_score': torch.tensor(0.1),
            },
        }

        stacked = stack_prompt_heads(heads, [3, 1])

        self.assertEqual(tuple(stacked['instance_score_map'].shape), (2, 2, 2))
        self.assertTrue(torch.equal(
            stacked['instance_score_map'][:, 0, 0],
            torch.tensor([3.0, 1.0])))
        self.assertTrue(torch.equal(
            stacked['presence_score'], torch.tensor([0.3, 0.1])))

    def test_batch_calibration_matches_individual_calibration(self):
        model = FakeCalibrationModel()
        heads = {
            2: {
                'instance_score_map': torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
                'semantic_score_map': torch.tensor([[0.3, 0.5], [0.7, 0.9]]),
                'presence_score': torch.tensor(0.4),
            },
            0: {
                'instance_score_map': torch.tensor([[0.1, 0.3], [0.5, 0.7]]),
                'semantic_score_map': torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
                'presence_score': torch.tensor(0.6),
            },
        }
        prompt_indices = [2, 0]
        stacked = stack_prompt_heads(heads, prompt_indices)

        batch_logits = model.calibrate_prompt_head_logits_batch(
            prompt_indices, stacked)
        individual_logits = torch.stack([
            model.calibrate_prompt_head_logits(prompt_idx, heads[prompt_idx])
            for prompt_idx in prompt_indices
        ])

        self.assertTrue(torch.allclose(
            batch_logits, individual_logits, atol=1e-6, rtol=1e-6))

    def test_batch_losses_match_individual_losses(self):
        model = FakeCalibrationModel()
        prompt_indices = [2, 0]
        heads = {
            2: {
                'instance_score_map': torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
                'semantic_score_map': torch.tensor([[0.3, 0.5], [0.7, 0.9]]),
                'presence_score': torch.tensor(0.4),
            },
            0: {
                'instance_score_map': torch.tensor([[0.1, 0.3], [0.5, 0.7]]),
                'semantic_score_map': torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
                'presence_score': torch.tensor(0.6),
            },
        }
        targets = {
            2: torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            0: torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        }
        valid_masks = {
            2: torch.tensor([[True, True], [True, False]]),
            0: torch.tensor([[True, True], [True, False]]),
        }
        args = SimpleNamespace(
            max_positive_weight=20.0,
            recall_target=0.3,
            tversky_alpha_fp=0.6,
            tversky_beta_fn=0.4,
            loss_name='soft_iou',
            bce_loss_weight=0.2,
            soft_iou_loss_weight=1.0,
            recall_loss_weight=0.1,
            param_reg_weight=0.01)
        stacked = stack_prompt_heads(heads, prompt_indices)
        logits = model.calibrate_prompt_head_logits_batch(
            prompt_indices, stacked)

        batch_outputs = prompt_batch_outputs(
            model, prompt_indices, logits, targets, valid_masks, args)

        for batch_idx, prompt_idx in enumerate(prompt_indices):
            individual = __import__(
                'train_prompt_params_independent').prompt_loss(
                    model,
                    prompt_idx,
                    heads[prompt_idx],
                    targets[prompt_idx],
                    valid_masks[prompt_idx],
                    args,
                    logits=logits[batch_idx])
            self.assertTrue(torch.allclose(
                batch_outputs[prompt_idx][0], individual[0],
                atol=1e-6, rtol=1e-6))
            for name, value in batch_outputs[prompt_idx][3].items():
                self.assertAlmostEqual(
                    float(value), float(individual[3][name]), places=6)

    def test_device_metrics_match_original_metrics(self):
        class MetricModel:
            device = torch.device('cpu')
            TRAINABLE_PROMPT_PARAMETER_NAMES = ('value',)

        probabilities = torch.tensor([0.8, 0.4, 0.7, 0.1])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        components = {
            'bce': torch.tensor(0.2),
            'soft_iou': torch.tensor(0.3),
            'tversky': torch.tensor(0.4),
            'recall': torch.tensor(0.1),
            'regularization': torch.tensor(0.0),
        }
        state = empty_device_metric_state(MetricModel())
        update_device_metric_state(
            state,
            torch.tensor(0.5),
            probabilities,
            target,
            0.5,
            components,
            gradients={'value': torch.tensor(0.25)})

        result = finalize_device_metric_states({7: state})[7]

        self.assertEqual(result['tp'], 2)
        self.assertEqual(result['fp'], 0)
        self.assertEqual(result['fn'], 0)
        self.assertEqual(result['tn'], 2)
        self.assertAlmostEqual(result['loss'], 0.5)
        self.assertAlmostEqual(result['loss_components']['bce'], 0.2, places=6)
        self.assertAlmostEqual(
            result['gradients']['value'], 0.25, places=6)


if __name__ == '__main__':
    unittest.main()
