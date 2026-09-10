import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

if os.environ.get('SEGOV3_LIGHTWEIGHT_TRAIN') == '1':
    BaseSegmentor = nn.Module

    class PixelData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _DummyRegistry:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator

    MODELS = _DummyRegistry()
else:
    from mmseg.models.segmentors import BaseSegmentor
    from mmengine.structures import PixelData
    from mmseg.registry import MODELS

from concept_utils import is_json_path, load_concepts
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.perflib.connected_components import connected_components


FOUR_PARAMETER_NAMES = (
    'semantic_weight',
    'presence_gain',
    'presence_bias',
    'log_prompt_weight',
)
IDENTIFIABLE_PARAMETER_NAMES = (
    'raw_semantic_weight',
    'raw_presence_gain',
    'presence_intercept',
)
# Compatibility name retained for callers that still import the old symbol.
PROMPT_PARAMETER_NAMES = FOUR_PARAMETER_NAMES

SCORE_EPS = 1e-6
LOG_SCORE_EPS = math.log(SCORE_EPS)


def inverse_softplus(value):
    """Return a numerically stable inverse of softplus for positive tensors."""
    value = torch.as_tensor(value).clamp_min(SCORE_EPS)
    return value + torch.log(-torch.expm1(-value))


def legacy_to_identifiable(semantic_weight, presence_gain, presence_bias,
                           log_prompt_weight):
    """Convert the legacy four parameters without changing prompt logits."""
    semantic_weight = torch.as_tensor(semantic_weight).clamp(
        SCORE_EPS, 1.0 - SCORE_EPS)
    presence_gain = torch.as_tensor(presence_gain).clamp_min(SCORE_EPS)
    presence_bias = torch.as_tensor(presence_bias)
    log_prompt_weight = torch.as_tensor(log_prompt_weight)
    return dict(
        raw_semantic_weight=torch.logit(semantic_weight),
        raw_presence_gain=inverse_softplus(presence_gain),
        presence_intercept=(
            log_prompt_weight
            + presence_gain * (0.5 - presence_bias)),
    )


class PromptCalibration(nn.Module):
    def __init__(self, values, parameter_names):
        super().__init__()
        self.parameter_names = tuple(parameter_names)
        for name in self.parameter_names:
            value = values[name].detach().clone().reshape(())
            self.register_parameter(name, nn.Parameter(value))
            self.register_buffer(f'_initial_{name}', value.clone())

    def parameter_items(self):
        return [(name, getattr(self, name)) for name in self.parameter_names]

    def initial_value(self, name):
        return getattr(self, f'_initial_{name}')


class PromptCalibrationBank(nn.Module):
    def __init__(self, parameter_vectors, parameter_names=None):
        super().__init__()
        self.parameter_names = tuple(
            parameter_names or tuple(parameter_vectors.keys()))
        lengths = {len(values) for values in parameter_vectors.values()}
        if len(lengths) != 1:
            raise ValueError(
                f'Prompt parameter vectors must have equal lengths: {lengths}')
        num_prompts = lengths.pop()
        self.prompts = nn.ModuleList([
            PromptCalibration({
                name: parameter_vectors[name][prompt_idx]
                for name in self.parameter_names
            }, self.parameter_names)
            for prompt_idx in range(num_prompts)
        ])

    def values(self, name):
        return torch.stack([getattr(prompt, name) for prompt in self.prompts])

    def initial_values(self, name):
        return torch.stack([
            prompt.initial_value(name) for prompt in self.prompts])

    def parameter_group(self, prompt_idx):
        return [parameter for _, parameter in self.prompts[prompt_idx].parameter_items()]

    def parameter_group_items(self, prompt_idx):
        return self.prompts[prompt_idx].parameter_items()

    def copy_vector_(self, name, values):
        if values.shape != (len(self.prompts),):
            raise ValueError(
                f'{name} shape mismatch: expected {(len(self.prompts),)}, '
                f'got {values.shape}.')
        with torch.no_grad():
            for prompt_idx, value in enumerate(values):
                getattr(self.prompts[prompt_idx], name).copy_(value)


@MODELS.register_module()
class SegEarthOV3LearnableSegmentation(BaseSegmentor):
    TRAINABLE_PROMPT_PARAMETER_NAMES = FOUR_PARAMETER_NAMES
    PROMPT_SCORE_VERSION = 'log_odds_log_weight_v2'
    COMPATIBLE_PROMPT_SCORE_VERSIONS = {
        'log_odds_v1',
        PROMPT_SCORE_VERSION,
        'identifiable_three_v1',
    }

    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 fusion_mode='max',
                 prompt_fusion_mode='max',
                 prompt_aggregation=None,
                 semantic_weight=0.5,
                 presence_gain=1.0,
                 presence_bias=0.5,
                 prompt_weight=1.0,
                 prompt_threshold=0.0,
                 concept_threshold=0.0,
                 concept_weight=1.0,
                 min_prompt_component_area=0.0,
                 instance_filter_use_presence=True,
                 prompt_parameterization='four_parameter',
                 freeze_sam=True,
                 calibration_checkpoint=None,
                 **kwargs):
        super().__init__()

        self.device = device
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path='weights/sam3/sam3.pt',
            device="cuda"
        )
        if freeze_sam:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

        self.processor = Sam3Processor(
            model, confidence_threshold=confidence_threshold, device=device)
        cls_info = get_cls_idx(classname_path)
        self.query_words, self.query_idx = cls_info[:2]
        self.concepts = cls_info[2] if len(cls_info) > 2 else None
        self.is_concept_json = self.concepts is not None
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.tensor(
            self.query_idx, dtype=torch.int64, device=device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        self.fusion_mode = fusion_mode
        self.prompt_fusion_mode = prompt_aggregation or prompt_fusion_mode
        self.instance_filter_use_presence = instance_filter_use_presence
        if prompt_parameterization not in (
                'four_parameter', 'identifiable_three'):
            raise ValueError(
                f'Unsupported prompt_parameterization: '
                f'{prompt_parameterization}')
        self.prompt_parameterization = prompt_parameterization

        initial_prompt_weight = self._expand_prompt_value(
            prompt_weight, 1.0).clamp_min(SCORE_EPS)
        initial_semantic_weight = self._expand_prompt_value(
            semantic_weight, 0.5)
        initial_presence_gain = self._expand_prompt_value(
            presence_gain, 1.0)
        initial_presence_bias = self._expand_prompt_value(
            presence_bias, 0.5)
        initial_log_prompt_weight = torch.log(initial_prompt_weight)
        if prompt_parameterization == 'identifiable_three':
            parameter_vectors = legacy_to_identifiable(
                initial_semantic_weight,
                initial_presence_gain,
                initial_presence_bias,
                initial_log_prompt_weight)
            parameter_names = IDENTIFIABLE_PARAMETER_NAMES
            self.PROMPT_SCORE_VERSION = 'identifiable_three_v1'
        else:
            parameter_vectors = dict(
                semantic_weight=initial_semantic_weight,
                presence_gain=initial_presence_gain,
                presence_bias=initial_presence_bias,
                log_prompt_weight=initial_log_prompt_weight)
            parameter_names = FOUR_PARAMETER_NAMES
            self.PROMPT_SCORE_VERSION = 'log_odds_log_weight_v2'
        self.TRAINABLE_PROMPT_PARAMETER_NAMES = parameter_names
        self.prompt_calibrations = PromptCalibrationBank(
            parameter_vectors, parameter_names)
        self.register_buffer(
            'prompt_threshold',
            self._expand_prompt_value(prompt_threshold, 0.0))
        self.register_buffer(
            'min_prompt_component_area',
            self._expand_prompt_value(min_prompt_component_area, 0.0))
        self.register_buffer(
            'concept_threshold',
            self._expand_concept_value(concept_threshold, 0.0))
        self.register_buffer(
            'concept_weight',
            self._expand_concept_value(concept_weight, 1.0))
        if calibration_checkpoint:
            self.load_calibration_checkpoint(calibration_checkpoint)

    def prompt_parameter_items(self):
        return [
            (name, self.prompt_calibrations.values(name))
            for name in self.TRAINABLE_PROMPT_PARAMETER_NAMES
        ]

    def prompt_parameters(self):
        return list(self.prompt_calibrations.parameters())

    def prompt_parameter_group_items(self, prompt_idx):
        return self.prompt_calibrations.parameter_group_items(prompt_idx)

    def prompt_parameter_group(self, prompt_idx):
        return self.prompt_calibrations.parameter_group(prompt_idx)

    def initial_prompt_parameter(self, name):
        if name not in self.TRAINABLE_PROMPT_PARAMETER_NAMES:
            raise KeyError(f'Unknown trainable prompt parameter: {name}')
        return self.prompt_calibrations.initial_values(name)

    def initial_prompt_parameter_group(self, prompt_idx):
        prompt = self.prompt_calibrations.prompts[prompt_idx]
        return {
            name: prompt.initial_value(name)
            for name in self.TRAINABLE_PROMPT_PARAMETER_NAMES
        }

    def set_prompt_parameter_group(self, prompt_idx, values):
        prompt = self.prompt_calibrations.prompts[prompt_idx]
        if (self.prompt_parameterization == 'identifiable_three'
                and not any(
                    name in values for name in IDENTIFIABLE_PARAMETER_NAMES)):
            semantic_weight = values.get(
                'semantic_weight', self.semantic_weight[prompt_idx])
            presence_gain = values.get(
                'presence_gain', self.presence_gain[prompt_idx])
            presence_bias = values.get(
                'presence_bias', self.presence_bias[prompt_idx])
            log_prompt_weight = values.get('log_prompt_weight')
            if log_prompt_weight is None:
                prompt_weight = values.get(
                    'prompt_weight', self.prompt_weight[prompt_idx])
                log_prompt_weight = torch.log(torch.as_tensor(
                    prompt_weight, device=self.device).clamp_min(SCORE_EPS))
            values = legacy_to_identifiable(
                semantic_weight,
                presence_gain,
                presence_bias,
                log_prompt_weight)
        with torch.no_grad():
            for name in self.TRAINABLE_PROMPT_PARAMETER_NAMES:
                if name in values:
                    getattr(prompt, name).copy_(torch.as_tensor(
                        values[name],
                        device=getattr(prompt, name).device,
                        dtype=getattr(prompt, name).dtype))
            if (self.prompt_parameterization == 'four_parameter'
                    and 'log_prompt_weight' not in values
                    and 'prompt_weight' in values):
                weight = torch.as_tensor(
                    values['prompt_weight'],
                    device=prompt.log_prompt_weight.device,
                    dtype=prompt.log_prompt_weight.dtype)
                prompt.log_prompt_weight.copy_(
                    torch.log(weight.clamp_min(SCORE_EPS)))

    @property
    def semantic_weight(self):
        if self.prompt_parameterization == 'identifiable_three':
            return torch.sigmoid(
                self.prompt_calibrations.values('raw_semantic_weight'))
        return self.prompt_calibrations.values('semantic_weight')

    @property
    def presence_gain(self):
        if self.prompt_parameterization == 'identifiable_three':
            return F.softplus(
                self.prompt_calibrations.values('raw_presence_gain'))
        return self.prompt_calibrations.values('presence_gain')

    @property
    def presence_bias(self):
        if self.prompt_parameterization == 'identifiable_three':
            return torch.full_like(self.presence_gain, 0.5)
        return self.prompt_calibrations.values('presence_bias')

    @property
    def log_prompt_weight(self):
        if self.prompt_parameterization == 'identifiable_three':
            return self.prompt_calibrations.values('presence_intercept')
        return self.prompt_calibrations.values('log_prompt_weight')

    @property
    def prompt_weight(self):
        return torch.exp(self.log_prompt_weight)

    def calibration_parameters(self):
        # Kept as a compatibility alias for the training script.
        return self.prompt_parameters()

    def load_calibration_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        score_version = checkpoint.get('prompt_score_version')
        if score_version not in self.COMPATIBLE_PROMPT_SCORE_VERSIONS:
            raise ValueError(
                f'Calibration checkpoint score version {score_version!r} is '
                f'incompatible with {self.PROMPT_SCORE_VERSION!r}.')
        if ('presence_strength' in checkpoint
                and 'presence_gain' not in checkpoint):
            print(
                'Legacy checkpoint presence_strength is not loaded because '
                'it is not equivalent to presence_gain/presence_bias.')
        if self.prompt_parameterization == 'identifiable_three':
            if all(name in checkpoint for name in IDENTIFIABLE_PARAMETER_NAMES):
                converted = {
                    name: checkpoint[name]
                    for name in IDENTIFIABLE_PARAMETER_NAMES
                }
            else:
                old_log_weight = checkpoint.get('log_prompt_weight')
                if old_log_weight is None and 'prompt_weight' in checkpoint:
                    old_log_weight = torch.log(
                        checkpoint['prompt_weight'].clamp_min(SCORE_EPS))
                converted = legacy_to_identifiable(
                    checkpoint.get('semantic_weight', self.semantic_weight),
                    checkpoint.get('presence_gain', self.presence_gain),
                    checkpoint.get('presence_bias', self.presence_bias),
                    old_log_weight if old_log_weight is not None
                    else self.log_prompt_weight)
            for name, value in converted.items():
                target = self.prompt_calibrations.values(name)
                value = value.to(device=target.device, dtype=target.dtype)
                if value.shape != target.shape:
                    raise ValueError(
                        f'{name} shape mismatch: expected {target.shape}, '
                        f'got {value.shape}.')
                self.prompt_calibrations.copy_vector_(name, value)

        for name in [
                'prompt_threshold',
                'min_prompt_component_area',
                'concept_threshold',
                'concept_weight']:
            if name not in checkpoint:
                continue
            target = getattr(self, name)
            value = checkpoint[name].to(
                device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f'{name} shape mismatch: expected {target.shape}, got {value.shape}.')
            with torch.no_grad():
                target.copy_(value)
        if self.prompt_parameterization == 'four_parameter':
            for name in FOUR_PARAMETER_NAMES:
                if name in checkpoint:
                    target = self.prompt_calibrations.values(name)
                    value = checkpoint[name].to(
                        device=target.device, dtype=target.dtype)
                    if value.shape != target.shape:
                        raise ValueError(
                            f'{name} shape mismatch: expected {target.shape}, '
                            f'got {value.shape}.')
                    self.prompt_calibrations.copy_vector_(name, value)
        if (self.prompt_parameterization == 'four_parameter'
                and 'log_prompt_weight' not in checkpoint
                and 'prompt_weight' in checkpoint):
            prompt_weight = checkpoint['prompt_weight'].to(
                device=self.log_prompt_weight.device,
                dtype=self.log_prompt_weight.dtype)
            self.prompt_calibrations.copy_vector_(
                'log_prompt_weight',
                torch.log(prompt_weight.clamp_min(SCORE_EPS)))

    def _concept_ids(self):
        if not self.concepts:
            return [str(i) for i in range(self.num_cls)]
        return [
            str(concept.get('concept_id') or concept.get('concept_name') or i)
            for i, concept in enumerate(self.concepts)
        ]

    def _prompt_keys(self):
        keys = []
        concept_ids = self._concept_ids()
        per_class_counts = {class_id: 0 for class_id in range(self.num_cls)}
        for prompt, class_id in zip(self.query_words, self.query_idx.tolist()):
            prompt_pos = per_class_counts[class_id]
            per_class_counts[class_id] += 1
            keys.append((prompt, concept_ids[class_id], prompt_pos))
        return keys

    def _expand_prompt_value(self, value, default):
        return self._expand_value(value, self.num_queries, default,
                                  self._prompt_keys())

    def _expand_concept_value(self, value, default):
        keys = [(concept_id, concept_id, idx)
                for idx, concept_id in enumerate(self._concept_ids())]
        return self._expand_value(value, self.num_cls, default, keys)

    def _expand_value(self, value, length, default, keys):
        values = torch.full(
            (length,), float(default), dtype=torch.float32, device=self.device)
        if value is None:
            return values
        if isinstance(value, (int, float)):
            values.fill_(float(value))
            return values
        if isinstance(value, (list, tuple)):
            if len(value) != length:
                raise ValueError(f'Expected {length} values, got {len(value)}.')
            return torch.tensor(value, dtype=torch.float32, device=self.device)
        if isinstance(value, dict):
            for i, (prompt, concept_id, prompt_pos) in enumerate(keys):
                qualified_keys = (
                    f'{concept_id}/{prompt}',
                    f'{concept_id}.{prompt}',
                    f'{concept_id}:{prompt}',
                    f'{concept_id}/{prompt_pos}',
                    f'{concept_id}.{prompt_pos}',
                    f'{concept_id}:{prompt_pos}',
                )
                matched_key = next(
                    (key for key in qualified_keys if key in value), None)
                if matched_key is not None:
                    values[i] = float(value[matched_key])
                elif prompt in value:
                    values[i] = float(value[prompt])
                elif concept_id in value:
                    concept_value = value[concept_id]
                    if isinstance(concept_value, dict):
                        values[i] = float(
                            concept_value.get(prompt, concept_value.get(
                                str(prompt_pos), values[i].item())))
                    else:
                        values[i] = float(concept_value)
            return values
        raise TypeError(f'Unsupported calibration value type: {type(value)}')

    def _bounded_prompt_values(self):
        if self.prompt_parameterization == 'identifiable_three':
            semantic_weight = self.semantic_weight
            presence_gain = self.presence_gain
            log_prompt_weight = self.log_prompt_weight
        else:
            semantic_weight = self.semantic_weight.clamp(0.0, 1.0)
            presence_gain = self.presence_gain.clamp_min(0.0)
            log_prompt_weight = self.log_prompt_weight.clamp_min(LOG_SCORE_EPS)
        values = dict(
            semantic_weight=semantic_weight,
            presence_gain=presence_gain,
            presence_bias=self.presence_bias.clamp(0.0, 1.0),
            log_prompt_weight=log_prompt_weight,
            prompt_weight=torch.exp(log_prompt_weight),
            prompt_threshold=self.prompt_threshold.clamp_min(0.0),
            min_prompt_component_area=self.min_prompt_component_area.clamp_min(0.0),
            concept_threshold=self.concept_threshold.clamp_min(0.0),
            concept_weight=self.concept_weight.clamp_min(0.0),
        )
        if self.prompt_parameterization == 'identifiable_three':
            values.update(
                raw_semantic_weight=self.prompt_calibrations.values(
                    'raw_semantic_weight'),
                raw_presence_gain=self.prompt_calibrations.values(
                    'raw_presence_gain'),
                presence_intercept=self.prompt_calibrations.values(
                    'presence_intercept'))
        return values

    def constrain_prompt_parameter_group(self, prompt_idx,
                                         presence_gain_max=8.0,
                                         prompt_weight_max=3.0):
        if self.prompt_parameterization == 'identifiable_three':
            return
        prompt = self.prompt_calibrations.prompts[prompt_idx]
        with torch.no_grad():
            prompt.semantic_weight.clamp_(0.0, 1.0)
            prompt.presence_gain.clamp_(0.0, presence_gain_max)
            prompt.presence_bias.clamp_(0.0, 1.0)
            prompt.log_prompt_weight.clamp_(
                LOG_SCORE_EPS, math.log(prompt_weight_max))

    def constrain_prompt_parameters(self, presence_gain_max=8.0,
                                    prompt_weight_max=3.0):
        for prompt_idx in range(self.num_queries):
            self.constrain_prompt_parameter_group(
                prompt_idx,
                presence_gain_max=presence_gain_max,
                prompt_weight_max=prompt_weight_max)

    def _instance_score(self, inference_state, inst_id):
        key = 'object_score' if self.instance_filter_use_presence else 'object_score_raw'
        score = inference_state[key][inst_id]
        while score.dim() > 0:
            score = score.squeeze(0)
        return score

    def _fuse_heads(self, query_idx, instance_score, semantic_score, params):
        if self.fusion_mode == 'max':
            return torch.max(instance_score, semantic_score)
        if self.fusion_mode in ('weighted', 'weighted_sum'):
            weight = params['semantic_weight'][query_idx]
            return weight * semantic_score + (1.0 - weight) * instance_score
        raise ValueError(f'Unsupported fusion_mode: {self.fusion_mode}')

    def _extract_prompt_heads_from_state(self, inference_state, query_idx,
                                         width, height):
        self.processor.reset_all_prompts(inference_state)
        inference_state = self.processor.set_text_prompt(
            state=inference_state, prompt=self.query_words[query_idx])

        instance_score_map = torch.zeros(
            (height, width), device=self.device)
        semantic_score_map = torch.zeros(
            (height, width), device=self.device)

        if (self.use_transformer_decoder
                and inference_state['masks_logits'].shape[0] > 0):
            inst_len = inference_state['masks_logits'].shape[0]
            for inst_id in range(inst_len):
                instance_logits = inference_state[
                    'masks_logits'][inst_id].squeeze()
                instance_score = self._instance_score(
                    inference_state, inst_id)

                if instance_logits.shape != (height, width):
                    instance_logits = F.interpolate(
                        instance_logits.view(
                            1, 1, *instance_logits.shape),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False).squeeze()

                instance_score_map = torch.max(
                    instance_score_map,
                    instance_logits * instance_score)

        if self.use_sem_seg:
            semantic_logits = inference_state['semantic_mask_logits']
            if semantic_logits.shape != (height, width):
                semantic_logits = F.interpolate(
                    semantic_logits,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False).squeeze()
            semantic_score_map = semantic_logits

        presence_score = inference_state['presence_score'].clone().squeeze()
        heads = dict(
            instance_score_map=instance_score_map.clone(),
            semantic_score_map=semantic_score_map.clone(),
            presence_score=presence_score)
        return inference_state, heads

    @staticmethod
    def probability_to_logits(probability):
        """Convert a bounded score probability to numerically stable logits."""
        return torch.logit(probability.clamp(SCORE_EPS, 1.0 - SCORE_EPS))

    def calibrate_prompt_head_logits(self, query_idx, heads,
                                     apply_prompt_weight=True):
        """Calibrate one prompt in log-odds space and return true logits."""
        params = self._bounded_prompt_values()
        fused_probability = self._fuse_heads(
            query_idx,
            heads['instance_score_map'],
            heads['semantic_score_map'],
            params)
        prompt_logits = self.probability_to_logits(fused_probability)
        if self.use_presence_score:
            gain = params['presence_gain'][query_idx]
            bias = params['presence_bias'][query_idx]
            presence_logit = gain * (heads['presence_score'] - bias)
            if self.prompt_parameterization == 'four_parameter':
                presence_logit = presence_logit.clamp(-8.0, 8.0)
            prompt_logits = prompt_logits + presence_logit
        if apply_prompt_weight:
            prompt_logits = (
                prompt_logits + params['log_prompt_weight'][query_idx])
        return prompt_logits

    def calibrate_prompt_head_logits_batch(self, query_indices, heads,
                                           apply_prompt_weight=True):
        """Calibrate stacked prompt heads without coupling prompt parameters."""
        query_indices = torch.as_tensor(
            query_indices, dtype=torch.long, device=self.device)
        params = self._bounded_prompt_values()
        instance_scores = heads['instance_score_map']
        semantic_scores = heads['semantic_score_map']
        if self.fusion_mode == 'max':
            fused_probability = torch.maximum(
                instance_scores, semantic_scores)
        elif self.fusion_mode in ('weighted', 'weighted_sum'):
            weights = params['semantic_weight'][query_indices]
            weights = weights.reshape(
                (len(query_indices),)
                + (1,) * (instance_scores.ndim - 1))
            fused_probability = (
                weights * semantic_scores
                + (1.0 - weights) * instance_scores)
        else:
            raise ValueError(f'Unsupported fusion_mode: {self.fusion_mode}')

        prompt_logits = self.probability_to_logits(fused_probability)
        broadcast_shape = (
            (len(query_indices),)
            + (1,) * (prompt_logits.ndim - 1))
        if self.use_presence_score:
            gains = params['presence_gain'][query_indices]
            biases = params['presence_bias'][query_indices]
            presence_scores = heads['presence_score'].reshape(-1)
            presence_logits = gains * (presence_scores - biases)
            if self.prompt_parameterization == 'four_parameter':
                presence_logits = presence_logits.clamp(-8.0, 8.0)
            prompt_logits = prompt_logits + presence_logits.reshape(
                broadcast_shape)
        if apply_prompt_weight:
            prompt_logits = prompt_logits + params[
                'log_prompt_weight'][query_indices].reshape(broadcast_shape)
        return prompt_logits

    def calibrate_prompt_heads(self, query_idx, heads,
                               apply_prompt_weight=True):
        """Return the calibrated prompt probability used by inference."""
        return torch.sigmoid(self.calibrate_prompt_head_logits(
            query_idx, heads, apply_prompt_weight=apply_prompt_weight))

    def extract_prompt_heads_batch(self, image, query_indices,
                                   ori_shape=None):
        query_indices = list(query_indices)
        for query_idx in query_indices:
            if query_idx < 0 or query_idx >= self.num_queries:
                raise IndexError(f'Invalid prompt index: {query_idx}')
        if self.slide_crop > 0 and (
                self.slide_crop < image.size[0]
                or self.slide_crop < image.size[1]):
            raise ValueError(
                'Independent prompt caching currently requires images no '
                'larger than slide_crop. Cache the configured 512x512 tiles.')

        width, height = image.size
        with torch.no_grad(), torch.autocast(
                device_type='cuda', dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            all_heads = []
            for query_idx in query_indices:
                inference_state, heads = self._extract_prompt_heads_from_state(
                    inference_state, query_idx, width, height)
                all_heads.append(heads)

        if ori_shape is not None:
            for heads in all_heads:
                for key in ('instance_score_map', 'semantic_score_map'):
                    if heads[key].shape != ori_shape:
                        heads[key] = F.interpolate(
                            heads[key].view(1, 1, *heads[key].shape),
                            size=ori_shape,
                            mode='bilinear',
                            align_corners=False).squeeze()
        return all_heads

    def extract_prompt_heads(self, image, query_idx, ori_shape=None):
        return self.extract_prompt_heads_batch(
            image, [query_idx], ori_shape=ori_shape)[0]

    def score_prompt(self, image, query_idx, ori_shape=None):
        heads = self.extract_prompt_heads(
            image, query_idx, ori_shape=ori_shape)
        return self.calibrate_prompt_heads(query_idx, heads)

    def _inference_single_view(self, image):
        w, h = image.size
        prompt_scores = []

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

        for query_idx in range(self.num_queries):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_state, heads = self._extract_prompt_heads_from_state(
                    inference_state, query_idx, w, h)
            prompt_scores.append(self.calibrate_prompt_heads(
                query_idx, heads, apply_prompt_weight=False))

        return torch.stack(prompt_scores, dim=0)

    def aggregate_prompt_scores(self,
                                prompt_scores: torch.Tensor,
                                mode: str = 'max',
                                prompt_weights: torch.Tensor | None = None
                                ) -> torch.Tensor:
        class_scores = []

        for class_id in range(self.num_cls):
            mask = self.query_idx == class_id
            current_scores = prompt_scores[mask]
            current_weights = (
                prompt_weights[mask].clamp_min(0.0)
                if prompt_weights is not None
                else torch.ones(
                    current_scores.shape[0],
                    device=current_scores.device,
                    dtype=current_scores.dtype))
            weighted_scores = (
                current_scores * current_weights[:, None, None])

            if current_scores.shape[0] == 0:
                score = torch.zeros_like(prompt_scores[0])
            elif mode == 'max':
                score = weighted_scores.max(dim=0).values
            elif mode in ('weighted_sum', 'weighted'):
                score = weighted_scores.mean(dim=0)
            elif mode == 'weighted_max':
                score = weighted_scores.max(dim=0).values
            elif mode == 'mean':
                score = weighted_scores.mean(dim=0)
            else:
                raise ValueError(f'Unsupported prompt fusion mode: {mode}')

            class_scores.append(score)

        return torch.stack(class_scores, dim=0)

    def filter_prompt_scores(self, prompt_scores):
        params = self._bounded_prompt_values()
        thresholds = params['prompt_threshold'][:, None, None]
        filtered_scores = torch.where(
            prompt_scores >= thresholds,
            prompt_scores,
            torch.zeros_like(prompt_scores))

        min_areas = params['min_prompt_component_area']
        if torch.all(min_areas <= 0):
            return filtered_scores

        kept_scores = filtered_scores.clone()
        for query_idx, min_area in enumerate(min_areas):
            area_value = int(torch.floor(min_area).item())
            if area_value <= 0:
                continue
            mask = filtered_scores[query_idx] > 0
            _, component_area = connected_components(mask[None, None])
            component_area = component_area.squeeze(0).squeeze(0)
            kept_scores[query_idx] = torch.where(
                component_area >= area_value,
                kept_scores[query_idx],
                torch.zeros_like(kept_scores[query_idx]))
        return kept_scores

    def slide_inference(self, image, stride, crop_size):
        w_img, h_img = image.size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_seg_logit = self._inference_single_view(crop_img)

                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."

        preds = preds / count_mat
        return preds

    def score_image(self, image, ori_shape=None):
        if self.slide_crop > 0 and (
                self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
            prompt_scores = self.slide_inference(
                image, self.slide_stride, self.slide_crop)
        else:
            prompt_scores = self._inference_single_view(image)

        if ori_shape is not None and prompt_scores.shape[-2:] != ori_shape:
            prompt_scores = F.interpolate(
                prompt_scores.unsqueeze(0),
                size=ori_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        prompt_scores = self.filter_prompt_scores(prompt_scores)
        params = self._bounded_prompt_values()
        calibrated_prompt_scores = torch.sigmoid(
            self.probability_to_logits(prompt_scores)
            + params['log_prompt_weight'][:, None, None])
        class_scores = self.aggregate_prompt_scores(
            calibrated_prompt_scores,
            mode=self.prompt_fusion_mode)
        class_scores = class_scores * params['concept_weight'][:, None, None]
        return class_scores, calibrated_prompt_scores

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        params = self._bounded_prompt_values()
        for i, meta in enumerate(batch_img_metas):
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            class_scores, raw_prompt_scores = self.score_image(
                image, ori_shape=ori_shape)
            seg_pred = torch.argmax(class_scores, dim=0)

            max_vals = class_scores.max(0)[0]
            thresholds = params['concept_threshold'].view(-1, 1, 1)
            below_concept_threshold = torch.gather(
                thresholds.expand(-1, *seg_pred.shape),
                0,
                seg_pred.unsqueeze(0)).squeeze(0)
            seg_pred[max_vals < torch.maximum(
                below_concept_threshold,
                torch.tensor(self.prob_thd, device=self.device))] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': class_scores}),
                'raw_prompt_scores': PixelData(**{'data': raw_prompt_scores}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })

        return data_samples

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    if is_json_path(path):
        concepts = load_concepts(path)
        class_names, class_indices = [], []
        for idx, concept in enumerate(concepts):
            prompts = concept.get('prompts', [])
            if isinstance(prompts, str):
                prompts = [prompts]
            prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
            class_names += prompts
            class_indices += [idx for _ in range(len(prompts))]
        return class_names, class_indices, concepts

    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
