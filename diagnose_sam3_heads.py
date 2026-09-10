from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import deque
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


CHECKPOINT_PATH = "weights/sam3/sam3.pt"
BPE_PATH = "./sam3/assets/bpe_simple_vocab_16e6.txt.gz"
SCORE_EPS = 1e-6

MAX_IMAGES = 20
DEFAULT_PARAMS_PATH = Path("configs/prams_cls_penglai_origin_multiple_5.json")

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

DEFAULT_SCORE_PARAMS = {
    "confidence_threshold": 0.1,
    "use_sem_seg": True,
    "use_transformer_decoder": True,
    "use_presence_score": True,
    "instance_filter_use_presence": True,
    "fusion_mode": "max",
    "semantic_weight": 0.5,
    "presence_gain": 0.0,
    "presence_bias": 0.5,
    "prompt_weight": 1.0,
    "prompt_threshold": 0.0,
    "min_prompt_component_area": 0.0,
    "prompt_fusion_mode": "max",
    "concept_weight": 1.0,
    "concept_threshold": 0.0,
    "prob_thd": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose SAM3 semantic, instance, and fused heads."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        type=Path,
        help="Optional directory containing GT label rasters matched by image stem.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for diagnostic outputs.",
    )
    parser.add_argument(
        "--prompt-path",
        "--prompts-path",
        required=True,
        type=Path,
        help="Prompt source. Use txt for one prompt per line, or json concept config.",
    )
    parser.add_argument(
        "--params-path",
        default=DEFAULT_PARAMS_PATH,
        type=Path,
        help=(
            "JSON file containing final-score calibration parameters. "
            f"Default: {DEFAULT_PARAMS_PATH}"
        ),
    )
    parser.add_argument(
        "--analysis-threshold",
        required=True,
        type=float,
        help="Threshold used for response-area statistics and GT metrics.",
    )
    parser.add_argument(
        "--max-images",
        default=MAX_IMAGES,
        type=int,
        help=f"Maximum number of images to diagnose. Default: {MAX_IMAGES}.",
    )
    parser.add_argument(
        "--ignore-index",
        default=255,
        type=int,
        help="Ignore index in GT label rasters. Default: 255.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        default=0,
        type=int,
        help=argparse.SUPPRESS,
    )
    args, _ = parser.parse_known_args()
    return args


def load_prompt_entries(prompt_path: Path) -> list[dict[str, Any]]:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt input does not exist: {prompt_path}")

    suffix = prompt_path.suffix.lower()
    if suffix == ".txt":
        entries = []
        for prompt_index, line in enumerate(
            prompt_path.read_text(encoding="utf-8").splitlines()
        ):
            prompt = line.strip()
            if not prompt:
                continue
            entries.append(
                {
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "prompt_position": prompt_index,
                    "concept_index": None,
                    "concept_id": None,
                    "concept_name": None,
                    "tnls_dl3": [],
                    "isdeleted": False,
                }
            )
    elif suffix == ".json":
        with prompt_path.open("r", encoding="utf-8") as file:
            entries = load_prompt_entries_from_json(json.load(file))
    else:
        raise ValueError(
            f"Unsupported prompt input format: {prompt_path.suffix}. "
            "Expected .txt or .json."
        )

    if not entries:
        raise RuntimeError(f"No prompts found in: {prompt_path}")

    return entries


def load_prompt_entries_from_json(data: object) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("JSON prompt config must be an object.")

    groups = data.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("JSON prompt config field 'groups' must be a list.")

    entries: list[dict[str, Any]] = []
    prompt_index = 0
    concept_index = 0
    for group in groups:
        if not isinstance(group, dict):
            continue

        concepts = group.get("concepts", [])
        if not isinstance(concepts, list):
            continue

        for concept in concepts:
            if not isinstance(concept, dict):
                continue

            concept_prompts = concept.get("prompts", [])
            if isinstance(concept_prompts, str):
                concept_prompts = [concept_prompts]
            if not isinstance(concept_prompts, list):
                concept_index += 1
                continue

            for prompt_position, prompt in enumerate(concept_prompts):
                if not isinstance(prompt, str) or not prompt.strip():
                    continue

                entries.append(
                    {
                        "prompt": prompt.strip(),
                        "prompt_index": prompt_index,
                        "prompt_position": prompt_position,
                        "concept_index": concept_index,
                        "concept_id": concept.get("concept_id"),
                        "concept_name": concept.get("concept_name"),
                        "tnls_dl3": [
                            int(label) for label in concept.get("tnls_dl3", [])
                        ],
                        "isdeleted": bool(concept.get("isdeleted", False)),
                    }
                )
                prompt_index += 1
            concept_index += 1

    return entries


def load_prompts(prompt_path: Path) -> list[str]:
    return [entry["prompt"] for entry in load_prompt_entries(prompt_path)]


def load_prompts_from_json(data: object) -> list[str]:
    return [entry["prompt"] for entry in load_prompt_entries_from_json(data)]


def load_score_params(params_path: Path | None) -> dict[str, Any]:
    params = dict(DEFAULT_SCORE_PARAMS)
    if params_path is None:
        return params
    if not params_path.exists() or params_path.stat().st_size == 0:
        return params

    with params_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Score parameter JSON must be an object.")

    if any(key in data for key in ("default", "model_params", "prompt_params", "concept_params")):
        default_params = data.get("default", data.get("model_params", {}))
        if not isinstance(default_params, dict):
            raise ValueError(
                "Score parameter JSON field 'default' must be an object."
            )
        params.update(default_params)

        for section_name in ("prompt_params", "concept_params"):
            section = data.get(section_name, {})
            if not isinstance(section, dict):
                raise ValueError(
                    f"Score parameter JSON field '{section_name}' must be an object."
                )
            params.update(section)
    else:
        params.update(data)
    return params


def entry_param(
    params: dict[str, Any],
    name: str,
    entry: dict[str, Any],
    default: Any,
) -> Any:
    value = params.get(name, default)
    if isinstance(value, dict):
        prompt = entry.get("prompt")
        concept_id = entry.get("concept_id")
        prompt_position = entry.get("prompt_position")
        keys: list[str] = []
        if concept_id is not None and prompt is not None:
            keys.extend(
                [
                    f"{concept_id}/{prompt}",
                    f"{concept_id}.{prompt}",
                    f"{concept_id}:{prompt}",
                ]
            )
        if concept_id is not None and prompt_position is not None:
            keys.extend(
                [
                    f"{concept_id}/{prompt_position}",
                    f"{concept_id}.{prompt_position}",
                    f"{concept_id}:{prompt_position}",
                ]
            )
        if prompt is not None:
            keys.append(prompt)

        matched_key = next((key for key in keys if key in value), None)
        if matched_key is not None:
            return value[matched_key]

        if concept_id in value:
            concept_value = value[concept_id]
            if isinstance(concept_value, dict):
                if prompt in concept_value:
                    return concept_value[prompt]
                if str(prompt_position) in concept_value:
                    return concept_value[str(prompt_position)]
                return default
            return concept_value
        return default

    if isinstance(value, list):
        prompt_index = int(entry.get("prompt_index", -1))
        if prompt_index < 0 or prompt_index >= len(value):
            return default
        return value[prompt_index]

    return value


def bool_param(
    params: dict[str, Any],
    name: str,
    entry: dict[str, Any],
    default: bool,
) -> bool:
    return bool(entry_param(params, name, entry, default))


def float_param(
    params: dict[str, Any],
    name: str,
    entry: dict[str, Any],
    default: float,
) -> float:
    return float(entry_param(params, name, entry, default))


def safe_name(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def tensor_to_map(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().squeeze()

    if value.ndim != 2:
        raise ValueError(
            f"Expected a 2D probability map, got shape: {tuple(value.shape)}"
        )

    return value.clamp(0.0, 1.0)


def resize_score_map(
    value: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if value.shape == (height, width):
        return value.clamp(0.0, 1.0)

    return F.interpolate(
        value.view(1, 1, *value.shape),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze().clamp(0.0, 1.0)


def prepare_instance_masks(
    masks: torch.Tensor,
) -> torch.Tensor:
    masks = masks.detach().float()

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    if masks.ndim != 3:
        raise ValueError(f"Unsupported instance mask shape: {tuple(masks.shape)}")

    return masks.clamp(0.0, 1.0)


def aggregate_instance_map(
    masks: torch.Tensor,
    scores: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if masks.numel() == 0 or masks.shape[0] == 0:
        return torch.zeros(
            (height, width),
            dtype=torch.float32,
            device=masks.device,
        )

    masks = prepare_instance_masks(masks)
    if masks.shape[-2:] != (height, width):
        masks = F.interpolate(
            masks[:, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    scores = scores.detach().float().reshape(-1)

    if masks.shape[0] != scores.shape[0]:
        raise ValueError(
            "Instance count and score count mismatch: "
            f"{masks.shape[0]} vs {scores.shape[0]}"
        )

    weighted_masks = masks * scores[:, None, None]
    instance_map = weighted_masks.amax(dim=0)

    return instance_map.clamp(0.0, 1.0)


def fuse_score_maps(
    semantic_map: torch.Tensor,
    instance_map: torch.Tensor,
    semantic_weight: float,
    fusion_mode: str,
) -> torch.Tensor:
    if fusion_mode == "max":
        return torch.maximum(semantic_map, instance_map)
    if fusion_mode in ("weighted", "weighted_sum"):
        weight = float(np.clip(semantic_weight, 0.0, 1.0))
        return weight * semantic_map + (1.0 - weight) * instance_map
    raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")


def filter_small_components(
    score_map: np.ndarray,
    threshold: float,
    min_area: float,
) -> np.ndarray:
    area_threshold = int(np.floor(float(min_area)))
    if area_threshold <= 0:
        return score_map

    mask = score_map >= threshold
    if not mask.any():
        return score_map

    keep = np.zeros(mask.shape, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue

            component = []
            queue = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not visited[ny, nx]
                        and mask[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(component) >= area_threshold:
                ys, xs = zip(*component)
                keep[ys, xs] = True

    filtered = score_map.copy()
    filtered[~keep] = 0.0
    return filtered


def apply_final_score_params(
    semantic_map: torch.Tensor,
    instance_raw_map: torch.Tensor,
    instance_gated_map: torch.Tensor,
    presence_score: float,
    entry: dict[str, Any],
    params: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    use_sem_seg = bool_param(params, "use_sem_seg", entry, True)
    use_transformer_decoder = bool_param(
        params, "use_transformer_decoder", entry, True
    )
    use_presence_score = bool_param(params, "use_presence_score", entry, True)
    instance_filter_use_presence = bool_param(
        params, "instance_filter_use_presence", entry, True
    )
    fusion_mode = str(entry_param(params, "fusion_mode", entry, "max"))
    semantic_weight = float_param(params, "semantic_weight", entry, 0.5)
    presence_gain = float_param(params, "presence_gain", entry, 0.0)
    presence_bias = float_param(params, "presence_bias", entry, 0.5)
    prompt_weight = float_param(params, "prompt_weight", entry, 1.0)
    prompt_threshold = float_param(params, "prompt_threshold", entry, 0.0)
    min_prompt_component_area = float_param(
        params, "min_prompt_component_area", entry, 0.0
    )
    prompt_fusion_mode = str(entry_param(params, "prompt_fusion_mode", entry, "max"))
    concept_weight = float_param(params, "concept_weight", entry, 1.0)
    concept_threshold = float_param(params, "concept_threshold", entry, 0.0)
    prob_thd = float_param(params, "prob_thd", entry, 0.1)

    semantic_for_final = (
        semantic_map if use_sem_seg else torch.zeros_like(semantic_map)
    )
    instance_source = (
        instance_gated_map if instance_filter_use_presence else instance_raw_map
    )
    instance_for_final = (
        instance_source
        if use_transformer_decoder
        else torch.zeros_like(instance_source)
    )
    fused = fuse_score_maps(
        semantic_for_final,
        instance_for_final,
        semantic_weight,
        fusion_mode,
    )

    final_logits = torch.logit(fused.clamp(SCORE_EPS, 1.0 - SCORE_EPS))
    effective_presence = 1.0
    presence_logit = 0.0
    if use_presence_score:
        presence_logit = float(
            np.clip(presence_gain * (presence_score - presence_bias), -8.0, 8.0)
        )
        effective_presence = float(np.exp(presence_logit))
        final_logits = final_logits + presence_logit

    if prompt_weight > 0.0:
        final_logits = final_logits + float(np.log(prompt_weight))
        final_score = torch.sigmoid(final_logits)
    else:
        final_score = torch.zeros_like(fused)
    final_score = final_score * max(concept_weight, 0.0)
    final_score = final_score.clamp(0.0, 1.0)

    final_np = final_score.detach().cpu().numpy()
    if prompt_threshold > 0:
        final_np = np.where(final_np >= prompt_threshold, final_np, 0.0)
    final_np = filter_small_components(
        final_np,
        max(prompt_threshold, 1e-6),
        min_prompt_component_area,
    )
    final_score = torch.from_numpy(final_np).to(
        device=fused.device,
        dtype=fused.dtype,
    )

    resolved = {
        "use_sem_seg": use_sem_seg,
        "use_transformer_decoder": use_transformer_decoder,
        "use_presence_score": use_presence_score,
        "instance_filter_use_presence": instance_filter_use_presence,
        "fusion_mode": fusion_mode,
        "semantic_weight": semantic_weight,
        "presence_gain": presence_gain,
        "presence_bias": presence_bias,
        "presence_logit": presence_logit,
        "effective_presence": float(effective_presence),
        "presence_domain": "odds",
        "prompt_weight": prompt_weight,
        "prompt_threshold": prompt_threshold,
        "min_prompt_component_area": min_prompt_component_area,
        "prompt_fusion_mode": prompt_fusion_mode,
        "concept_weight": concept_weight,
        "concept_threshold": concept_threshold,
        "prob_thd": prob_thd,
    }
    return fused, final_score, resolved


def save_probability_map(
    probability: np.ndarray,
    output_path: Path,
) -> None:
    plt.imsave(
        output_path,
        probability,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )


def save_score_npz(
    score: np.ndarray,
    output_path: Path,
) -> None:
    np.savez_compressed(
        output_path,
        score=score.astype(np.float16, copy=False),
    )


def render_binary_overlay(
    image: Image.Image,
    mask: np.ndarray | None,
    color: tuple[int, int, int],
    alpha: float = 0.55,
) -> np.ndarray:
    raster = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = raster.copy()
    if mask is None:
        return overlay
    mask = mask.astype(bool)
    color_array = np.asarray(color, dtype=np.float32)
    blended = (
        raster.astype(np.float32) * (1.0 - alpha)
        + color_array[None, None, :] * alpha
    )
    overlay[mask] = blended[mask].astype(np.uint8)
    return overlay


def gt_label_color(label: int) -> tuple[int, int, int]:
    return (
        int((37 * label + 41) % 256),
        int((97 * label + 113) % 256),
        int((53 * label + 197) % 256),
    )


def save_gt_overview(
    image: Image.Image,
    gt_label: np.ndarray | None,
    ignore_index: int,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    if gt_label is None:
        return

    raster = np.asarray(image.convert("RGB"), dtype=np.float32)
    valid_mask = gt_label != ignore_index
    labels = [
        int(label)
        for label in sorted(np.unique(gt_label[valid_mask]).tolist())
    ] if valid_mask.any() else []

    overlay = raster.copy()
    for label in labels:
        mask = gt_label == label
        color = np.asarray(gt_label_color(label), dtype=np.float32)
        overlay[mask] = raster[mask] * (1.0 - alpha) + color * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    legend_height = max(5.0, 0.34 * max(len(labels), 1) + 1.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, legend_height),
        gridspec_kw={"width_ratios": [3.0, 1.0]},
    )

    axes[0].imshow(overlay)
    axes[0].set_title("Input + GT overlay")
    axes[0].axis("off")

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    axes[1].set_title("GT legend")

    if labels:
        step = 1.0 / (len(labels) + 1)
        for index, label in enumerate(labels):
            y_pos = 1.0 - (index + 1) * step
            color = np.asarray(gt_label_color(label), dtype=np.float32) / 255.0
            axes[1].scatter(
                [0.12],
                [y_pos],
                marker="s",
                s=140,
                color=[color],
            )
            axes[1].text(
                0.22,
                y_pos,
                f"GT {label}",
                ha="left",
                va="center",
                family="monospace",
                fontsize=10,
            )
    else:
        axes[1].text(
            0.05,
            0.95,
            "No valid GT labels",
            ha="left",
            va="top",
            family="monospace",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_normal_panel(
    image: Image.Image,
    prompt: str,
    output_path: Path,
    gt_mask: np.ndarray | None = None,
    pred_mask: np.ndarray | None = None,
) -> None:
    """Save the compact GT/prediction overlay view used during validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(render_binary_overlay(image, gt_mask, (255, 64, 64)))
    axes[0].set_title("GT mask")

    axes[1].imshow(render_binary_overlay(image, pred_mask, (64, 255, 96)))
    axes[1].set_title("Pred mask")

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(f'Prompt: "{prompt}"', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def metric_text(metrics: dict[str, Any] | None, threshold: float) -> str:
    if not metrics:
        return "GT metrics unavailable"

    def pct(value: Any) -> str:
        if value is None:
            return "nan"
        if isinstance(value, float) and np.isnan(value):
            return "nan"
        return f"{100.0 * float(value):.2f}"

    return "\n".join(
        [
            f"threshold = {threshold:.3f}",
            f"mIoU = {pct(metrics.get('miou'))}",
            f"IoU = {pct(metrics.get('iou'))}",
            f"Precision = {pct(metrics.get('precision'))}",
            f"Recall = {pct(metrics.get('recall'))}",
            f"F1 = {pct(metrics.get('fscore'))}",
            f"GT area = {pct(metrics.get('label_area_ratio'))}",
            f"Pred area = {pct(metrics.get('pred_area_ratio'))}",
        ]
    )


def save_panel(
    image: Image.Image,
    semantic: np.ndarray,
    instance_raw: np.ndarray,
    instance_gated: np.ndarray,
    fused_before_presence: np.ndarray,
    final_score: np.ndarray,
    presence_score: float,
    prompt: str,
    output_path: Path,
    gt_mask: np.ndarray | None = None,
    pred_mask: np.ndarray | None = None,
    final_metrics: dict[str, Any] | None = None,
    prediction_threshold: float = 0.0,
) -> None:
    if gt_mask is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Input image")

        axes[0, 1].imshow(
            semantic,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[0, 1].set_title("Semantic score")

        axes[0, 2].imshow(
            instance_raw,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[0, 2].set_title("Raw instance score")

        axes[1, 0].imshow(
            instance_gated,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[1, 0].set_title("Presence-gated instance")

        axes[1, 1].imshow(
            fused_before_presence,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[1, 1].set_title("Fusion before presence")

        axes[1, 2].imshow(
            final_score,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[1, 2].set_title(f"Final score\npresence={presence_score:.4f}")
    else:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Input image")

        axes[0, 1].imshow(render_binary_overlay(image, gt_mask, (255, 64, 64)))
        axes[0, 1].set_title("GT mask")

        axes[0, 2].imshow(
            semantic,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[0, 2].set_title("Semantic score")

        axes[1, 0].imshow(
            instance_raw,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[1, 0].set_title("Raw instance score")

        axes[1, 1].imshow(
            instance_gated,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[1, 1].set_title("Presence-gated instance")

        axes[1, 2].imshow(
            fused_before_presence,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[1, 2].set_title("Fusion before presence")

        axes[2, 0].imshow(
            final_score,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[2, 0].set_title(f"Final score\npresence={presence_score:.4f}")

        axes[2, 1].imshow(render_binary_overlay(image, pred_mask, (64, 255, 96)))
        axes[2, 1].set_title("Pred mask")

        axes[2, 2].text(
            0.02,
            0.98,
            metric_text(final_metrics, prediction_threshold),
            ha="left",
            va="top",
            family="monospace",
            fontsize=10,
        )
        axes[2, 2].set_title("Final metrics")

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(f'Prompt: "{prompt}"', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_gt_index(gt_dir: Path | None) -> dict[str, Path]:
    if gt_dir is None:
        return {}
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")

    candidates = sorted(
        path
        for path in gt_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    gt_index: dict[str, Path] = {}
    for path in sorted(candidates, key=lambda item: (len(item.parts), str(item))):
        gt_index.setdefault(path.stem, path)
    return gt_index


def load_gt_label(
    gt_path: Path | None,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    if gt_path is None:
        return None

    gt_image = Image.open(gt_path)
    if gt_image.size != (target_shape[1], target_shape[0]):
        resampling = getattr(Image, "Resampling", Image).NEAREST
        gt_image = gt_image.resize(
            (target_shape[1], target_shape[0]),
            resample=resampling,
        )
    gt = np.asarray(gt_image)
    if gt.ndim == 3:
        gt = gt[..., 0]
    return gt


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def binary_metrics(
    score_map: np.ndarray,
    gt_label: np.ndarray | None,
    positive_labels: list[int],
    threshold: float,
    ignore_index: int,
) -> dict[str, Any]:
    empty = {
        "iou": None,
        "miou": None,
        "background_iou": None,
        "precision": None,
        "recall": None,
        "fscore": None,
        "intersect": None,
        "union": None,
        "pred_area": None,
        "label_area": None,
        "valid_area": None,
        "pred_area_ratio": None,
        "label_area_ratio": None,
    }
    if gt_label is None or not positive_labels:
        return empty

    valid_mask = gt_label != ignore_index
    valid_area = int(valid_mask.sum())
    if valid_area == 0:
        return empty

    pred_mask = (score_map >= threshold) & valid_mask
    label_mask = np.isin(gt_label, positive_labels) & valid_mask

    tp = int((pred_mask & label_mask).sum())
    fp = int((pred_mask & ~label_mask & valid_mask).sum())
    fn = int((~pred_mask & label_mask & valid_mask).sum())
    tn = int((~pred_mask & ~label_mask & valid_mask).sum())

    pred_area = tp + fp
    label_area = tp + fn
    union = tp + fp + fn
    iou = safe_divide(tp, union)
    background_iou = safe_divide(tn, tn + fp + fn)
    valid_ious = [
        value for value in (iou, background_iou) if not np.isnan(value)
    ]
    miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    precision = safe_divide(tp, pred_area)
    recall = safe_divide(tp, label_area)
    if np.isnan(precision) or np.isnan(recall):
        fscore = float("nan")
    elif precision + recall == 0:
        fscore = 0.0
    else:
        fscore = safe_divide(2 * precision * recall, precision + recall)

    return {
        "iou": iou,
        "miou": miou,
        "background_iou": background_iou,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "intersect": tp,
        "union": union,
        "pred_area": pred_area,
        "label_area": label_area,
        "valid_area": valid_area,
        "pred_area_ratio": safe_divide(pred_area, valid_area),
        "label_area_ratio": safe_divide(label_area, valid_area),
    }


def prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def map_statistics(
    prefix: str,
    score_map: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    return {
        f"{prefix}_max": float(score_map.max()),
        f"{prefix}_mean": float(score_map.mean()),
        f"{prefix}_area_030": float((score_map >= threshold).mean()),
        f"{prefix}_area_050": float((score_map >= 0.50).mean()),
    }


def metric_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def stringify_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def copy_run_input(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return str(destination)
    return ""


def save_run_inputs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    copied_files = {
        "prompt_path": copy_run_input(
            args.prompt_path,
            output_dir / f"run_prompt_path{args.prompt_path.suffix}",
        ),
        "params_path": copy_run_input(
            args.params_path,
            output_dir / f"run_params_path{args.params_path.suffix}",
        ),
    }
    args_record = stringify_args(args)
    args_record["copied_files"] = copied_files

    args_path = output_dir / "run_args.json"
    with args_path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(args_record),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    return {
        "args": str(args_path),
        **copied_files,
    }


def finite_metric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def format_ratio(value: Any) -> str:
    number = finite_metric_value(value)
    if number is None:
        return "n/a"
    return f"{100.0 * number:.2f}"


def format_prompt_metric_row(
    row: dict[str, Any],
    image_index: int,
    num_images: int,
) -> str:
    final_prefix = "final_metric"
    semantic_prefix = "semantic_metric"
    fused_prefix = "fused_metric"
    gated_prefix = "instance_gated_metric"
    return (
        f"[{image_index}/{num_images}] "
        f"image={Path(str(row['image'])).name} "
        f"prompt#{row['prompt_index']:03d} "
        f"concept={row.get('concept_id') or 'none'} "
        f"prompt=\"{row['prompt']}\" "
        f"gt={'yes' if row.get('gt_available') else 'no'} "
        f"threshold={float(row['prediction_threshold']):.3f} "
        f"presence={float(row['presence_score']):.4f} "
        f"gate={float(row.get('param_effective_presence', 1.0)):.4f} "
        f"final(mIoU={format_ratio(row.get(f'{final_prefix}_miou'))}, "
        f"IoU={format_ratio(row.get(f'{final_prefix}_iou'))}, "
        f"P={format_ratio(row.get(f'{final_prefix}_precision'))}, "
        f"R={format_ratio(row.get(f'{final_prefix}_recall'))}, "
        f"F1={format_ratio(row.get(f'{final_prefix}_fscore'))}) "
        f"semantic(IoU={format_ratio(row.get(f'{semantic_prefix}_iou'))}, "
        f"mIoU={format_ratio(row.get(f'{semantic_prefix}_miou'))}) "
        f"inst_gated(IoU={format_ratio(row.get(f'{gated_prefix}_iou'))}) "
        f"fused(IoU={format_ratio(row.get(f'{fused_prefix}_iou'))})"
    )


def metric_average(rows: list[dict[str, Any]], key: str) -> tuple[int, float | None]:
    values = [
        value
        for row in rows
        if (value := finite_metric_value(row.get(key))) is not None
    ]
    if not values:
        return 0, None
    return len(values), float(np.mean(values))


def format_average_metric_row(
    rows: list[dict[str, Any]],
    prefix: str,
    label: str,
) -> str:
    miou_count, miou = metric_average(rows, f"{prefix}_miou")
    _, iou = metric_average(rows, f"{prefix}_iou")
    _, precision = metric_average(rows, f"{prefix}_precision")
    _, recall = metric_average(rows, f"{prefix}_recall")
    _, fscore = metric_average(rows, f"{prefix}_fscore")
    return (
        f"{label}: n={miou_count} "
        f"mIoU={format_ratio(miou)} "
        f"IoU={format_ratio(iou)} "
        f"Precision={format_ratio(precision)} "
        f"Recall={format_ratio(recall)} "
        f"F1={format_ratio(fscore)}"
    )


def main() -> None:
    args = parse_args()
    image_dir: Path = args.image_dir
    output_dir: Path = args.output_dir

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_files = save_run_inputs(args, output_dir)
    log_path = output_dir / "diagnosis_log.txt"
    log_file = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    prompt_entries = load_prompt_entries(args.prompt_path)
    prompts = [entry["prompt"] for entry in prompt_entries]
    score_params = load_score_params(args.params_path)
    analysis_threshold: float = args.analysis_threshold
    ignore_index: int = args.ignore_index
    gt_index = build_gt_index(args.gt_dir)

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )[: args.max_images]

    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    log(f"Run args JSON: {run_files['args']}")
    log(f"Copied prompt-path: {run_files['prompt_path']}")
    log(f"Copied params-path: {run_files['params_path']}")
    log(f"Log file: {log_path}")
    log(f"Found {len(image_paths)} diagnostic images.")
    log(f"Loaded {len(prompts)} prompts from {args.prompt_path}")
    if args.gt_dir is not None:
        log(f"Indexed {len(gt_index)} GT rasters from {args.gt_dir}")
    log(f"Loaded final-score parameters from {args.params_path}")

    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
    )

    processor = Sam3Processor(
        model,
        confidence_threshold=float(score_params["confidence_threshold"]),
        device="cuda",
    )

    summary_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for image_index, image_path in enumerate(image_paths, start=1):
            log(f"[{image_index}/{len(image_paths)}] image={image_path.name}")

            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            gt_path = gt_index.get(image_path.stem)
            gt_label = load_gt_label(gt_path, (height, width))

            state = processor.set_image(image)

            image_output_dir = output_dir / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)

            image.save(image_output_dir / "input.png")
            save_gt_overview(
                image=image,
                gt_label=gt_label,
                ignore_index=ignore_index,
                output_path=image_output_dir / "gt.png",
            )

            for prompt_index, entry in enumerate(prompt_entries):
                prompt = entry["prompt"]
                processor.reset_all_prompts(state)

                state = processor.set_text_prompt(
                    state=state,
                    prompt=prompt,
                )

                presence_score = float(
                    state["presence_score"].detach().float().cpu().item()
                )

                semantic_map = resize_score_map(
                    tensor_to_map(state["semantic_mask_logits"]),
                    height,
                    width,
                )
                masks = state["masks_logits"]

                if "object_score_raw" in state:
                    raw_object_scores = state["object_score_raw"]
                else:
                    gated_scores = state["object_score"].detach().float()
                    raw_object_scores = (
                        gated_scores / max(presence_score, 1e-6)
                    ).clamp(0.0, 1.0)

                gated_object_scores = state["object_score"]

                instance_raw_map = aggregate_instance_map(
                    masks=masks,
                    scores=raw_object_scores,
                    height=height,
                    width=width,
                )
                instance_gated_map = aggregate_instance_map(
                    masks=masks,
                    scores=gated_object_scores,
                    height=height,
                    width=width,
                )

                fused_before_presence, final_score, resolved_params = (
                    apply_final_score_params(
                        semantic_map=semantic_map,
                        instance_raw_map=instance_raw_map,
                        instance_gated_map=instance_gated_map,
                        presence_score=presence_score,
                        entry=entry,
                        params=score_params,
                    )
                )

                semantic_np = semantic_map.cpu().numpy()
                instance_raw_np = instance_raw_map.cpu().numpy()
                instance_gated_np = instance_gated_map.cpu().numpy()
                fused_np = fused_before_presence.cpu().numpy()
                final_np = final_score.cpu().numpy()

                positive_labels = entry.get("tnls_dl3", [])
                prediction_threshold = max(
                    analysis_threshold,
                    float(resolved_params["concept_threshold"]),
                    float(resolved_params["prob_thd"]),
                )
                final_metrics = binary_metrics(
                    final_np,
                    gt_label,
                    positive_labels,
                    prediction_threshold,
                    ignore_index,
                )
                gt_mask = None
                pred_mask = None
                if gt_label is not None and positive_labels:
                    valid_mask = gt_label != ignore_index
                    gt_mask = np.isin(gt_label, positive_labels) & valid_mask
                    pred_mask = (final_np >= prediction_threshold) & valid_mask

                prompt_name = safe_name(prompt)
                prompt_dir = image_output_dir / prompt_name
                prompt_dir.mkdir(parents=True, exist_ok=True)

                save_score_npz(semantic_np, prompt_dir / "semantic_score.npz")
                save_score_npz(instance_raw_np, prompt_dir / "instance_score_raw.npz")
                save_score_npz(
                    instance_gated_np,
                    prompt_dir / "instance_score_gated.npz",
                )
                save_score_npz(fused_np, prompt_dir / "fused_before_presence.npz")
                save_score_npz(final_np, prompt_dir / "final_score.npz")

                save_probability_map(semantic_np, prompt_dir / "semantic_score.png")
                save_probability_map(
                    instance_raw_np,
                    prompt_dir / "instance_score_raw.png",
                )
                save_probability_map(
                    instance_gated_np,
                    prompt_dir / "instance_score_gated.png",
                )
                save_probability_map(
                    fused_np,
                    prompt_dir / "fused_before_presence.png",
                )
                save_probability_map(final_np, prompt_dir / "final_score.png")

                panel_path = prompt_dir / "diagnostic_panel.png"
                save_panel(
                    image=image,
                    semantic=semantic_np,
                    instance_raw=instance_raw_np,
                    instance_gated=instance_gated_np,
                    fused_before_presence=fused_np,
                    final_score=final_np,
                    presence_score=presence_score,
                    prompt=prompt,
                    output_path=panel_path,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    final_metrics=final_metrics,
                    prediction_threshold=prediction_threshold,
                )
                shutil.copyfile(
                    panel_path,
                    result_dir
                    / f"{safe_name(image_path.stem)}_{prompt_name}_diagnostic_panel.png",
                )

                num_instances = int(prepare_instance_masks(masks).shape[0])

                row: dict[str, Any] = {
                    "image": str(image_path),
                    "gt_path": str(gt_path) if gt_path is not None else "",
                    "gt_available": gt_label is not None,
                    "concept_id": entry.get("concept_id"),
                    "concept_name": entry.get("concept_name"),
                    "concept_index": entry.get("concept_index"),
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "prompt_position": entry.get("prompt_position"),
                    "tnls_dl3": entry.get("tnls_dl3", []),
                    "isdeleted": entry.get("isdeleted", False),
                    "presence_score": presence_score,
                    "num_instances": num_instances,
                    "raw_object_score_max": (
                        float(
                            raw_object_scores.detach().float().max().cpu().item()
                        )
                        if raw_object_scores.numel() > 0
                        else 0.0
                    ),
                    "prediction_threshold": prediction_threshold,
                }
                for key, value in resolved_params.items():
                    row[f"param_{key}"] = value

                row.update(map_statistics("semantic", semantic_np, analysis_threshold))
                row.update(
                    map_statistics("instance_raw", instance_raw_np, analysis_threshold)
                )
                row.update(
                    map_statistics(
                        "instance_gated",
                        instance_gated_np,
                        analysis_threshold,
                    )
                )
                row.update(map_statistics("fused", fused_np, analysis_threshold))
                row.update(map_statistics("final", final_np, analysis_threshold))

                for metric_prefix, score_map in (
                    ("semantic_metric", semantic_np),
                    ("instance_raw_metric", instance_raw_np),
                    ("instance_gated_metric", instance_gated_np),
                    ("fused_metric", fused_np),
                    ("final_metric", final_np),
                ):
                    row.update(
                        prefixed_metrics(
                            metric_prefix,
                            binary_metrics(
                                score_map,
                                gt_label,
                                positive_labels,
                                prediction_threshold,
                                ignore_index,
                            ),
                        )
                    )

                log(format_prompt_metric_row(row, image_index, len(image_paths)))
                summary_rows.append(row)

                with open(
                    prompt_dir / "meta.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        json_safe(row),
                        file,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )

    csv_path = output_dir / "head_diagnosis_summary.csv"
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metric_fieldnames(summary_rows),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    log("")
    log(
        "Average metrics across processed image/prompt pairs "
        f"({len(image_paths)} images, {len(summary_rows)} rows):"
    )
    for prefix, label in (
        ("semantic_metric", "semantic"),
        ("instance_raw_metric", "instance_raw"),
        ("instance_gated_metric", "instance_gated"),
        ("fused_metric", "fused"),
        ("final_metric", "final"),
    ):
        log(format_average_metric_row(summary_rows, prefix, label))

    log("")
    log(f"Diagnosis complete: {output_dir}")
    log(f"Summary table: {csv_path}")
    log_file.close()


if __name__ == "__main__":
    main()
