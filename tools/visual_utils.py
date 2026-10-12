"""Shared raster visualizations; no model, training, config or SAM imports.

Callers prepare predictions and GT; this module only renders supplied arrays.
Legacy entry points re-export these helpers for compatibility.
"""
from __future__ import annotations

import os
import os.path as osp
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROMPT_FILENAME_SEPARATOR = '__PROMPT__'


def safe_filename(text, max_length=120):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(text))
    text = re.sub(r'\s+', '_', text).strip(' ._')
    return (text[:max_length].rstrip(' ._') or 'prompt')


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def panel_map(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.clip(np.asarray(value).squeeze(), 0.0, 1.0)


def render_labeled_overlay(raster, labels, names, palette, alpha=0.55,
                           ignore_index=255):
    """Blend class masks and put each class name inside its largest region."""
    import cv2

    labels = as_numpy(labels)
    layer = np.zeros_like(raster)
    valid = np.zeros(labels.shape, dtype=bool)
    present = []
    for index, (name, color) in enumerate(zip(names, palette)):
        mask = labels == index
        if index == ignore_index or not mask.any():
            continue
        layer[mask] = color
        valid |= mask
        present.append((name, mask, color))
    canvas = Image.fromarray(blend_overlay(raster, layer, valid, alpha))
    draw = ImageDraw.Draw(canvas)
    font = load_chinese_font(max(12, min(22, min(labels.shape) // 45)))
    for name, mask, color in present:
        # Place text on a real pixel of the largest component, even for roads.
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        yy, xx = np.nonzero(components == largest)
        center = int(np.argmin((xx - xx.mean())**2 + (yy - yy.mean())**2))
        text = str(name)
        if not font_supports_text(font, text):
            text = text.encode('ascii', 'replace').decode('ascii')
        box = draw.textbbox((0, 0), text, font=font)
        width, height = box[2] - box[0] + 8, box[3] - box[1] + 8
        x = max(0, min(int(xx[center]) - width // 2, canvas.width - width))
        y = max(0, min(int(yy[center]) - height // 2, canvas.height - height))
        text_background = tuple(int(0.45 * c + 0.55 * 255) for c in color)
        draw.rectangle((x, y, x + width, y + height), fill=text_background,
                       outline='black')
        draw.text((x + 4, y + 4 - box[1]), text, fill='black', font=font)
    return np.asarray(canvas)


def save_labeled_comparison(raster, gt, prediction, names, palette,
                            output_path, alpha=0.55, ignore_index=255,
                            right_title='Prediction'):
    """Two copies of the remote-sensing raster, with GT/prediction overlays."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    left = render_labeled_overlay(raster, gt, names, palette, alpha, ignore_index)
    right = render_labeled_overlay(
        raster, prediction, names, palette, alpha, ignore_index)
    save_compare_image(left, right, 'GT', right_title, output_path,
                       load_chinese_font())


def save_binary_comparison(raster, gt_mask, probability, name, color,
                           output_path, threshold=0.5, alpha=0.55,
                           valid_mask=None):
    """Render a concept/prompt independently (before multiclass argmax)."""
    probability = as_numpy(probability)
    valid = np.ones(probability.shape, dtype=bool) if valid_mask is None else valid_mask
    gt = np.where(as_numpy(gt_mask) & valid, 0, 255)
    prediction = np.where((probability >= threshold) & valid, 0, 255)
    save_labeled_comparison(
        raster, gt, prediction, [name], [color], output_path, alpha,
        right_title=f'{name} | threshold={threshold:g}')


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def load_chinese_font(size=18):
    font_candidates = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
        r'C:\Windows\Fonts\arialuni.ttf',
    ]
    for font_path in font_candidates:
        if osp.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def label_color(label):
    label = int(label)
    return np.array([
        (37 * label + 17) % 256,
        (67 * label + 29) % 256,
        (97 * label + 43) % 256
    ],
                    dtype=np.uint8)


def concept_color(index):
    index = int(index) + 1
    return np.array([
        (53 * index + 31) % 256,
        (97 * index + 47) % 256,
        (149 * index + 61) % 256
    ],
                    dtype=np.uint8)


def concept_gt_color(concept, fallback_index):
    labels = concept.get('tnls_dl3', [])
    if labels:
        return label_color(labels[0])
    return concept_color(fallback_index)


def font_supports_text(font, text):
    try:
        font.getbbox(text)
        return True
    except UnicodeEncodeError:
        return False


def render_gt_label(label, ignore_index):
    label = as_numpy(label)
    image = np.zeros((*label.shape, 3), dtype=np.uint8)
    for value in np.unique(label):
        if int(value) == ignore_index:
            continue
        image[label == value] = label_color(value)
    return image


def load_raster(image_path, shape):
    raster = Image.open(image_path).convert('RGB')
    raster = raster.resize((shape[1], shape[0]))
    return np.asarray(raster, dtype=np.uint8)


def blend_overlay(raster, color_layer, mask, alpha):
    overlay = raster.copy()
    blended = (
        raster.astype(np.float32) * (1 - alpha) +
        color_layer.astype(np.float32) * alpha)
    overlay[mask] = blended[mask].astype(np.uint8)
    return overlay


def render_gt_overlay(raster, label, ignore_index, alpha):
    label_np = as_numpy(label)
    label_vis = render_gt_label(label, ignore_index)
    valid_mask = label_np != ignore_index
    return blend_overlay(raster, label_vis, valid_mask, alpha)


def render_prediction(raster, pred_label, concepts, alpha):
    pred_label = as_numpy(pred_label)
    color_layer = np.zeros((*pred_label.shape, 3), dtype=np.uint8)
    valid_mask = np.zeros(pred_label.shape, dtype=bool)
    for concept_idx, concept in enumerate(concepts):
        concept_mask = pred_label == concept_idx
        color_layer[concept_mask] = concept_gt_color(
            concept, concept_idx)
        valid_mask |= concept_mask
    return blend_overlay(raster, color_layer, valid_mask, alpha)


def render_concept_mask(raster, pred_label, concept_idx, concept, alpha):
    pred_label = as_numpy(pred_label)
    mask = pred_label == concept_idx
    color_layer = np.zeros((*pred_label.shape, 3), dtype=np.uint8)
    color_layer[mask] = concept_gt_color(concept, concept_idx)
    return blend_overlay(raster, color_layer, mask, alpha)


def save_compare_image(left, right, left_title, right_title, out_file, font):
    title_height = 32
    h, w = left.shape[:2]
    canvas = np.full((h + title_height, w * 2, 3), 255, dtype=np.uint8)
    canvas[title_height:, :w] = left
    canvas[title_height:, w:] = right
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    if not font_supports_text(font, right_title):
        right_title = ''.join(char if ord(char) < 128 else '_'
                              for char in right_title)
    draw.text((6, 5), left_title, fill=(0, 0, 0), font=font)
    draw.text((w + 6, 5), right_title, fill=(0, 0, 0), font=font)
    image.save(out_file)


def save_tnls_concept_comparisons(data_sample,
                                  concepts,
                                  vis_dir,
                                  ignore_index=255,
                                  score_threshold=0.1,
                                  gt_alpha=0.55):
    if vis_dir is None:
        return

    vis_root = osp.join(vis_dir, 'vis_data')
    vis_image_dir = osp.join(vis_root, 'vis_image')
    prompt_dir = osp.join(vis_root, 'promot')
    visual_dir = osp.join(vis_root, 'visual')
    os.makedirs(vis_image_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
    pred_label = data_sample['pred_sem_seg']['data'].squeeze()
    gt_label = data_sample['gt_sem_seg']['data'].squeeze()
    raw_prompt_scores = data_sample.get('raw_prompt_scores', {}).get(
        'data', None)
    font = load_chinese_font()
    image_shape = gt_label.shape[-2:]
    raster = load_raster(data_sample['img_path'], image_shape)

    gt_vis = render_gt_overlay(raster, gt_label, ignore_index, gt_alpha)
    pred_vis = render_prediction(raster, pred_label, concepts, gt_alpha)
    save_compare_image(
        gt_vis, pred_vis, 'GT label', 'prediction',
        osp.join(vis_image_dir, f'{basename}.jpg'), font)

    if raw_prompt_scores is not None:
        prompt_name_counts = {}
        prompt_idx = 0
        for concept_idx, concept in enumerate(concepts):
            prompts = concept.get('prompts', [])
            if isinstance(prompts, str):
                prompts = [prompts]
            for prompt_text in prompts:
                score = raw_prompt_scores[prompt_idx].detach().float().cpu().numpy()
                prompt_mask = score > score_threshold
                prompt_layer = np.zeros(
                    (*prompt_mask.shape, 3), dtype=np.uint8)
                prompt_layer[prompt_mask] = concept_gt_color(
                    concept, concept_idx)
                prompt_vis = blend_overlay(
                    raster, prompt_layer, prompt_mask, gt_alpha)
                prompt_name = safe_filename(prompt_text)
                prompt_name_counts[prompt_name] = (
                    prompt_name_counts.get(prompt_name, 0) + 1)
                suffix = prompt_name
                if prompt_name_counts[prompt_name] > 1:
                    suffix = f'{prompt_name}_{prompt_name_counts[prompt_name]}'
                save_compare_image(
                    gt_vis, prompt_vis, 'GT label', prompt_text,
                    osp.join(
                        prompt_dir,
                        f'{basename}{PROMPT_FILENAME_SEPARATOR}{suffix}.jpg'),
                    font)
                prompt_idx += 1

    for concept_idx, concept in enumerate(concepts):
        concept_id = concept.get('concept_id')
        concept_name = concept.get('concept_name') or concept_id
        concept_vis = render_concept_mask(
            raster, pred_label, concept_idx, concept, gt_alpha)
        save_compare_image(
            gt_vis, concept_vis, 'GT label', concept_name,
            osp.join(visual_dir, f'{basename}_{concept_id}.jpg'), font)


def save_probability_map(
    probability: np.ndarray,
    output_path: Path,
) -> None:
    plt = _pyplot()
    plt.imsave(
        output_path,
        probability,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
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
    plt = _pyplot()
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
    plt = _pyplot()
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
    plt = _pyplot()
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



def draw_sem_seg(image, sem_seg, classes, palette, get_center_loc,
                 alpha=0.7, with_labels=True):
    """Rendering backend for the MMSeg adapter, independent of MMSeg."""
    import cv2
    num_classes = len(classes)

    sem_seg = as_numpy(sem_seg)
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    mask = np.zeros_like(image, dtype=np.uint8)
    for label, color in zip(labels, colors):
        mask[sem_seg[0] == label, :] = color

    if with_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.05
        fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
        fontColor = (0, 0, 0)
        if image.shape[0] < 300 or image.shape[1] < 300:
            thickness = 1
            rectangleThickness = 1
        else:
            thickness = 2
            rectangleThickness = 2
        lineType = 2

        masks = sem_seg[0] == labels[:, None, None]
        masks = masks.astype(np.uint8)
        for mask_num in range(len(labels)):
            classes_id = labels[mask_num]
            classes_color = colors[mask_num]
            loc = get_center_loc(masks[mask_num])
            text = classes[classes_id]
            (label_width, label_height), baseline = cv2.getTextSize(
                text, font, fontScale, thickness)
            mask = cv2.rectangle(mask, loc,
                                 (loc[0] + label_width + baseline,
                                  loc[1] + label_height + baseline),
                                 classes_color, -1)
            mask = cv2.rectangle(mask, loc,
                                 (loc[0] + label_width + baseline,
                                  loc[1] + label_height + baseline),
                                 (0, 0, 0), rectangleThickness)
            mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                               font, fontScale, fontColor, thickness,
                               lineType)

    color_seg = (image * (1 - alpha) + mask * alpha).astype(
        np.uint8)
    return color_seg


def save_training_comparisons(image_path, gt_label, pred_label, concepts,
                              save_dir, epoch, ignore_index=255, gt_alpha=0.55):
    """Legacy training comparison layout, accepting prepared predictions."""
    vis_root = osp.join(save_dir, 'vis_data')
    vis_image_dir = osp.join(vis_root, 'vis_image')
    visual_dir = osp.join(vis_root, 'visual')
    os.makedirs(vis_image_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    epoch_suffix = f'epoch_{epoch:04d}'
    basename = osp.splitext(osp.basename(image_path))[0]
    font = load_chinese_font()
    image_shape = gt_label.shape[-2:]
    raster = load_raster(image_path, image_shape)

    gt_vis = render_gt_overlay(raster, gt_label, ignore_index, gt_alpha)
    pred_vis = render_prediction(raster, pred_label, concepts, gt_alpha)
    save_compare_image(
        gt_vis, pred_vis, 'GT label', 'prediction',
        osp.join(vis_image_dir, f'{basename}_{epoch_suffix}.jpg'), font)

    for concept_idx, concept in enumerate(concepts):
        concept_id = safe_filename(concept.get('concept_id') or concept_idx)
        concept_name = concept.get('concept_name') or concept.get(
            'concept_id') or str(concept_idx)
        concept_vis = render_concept_mask(
            raster, pred_label, concept_idx, concept, gt_alpha)
        save_compare_image(
            gt_vis, concept_vis, 'GT label', concept_name,
            osp.join(
                visual_dir,
                f'{basename}_{concept_id}_{epoch_suffix}.jpg'), font)
