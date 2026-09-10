from typing import List, Optional
import os
import os.path as osp

import cv2
import numpy as np
import torch
from mmengine.structures import PixelData
from mmseg.registry import VISUALIZERS
from mmseg.visualization import SegLocalVisualizer
from PIL import Image, ImageDraw, ImageFont

from custom_savers import PROMPT_FILENAME_SEPARATOR, safe_filename


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
    label = label.detach().cpu().numpy()
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
    label_np = label.detach().cpu().numpy()
    label_vis = render_gt_label(label, ignore_index)
    valid_mask = label_np != ignore_index
    return blend_overlay(raster, label_vis, valid_mask, alpha)


def render_prediction(raster, pred_label, concepts, alpha):
    pred_label = pred_label.detach().cpu().numpy()
    color_layer = np.zeros((*pred_label.shape, 3), dtype=np.uint8)
    valid_mask = np.zeros(pred_label.shape, dtype=bool)
    for concept_idx, concept in enumerate(concepts):
        concept_mask = pred_label == concept_idx
        color_layer[concept_mask] = concept_gt_color(
            concept, concept_idx)
        valid_mask |= concept_mask
    return blend_overlay(raster, color_layer, valid_mask, alpha)


def render_concept_mask(raster, pred_label, concept_idx, concept, alpha):
    pred_label = pred_label.detach().cpu().numpy()
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


@VISUALIZERS.register_module()
class BlackTextSegLocalVisualizer(SegLocalVisualizer):
    """Segmentation visualizer with black class-label text."""

    def _draw_sem_seg(self,
                      image: np.ndarray,
                      sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List],
                      with_labels: Optional[bool] = True) -> np.ndarray:
        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data
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

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            masks = masks.astype(np.uint8)
            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(masks[mask_num])
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

        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg
