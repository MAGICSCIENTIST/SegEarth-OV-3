from typing import List, Optional

import numpy as np
from mmengine.structures import PixelData
from mmseg.registry import VISUALIZERS
from mmseg.visualization import SegLocalVisualizer

from tools.visual_utils import (
    draw_sem_seg,
    load_chinese_font,
    label_color,
    concept_color,
    concept_gt_color,
    font_supports_text,
    render_gt_label,
    load_raster,
    blend_overlay,
    render_gt_overlay,
    render_prediction,
    render_concept_mask,
    save_compare_image,
    save_tnls_concept_comparisons,
)

from custom_savers import PROMPT_FILENAME_SEPARATOR, safe_filename


@VISUALIZERS.register_module()
class BlackTextSegLocalVisualizer(SegLocalVisualizer):
    """Segmentation visualizer with black class-label text."""

    def _draw_sem_seg(self,
                      image: np.ndarray,
                      sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List],
                      with_labels: Optional[bool] = True) -> np.ndarray:
        color_seg = draw_sem_seg(
            image, sem_seg.cpu().data, classes, palette,
            self._get_center_loc, self.alpha, with_labels)
        self.set_image(color_seg)
        return color_seg
