import os
import os.path as osp

import numpy as np
from PIL import Image


from tools.visual_utils import PROMPT_FILENAME_SEPARATOR, safe_filename


def ensure_tnls_output_dirs(output_dir):
    dirs = dict(
        prompt=osp.join(output_dir, 'promot_response'),
        visual=osp.join(output_dir, 'visual_response'),
        pred=osp.join(output_dir, 'pred'))
    for save_dir in dirs.values():
        os.makedirs(save_dir, exist_ok=True)
    return dirs


def save_tnls_outputs(data_sample, concepts, output_dir):
    if output_dir is None:
        return

    dirs = ensure_tnls_output_dirs(output_dir)
    basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
    pred_label = data_sample['pred_sem_seg']['data'].squeeze()
    raw_prompt_scores = data_sample['raw_prompt_scores']['data']

    prompt_name_counts = {}
    prompt_idx = 0
    for concept in concepts:
        prompts = concept.get('prompts', [])
        if isinstance(prompts, str):
            prompts = [prompts]
        for prompt_text in prompts:
            score = raw_prompt_scores[prompt_idx].detach().cpu().numpy()
            prompt_name = safe_filename(prompt_text)
            prompt_name_counts[prompt_name] = (
                prompt_name_counts.get(prompt_name, 0) + 1)
            suffix = prompt_name
            if prompt_name_counts[prompt_name] > 1:
                suffix = f'{prompt_name}_{prompt_name_counts[prompt_name]}'
            score_file = osp.join(
                dirs['prompt'],
                f'{basename}{PROMPT_FILENAME_SEPARATOR}{suffix}.npz')
            np.savez_compressed(
                score_file, score=score.astype(np.float16, copy=False))
            prompt_idx += 1

    for concept_idx, concept in enumerate(concepts):
        concept_id = concept.get('concept_id')
        concept_mask = (
            (pred_label == concept_idx).detach().cpu().numpy().astype(
                np.uint8) * 255)
        Image.fromarray(concept_mask).save(
            osp.join(dirs['visual'], f'{basename}_{concept_id}.jpg'))

    output_mask = pred_label.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(output_mask).save(osp.join(dirs['pred'], f'{basename}.png'))
