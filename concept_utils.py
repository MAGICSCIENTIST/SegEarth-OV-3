import json
import os.path as osp


def load_concepts(path):
    if is_json_path(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        concepts = []
        for group in data.get('groups', []):
            for concept in group.get('concepts', []):
                concepts.append(concept)
        return concepts

    concepts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            prompts = [prompt.strip() for prompt in line.split(',')
                       if prompt.strip()]
            if not prompts:
                continue

            concept_name = prompts[0]
            concepts.append({
                'concept_id': concept_name,
                'concept_name': concept_name,
                'tnls_sources': [],
                'tnls_dl3': [],
                'prompts': prompts,
            })
    return concepts


def is_json_path(path):
    return osp.splitext(path)[1].lower() == '.json'
