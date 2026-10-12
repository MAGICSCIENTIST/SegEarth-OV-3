"""Audit saved CX-B results and write a Chinese per-configuration report.

Reads metrics/checkpoints only; does not train or run SAM inference.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
from openpyxl import load_workbook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cx_b.modules import load_variant
from cx_b.configuration import (
    COMBINATION_CONFIGS, PROMPT_METHODS, CONCEPT_METHODS, LOSS_METHODS,
    combination_name, resolve_combination_config,
)

METRICS = ('IoU', 'Acc', 'Precision', 'Recall', 'Fscore')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def f(value, signed=False, digits=3):
    if value is None or not math.isfinite(float(value)):
        return 'NA'
    return format(float(value), f'+.{digits}f' if signed else f'.{digits}f')


def table(headers, rows):
    clean = lambda v: str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(clean, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(clean, row)) + ' |' for row in rows])


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def link(path, label=None):
    return f'[{label or path.name}]({path.as_posix()})'


def points(row):
    return [(0, row['initial_metrics'])] + [(h['epoch'], h['metrics']) for h in row['history']]


def peak(row, metric):
    candidates = [(epoch, metrics[metric]) for epoch, metrics in points(row)
                  if metrics.get(metric) is not None]
    maximum = max(v for _, v in candidates)
    epochs = [e for e, v in candidates if abs(v - maximum) <= 1e-12]
    return f'{f(maximum * 100)} @ ' + ', '.join(map(str, epochs))


def load_baseline(root):
    book = load_workbook(root / 'results.xlsx', data_only=True, read_only=True)
    data = {}
    for sheet in book:
        rows = list(sheet.values)
        data[sheet.title] = [dict(zip(rows[0], row)) for row in rows[1:]]
    book.close()
    tree = ast.parse((root / 'cfg_loveda.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
    model = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
    return dict(summary=data['summary'][-1], model=model,
                concepts={r['Class']: r for r in data['per_class']},
                prompts={r['Prompt']: r for r in data['per_prompt']})


def inspect_run(directory, baseline):
    snapshot = read_json(directory / 'run_config.json')
    done = read_json(directory / 'completed.json')
    assert done['status'] == 'complete' and done['fingerprint'] == snapshot['fingerprint'], directory
    prompts = read_json(directory / 'prompt_metrics.json')
    concepts = read_json(directory / 'concept_metrics.json')
    summary = read_json(directory / 'summary_metrics.json')['metrics']
    cx = snapshot['config']['cx_b']
    assert cx['prompt_init'] == 'scratch', 'This audit expects scratch runs.'
    source_checks = {}
    for name, digest in snapshot['inputs'].get('implementation', {}).items():
        path = Path(name)
        source_checks[name] = ('一致' if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                               else '变化或不可访问')
    checks = []
    for row in prompts + concepts:
        final = row.get('best_metrics', row.get('metrics'))
        name = row.get('prompt', row.get('concept_id'))
        candidates = points(row)
        maximum = max(m['IoU'] for _, m in candidates)
        assert abs(maximum - final['IoU']) < 1e-10, (directory, name, 'peak mismatch')
        chosen = next(m for epoch, m in candidates if epoch == row['best_epoch'])
        for metric in METRICS:
            assert abs(chosen[metric] - final[metric]) < 1e-10
            assert abs(final[metric] - row['initial_metrics'][metric] - row['metric_gains'][metric]) < 1e-10
        tp, fp, fn = final['TP'], final['FP'], final['FN']
        assert abs(final['IoU'] - tp / (tp + fp + fn)) < 1e-10
        base_gt = baseline['prompts'][name if 'prompt' in row else next(
            p['prompt'] for p in prompts if p['concept_id'] == name)]['GTPixels']
        checks.append(final['GTPixels'] == base_gt)
    assert all(checks), 'Different GT pixel counts from baseline.'
    for metric, summary_key in [('IoU', 'mIoU'), ('Acc', 'mAcc'),
                                ('Fscore', 'mFscore'), ('Precision', 'mPrecision'),
                                ('Recall', 'mRecall')]:
        assert abs(mean([r['metrics'][metric] for r in concepts]) * 100 - summary[summary_key]) < 1e-8
    modules = {}
    hashes = {}
    prompt_folder = 'dynamic_prompt' if cx['prompt_method'] == 'dynamic_head' else 'prompt_checkpoints'
    for row in prompts:
        path = directory / prompt_folder / f'prompt_{row["prompt_index"]:03d}/best.pth'
        module, metadata = load_variant(str(path))
        assert metadata['run_fingerprint'] == snapshot['fingerprint']
        assert metadata['epoch'] == row['best_epoch']
        for metric in METRICS:
            assert abs(metadata['validation_metrics'][metric] - row['best_metrics'][metric]) < 1e-10
        modules[('prompt', row['prompt_index'])] = (module, metadata)
        # Hash tensor values only: checkpoint metadata varies across methods.
        digest = hashlib.sha256()
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        hashes[row['prompt']] = digest.hexdigest()
    for row in concepts:
        paths = list((directory / 'concept').glob(f'concept_{row["concept_index"]:03d}_*/best.pth'))
        assert len(paths) == 1
        module, metadata = load_variant(str(paths[0]))
        assert metadata['run_fingerprint'] == snapshot['fingerprint']
        assert metadata['epoch'] == row['best_epoch']
        for metric in METRICS:
            assert abs(metadata['validation_metrics'][metric] - row['metrics'][metric]) < 1e-10
        modules[('concept', row['concept_index'])] = (module, metadata)
    log = (directory / 'run.log').read_text(encoding='utf-8')
    duration = re.findall(r'CX-B training ended .*?elapsed=(\S+)', log)
    current, registered_path = resolve_combination_config(
        cx['prompt_method'], cx['concept_method'], cx['loss_method'])
    config_changes = []
    def compare(old, new, prefix=''):
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                compare(old.get(key), new.get(key), prefix + '.' + key if prefix else key)
        elif old != new:
            config_changes.append([prefix, json.dumps(old, ensure_ascii=False),
                                   json.dumps(new, ensure_ascii=False)])
    compare(snapshot['config'], current)
    return dict(name=directory.parent.name, directory=directory, snapshot=snapshot,
                done=done, prompts=prompts, concepts=concepts, summary=summary,
                cx=cx, modules=modules, source_checks=source_checks, prompt_hashes=hashes,
                config_changes=config_changes, registered_path=registered_path,
                duration=duration[-1] if duration else '未记录')


def stage_tables(rows, prompt=False):
    out = []
    for r in rows:
        name = r['prompt'] if prompt else r['concept_id']
        initial = r['initial_metrics']
        final = r['best_metrics'] if prompt else r['metrics']
        last = r['history'][-1] if r['history'] else dict(epoch=0, metrics=final)
        gain = (final['IoU'] - initial['IoU']) * 100
        if r['status'] in ('identity', 'fixed'):
            result = ('单 prompt 直通' if r['status'] == 'identity'
                      else '固定评分，无训练' if prompt else '固定聚合，无训练')
        elif r['best_epoch'] == 0:
            result = '训练未胜过 epoch 0'
        else:
            result = 'IoU 提升' if gain > 1e-8 else '无 IoU 提升'
        out.append([name, f(initial['IoU']*100), f(final['IoU']*100), f(gain, True),
                    r['best_epoch'], last['epoch'], f(last['metrics']['IoU']*100),
                    *[f((final[k]-initial[k])*100, True) for k in ('Acc', 'Precision', 'Recall', 'Fscore')],
                    result])
    return table(['Prompt' if prompt else 'Concept', '初始 IoU', '选中 IoU', 'ΔIoU',
                  '最佳 epoch', '训练末 epoch', '末轮 IoU', 'ΔAcc', 'ΔPrecision',
                  'ΔRecall', 'ΔF1', '判断'], out)


def metric_peak_table(rows, prompt=False):
    return table(['Prompt' if prompt else 'Concept', 'IoU 峰值 @ epoch',
                  'Precision 峰值 @ epoch', 'Recall/Acc 峰值 @ epoch', 'F1 峰值 @ epoch'],
                 [[r['prompt'] if prompt else r['concept_id'],
                   *[peak(r, k) for k in ('IoU', 'Precision', 'Recall', 'Fscore')]] for r in rows])


def recommendation(run, runs):
    cx, summary = run['cx'], run['summary']
    ref = next(r for r in runs if r['cx']['prompt_method']=='A3-2'
               and r['cx']['concept_method']=='A3-2' and r['cx']['loss_method']=='A3-2')
    key = (cx['prompt_method'], cx['concept_method'], cx['loss_method'])
    if cx['prompt_method'] == 'segov3':
        return '复原基线有明确价值：11个prompt和7个concept均无训练，epoch0即最终结果。此前全量1669张最终多分类预测与历史Eval逐像素一致，mIoU=47.414491%；本次独立二分类40.907263%可用作同口径基准。不能把固定路径解释为A3-2 loss有效，也不必为它追加epoch或比较三个loss。'
    if key == ('A3-2', 'A3-2', 'A3-2'):
        return '作为三参数校准加均值聚合的对照有价值，但不建议作为当前最终方案。均值融合明显稀释 barren 的强 prompt，background 虽从初始化大幅提升，仍伴随较多误检。'
    if cx['prompt_method']=='A3-2' and cx['concept_method']=='A3-2':
        delta=summary['mIoU']-ref['summary']['mIoU']
        foreground = mean([c['metrics']['IoU']*100 for c in run['concepts'] if c['concept_id']!='background'])
        ref_foreground = mean([c['metrics']['IoU']*100 for c in ref['concepts'] if c['concept_id']!='background'])
        return f'保留为损失消融结果：七类独立concept mIoU比A3-2损失低 {f(-delta)} pp，但去掉background后的六类均值反而高 {f(foreground-ref_foreground)} pp。七类排序主要受background驱动，不能认定此前景识别普遍更差；先做最终多分类复评，再决定是否继续扩展。'
    if cx['concept_method']=='max':
        return '适合作为低开销聚合对照。它没有 concept 可学习参数，收益来自 prompt 校准和固定 max 规则；max 对 forest 的误检较敏感，继续增加“concept 训练轮数”没有意义。'
    if cx['concept_method']=='logsumexp':
        return '聚合规则有一定价值，但温度训练的边际收益很小。应区分选择 logsumexp 本身与训练温度带来的收益；barren 选中 epoch 0，优先与固定温度和单 soil 对照，再决定是否保留温度训练。'
    if cx['concept_method']=='dynamic':
        return '有明确的 concept 聚合训练价值，主要体现在 forest 和 barren；building 相对 max 的收益很小，barren 仍低于单 soil。适合保留，并做类别级简化和统一多分类复评。'
    if cx['concept_method']=='reliability_dynamic':
        return '值得保留为主要候选：三个可训练 concept 均改善，barren 相对同配置单 soil 也有小幅收益。与 region_dynamic 总体差异很小，不能凭单 seed 宣称明显优劣。'
    if cx['concept_method']=='region_dynamic':
        return '当前已完成组合中独立 concept mIoU 最高，建议作为优先复评候选；优势主要来自 barren/forest。与 reliability_dynamic 的总体差距很小，需要重复 seed 和统一多分类评估确认。'
    if cx['concept_method']=='sam_region_dynamic':
        return '相对本配置初始聚合器有训练收益，但本次在线SAM区域特征未表现出优于region_dynamic或reliability_dynamic的独立concept mIoU。与普通dynamic的差距很小；在线与float16缓存还存在精度/执行方式混杂。考虑日志耗时，暂不优先扩大在线训练，应先验证最终多分类收益与推理成本。'
    if cx['prompt_method']=='dynamic_head':
        match=next(r for r in runs if r['cx']['prompt_method']=='A3-2'
                   and r['cx']['concept_method']=='A3-2' and r['cx']['loss_method']==cx['loss_method'])
        delta=summary['mIoU']-match['summary']['mIoU']
        if cx['loss_method']=='A3-2':
            return f'动态 prompt 头有价值：相对相同损失、相同均值聚合的 A3 prompt 提升 {f(delta)} 个百分点，barren/tree/water 受益。仍受均值聚合限制，尚不能等同于最终分割优于 base。'
        return f'动态 prompt 头相对同损失的标量 A3 头提升 {f(delta)} 个百分点，有局部价值；但当前损失与均值聚合组合的总体结果较弱，建议保留作消融，优先处理 background 与 barren，而不是全量延长训练。'
    return '需进一步统一口径复评。'


def synthesize(runs, incomplete, binary_base, verification_path):
    """Aggregate audited findings after all per-run chapters were written/read."""
    identity = lambda r: tuple(r['cx'][k] for k in ('prompt_method','concept_method','loss_method'))
    indexed = {identity(r): r for r in runs}
    done = set(indexed)
    pending = set(COMBINATION_CONFIGS)-done
    mean_ref = indexed[('A3-2','A3-2','A3-2')]
    max_ref = indexed[('A3-2','max','A3-2')]
    region = indexed[('A3-2','region_dynamic','A3-2')]
    reliability = indexed[('A3-2','reliability_dynamic','A3-2')]
    sam = indexed[('A3-2','sam_region_dynamic','A3-2')]
    sections = [
        '# 综合结论：现有实验支持哪些方法',
        '本节在前述逐配置章节生成并从磁盘重新读取后汇总。统计单位是方法组合，不把相同固定规则的loss标签当成独立有效性证据。',
        '## 1. 可以确认的事实与不能扩大的结论',
        table(['问题','现有证据','结论边界'],[
            ['是否复原Eval LoveDA','segov3+max：全量1669张最终预测零差异，最终mIoU=47.414491%；独立二分类=40.907263%','复原成立；没有训练收益'],
            ['prompt参数优化是否有效','A3-2/A3-2/A3-2中'+str(sum(p['iou_gain_vs_initial']>1e-10 for p in mean_ref['prompts']))+'/11 prompt的IoU高于自身epoch0','只证明部分prompt联合参数优化有效；不能分别归因于α/gain/intercept'],
            ['dynamic_head是否优于标量A3','相同mean concept、相同loss下，三个loss的已完成配对均有正的宏平均增益','单seed、当前验证集和三组损失配置内成立；不是任何concept下都成立'],
            ['可训练concept是否有价值','dynamic/reliability/region/sam_region均高于本配置初始聚合器和本配置同prompt的max','当前二分类验证有正证据，最终多分类收益未确认'],
            ['哪个loss目前更合适','A3-2在标量和动态prompt、固定mean concept的两组配对中均最高','支持先用A3-2；尚未验证动态concept上的loss排序'],
            ['region一定优于reliability吗',f'独立concept mIoU仅差 {f(region["summary"]["mIoU"]-reliability["summary"]["mIoU"],digits=6)} pp','单seed无法证明稳定优越；都保留为候选'],
            ['SAM区域分支是否值得优先扩展',f'sam_region={f(sam["summary"]["mIoU"])}%，比region {f(sam["summary"]["mIoU"]-region["summary"]["mIoU"],True)} pp','当前未显示必要性；执行方式/精度也有混杂，不能归因为区域特征无用'],
            ['是否已有训练组合超过最终base','只有segov3复原组合有全量最终多分类对照，结果相同；其余12个缺此证据','当前不能宣称超过base，也不能宣称所有方法无效'],
        ]),
        '注意：各模块都使用同一验证集选择最佳epoch，最终结果由不同epoch的模块拼合，不存在单一“整套模型最优epoch”。还需要独立评估数据或固定选择规则后的重复验证，避免把验证集选优当成泛化证明。',
        '## 2. Prompt方法与loss的配对证据',
    ]
    loss_rows = []
    for prompt in ('A3-2','dynamic_head'):
        reference = indexed[(prompt,'A3-2','A3-2')]
        for loss in LOSS_METHODS:
            r = indexed[(prompt,'A3-2',loss)]
            loss_rows.append([prompt,loss,f(r['summary']['mIoU']),
                              f(mean([c['metrics']['IoU']*100 for c in r['concepts'] if c['concept_id']!='background'])),
                              f(next(c['metrics']['IoU']*100 for c in r['concepts'] if c['concept_id']=='background')),
                              f(r['summary']['mIoU']-reference['summary']['mIoU'],True),
                              sum(p['iou_gain_vs_initial']>1e-10 for p in r['prompts']),
                              ', '.join(f"{p['prompt']}:{p['best_epoch']}" for p in r['prompts'] if p['best_epoch']>0)])
    sections += [table(['prompt方法','loss（concept固定mean）','七类mIoU','去掉background六类mIoU','background IoU','相对本prompt的A3-2 loss','提升prompt数/11','非零最佳epoch'],loss_rows),
                 table(['相同loss','A3 prompt + mean','dynamic_head + mean','动态prompt增益'],[
                     [loss,f(indexed[('A3-2','A3-2',loss)]['summary']['mIoU']),
                      f(indexed[('dynamic_head','A3-2',loss)]['summary']['mIoU']),
                      f(indexed[('dynamic_head','A3-2',loss)]['summary']['mIoU']-indexed[('A3-2','A3-2',loss)]['summary']['mIoU'],True)]
                     for loss in LOSS_METHODS]),
                 '这些六组实验的concept是固定mean，因此loss差异实际主要发生在prompt训练。不能用它们断言concept训练也应采用相同loss排序。dynamic_head同时有动态空间网络和可训练base标量；没有冻结base的消融，不能把全部增益归因于空间网络本身。',
                 '**loss排序对background敏感：** 标量A3 prompt下，两个BCE类损失的六个前景类别平均IoU反而高于A3-2 loss；A3-2的七类优势主要由background二分类提升贡献。A3-2的background Pred/GT约2.544、Precision仅约36.23%，其高Recall可能在最终argmax中引入竞争。因此“A3-2目前优先”仅是当前七类二分类标准下的选择，不能提前断言其最终分割优于BCE。',
                 '## 3. Concept方法证据：区分规则、初始化与训练',
                 table(['concept方法（A3 prompt/A3 loss）','初始mIoU','最终mIoU','聚合训练净增益','同prompt max','相对自身max','多prompt concept最佳epoch','prompt权重与固定max是否相同','日志耗时'],[
                     [concept,f(r['summary']['InitialmIoU']),f(r['summary']['mIoU']),f(r['summary']['mIoUGain'],True),
                      f(mean([c['max_baseline']['IoU']*100 for c in r['concepts']])),
                      f(mean([c['iou_delta_vs_max']*100 for c in r['concepts']]),True),
                      ', '.join(f"{c['concept_id']}:{c['best_epoch']}" for c in r['concepts'] if c['prompt_count']>1),
                      '逐张量相同' if r['prompt_hashes']==max_ref['prompt_hashes'] else '不同',r['duration']]
                     for concept in CONCEPT_METHODS for r in [indexed[('A3-2',concept,'A3-2')]]]),
                 '固定mean和max的concept训练增益必为0，但改变聚合规则仍会改变结果。logsumexp的较高初始分数不是训练贡献，其温度训练净增益仅约0.106 pp。region/reliability/dynamic的训练净增益超过2 pp，支持保留可学习聚合。六个cache/A3-loss组合的prompt权重逐张量相同，使concept方法的比较更有说服力。sam_region的prompt权重不同，在线SAM与缓存还存在float32/float16原始heads精度差异，不能当作仅增加区域特征的严格单因素实验；日志耗时受硬件、并发、暂停和I/O影响，不是严格速度基准。',
                 '## 4. 哪些类别最值得继续做',
                 table(['Concept','region训练净ΔIoU','region vs自身max','reliability训练净ΔIoU','reliability vs自身max','SAM region训练净ΔIoU'],[
                     [name,*[f(value,True) for value in (
                         next(c for c in region['concepts'] if c['concept_id']==name)['iou_gain_vs_initial']*100,
                         next(c for c in region['concepts'] if c['concept_id']==name)['iou_delta_vs_max']*100,
                         next(c for c in reliability['concepts'] if c['concept_id']==name)['iou_gain_vs_initial']*100,
                         next(c for c in reliability['concepts'] if c['concept_id']==name)['iou_delta_vs_max']*100,
                         next(c for c in sam['concepts'] if c['concept_id']==name)['iou_gain_vs_initial']*100)]]
                     for name in ('building','barren','forest')]),
                 '只有building、barren、forest拥有多个prompt并真正训练concept；background、road、water、agricultural为单prompt，任何concept方法都直通。background尤其需要结合最终低分回退评估，不能只凭其独立二分类IoU决定是否保留。',
                 '# 尚未完成的参数组合与下一步计划',
                 f'当前注册空间：{len(PROMPT_METHODS)} prompt × {len(CONCEPT_METHODS)} concept × {len(LOSS_METHODS)} loss = {len(COMBINATION_CONFIGS)} 种。已完成 {len(done)} 种，未完成 {len(pending)} 种；已发现但未完成运行 {len(incomplete)} 个。当前所有完成结果只有seed 3407，不能把另一个seed计为新方法组合。',
                 '## 1. 完成矩阵',
                 table(['prompt方法','concept方法',*LOSS_METHODS],[
                     [p,c,*['已完成' if (p,c,l) in done else '未做' for l in LOSS_METHODS]]
                     for p in PROMPT_METHODS for c in CONCEPT_METHODS]),
                 '## 2. 全部未完成组合（逐项列出）',
                 table(['序号','prompt_method','concept_method','loss_method','组合目录/配置文件名','备注'],[
                     [i,p,c,l,combination_name(p,c,l),
                      '与已完成segov3+max相同固定预测，不需为loss重复跑' if p=='segov3' and c=='max'
                      else '固定mean，无训练；三个loss结果相同，只需一个规则对照' if p=='segov3' and c=='A3-2'
                      else '在线SAM，成本较高' if c=='sam_region_dynamic' else '有待验证']
                     for i,(p,c,l) in enumerate((key for key in COMBINATION_CONFIGS if key in pending),1)]),
                 '每一行对应 `configs/cx_b/<组合名>.py`。上述“未做”限定当前指定结果根目录，不推断其他目录中是否做过。',
                 '## 3. 推荐顺序',
                 '**P0：先补评估，再增加训练。** 对已有11个缓存训练组合复用checkpoint与原始heads，生成最终多分类指标；在线sam_region随后单独评估。保留“各运行原有决策参数”的结果，并额外做统一argmax、阈值0.5、低分回退背景0的对照，明确这是评估覆盖，不修改训练快照。GT映射、1669张图像和标签范围保持一致。不同SAM过滤阈值仍属于管线差异，不能声称统一了上游输入。优先region/reliability/dynamic/A3+max/dynamic_head+mean，最终判据是对47.414491%的增益和逐类别变化。',
                 '**P1：最小新增训练组，共4组。**',
                 table(['优先级','组合','要回答的问题'],[
                     ['1','segov3 + region_dynamic + A3-2','冻结原始Eval prompt，只训练concept，能否超过segov3+max？'],
                     ['2','segov3 + region_dynamic + soft_iou_bce','在相同固定prompt下检验concept loss，排除prompt训练混杂'],
                     ['3','segov3 + region_dynamic + bce_tversky','补齐同条件三个concept loss的比较'],
                     ['4','dynamic_head + region_dynamic + A3-2','动态prompt与当前最佳concept是否互补，而非只在mean聚合下有价值？'],
                 ]),
                 'P1先采用共同20 epoch、patience=7次验证、每2轮验证作为筛选，记录epoch0/2/4/…/20和实际停止点；它只回答短预算表现。现有forest最佳可能在48/58/78/80，不能把20轮筛选中的未胜出当作收敛后无效。短跑胜者与公平对照再用相同80轮预算确认；不要拿20轮新结果直接定论优于/劣于80轮旧结果。只训练concept时，segov3 prompt本身固定，prompt epochs不会影响结果。',
                 '**P2：验证稳定性和归因。** P0/P1筛出的1–2种方法及其直接对照至少使用3个配对seed（已有3407加2个新seed），报告均值、标准差和每类结果；追加segov3+reliability_dynamic+A3-2判断region/reliability近似持平是否稳定。固定选择策略后再做独立评估；多seed不替代独立数据。',
                 '**暂不优先：** 穷举50个缺失组合；为segov3+max重复三个loss；延长固定max/mean或单prompt concept的训练；在没有最终多分类优势前扩展昂贵SAM区域组合；仅凭0.037 pp单seed差距淘汰reliability。',
                 '**停止/继续标准：** 至少同时看最终多分类mIoU、各类IoU/Precision/Recall、背景误报和推理成本。二分类变好但最终多分类下降，说明校准或类别竞争出了问题，应先处理这些问题，不继续盲目加epoch。',
                 '复原基线全量证据：'+link(verification_path)+'。本报告未启动上述建议实验。',
                 ]
    assert len(done)+len(pending)==len(COMBINATION_CONFIGS)
    return '\n\n'.join(sections)


def write_report(root, base_root, output):
    baseline = load_baseline(base_root)
    discovered = sorted(p.parent for p in root.glob('*/seed_*/run_config.json'))
    directories = [p for p in discovered if (p/'completed.json').is_file()
                   and read_json(p/'completed.json').get('status') == 'complete']
    incomplete = [p for p in discovered if p not in directories]
    runs = [inspect_run(p, baseline) for p in directories]
    assert runs, 'No completed results found.'
    restored = next(r for r in runs if (r['cx']['prompt_method'], r['cx']['concept_method']) == ('segov3', 'max'))
    binary_base = restored['summary']
    restored_concepts = {c['concept_id']: c['metrics'] for c in restored['concepts']}
    verification_path = REPO/'out/cx_b_segov3_comparison/comparison.json'
    verification = read_json(verification_path)
    assert verification['images'] == 1669 and verification['mismatched_pixels'] == 0
    assert all(p['strict_matches_baseline'] and p['inclusive_matches_training'] for p in verification['prompts'])
    for p, v in zip(restored['prompts'], verification['prompts']):
        assert p['prompt'] == v['prompt']
        assert abs(p['best_metrics']['IoU']*100-v['cx_b_iou_percent']) < 1e-9
    bsum = baseline['summary']
    report = ['**CX-B 训练结果逐配置分析**',
              f'生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}（本机时间）。分析 {len(runs)} 个已完成运行；未完成 {len(incomplete)} 个；本次不排除任何组合。全部运行seed情况：{sorted(set(r["snapshot"]["seed"] for r in runs))}。',
              f'实验目录：`{root}`。base：{link(base_root / "results.xlsx")}，配置快照：{link(base_root / "cfg_loveda.py")}。',
              '**阅读口径（适用于每一章）**',
              '- 指标统一以百分数显示，所有差值为百分点，不是相对百分比。Acc 在这里等于类别 Recall；F1 对应 Fscore。\n'
              '- prompt 初始值是该 prompt 训练前 epoch 0；concept 初始值是 prompt 已选优并冻结后、聚合器尚未训练的状态。`InitialmIoU` 不能解释为整个系统训练前的 mIoU。\n'
              '- base 的 summary/per_class 是最终互斥多分类；CX-B concept 是各类独立阈值化二分类，允许同一像素被多类判正。concept 与 base 的差值只作数值参照，不能证明最终多分类提升或下降。尤其 CX-B 的 aAcc 是 ΣTP/ΣGT 的微平均召回，不是 base 的最终像素准确率。\n'
              '- base per_prompt 与 CX-B prompt 都是逐 prompt 二分类，可作同任务的管线结果比较；但 base/segov3 confidence_threshold=0.5、A3/dynamic_head=0.1，融合与 presence 公式也不同，跨管线收益不能全部归因于训练。base 使用 >0.5、CX-B 使用 >=0.5，阈值边界亦有差别。\n'
              '- 已核对所有 prompt/concept 的 GTPixels 与 base 对应 prompt 完全一致，且配置均指向 LoveDA val；这支持同一标签集合，但不能代替逐图内容哈希一致性验证。base 0起始映射+255忽略 与训练原始1–7标签+0忽略语义一致。\n'
              '- 每个模块按验证 IoU 选 best；epoch 0 是合法候选，status=trained 不代表有效。峰值表包括 epoch 0；并列按1e-12容差统计，列出全部已评估的并列 epoch。通常隔2轮验证，未验证的epoch无法推断是否更优。Precision/Recall 峰值可能不属于最终所选权重。\n'
              '- 可学习参数的数值增减不等于性能改善。当前结果只有联合优化证据，没有逐参数冻结/消融，不能给每个标量或卷积权重单独归因。以下同时报告超参数、权重变化、模块效果与各指标峰值。\n'
              '- 本报告分析现有文件，没有重新训练。复用此前segov3的全量1669张多分类核对结果；其余12个组合尚无全量最终多分类结果，不把独立二分类排名当作最终分割排名。',
              '**新增的可比基准**：`segov3 + max + A3-2` 已全量复原历史Eval，最终mIoU=47.414491%；其独立concept mIoU='+f(binary_base['mIoU'],digits=6)+'%。以下新增相对此基准的同任务差值；SAM过滤与评分公式仍是跨管线变化，不能把差值全部归因为训练。证据：'+link(verification_path)+'。',
              '**全部已完成组合概览（按独立 concept mIoU 排序）**',
              table(['配置参数', '配置', 'seed', 'concept mIoU（独立二分类）', 'concept训练 ΔmIoU',
                     '同prompt的max mIoU', 'vs max', 'vs复原基线40.907（同口径）',
                     'LoveDA mIoU（最终多分类）', '当前二分类−LoveDA（异口径数值差）',
                     '当前最终多分类与LoveDA核对', 'prompt有效数'],
                    [[f'prompt: {r["cx"]["prompt_method"]};<br>concept: {r["cx"]["concept_method"]};<br>loss: {r["cx"]["loss_method"]}',
                      r['name'], r['snapshot']['seed'], f(r['summary']['mIoU']),
                      f(r['summary']['mIoUGain'], True),
                      f(mean([c['max_baseline']['IoU']*100 for c in r['concepts']])),
                      f(mean([c['iou_delta_vs_max']*100 for c in r['concepts']]), True),
                      f(r['summary']['mIoU']-binary_base['mIoU'], True),
                      f(verification['baseline_multiclass']['summary']['mIoU'], digits=6),
                      f(r['summary']['mIoU']-verification['baseline_multiclass']['summary']['mIoU'], True),
                      (f'{f(verification["current_multiclass"]["summary"]["mIoU"],digits=6)}%；Δ=0；逐像素一致'
                       if r is restored else '待全量多分类评估'),
                      f"{sum(p['iou_gain_vs_initial']>1e-10 for p in r['prompts'])}/{len(r['prompts'])}"]
                     for r in sorted(runs, key=lambda x:x['summary']['mIoU'], reverse=True)]),
              '表内mIoU单位为%，差值单位为百分点。LoveDA原报告显示47.41%，历史预测图全量重算为47.414491%。'
              '“当前二分类−LoveDA”仅是异口径数值参照，不能作为性能提升/下降结论；'
              '最终同口径比较看“当前最终多分类与LoveDA核对”列，待评估项不填造结果。配置参数使用单元格内换行显示。',
              '当前优先候选为 region_dynamic、reliability_dynamic 和 dynamic；排序只针对现有独立二分类报告。region 与 reliability 的差距约 0.037 个百分点，单 seed 不足以证明稳定优势。base mIoU=47.410 是另一评估口径，不能据此宣布所有训练无效。']
    for run in runs:
        d, snapshot, cx = run['directory'], run['snapshot'], run['cx']
        cfg, prompts, concepts, summary = snapshot['config'], run['prompts'], run['concepts'], run['summary']
        header = run['name'] if sum(r['name']==run['name'] for r in runs)==1 else f"{run['name']} / {d.name}"
        report += [f'# {header}', '## 1. 输入参数与训练设置',
                   f'运行目录：`{d}`。状态：complete，完成时间 {run["done"]["completed_at"]}；日志记录总耗时 {run["duration"]}（受并发、I/O和硬件影响，不能视作纯算法耗时）。',
                   f'读取并交叉校验：{link(d/"run_config.json")}、{link(d/"prompt_metrics.json")}、{link(d/"concept_metrics.json")}、{link(d/"summary_metrics.json")} 以及所有 prompt/concept 的 `best.pth`。完成 fingerprint、权重 epoch、历史最大 IoU、指标差值及汇总均值均一致。',
                   f'运行 fingerprint：`{snapshot["fingerprint"]}`。历史快照记录的 {len(run["source_checks"])} 个实现/概念文件 SHA256 与当前文件核对：'
                   + ('全部一致。' if all(v=='一致' for v in run['source_checks'].values()) else json.dumps(run['source_checks'],ensure_ascii=False))
                   + f' SAM 权重记录：`{snapshot["inputs"].get("sam_checkpoint", {}).get("path")}`；未重新计算大型 SAM 权重哈希。',
                   '**按快照重建的调用参数**（并非声称恢复了原始 shell 命令文本）：',
                   '```text\npython train_batch_prompt_concept_variants.py '
                   f'--prompt_methd={cx["prompt_method"]} --concept_method={cx["concept_method"]} '
                   f'--loss_method={cx["loss_method"]} --prompt-init={cx["prompt_init"]} '
                   f'--output-root="{root.as_posix()}" --seed={snapshot["seed"]} '
                   f'--prompt-head-cache="{snapshot["inputs"]["prompt_head_cache"]}" --cfg-options '
                   + ' '.join(f'cx_b.{stage}.{key}={cx[stage][key]}'
                              for stage in ('prompt_train','concept_train')
                              for key in ('epochs','patience','lr','validate_every_epochs','param_reg_weight'))+'\n```',
                   f'对应配置：`{snapshot["config_path"]}`；选中 concept={snapshot["inputs"].get("selected_concepts")}（null 表示全部），7 concepts / 11 prompts，SAM 冻结；本次无 A3 旧权重初始化。',
                   '**历史实际输入与当前注册配置默认值的差异**（仅对照，不修改历史运行）：',
                   table(['参数路径','历史运行实际值','当前配置默认值'],run['config_changes']) if run['config_changes'] else '历史配置与当前注册配置的字段值全部一致。',
                   '**保存的实际配置参数**（历史快照优先于今天文件中的默认值）：',
                   '```json\n'+json.dumps(dict(cx_b=cx, model=cfg['model'],
                       train_dataset=cfg['train_dataloader']['dataset'],
                       validation_dataset=cfg['validation_dataloader']['dataset']), ensure_ascii=False, indent=2)+'\n```',
                   '参数生效说明：有可学习参数时使用Adam。'+ '；'.join(
                       f'{stage}: epochs={cx[stage]["epochs"]}, lr={cx[stage]["lr"]}, validate_every={cx[stage]["validate_every_epochs"]}, patience={cx[stage]["patience"]}次验证, 正则={cx[stage]["param_reg_weight"]}'
                       for stage in ('prompt_train','concept_train'))+
                   '。patience按验证次数累计，不是训练epoch数；固定模块跳过优化。prompt_train.batch_size虽保存在配置，当前优化循环逐样本更新，未消费此字段。selection_metric=hard_iou实际按IoU选优。',
                   'A3-2 损失使用 SoftIoU + 0.1×max(0,0.3−Recall)；soft_iou_bce 使用 SoftIoU+BCE；bce_tversky 使用 BCE+(1−Tversky)，FP/FN权重0.6/0.4。无正样本图像均使用平均预测概率惩罚。配置中其他 loss 字段仅在对应分支生效。',
                   '保存的model.fusion_mode/prompt_fusion_mode等learnable模型参数不直接决定CX-B输出：本流程提取原始heads后，实际评分/聚合由cx_b.prompt_method和concept_method的模块执行。不能把快照中的weighted_sum字样误读为segov3或max的实际融合规则。',
                   '单 prompt 的聚合器恒等直通；A3-2 concept 是固定均值，max 是固定最大值，二者不进行 concept 梯度训练。hidden_channels 对 A3 标量头/固定聚合无效；temperature 仅用于 logsumexp。segov3 prompt固定且无可训练参数。'+
                   ('本次在线SAM运行，sam_region_cfg生效；不读写SAM region/head输出缓存。' if cx['data_mode']=='online_sam' else '本次cache运行，sam_region_cfg不参与计算。')]
        # Actual parameter values, including the dynamic heads' learned base.
        param_rows=[]
        for row in prompts:
            module, meta=run['modules'][('prompt',row['prompt_index'])]
            base=getattr(module,'base',module)
            if cx['prompt_method'] == 'segov3':
                param_rows.append([row['prompt'],row['best_epoch'], '不适用','不适用','不适用','不适用','不适用','不适用','固定max(S,I)×presence'])
                continue
            values=[float(base.semantic_weight.detach()),float(base.presence_gain.detach()),float(base.presence_intercept.detach())]
            tail='—'
            if hasattr(module,'output_projection'):
                weight=module.output_projection.weight.detach().float()
                tail=f'输出层weight RMS={f(weight.square().mean().sqrt())}；bias={f(module.output_projection.bias.item())}'
            param_rows.append([row['prompt'], row['best_epoch'],
                               f(values[0]), f(values[0]-.5,True),f(values[1]),f(values[1]-1,True),
                               f(values[2]),f(values[2],True),tail])
        report += ['### 1.1 选中权重中的可学习参数',
                   ('segov3没有α/gain/intercept参数，沿用原始max(S,I)×presence；不存在训练初末参数差值。' if cx['prompt_method']=='segov3' else 'scratch 输入为 semantic_weight=0.5、presence_gain=1、presence_intercept=0。下面的 Δ 都相对这三个初值。A3 概率为 sigmoid(logit(α×semantic+(1−α)×instance)+gain×(presence−0.5)+intercept)。动态头表中的 α 是 base α，实际像素 α 还受网络残差影响。'),
                   table(['Prompt','epoch','α','Δα','gain','Δgain','intercept','Δintercept','动态头状态'],param_rows),
                   '参数变化只证明所选权重发生改变；不能将 IoU 增益分摊给 α、gain 或 intercept。base SAM3 没有这些可直接对齐的校准参数，因此对 base 的参数差异是“新增模块”，不是可相减的同名训练参数。']
        agg_rows=[]
        for row in concepts:
            module,meta=run['modules'][('concept',row['concept_index'])]
            details='无可学习参数'
            if hasattr(module,'temperature'):
                t=float(module.temperature.detach())
                details=f'temperature 0.100000 → {f(t,digits=6)}；Δ={f(t-.1,True,6)}'
            elif hasattr(module,'raw_response_scale'):
                scale=float(torch.nn.functional.softplus(module.raw_response_scale).detach())
                bias=(module.prompt_bias-module.prompt_bias.mean()).detach().tolist()
                details=f'response_scale {f(math.log(2))} → {f(scale)}；中心化prompt_bias={json.dumps([round(v,5) for v in bias])}'
                weights=torch.cat([p.detach().flatten().float() for p in module.network.parameters()])
                details+=f'；network参数RMS={f(weights.square().mean().sqrt())}'
            count=sum(p.numel() for p in module.parameters())
            agg_rows.append([row['concept_id'], row['prompt_count'],row['best_epoch'],type(module).__name__,count,details])
        report += [table(['Concept','prompt数','epoch','实际聚合器','参数量','选中参数/权重统计'],agg_rows),
                   '动态聚合网络 epoch 0 的全部随机权重未单独保存在结果报告中，因此这里只列 best 权重统计，不伪造逐权重初末差值。prompt_bias 按成员 prompt 顺序列出；它是 softmax 前的偏置，不等于固定融合权重。',
                   '## 2. 配置内部的训练效果与最佳 epoch',
                   '### 2.1 Prompt 阶段：对训练前 epoch 0 的变化', stage_tables(prompts,True),
                   '### 2.2 Concept 阶段：对冻结 prompt 后的初始聚合器的变化', stage_tables(concepts),
                   '### 2.3 每项指标的历史峰值与达到的 epoch',
                   '下表按各指标分别取最大值，展示“峰值百分数 @ 全部并列已评估epoch”。这些峰值不能拼成一套实际存在的权重；实际 best.pth 仍按 IoU 选择。',
                   metric_peak_table(prompts,True),metric_peak_table(concepts)]
        better=[p for p in prompts if p['iou_gain_vs_initial']>1e-10]
        unchanged=[p['prompt'] for p in prompts if p['best_epoch']==0]
        caps=[p['prompt'] for p in prompts if p['best_epoch']==cx['prompt_train']['epochs']]
        caps += [f'concept:{c["concept_id"]}' for c in concepts if c['best_epoch']==cx['concept_train']['epochs']]
        report += [f'Prompt 有效提升 {len(better)}/{len(prompts)}；epoch 0 保留项：{", ".join(unchanged) or "无"}。'
                   f' Concept阶段宏平均 IoU 为 {f(summary["InitialmIoU"])} → {f(summary["mIoU"])}，变化 {f(summary["mIoUGain"],True)} 个百分点。',
                   '达到训练上限才取得最佳值的模块：'+(', '.join(caps) if caps else '无')+'。末轮最佳只说明本次上限处仍有改善，不能保证延长训练有收益。',
                   '### 2.4 与相同已训练 prompt 的 max 及最佳单 prompt 对照']
        contrast=[]
        for c in concepts:
            members=[p for p in prompts if p['concept_id']==c['concept_id']]
            best=max(members,key=lambda p:p['best_metrics']['IoU'])
            contrast.append([c['concept_id'],f(c['metrics']['IoU']*100),
                             f(c['max_baseline']['IoU']*100),f(c['iou_delta_vs_max']*100,True),
                             best['prompt'],f(best['best_metrics']['IoU']*100),
                             f((c['metrics']['IoU']-best['best_metrics']['IoU'])*100,True)])
        report += [table(['Concept','当前IoU','max IoU','vs max','最佳单prompt','单prompt IoU','vs 单prompt'],contrast),
                   '这里的 max_baseline 使用本配置已经选优的 prompt，属于比外部 base 更可比的聚合对照。最佳单 prompt 是本验证集上的事后选择，只作诊断，不能当作独立测试结果。',
                   '## 3. 与 cfg_loveda base 的差异',
                   '### 3.1 输入参数和评估规则差异',
                   table(['项目','base cfg_loveda','当前配置','影响'],[
                       ['SAM confidence_threshold','0.5',cfg['model']['confidence_threshold'],'相同' if cfg['model']['confidence_threshold']==.5 else '上游候选实例过滤不同；epoch0也可能不同'],
                       ['概率/最终阈值','prob_thd=0.5',f"model.prob_thd={cfg['model']['prob_thd']}；训练metric_threshold={cx['metric_threshold']}",'训练二分类使用后者；前者不是当前训练汇总阈值'],
                       ['逐prompt阈值判定','>0.5',f">={cx['metric_threshold']}",'临界值口径不同'],
                       ['prompt分支融合','max(semantic,instance)后乘presence（当前base实现）','相同原始规则' if cx['prompt_method']=='segov3' else cx['prompt_method']+'校准；logit域presence偏移','与base相同' if cx['prompt_method']=='segov3' else '评分公式改变，非仅训练权重差异'],
                       ['concept融合','max（base构造函数默认）',cx['concept_method'],'A3-2在此代表均值，不是再次训练A3头'],
                       ['最终类别决策','argmax后低分回退background（当前base实现默认bg_idx=0）','训练报告无跨类argmax','concept表为不同评估任务'],
                       ['GT编码','reduce_zero_label=True；ignore=255','原始1–7；ignore=0','经正确映射后同一标签语义'],
                       ['训练/参数优化','无任务校准训练','固定规则，无训练' if cx['prompt_method']=='segov3' and cx['concept_method']=='max' else 'SAM冻结；prompt独立训练，再训练concept','不是SAM主干微调'],
                       ['数据划分','LoveDA img_dir/val','LoveDA img_dir/val','GTPixels已逐项核对一致'],
                       ['网络结构/训练超参数','无对应Adam/lr/epoch/hidden训练配置','见本章第1节完整JSON','base没有相应可学习参数，数值差不适用']]),
                   'base 的显式阈值/数据配置来自保存的 cfg_loveda.py；融合和低分回退规则来自当前 segearthov3_segmentor.py，因为 base 目录未保存该源文件的历史副本，不能证明代码从运行时至今完全未改。',
                   '**逐项 model 输入参数对照**：base 未显式保存的参数不填造历史数值；“新增”不表示 base 的数学作用等于0。训练专属 cx_b 参数全部已在第1节记录，base 无对应训练配置。',
                   table(['参数','base显式输入','当前显式输入','配置差异'],[
                       [key, json.dumps(baseline['model'].get(key,'未显式设置'),ensure_ascii=False),
                        json.dumps(cfg['model'].get(key,'未显式设置'),ensure_ascii=False),
                        ('相同' if baseline['model'].get(key,'未显式设置')==cfg['model'].get(key,'未显式设置')
                         else f"数值差={f(cfg['model'][key]-baseline['model'][key],True)}"
                         if isinstance(baseline['model'].get(key), (int,float)) and isinstance(cfg['model'].get(key),(int,float))
                         else '新增/方法改变；无直接可比数值差')]
                       for key in sorted(set(baseline['model']) | set(cfg['model']))]),
                   '### 3.2 总体指标差异（异口径参照）',
                   table(['指标','base多分类','当前独立concept','数值差（不可视作同口径增益）'],
                         [[k,f(bsum[k]),f(summary[k]),f(summary[k]-bsum[k],True)]
                          for k in ['aAcc','mIoU','mAcc','mFscore','mPrecision','mRecall']]),
                   '### 3.3 各 concept 与 base 最终类别结果对照（异口径）']
        report += [table(['Concept','base IoU','当前IoU','ΔIoU','ΔAcc','ΔPrecision','ΔRecall','ΔF1','当前Pred/GT面积比'],
                         [[c['concept_id'],f(baseline['concepts'][c['concept_id']]['IoU']),f(c['metrics']['IoU']*100),
                           *[f(c['metrics'][k]*100-baseline['concepts'][c['concept_id']][k],True) for k in METRICS],
                           f(c['metrics']['PredGTAreaRatio'])] for c in concepts]),
                   '上表每类的差值都受跨类竞争/回退与独立阈值化差异影响，不能作为该类最终分割提升的结论。',
                   '### 3.3.1 与复原base的独立二分类对照（同统计口径）',
                   '基准为经过全量复原验证的segov3+max报告，各类均按>=0.5独立判定。当前与该基准的差值可比较二分类管线效果；除segov3本身外，上游confidence阈值、prompt评分和训练均可能贡献差异。',
                   table(['指标','复原base独立二分类','当前','当前−基准'],[
                       [k,f(binary_base[k]),f(summary[k]),f(summary[k]-binary_base[k],True)]
                       for k in ('mIoU','mAcc','mPrecision','mRecall','mFscore','aAcc')]),
                   table(['Concept','基准IoU','当前IoU','ΔIoU','ΔAcc','ΔPrecision','ΔRecall','ΔF1','ΔTP','ΔFP','ΔFN'],[
                       [c['concept_id'],f(restored_concepts[c['concept_id']]['IoU']*100),f(c['metrics']['IoU']*100),
                        *[f((c['metrics'][k]-restored_concepts[c['concept_id']][k])*100,True) for k in METRICS],
                        *[f"{int(c['metrics'][k]-restored_concepts[c['concept_id']][k]):+d}" for k in ('TP','FP','FN')]]
                       for c in concepts]),
                   '### 3.3.2 最终多分类与历史Eval对照',
                   ('全量1669张、1,750,073,344个像素差异为0；mIoU='+f(verification['current_multiclass']['summary']['mIoU'],digits=6)+
                    '%，aAcc='+f(verification['current_multiclass']['summary']['aAcc'],digits=6)+'%，与历史预测重算值完全相同。11个prompt统一为>0.5后TP/FP/FN也全部一致；>=与>差异合计39198个分数等于0.5的有效像素（按prompt分别计数）。'
                    if cx['prompt_method']=='segov3' else '未发现本运行的全量最终多分类评估结果；不得用本章二分类分数减47.414491%来宣称最终分割提升/下降。此处是待完成的评估，不是尚未执行的训练组合。'),
                   '### 3.4 各 prompt 与 base 二分类结果对照',
                   table(['Prompt','base IoU','训练epoch0 IoU','选中IoU','epoch0−base','训练净ΔIoU',
                          '最终−base','ΔAcc vs base','ΔPrecision vs base','ΔRecall vs base','ΔF1 vs base'],
                         [[p['prompt'],f(baseline['prompts'][p['prompt']]['IoU']),
                           f(p['initial_metrics']['IoU']*100),f(p['best_metrics']['IoU']*100),
                           f(p['initial_metrics']['IoU']*100-baseline['prompts'][p['prompt']]['IoU'],True),
                           f(p['iou_gain_vs_initial']*100,True),
                           *[f(p['best_metrics'][k]*100-baseline['prompts'][p['prompt']][k],True) for k in METRICS]] for p in prompts]),
                   '“最终−base = epoch0−base + 训练净ΔIoU”。这能区分初始评分/上游设置变化与本次参数优化的增益，避免把 epoch0 已经存在的差异算作训练贡献。',
                   '**各 prompt 的误检、漏检与面积变化**（相对 base per_prompt）：',
                   table(['Prompt','ΔTP','ΔFP','ΔFN','base Pred/GT','当前 Pred/GT','Δ面积比'],[
                       [p['prompt'], *[f"{int(p['best_metrics'][k]-baseline['prompts'][p['prompt']][k]):+d}" for k in ('TP','FP','FN')],
                        f(baseline['prompts'][p['prompt']]['PredGTAreaRatio']),
                        f(p['best_metrics']['PredGTAreaRatio']),
                        f(p['best_metrics']['PredGTAreaRatio']-baseline['prompts'][p['prompt']]['PredGTAreaRatio'],True)]
                       for p in prompts]),
                   'base per_class 未保存 TP/FP/FN，因此不能计算可靠的逐 concept 像素计数差；未把 prompt 计数冒充多分类计数。',
                   '### 3.5 与其他已完成 CX-B 配置的同口径对照']
        peers=[]
        for other in runs:
            if other is run: continue
            oc=other['cx']
            diffs=[k for k in ['prompt_method','concept_method','loss_method'] if oc[k]!=cx[k]]
            if len(diffs)==1:
                identical=all(abs(a['best_metrics']['IoU']-b['best_metrics']['IoU'])<1e-12
                              for a,b in zip(prompts,other['prompts']))
                weights_equal = run['prompt_hashes'] == other['prompt_hashes']
                same_upstream = (cfg['model']==other['snapshot']['config']['model'] and
                                 cx['data_mode']==oc['data_mode'])
                peers.append([other['name'],diffs[0],f(other['summary']['mIoU']),
                              f(summary['mIoU']-other['summary']['mIoU'],True),
                              ('prompt权重逐张量一致' if weights_equal else 'prompt权重不同')+'；'+
                              ('prompt指标一致' if identical else 'prompt指标不同')+'；'+
                              ('上游model/data_mode一致' if same_upstream else '上游设置/执行模式也改变')])
        report += [table(['参照配置','改变因素','参照mIoU','当前−参照','说明'],peers),
                   '这些仍是单 seed 验证结果；只有一个方法因素变化，不等于有统计显著性。',
                   '## 4. 综合判断：这次训练是否有价值',recommendation(run,runs)]
        top=sorted(better,key=lambda p:p['iou_gain_vs_initial'],reverse=True)[:3]
        report += ['- 最有收益的 prompt：'+('；'.join(f"{p['prompt']} +{f(p['iou_gain_vs_initial']*100)} pp @ epoch {p['best_epoch']}" for p in top) or '无')+'。']
        for c in concepts:
            if c['prompt_count']<=1: continue
            best=max((p for p in prompts if p['concept_id']==c['concept_id']),key=lambda p:p['best_metrics']['IoU'])
            gap=(c['metrics']['IoU']-best['best_metrics']['IoU'])*100
            report += [f"- {c['concept_id']}：聚合训练净收益 {f(c['iou_gain_vs_initial']*100,True)} pp，"
                       f"相对max {f(c['iou_delta_vs_max']*100,True)} pp，相对最佳单prompt `{best['prompt']}` {f(gap,True)} pp；"
                       f"当前Precision={f(c['metrics']['Precision']*100)}%、Recall={f(c['metrics']['Recall']*100)}%。"]
        bg=next(c for c in concepts if c['concept_id']=='background')
        report += [f"- background：IoU={f(bg['metrics']['IoU']*100)}%、Precision={f(bg['metrics']['Precision']*100)}%、Recall={f(bg['metrics']['Recall']*100)}%、Pred/GT={f(bg['metrics']['PredGTAreaRatio'])}。必须结合误检与召回判断，不能只看相对极低初始化的巨大增益。",
                   ('- 已验证原始Eval的最终预测复原；本组合没有训练收益。' if cx['prompt_method']=='segov3' else '- 证据范围是本验证集的独立二分类和模块联合选优；尚未证明对base最终多分类提升、独立测试集泛化或单个参数的因果贡献。'),
                   ('- 作为已完成全量核对的对照保留；下一步固定此prompt，训练region/reliability等concept，隔离聚合收益。' if cx['prompt_method']=='segov3' else '- 后续先统一预测阈值、GT映射、argmax/背景回退规则做全验证集多分类复评，保留SAM候选过滤不同的说明；若统一SAM过滤阈值需重新提取heads，不能直接复用旧缓存冒充同输入。对无收益epoch0项保留初值，对临近上限最佳的模块再考虑定向延长。无需为固定均值/max或单prompt concept增加训练轮数。')]
    text='\n\n'.join(report)+'\n'
    assert text.count('\n# ')==len(runs)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text,encoding='utf-8')
    # Read back every completed per-configuration analysis before synthesizing.
    reread = output.read_text(encoding='utf-8')
    for run in runs:
        assert f'# {run["name"]}' in reread and '## 4. 综合判断' in reread
    synthesis = synthesize(runs, incomplete, binary_base, verification_path)
    output.write_text(reread+'\n\n'+synthesis+'\n',encoding='utf-8')
    text = output.read_text(encoding='utf-8')
    print(json.dumps(dict(report=str(output),configurations=len(runs),characters=len(text),
                          lines=len(text.splitlines()),validated_prompt_modules=sum(len(r['prompts']) for r in runs),
                          validated_concept_modules=sum(len(r['concepts']) for r in runs)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    write_report(args.root,args.base,args.output)
