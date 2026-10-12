"""Unified entry point for CX-B prompt/concept/loss combinations."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import hashlib
from datetime import datetime, timezone, timedelta

from cx_b.configuration import (
    COMBINATION_CONFIGS, CONCEPT_METHODS, LOSS_METHODS, PROMPT_INIT_MODES,
    PROMPT_METHODS, JOINT_MODES, combination_name, parse_cfg_options,
    resolve_combination_config, validate_config,
    refiner_settings, effective_joint_train,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_config', nargs='?', default=None)
    parser.add_argument('--prompt_methd', '--prompt_method',
                        dest='prompt_method', choices=PROMPT_METHODS)
    parser.add_argument('--concept_method', choices=CONCEPT_METHODS)
    parser.add_argument('--loss_method', choices=LOSS_METHODS)
    parser.add_argument('--joint-mode', '--joint_mode', choices=JOINT_MODES,
                        help='Optional final joint refinement after the original two stages; default independent.')
    parser.add_argument('--joint-epochs', type=int)
    parser.add_argument('--joint-patience', type=int)
    parser.add_argument('--joint-lr', type=float)
    parser.add_argument('--joint-refiner', choices=('none', 'ov_class_spatial'), default=None)
    parser.add_argument('--joint-refiner-strength', type=float, default=None)
    parser.add_argument('--refiner-epochs', type=int)
    parser.add_argument('--refiner-patience', type=int)
    parser.add_argument('--refiner-lr', type=float)
    parser.add_argument('--joint-init', choices=('scratch', 'checkpoint'), default='scratch',
                        help='scratch keeps the two independent stages; checkpoint skips them and fine-tunes a full joint bundle.')
    parser.add_argument('--joint-init-checkpoint',
                        help='Source run joint/*.pth (cx_b_joint_v1); optimizer is initialized afresh.')
    parser.add_argument(
        '--prompt-init', '--prompt_init', dest='prompt_init',
        choices=PROMPT_INIT_MODES, default=None,
        help=('Prompt initialization: scratch trains from unified defaults '
              '(segov3 uses fixed original scoring); '
              'a3_warmstart trains from an A3 checkpoint; a3_reuse may reuse '
              'a completed A3 prompt only for the exact A3-2/A3-2 case.'))
    parser.add_argument('--a3-checkpoint-root')
    parser.add_argument('--prompt-head-cache')
    parser.add_argument('--output-root', default='weights/stage_cx_b')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--concept-indices', nargs='*', type=int)
    parser.add_argument(
        '--export-probabilities', action=argparse.BooleanOptionalAction,
        default=None,
        help='Export validation concept probability NPY files after training.')
    parser.add_argument(
        '--export-probabilities-only', action='store_true',
        help='Load completed prompt/concept pth files and export NPY files '
             'without training again.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--cfg-options', nargs='*', default=[])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list-methods', action='store_true')
    return parser.parse_args(argv)


def list_methods():
    print('prompt_methd:', ', '.join(PROMPT_METHODS))
    print('prompt_init:', ', '.join(PROMPT_INIT_MODES))
    print('concept_method:', ', '.join(CONCEPT_METHODS))
    print('loss_method:', ', '.join(LOSS_METHODS))
    print('joint_mode:', ', '.join(JOINT_MODES))
    print('joint_init: scratch, checkpoint')
    print('joint_refiner: none, ov_class_spatial (strength=0 also disables)')
    print(f'combinations: {len(COMBINATION_CONFIGS)}')


def main(argv=None):
    args = parse_args(argv)
    if args.list_methods:
        list_methods()
        return 0
    missing = [name for name, value in (
        ('--prompt_methd', args.prompt_method),
        ('--concept_method', args.concept_method),
        ('--loss_method', args.loss_method)) if value is None]
    if missing:
        raise SystemExit('Missing required method selectors: ' + ', '.join(missing))
    overrides = parse_cfg_options(args.cfg_options)
    config, config_path = resolve_combination_config(
        args.prompt_method, args.concept_method, args.loss_method,
        overrides=overrides, alternate_base=args.base_config)
    if args.prompt_init is not None:
        config['cx_b']['prompt_init'] = args.prompt_init
    if args.joint_mode is not None:
        config['cx_b']['joint_mode'] = args.joint_mode
    for field in ('joint_refiner', 'joint_refiner_strength'):
        if getattr(args, field) is not None:
            config['cx_b'][field] = getattr(args, field)
    for field in ('epochs', 'patience', 'lr'):
        if getattr(args, 'refiner_'+field) is not None:
            config['cx_b'].setdefault('refiner_train', {})[field] = getattr(args, 'refiner_'+field)
    for field in ('epochs', 'patience', 'lr'):
        value = getattr(args, 'joint_'+field)
        if value is not None:
            config['cx_b'].setdefault('joint_train', {})[field] = value
    if args.export_probabilities is not None:
        config['cx_b']['export_probabilities'] = args.export_probabilities
    if args.export_probabilities_only and args.export_probabilities is not None:
        raise SystemExit(
            '--export-probabilities-only cannot be combined with '
            '--export-probabilities/--no-export-probabilities.')
    validate_config(
        config, (args.prompt_method, args.concept_method, args.loss_method))
    prompt_init = config['cx_b'].get('prompt_init', 'scratch')
    if prompt_init == 'scratch' and args.a3_checkpoint_root:
        raise SystemExit(
            '--a3-checkpoint-root cannot be used with --prompt-init=scratch. '
            'Choose a3_warmstart/a3_reuse or remove the checkpoint argument.')
    if prompt_init != 'scratch' and not args.a3_checkpoint_root:
        raise SystemExit(
            f'--prompt-init={prompt_init} requires --a3-checkpoint-root.')
    if args.prompt_head_cache:
        config['cx_b']['prompt_head_cache'] = osp.abspath(
            args.prompt_head_cache)
    if args.joint_init == 'checkpoint':
        if config['cx_b'].get('joint_mode', 'independent') == 'independent':
            raise SystemExit('--joint-init=checkpoint requires a joint mode.')
        if prompt_init != 'scratch' or args.a3_checkpoint_root:
            raise SystemExit('Joint initialization cannot be combined with A3 prompt warmstart/reuse.')
        if args.export_probabilities_only:
            raise SystemExit('Use eval_cx_b.py for exports of a completed joint-initialized run.')
        if not args.joint_init_checkpoint or not osp.isfile(args.joint_init_checkpoint):
            raise SystemExit('--joint-init=checkpoint requires an existing --joint-init-checkpoint file.')
        source = osp.abspath(args.joint_init_checkpoint)
        digest = hashlib.sha256()
        with open(source, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        config['cx_b'].update(joint_init='checkpoint', joint_init_checkpoint=source,
                              joint_init_sha256=digest.hexdigest())
    elif args.joint_init_checkpoint:
        raise SystemExit('--joint-init-checkpoint requires --joint-init=checkpoint.')
    refiner_method, refiner_strength = refiner_settings(config['cx_b'])
    if refiner_method != 'none':
        if args.joint_init != 'checkpoint':
            raise SystemExit('R1 requires --joint-init=checkpoint; independent stages stay frozen')
        if args.export_probabilities_only:
            raise SystemExit('Use eval_cx_b.py to export the complete R1 pipeline')
    name = combination_name(
        args.prompt_method, args.concept_method, args.loss_method)
    joint_mode = config['cx_b'].get('joint_mode', 'independent')
    if joint_mode != 'independent':
        if args.concept_indices is not None:
            raise SystemExit('Joint refinement requires all concepts; remove --concept-indices.')
        name += '__j_' + joint_mode
        # Saved final inference and joint validation must use the same decision.
        config['model']['prob_thd'] = config['cx_b']['joint_train']['threshold']
        config['cx_b']['prediction_background'] = config['cx_b']['joint_train']['background']
    run_folder = f'seed_{args.seed}'
    if refiner_method != 'none':
        name += f'__r_{refiner_method}__s_{refiner_strength:g}__u_refiner_only'
    if args.joint_init == 'checkpoint':
        name += '__init_ckpt_' + config['cx_b']['joint_init_sha256'][:8]
        run_folder += '_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output_dir = osp.abspath(osp.join(args.output_root, name, run_folder))
    if args.joint_init == 'checkpoint':
        base_output = output_dir
        serial = 2
        while osp.exists(output_dir):
            output_dir = base_output + f'_{serial:02d}'
            serial += 1
    resolved = dict(
        combination=name,
        config_path=osp.relpath(config_path, osp.dirname(__file__)),
        config_path_absolute=config_path,
        output_dir=output_dir,
        seed=args.seed,
        prompt_init=prompt_init,
        joint_mode=joint_mode,
        joint_init=args.joint_init,
        joint_init_checkpoint=args.joint_init_checkpoint,
        export_probabilities=config['cx_b'].get(
            'export_probabilities', False),
        export_probabilities_only=args.export_probabilities_only,
        data_mode=config['cx_b']['data_mode'],
        config=config)
    if refiner_method != 'none':
        resolved.update(joint_refiner=refiner_method, joint_refiner_strength=refiner_strength,
                        train_scope='refiner_only', effective_joint_train=effective_joint_train(config['cx_b']))
    if args.dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        return 0
    os.makedirs(output_dir, exist_ok=args.joint_init == 'scratch')
    # Delay heavyweight SAM/MMSeg imports until routing and dry-run complete.
    from cx_b.trainer import CXBTrainer
    trainer = CXBTrainer(
        config, config_path, output_dir, seed=args.seed,
        a3_checkpoint_root=args.a3_checkpoint_root,
        selected_concepts=args.concept_indices, resume=args.resume, cli_args=vars(args))
    if args.export_probabilities_only:
        with trainer.report:
            result = trainer.export_probabilities_only()
    else:
        result = trainer.run()
    trainer.report.done = True
    trainer.report.line(f'[done] status={result["status"]} output_dir={output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
