#!/usr/bin/env python3
"""Sequential experiment runner.

Usage:
    python tools/run_experiments.py experiments/aachen_day_sweep.yaml
    python tools/run_experiments.py experiments/aachen_day_sweep.yaml --only A_mean_only F_plm_full
    python tools/run_experiments.py experiments/aachen_day_sweep.yaml --skip H_rank0_mean_only
    python tools/run_experiments.py experiments/aachen_day_sweep.yaml --summary-only
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def deep_set(d: dict, dotkey: str, value) -> None:
    """Set d[a][b][c] from dotkey 'a.b.c'."""
    keys = dotkey.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(base: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        deep_set(cfg, k, v)
    return cfg


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def already_done(out_dir: Path) -> bool:
    return (out_dir / 'metrics.json').exists()


def print_summary(outputs_root: Path, experiments: list[dict]) -> None:
    rows = []
    for exp in experiments:
        name = exp['name']
        out_dir = outputs_root / name
        mfile = out_dir / 'metrics.json'
        if not mfile.exists():
            rows.append({'name': name, 'status': 'missing'})
            continue
        with open(mfile) as f:
            m = json.load(f)
        summary = m.get('summary', m)
        rows.append({
            'name': name,
            'status': 'done',
            'success_rate': summary.get('success_rate', float('nan')),
            'med_rot_deg': summary.get('median_rot_err_deg', float('nan')),
            'med_trans_m': summary.get('median_trans_err_m', float('nan')),
            'n_success': summary.get('num_success', '?'),
            'n_queries': summary.get('num_queries', '?'),
        })

    print()
    print(f"{'Experiment':<30} {'Status':<8} {'Success%':>9} {'MedRot°':>9} {'MedTrans m':>11} {'Pass/Total':>12}")
    print('-' * 85)
    for r in rows:
        if r['status'] == 'missing':
            print(f"{r['name']:<30} {'missing':<8}")
            continue
        sr = r['success_rate']
        print(
            f"{r['name']:<30} {'done':<8} "
            f"{sr*100:>8.1f}% "
            f"{r['med_rot_deg']:>9.2f} "
            f"{r['med_trans_m']:>11.3f} "
            f"{str(r['n_success'])+'/'+str(r['n_queries']):>12}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description='Sequential experiment runner')
    parser.add_argument('manifest', type=str, help='Path to experiment manifest YAML')
    parser.add_argument('--only', nargs='+', default=None, help='Run only these experiment names')
    parser.add_argument('--skip', nargs='+', default=None, help='Skip these experiment names')
    parser.add_argument('--rerun', action='store_true', help='Rerun even if metrics.json exists')
    parser.add_argument('--summary-only', action='store_true', help='Print summary table and exit')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_yaml(manifest_path)

    base_cfg_path = Path(manifest['base_config'])
    base_cfg = load_yaml(base_cfg_path)

    dataset_root = manifest.get('dataset_root') or base_cfg.get('dataset_root')
    pipeline = manifest.get('pipeline', 'plm_match.pipelines.hloc_localize')
    outputs_root = Path(manifest.get('outputs_root', './outputs/sweep'))
    experiments = manifest['experiments']

    if args.only:
        experiments = [e for e in experiments if e['name'] in args.only]
    if args.skip:
        experiments = [e for e in experiments if e['name'] not in args.skip]

    outputs_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print_summary(outputs_root, experiments)
        return

    skip_set = set(args.skip or [])
    total = len(experiments)
    passed = 0
    failed = []
    skipped = 0

    print(f'\nRunning {total} experiments from {manifest_path.name}')
    print(f'Outputs root: {outputs_root}\n')

    for i, exp in enumerate(experiments):
        name = exp['name']
        out_dir = outputs_root / name

        if name in skip_set:
            print(f'[{i+1}/{total}] SKIP  {name}')
            skipped += 1
            continue

        if not args.rerun and already_done(out_dir):
            print(f'[{i+1}/{total}] DONE  {name}  (already complete, use --rerun to force)')
            skipped += 1
            continue

        overrides = exp.get('overrides', {})
        cfg = apply_overrides(base_cfg, overrides)
        cfg['out_dir'] = str(out_dir)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', prefix=f'exp_{name}_', delete=False
        ) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        print(f'[{i+1}/{total}] START {name}')
        if overrides:
            for k, v in overrides.items():
                print(f'          {k}: {v}')

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, '-m', pipeline,
             '--config', tmp_path,
             '--dataset_root', str(dataset_root),
             '--out_dir', str(out_dir)],
            cwd=Path(__file__).parent.parent,
        )
        elapsed = time.time() - t0

        Path(tmp_path).unlink(missing_ok=True)

        if result.returncode == 0:
            passed += 1
            mfile = out_dir / 'metrics.json'
            if mfile.exists():
                with open(mfile) as f:
                    m = json.load(f)
                summary = m.get('summary', m)
                sr = summary.get('success_rate', float('nan'))
                print(f'[{i+1}/{total}] OK    {name}  success={sr*100:.1f}%  ({elapsed:.0f}s)')
            else:
                print(f'[{i+1}/{total}] OK    {name}  ({elapsed:.0f}s)')
        else:
            failed.append(name)
            print(f'[{i+1}/{total}] FAIL  {name}  returncode={result.returncode}  ({elapsed:.0f}s)')

    print(f'\n{"="*60}')
    print(f'Done: {passed}/{total} passed, {len(failed)} failed, {skipped} skipped')
    if failed:
        print(f'Failed: {failed}')

    print_summary(outputs_root, manifest['experiments'])


if __name__ == '__main__':
    main()
