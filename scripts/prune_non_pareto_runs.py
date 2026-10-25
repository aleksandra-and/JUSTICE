#!/usr/bin/env python3
"""
Remove run directories that are NOT on the Pareto front.

Pareto front is determined by computing avg temperature (min) and cumulative
net economic output (max) from the npy evaluation files for every run, then
applying standard Pareto dominance filtering.

Usage:
    python scripts/prune_non_pareto_runs.py            # dry-run (safe default)
    python scripts/prune_non_pareto_runs.py --delete   # actually delete
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths (mirror the notebook config) ───────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
EVAL_ROOT  = REPO_ROOT / 'results/detailed_analysis/evaluation/algo_comparison_v3_fixed/seeds'
ALGO_ROOTS = {
    'HAPPO': EVAL_ROOT / 'happo',
    'HASAC': EVAL_ROOT / 'hasac',
    'MAPPO': EVAL_ROOT / 'mappo',
}
SCENARIO = 'SSP245'


# ── Helpers ───────────────────────────────────────────────────────────────────
def latest_export_prefix(run_dir: Path) -> str | None:
    candidates = list(run_dir.glob(f'*_idx*_{SCENARIO}_global_temperature.npy'))
    if not candidates:
        return None
    def _idx(p: Path) -> int:
        return int(p.stem.rsplit(f'_{SCENARIO}_global_temperature', 1)[0].split('_idx')[-1])
    best = max(candidates, key=_idx)
    return best.stem.rsplit(f'_{SCENARIO}_global_temperature', 1)[0]


def compute_objectives(algo_root: Path) -> pd.DataFrame:
    from justice.util.model_time import TimeHorizon
    th = TimeHorizon(start_year=2015, end_year=2300, data_timestep=5, timestep=1)
    n_years = len(th.model_time_horizon)

    records = []
    for run_dir in sorted(algo_root.iterdir()):
        if not run_dir.is_dir():
            continue
        prefix = latest_export_prefix(run_dir)
        if prefix is None:
            continue
        temp_path = run_dir / f'{prefix}_{SCENARIO}_global_temperature.npy'
        neo_path  = run_dir / f'{prefix}_{SCENARIO}_net_economic_output.npy'
        if not temp_path.exists() or not neo_path.exists():
            continue

        t = np.load(temp_path)
        if t.ndim == 1:
            avg_temp = float(t.mean())
        elif t.ndim == 2:
            avg_temp = float(np.median(t.mean(axis=0 if t.shape[0] == n_years else 1)))
        else:
            avg_temp = float(np.median(t.mean()))

        n = np.load(neo_path)
        if n.ndim == 3:
            cum_neo = float(np.median(n.sum(axis=0).sum(axis=0)))
        elif n.ndim == 2:
            cum_neo = float(n.sum(axis=0).sum() if n.shape[1] == n_years else np.median(n.sum(axis=0)))
        else:
            cum_neo = float(n.sum())

        records.append({'run_name': run_dir.name, 'run_dir': run_dir,
                        'avg_temperature': avg_temp, 'cumulative_neo': cum_neo})
    return pd.DataFrame(records)


def is_pareto_efficient(points: np.ndarray, directions: list[str]) -> np.ndarray:
    adjusted = points.copy()
    for j, d in enumerate(directions):
        if d == 'min':
            adjusted[:, j] = -adjusted[:, j]
    mask = np.ones(len(adjusted), dtype=bool)
    for i in range(len(adjusted)):
        for j in range(len(adjusted)):
            if i != j and np.all(adjusted[j] >= adjusted[i]) and np.any(adjusted[j] > adjusted[i]):
                mask[i] = False
                break
    return mask


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--delete', action='store_true',
                        help='Actually delete non-Pareto directories (default: dry-run)')
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))

    to_delete: list[Path] = []
    to_keep:   list[Path] = []

    for algo, root in ALGO_ROOTS.items():
        if not root.exists():
            print(f'[{algo}] root not found: {root}')
            continue

        print(f'\n[{algo}] computing objectives in {root} ...')
        df = compute_objectives(root)
        if df.empty:
            print(f'  no runs found')
            continue

        pts   = df[['avg_temperature', 'cumulative_neo']].values
        mask  = is_pareto_efficient(pts, ['min', 'max'])
        pareto_names = set(df.loc[mask, 'run_name'])

        print(f'  {mask.sum()} / {len(df)} runs on Pareto front:')
        for name in sorted(pareto_names):
            print(f'    KEEP  {name}')

        for _, row in df.iterrows():
            if row['run_name'] in pareto_names:
                to_keep.append(row['run_dir'])
            else:
                to_delete.append(row['run_dir'])

    print(f'\n{"=" * 60}')
    print(f'Runs to KEEP  : {len(to_keep)}')
    print(f'Runs to DELETE: {len(to_delete)}')

    if not to_delete:
        print('Nothing to delete.')
        return

    print('\nDirectories that will be DELETED:')
    for p in sorted(to_delete):
        print(f'  {p}')

    if not args.delete:
        print('\n[DRY-RUN] No files were deleted. Re-run with --delete to apply.')
        return

    confirm = input(f'\nDelete {len(to_delete)} directories? [yes/N] ').strip().lower()
    if confirm != 'yes':
        print('Aborted.')
        return

    for p in to_delete:
        shutil.rmtree(p)
        print(f'  Deleted: {p}')

    print(f'\nDone. {len(to_delete)} directories removed.')


if __name__ == '__main__':
    main()
