"""Collect the per-run GA files into one table of budget checkpoints.

    python collect_ga.py --input results --output results/budget_ls_sweep.csv

Keeps the 15k, 30k, 60k and 120k offspring checkpoints of every run, which is
the layout of data/budget_ls_sweep.csv. The archived file is not overwritten by
default: wall-clock columns from a rerun differ by machine.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BUDGETS = (15000, 30000, 60000, 120000)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', default='results')
    parser.add_argument('--output', default='results/budget_ls_sweep.csv')
    args = parser.parse_args()

    files = sorted((ROOT / args.input).glob('seed*_pop*_ls*_mv*.csv'))
    if not files:
        raise SystemExit(f'no run files in {ROOT / args.input}')
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data = data[data.offspring.isin(BUDGETS)]
    data['ls'] = (data.ls_fraction * 100).round().astype(int)
    data = data.sort_values(['seed', 'pop', 'ls_fraction', 'offspring'])

    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, index=False)
    print(f'{len(data)} records from {len(files)} runs -> {target}')


if __name__ == '__main__':
    main()
