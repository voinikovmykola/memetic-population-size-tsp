"""Iterated local search: the non-population control for the GA sweep.

Nearest-neighbour start, double-bridge perturbation, candidate-list 2-opt run to
a local optimum, accept if not worse. Uses the same operators and the same
20-neighbour candidate lists as run_ga.py, so the two are compared at matched
2-opt calls and matched edge checks.

    python run_ils.py --seeds 10 --seed-start 85001 --calls 12000

This is a control, not a tuned reference implementation.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

import tsp
from tsp import OPTIMUM, UNBOUNDED

ROOT = Path(__file__).resolve().parent


def double_bridge(route, r1, r2, r3):
    """4-opt double bridge at three cut points: A B C D -> A C B D."""
    return np.concatenate((route[:r1], route[r2:r3], route[r1:r2], route[r3:]))


def cut_points(rng, n, max_segment):
    """Three ordered cuts; max_segment=0 draws them uniformly over the tour."""
    if max_segment:
        lengths = [int(rng.integers(1, max_segment)) for _ in range(3)]
        first = int(rng.integers(1, n - sum(lengths) - 1))
        return first, first + lengths[0], first + lengths[0] + lengths[1]
    return tuple(int(v) for v in np.sort(rng.choice(np.arange(1, n), 3, replace=False)))


def run(seed, checkpoints, dist, neighbours, max_segment):
    n = len(dist)
    rng = tsp.rng_for(seed, 31337)
    current = tsp.nearest_neighbour(dist, int(rng.integers(n)), np.zeros(n, np.int64), 3)
    current, c, _ = tsp.two_opt(current, dist, neighbours, UNBOUNDED, UNBOUNDED)
    best = incumbent = tsp.tour_length(current, dist)
    checks = int(c)
    calls = 1
    started = time.perf_counter()
    records = []
    while calls < max(checkpoints):
        kicked = double_bridge(current, *cut_points(rng, n, max_segment))
        candidate, c, _ = tsp.two_opt(kicked, dist, neighbours, UNBOUNDED, UNBOUNDED)
        calls += 1
        checks += int(c)
        length = tsp.tour_length(candidate, dist)
        if length <= incumbent:
            current, incumbent = candidate, length
        best = min(best, length)
        if calls in checkpoints:
            records.append(dict(
                seed=seed, kick='segment' if max_segment else 'unrestricted',
                calls=calls, gap=100 * (best - OPTIMUM) / OPTIMUM, checks=checks,
                seconds=time.perf_counter() - started))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=85001)
    parser.add_argument('--calls', type=int, nargs='*', default=[12000, 30000, 60000, 120000])
    parser.add_argument('--max-segment', type=int, default=0,
                        help='0 for an unrestricted kick, otherwise the segment bound')
    parser.add_argument('--instance', default='d493.tsp')
    parser.add_argument('--output', default='results/ils_runs.csv')
    args = parser.parse_args()

    _, dist = tsp.load_instance(ROOT / args.instance)
    neighbours = tsp.candidate_lists(dist)
    tsp.warm_up(dist, neighbours)

    checkpoints = set(args.calls)
    rows = [record
            for seed in range(args.seed_start, args.seed_start + args.seeds)
            for record in run(seed, checkpoints, dist, neighbours, args.max_segment)]

    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for budget in sorted(checkpoints):
        gaps = [r['gap'] for r in rows if r['calls'] == budget]
        checks = [r['checks'] for r in rows if r['calls'] == budget]
        print(f'{budget:>7} calls: gap {np.mean(gaps):.3f}% '
              f'(sd {np.std(gaps, ddof=1):.3f}, n={len(gaps)}), '
              f'checks {np.mean(checks) / 1e6:.1f}M')


if __name__ == '__main__':
    main()
