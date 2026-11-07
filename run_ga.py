"""Memetic GA sweep: population size x local-search fraction x evaluation budget.

Generational GA with order crossover, inversion and candidate-list 2-opt. The
controlled budget is the number of offspring generated; one run records every
budget checkpoint, so a single sweep yields all budget slices.

    python run_ga.py --seeds 24 --seed-start 86001 --budget 120000 \
                     --pop 20 40 80 200 --ls 0.1 0.2 0.4

Three switches record methodological decisions rather than tuning knobs:

    --init nn|rcl3   constructive starts: nearest neighbour, or randomized with
                     a restricted candidate list of 3 at every step
    --heur N|frac    number of constructive starts: fixed, or 0.20*P
    --ls-moves N     accepted-move cap per 2-opt call; 0 runs to a local optimum

Published results use --init nn --heur 4 --ls-moves 0.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numba
import numpy as np

import tsp
from tsp import OPTIMUM, UNBOUNDED

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = (15000, 30000, 60000, 120000, 240000)
ELITE_FRACTION = 0.025
CROSSOVER_RATE = 0.9
INVERSION_RATE = 0.25


def initial_population(dist, seed, size, init, heur):
    """Constructive starts plus random permutations, shuffled into the population."""
    n = len(dist)
    construction = tsp.rng_for(seed, 20261106)
    n_constructed = round(0.20 * size) if heur == 'frac' else min(int(heur), size)
    routes = []
    for _ in range(n_constructed):
        start = int(construction.integers(n))
        stream = (construction.integers(0, 3, size=n, dtype=np.int64) if init == 'rcl3'
                  else np.zeros(n, dtype=np.int64))
        routes.append(tsp.nearest_neighbour(dist, start, stream, 3))
    filler = tsp.rng_for(seed, 412)
    routes += [filler.permutation(n).astype(np.int32) for _ in range(size - n_constructed)]
    order = tsp.rng_for(seed, 888).permutation(size)
    return np.stack(routes)[order].copy()


def edge_statistics(population):
    """Shannon entropy and count of the undirected edges present in the population."""
    n = population.shape[1]
    nxt = np.roll(population, -1, axis=1)
    edges = np.minimum(population, nxt).astype(np.int64) * n + np.maximum(population, nxt)
    _, counts = np.unique(edges, return_counts=True)
    p = counts / (len(population) * n)
    return float(-(p * np.log(p)).sum()), int(len(counts))


def batch_size(population_size, elites, generated, budget):
    """Offspring to produce now: never overshoot the budget or a checkpoint."""
    remaining = [population_size - elites, budget - generated]
    ahead = [c - generated for c in CHECKPOINTS if c > generated]
    if ahead:
        remaining.append(min(ahead))
    return min(remaining)


def breed(population, fitness, count, dist, rng):
    """Tournament selection of size two, order crossover, inversion."""
    offspring = np.empty((count, population.shape[1]), dtype=np.int32)
    for j in range(count):
        parents = []
        for _ in range(2):
            a, b = rng.integers(len(population), size=2)
            parents.append(population[a if fitness[a] <= fitness[b] else b])
        child = (tsp.order_crossover(parents[0], parents[1], rng)
                 if rng.random() < CROSSOVER_RATE else parents[0].copy())
        if rng.random() < INVERSION_RATE:
            tsp.invert(child, rng)
        offspring[j] = child
    return offspring


def run(seed, size, ls_fraction, budget, dist, neighbours, init, heur, ls_moves, log_every):
    rng = tsp.rng_for(seed, 999)
    population = initial_population(dist, seed, size, init, heur)
    elites = max(1, int(math.floor(ELITE_FRACTION * size + 0.5)))
    fitness = tsp.evaluate(population, dist)
    best = int(fitness.min())
    initial_gap = 100 * (best - OPTIMUM) / OPTIMUM
    move_cap = ls_moves if ls_moves > 0 else UNBOUNDED
    check_cap = 3000 if ls_moves > 0 else UNBOUNDED

    generated = calls = checks = moves = generations = 0
    ls_seconds = 0.0
    started = time.perf_counter()
    records = []
    while generated < budget:
        generations += 1
        keep = np.argsort(fitness, kind='stable')[:elites]
        elite = population[keep].copy()
        count = batch_size(size, elites, generated, budget)
        offspring = breed(population, fitness, count, dist, rng)
        scores = tsp.evaluate(offspring, dist)

        # Cumulative quota, so the number of 2-opt calls per budget does not
        # depend on how the offspring happen to split across generations.
        quota = int(math.floor(ls_fraction * (generated + count) + 1e-9)) - calls
        if quota > 0:
            started_ls = time.perf_counter()
            for idx in np.argsort(scores, kind='stable')[:quota]:
                offspring[idx], c, m = tsp.two_opt(
                    offspring[idx], dist, neighbours, check_cap, move_cap)
                calls += 1
                checks += int(c)
                moves += int(m)
            ls_seconds += time.perf_counter() - started_ls

        if count == size - elites:
            population = np.concatenate((elite, offspring))
        else:
            # Short final batch: top up from the previous generation rather than
            # spending offspring that the budget no longer allows.
            survivors = population[np.argsort(fitness, kind='stable')[elites:]]
            population = np.concatenate(
                (elite, offspring, survivors[:size - elites - count]))
        fitness = tsp.evaluate(population, dist)
        best = min(best, int(fitness.min()))
        generated += count

        if generations % log_every == 0 or generated in CHECKPOINTS or generated == budget:
            entropy, distinct = edge_statistics(population)
            records.append(dict(
                seed=seed, pop=size, ls_fraction=ls_fraction, ls_moves=ls_moves,
                init=init, heur=str(heur), offspring=generated, generations=generations,
                init_best=initial_gap, best=best, gap=100 * (best - OPTIMUM) / OPTIMUM,
                entropy=entropy, unique_edges=distinct, calls=calls, checks=checks,
                moves=moves, ls_seconds=ls_seconds, seconds=time.perf_counter() - started))
    return records


def write_csv(path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seeds', type=int, default=24)
    parser.add_argument('--seed-start', type=int, default=86001)
    parser.add_argument('--budget', type=int, default=120000)
    parser.add_argument('--pop', type=int, nargs='*', default=[20, 40, 80, 200])
    parser.add_argument('--ls', type=float, nargs='*', default=[0.1, 0.2, 0.4])
    parser.add_argument('--init', choices=['nn', 'rcl3'], default='nn')
    parser.add_argument('--heur', default='4')
    parser.add_argument('--ls-moves', type=int, nargs='*', default=[0])
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--instance', default='d493.tsp')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    _, dist = tsp.load_instance(ROOT / args.instance)
    neighbours = tsp.candidate_lists(dist)
    tsp.warm_up(dist, neighbours)

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(vars(args), python=sys.version.split()[0],
                      numpy=np.__version__, numba=numba.__version__)
    (out / 'config.json').write_text(json.dumps(provenance, indent=2))

    jobs = [(seed, size, fraction, cap)
            for seed in range(args.seed_start, args.seed_start + args.seeds)
            for size in args.pop
            for fraction in args.ls
            for cap in (args.ls_moves if fraction > 0 else args.ls_moves[:1])]
    for k, (seed, size, fraction, cap) in enumerate(jobs, 1):
        target = out / f'seed{seed}_pop{size}_ls{round(100 * fraction):02d}_mv{cap}.csv'
        if target.exists():          # completed runs are never recomputed
            print(f'[{k}/{len(jobs)}] skip {target.name}', flush=True)
            continue
        rows = run(seed, size, fraction, args.budget, dist, neighbours,
                   args.init, args.heur, cap, args.log_every)
        write_csv(target, rows)
        print(f'[{k}/{len(jobs)}] P={size} LS={fraction:g} mv={cap} seed={seed} '
              f'gap={rows[-1]["gap"]:.3f}% t={rows[-1]["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
