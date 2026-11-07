"""Instance loading, permutation operators and candidate-list 2-opt.

Shared by run_ga.py and run_ils.py. Distances follow TSPLIB EUC_2D:
d(i,j) = floor(sqrt(dx^2 + dy^2) + 0.5), integer.
"""
from pathlib import Path

import numpy as np
from numba import njit

OPTIMUM = 35002           # d493, TSPLIB
UNBOUNDED = 10 ** 8


def load_instance(path):
    """Read a TSPLIB EUC_2D file; return coordinates and the distance matrix."""
    meta, coords, in_coords = {}, [], False
    for line in (s.strip() for s in Path(path).read_text(encoding='utf-8').splitlines()):
        if line == 'NODE_COORD_SECTION':
            in_coords = True
        elif line in {'EOF', 'DISPLAY_DATA_SECTION'}:
            break
        elif in_coords and line:
            i, x, y = line.split()
            coords.append((int(i), float(x), float(y)))
        elif ':' in line:
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip()
    if meta.get('EDGE_WEIGHT_TYPE') != 'EUC_2D':
        raise ValueError('only EUC_2D instances are supported')
    if len(coords) != int(meta['DIMENSION']):
        raise ValueError('DIMENSION does not match the coordinate count')
    if sorted(c[0] for c in coords) != list(range(1, len(coords) + 1)):
        raise ValueError('expected TSPLIB node indices 1..DIMENSION')
    coords.sort()
    xy = np.array([(x, y) for _, x, y in coords])
    delta = xy[:, None, :] - xy[None, :, :]
    return xy, np.floor(np.sqrt((delta ** 2).sum(axis=2)) + 0.5).astype(np.int32)


def candidate_lists(dist, k=20):
    """The k nearest cities of each city, excluding itself."""
    return np.argsort(dist, axis=1, kind='stable')[:, 1:k + 1].astype(np.int32)


def tour_length(tour, dist):
    return int(dist[tour, np.roll(tour, -1)].sum())


def evaluate(population, dist):
    return dist[population, np.roll(population, -1, axis=1)].sum(axis=1).astype(np.int32)


def is_tour(tour, n):
    return len(tour) == n and np.array_equal(np.sort(tour), np.arange(n))


def rng_for(seed, stream):
    """Independent generator per (seed, stream) so the streams stay separable."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


@njit
def nearest_neighbour(dist, start, choice_stream, rcl):
    """Nearest-neighbour construction; choice_stream picks among the rcl best at each step.

    A stream of zeros gives the deterministic nearest-neighbour tour.
    """
    n = len(dist)
    route = np.empty(n, dtype=np.int32)
    used = np.zeros(n, dtype=np.uint8)
    current = int(start)
    for k in range(n):
        route[k] = current
        used[current] = 1
        if k == n - 1:
            break
        best_ids = np.full(rcl, -1, dtype=np.int32)
        best_cost = np.full(rcl, 2147483647, dtype=np.int64)
        for j in range(n):
            if used[j] != 0:
                continue
            d = int(dist[current, j])
            for q in range(rcl):
                if d < best_cost[q] or (d == best_cost[q] and j < best_ids[q]):
                    for h in range(rcl - 1, q, -1):
                        best_cost[h] = best_cost[h - 1]
                        best_ids[h] = best_ids[h - 1]
                    best_cost[q] = d
                    best_ids[q] = j
                    break
        valid = 0
        while valid < rcl and best_ids[valid] >= 0:
            valid += 1
        current = int(best_ids[int(choice_stream[k]) % valid])
    return route


@njit
def _ox_fill(parent1, parent2, left, right):
    n = len(parent1)
    child = np.full(n, -1, dtype=np.int32)
    used = np.zeros(n, dtype=np.uint8)
    for j in range(left, right + 1):
        child[j] = parent1[j]
        used[parent1[j]] = 1
    pos = (right + 1) % n
    for shift in range(n):
        gene = parent2[(right + 1 + shift) % n]
        if used[gene] == 0:
            child[pos] = gene
            pos = (pos + 1) % n
    return child


def order_crossover(parent1, parent2, rng):
    left, right = sorted(map(int, rng.choice(len(parent1), size=2, replace=False)))
    return _ox_fill(parent1, parent2, left, right)


def invert(tour, rng):
    """Reverse a random segment in place."""
    left, right = sorted(map(int, rng.choice(len(tour), size=2, replace=False)))
    tour[left:right + 1] = tour[left:right + 1][::-1]


@njit
def two_opt(tour, dist, neighbours, max_checks, max_moves):
    """First-improvement 2-opt over candidate lists with don't-look bits.

    Returns the tour, the edge-exchange checks performed and the accepted moves.
    Budgets set high run the search to a local optimum of the candidate
    neighbourhood.
    """
    route = tour.copy()
    n = len(route)
    pos = np.empty(n, np.int32)
    dlb = np.zeros(n, np.uint8)
    checks = 0
    moves = 0
    while checks < max_checks and moves < max_moves:
        for k in range(n):
            pos[route[k]] = k
        changed = False
        for i in range(n - 1):
            a = int(route[i])
            b = int(route[i + 1])
            if dlb[a]:
                continue
            for h in range(neighbours.shape[1]):
                c = int(neighbours[a, h])
                j = int(pos[c])
                if j <= i + 1 or (i == 0 and j == n - 1):
                    continue
                d = int(route[(j + 1) % n])
                checks += 1
                if int(dist[a, c]) + int(dist[b, d]) - int(dist[a, b]) - int(dist[c, d]) < 0:
                    # Every city on the reversed segment gains a new neighbourhood.
                    dlb[a] = 0
                    dlb[d] = 0
                    for k in range(i + 1, j + 1):
                        dlb[route[k]] = 0
                    route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                    moves += 1
                    changed = True
                    break
                if checks >= max_checks:
                    break
            if changed or checks >= max_checks:
                break
            dlb[a] = 1
        if not changed:
            break
    return route, checks, moves


def warm_up(dist, neighbours):
    """Compile the kernels so the first timed run is not distorted."""
    probe = np.arange(len(dist), dtype=np.int32)
    _ox_fill(probe, probe, 10, 20)
    nearest_neighbour(dist, 0, np.zeros(len(dist), np.int64), 3)
    two_opt(probe, dist, neighbours, 3000, 30)
