# Population size and local-search intensity in a memetic GA for the TSP

Code and data for the study of how population size interacts with local-search
intensity and the evaluation budget in a memetic genetic algorithm, measured on
the TSPLIB95 instance `d493`.

The result is an interaction, not a new solver: the advantage of a small
population appears only at a high local-search fraction and a short budget, and
weakens as the budget grows.

## Contents

| Path | Purpose |
|---|---|
| `tsp.py` | instance loading, OX, inversion, candidate-list 2-opt with don't-look bits |
| `run_ga.py` | the memetic GA sweep (population x local-search fraction x budget) |
| `run_ils.py` | iterated local search, the non-population control |
| `collect_ga.py` | collects the per-run GA files into one table of budget checkpoints |
| `make_figures.py` | regenerates the three published figures from the CSV files |
| `data/budget_ls_sweep.csv` | 1152 records from 288 GA runs, all budget checkpoints; `ls` repeats `ls_fraction` in percent |
| `data/ils_runs.csv` | 40 records from 10 ILS runs |
| `figures/population_15k_120k.png` | Fig. 1 of the theses |
| `figures/paired_budget_effect.png` | Fig. 2 of the theses |
| `figures/time_quality_120k.png` | Fig. 3 of the theses |
| `CITATION.cff` | citation metadata (also read by Zenodo) |
| `LICENSE`, `LICENSE-DATA` | MIT for code, CC BY 4.0 for data, figures and documentation |

## Experimental setup

- TSPLIB95 EUC_2D instance `d493`, 493 cities, optimum 35 002.
- Single population, permutation encoding, order crossover (rate 0.9),
  inversion mutation (rate 0.25), tournament selection of size two, elitism of
  2.5% of the population rounded to whole individuals with a minimum of one, so
  P = 20 keeps one elite (5%).
- 2-opt restricted to the 20 nearest cities of each endpoint, with don't-look
  bits, run to a local optimum of that candidate neighbourhood (`--ls-moves 0`).
  This is not full 2-opt over all pairs of edges.
- Population sizes 20, 40, 80, 200; local-search fractions 0.1, 0.2, 0.4.
- 24 paired initial generator states, 86001 to 86024.
- Four nearest-neighbour constructive starts in every initial population,
  independent of P, so population size is not confounded with the number of
  constructive draws (`--init nn --heur 4`).
- Each run records checkpoints at 15k, 30k, 60k and 120k offspring, so the
  budget axis comes from one sweep rather than four.
- At a fixed local-search fraction the number of 2-opt calls is equal across
  population sizes by construction; measured edge checks differ by under 3%.
  Wall time is recorded but not equalised.

## Reproducing

The TSPLIB instance is not redistributed here. Download `d493.tsp` from
<https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/> and place it beside
the scripts; the file used in this study has SHA-256

```
0360e94989bfa4bbd37a7f24e3ec578115fdd5ad7a2d2f765e10703deab78ade
```

```bash
pip install -r requirements.txt
python run_ga.py --seeds 24 --seed-start 86001 --budget 120000 \
                 --pop 20 40 80 200 --ls 0.1 0.2 0.4 --output results
python run_ils.py --seeds 10 --seed-start 85001 --calls 12000 30000 60000 120000
python collect_ga.py
python make_figures.py
```

Reruns write to `results/` and leave the archived `data/` files untouched.
`make_figures.py` draws from `data/`; to plot a rerun, change the path in its
`load()` to `results/budget_ls_sweep.csv`.

`SHA256SUMS.txt` covers the archived copies of the files as deposited. A
regenerated figure can be visually identical and still have a different hash,
because PNG files carry metadata such as the Matplotlib version; compare
regenerated figures by eye, not by checksum.

`run_ga.py` skips configurations whose output file already exists, so an
interrupted sweep resumes without recomputing. Library versions are written to
`results/config.json` at run time.

Wall-clock readings depend on the machine and are not portable benchmarks.
Tour quality is deterministic given the generator states: a rerun reproduced the
archived record for seed 86001, P = 40, q = 0.2 exactly in tour length, gap,
2-opt calls, edge checks, accepted moves, edge entropy and distinct edges.

## Scope of the claims

The primary statistics are two interactions stated before the sweep: population
size with local-search fraction, and population size with budget. The 36
cellwise contrasts are descriptive; three of them pass a Bonferroni threshold of
0.05/36, all for P = 200 against P = 20 at a local-search fraction of 0.4, at
budgets of 15k, 30k and 60k offspring.

Edge counts and edge entropy are recorded and reported as description. No causal
claim is made about diversity and final tour quality.

The ILS control shows that the population component is not redundant under these
settings. It is a control, not a tuned reference implementation, and no claim is
made against specialised TSP solvers.

## Citation

If you use this code or data, please cite this archive (metadata in
`CITATION.cff`):

> Voinikov, M. (2026). _Population size and local-search intensity in a
> memetic genetic algorithm for the TSP: code and data_ (Version 1.0.0)
> [Software]. Zenodo. https://doi.org/10.5281/zenodo.24044880

Cite the DOI of this version rather than the concept DOI, which always
resolves to the latest version.

## License

The source code (`*.py`) is released under the MIT License (`LICENSE`). The
data, figures and documentation are released under CC BY 4.0
(`LICENSE-DATA`). The TSPLIB instance is not included and keeps its own terms.
