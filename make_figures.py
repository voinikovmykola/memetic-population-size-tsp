"""Regenerate the three published figures from the archived CSV files.

    python make_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
POPULATIONS = [20, 40, 80, 200]
BUDGETS = [15000, 30000, 60000, 120000]
COLOUR = {10: '#4C72B0', 20: '#DD8452', 40: '#C44E52'}
STYLE = {'figure.dpi': 300, 'font.family': 'DejaVu Sans', 'font.size': 9,
         'axes.spines.top': False, 'axes.spines.right': False,
         'axes.titlesize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 8,
         'ytick.labelsize': 8, 'axes.labelsize': 9}


def load():
    data = pd.read_csv(ROOT / 'data' / 'budget_ls_sweep.csv')
    data['ls'] = (data.ls_fraction * 100).round().astype(int)
    return data[data.offspring.isin(BUDGETS)]


def half_width(series):
    """Half-width of the 95% t-interval of the mean."""
    n = series.count()
    return stats.t.ppf(0.975, n - 1) * series.std() / np.sqrt(n)


def shared_legend(fig, ax, top=0.91, handlelength=2.2):
    """One-row legend of the 2-opt fractions, centred above the axes."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               title='Частка нащадків з 2-opt:', title_fontsize=7,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.5,
               handlelength=handlelength)
    fig.tight_layout(rect=(0, 0, 1, top))


def population_panels(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3))
    for ax, letter, budget in zip(axes, 'аб', (15000, 120000)):
        slice_ = data[data.offspring == budget]
        for fraction in COLOUR:
            group = slice_[slice_.ls == fraction].groupby('pop')['gap']
            mean = group.mean().reindex(POPULATIONS)
            band = group.apply(half_width).reindex(POPULATIONS)
            ax.plot(POPULATIONS, mean, 'o-', color=COLOUR[fraction], lw=1.5,
                    ms=4.5, label=f'{fraction}%')
            ax.fill_between(POPULATIONS, mean - band, mean + band,
                            color=COLOUR[fraction], alpha=0.14, lw=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(POPULATIONS)
        ax.set_xticklabels(POPULATIONS)
        ax.minorticks_off()
        ax.set_xlim(16, 245)
        ax.set_xlabel('розмір популяції $P$')
        ax.set_title(f'({letter}) бюджет {budget // 1000} тис. нащадків', pad=6)
    axes[0].set_ylabel('відхилення від оптимуму, %')
    shared_legend(fig, axes[0], top=0.9)
    fig.savefig(path)


def paired_effect(data, path):
    wide = data.pivot_table(index=['offspring', 'ls', 'seed'], columns='pop', values='gap')
    positions = np.arange(len(BUDGETS))
    dodge = 0.14
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.axhline(0, color='0.6', lw=0.8, ls='--', zorder=0)
    for k, fraction in enumerate(COLOUR):
        means, bars = [], []
        for budget in BUDGETS:
            paired = wide.loc[(budget, fraction)][200] - wide.loc[(budget, fraction)][20]
            means.append(paired.mean())
            bars.append(half_width(paired))
        ax.errorbar(positions + (k - 1) * dodge, means, yerr=bars, fmt='o-',
                    color=COLOUR[fraction], lw=1.5, ms=4.5, elinewidth=1,
                    capsize=2.5, capthick=1, label=f'{fraction}%')
    ax.set_xticks(positions)
    ax.set_xticklabels([b // 1000 for b in BUDGETS])
    ax.set_xlim(-0.4, len(BUDGETS) - 0.6)
    ax.set_xlabel('бюджет, тис. нащадків')
    ax.set_ylabel(r'$\Delta$Gap, в. п.')
    shared_legend(fig, ax)
    fig.savefig(path)


def time_quality(data, path):
    slice_ = data[data.offspring == 120000]
    nudge = {(10, 80): (4, -11), (10, 40): (5, 4), (20, 20): (5, -11), (40, 40): (5, -11)}
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    for fraction in COLOUR:
        means = slice_[slice_.ls == fraction].groupby('pop')[['seconds', 'gap']].mean()
        means = means.reindex(POPULATIONS)
        ax.plot(means.seconds, means.gap, 'o', color=COLOUR[fraction], ms=7,
                label=f'{fraction}%')
        for size, (x, y) in zip(POPULATIONS, means.values):
            ax.annotate(str(size), (x, y), textcoords='offset points',
                        xytext=nudge.get((fraction, size), (5, 4)), size=7,
                        color=COLOUR[fraction])
    ax.margins(x=0.06, y=0.12)
    ax.set_xlabel('фактичний час прогону, с')
    ax.set_ylabel('відхилення від оптимуму, %')
    shared_legend(fig, ax, handlelength=2.0)
    fig.savefig(path)


def main():
    mpl.rcParams.update(STYLE)
    data = load()
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    population_panels(data, figures / 'population_15k_120k.png')
    paired_effect(data, figures / 'paired_budget_effect.png')
    time_quality(data, figures / 'time_quality_120k.png')
    print(f'wrote three figures to {figures}')


if __name__ == '__main__':
    main()
