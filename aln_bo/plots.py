"""The package's figures: projection maps of a surrogate over every pair of free knobs.

A projection is the surrogate evaluated on a full grid over the FREE knobs (context knobs at their fixed
values) and averaged over the other knobs for every pair. Measured runs are drawn on top (one marker per
distinct setting, sized by the number of runs stacked there) and the suggested next run as a red star on
the acquisition figure. Works for any surrogate (GP, BNN, random forest).

The two panel layouts were adapted from plot_surrogate and plot_gp_mean_projections in
https://github.com/sumner-harris/alchemy (Copyright (c) 2025 sumner-harris, MIT License).

    grids, XG = full_grid(n)                        # the grid (list of 1-D axes) and the recipes on it
    mean, var = model mean / variance on XG         # any model with .predict(X) -> (mean, std)
    surrogate_figures(mean, var, grids, X, Y, names, "GP", out, marks)      -> <out>/<label>_mean.png, _variance.png
    acquisition_figure(acq, grids, X, Y, names, "EI (GP)", path, marks)    -> one figure, star on the suggestion
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter

from .config import PARAMS, FREE_IDX, apply_fixed

MAX_GRID_POINTS = 400_000


def full_grid(n: int = 12):
    """One linspace per free knob; the full product grid as recipes (context fixed). n is reduced if the
    product would exceed MAX_GRID_POINTS (a full grid is exponential in the knob count)."""
    free = [(i, PARAMS[i]) for i in FREE_IDX]
    d = max(len(free), 1)
    n = max(3, min(n, int(MAX_GRID_POINTS ** (1.0 / d))))
    grids = [np.linspace(p.low, p.high, n) for _, p in free]
    mesh = np.meshgrid(*grids, indexing="ij")
    G = np.column_stack([m.ravel() for m in mesh])
    XG = np.repeat(apply_fixed(np.zeros((1, len(PARAMS)))), len(G), axis=0)
    for k, (i, _) in enumerate(free):
        XG[:, i] = G[:, k]
    return grids, XG


def free_names() -> list[str]:
    return [f"{PARAMS[i].name} [{PARAMS[i].unit}]" if PARAMS[i].unit else PARAMS[i].name for i in FREE_IDX]


def evaluate(fn, XG: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """fn(X) -> values, evaluated in chunks (keeps torch / jax memory small)."""
    return np.concatenate([np.asarray(fn(XG[k:k + chunk]), float).ravel() for k in range(0, len(XG), chunk)])


def _project(values: np.ndarray, grids, i: int, j: int) -> np.ndarray:
    arr = np.asarray(values, float).reshape(*[len(g) for g in grids])
    other = tuple(k for k in range(len(grids)) if k not in (i, j))
    return arr.mean(axis=other) if other else arr


def _panel_finish(ax, gi, gj, names, i, j):
    ax.set_box_aspect(1)
    ax.margins(0)
    ax.set_xlim(gi[0], gi[-1]); ax.set_ylim(gj[0], gj[-1])
    fmt = ScalarFormatter(useOffset=False); fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(names[i]); ax.set_ylabel(names[j])
    ax.set_title(f"{names[i]} vs {names[j]}", pad=2, fontsize="small")


def _colourbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    lo, hi = cb.ax.get_ylim()
    if hi - lo < 1e-4 * max(abs(lo), abs(hi), 1e-12):
        cb.ax.set_yticks([lo, hi]); cb.ax.set_yticklabels([f"{lo:.4g}", f"{hi:.4g}"])
        cb.ax.set_title(f"flat\n(range {hi - lo:.0e})", fontsize=7)
    else:
        cb.ax.yaxis.set_major_locator(MaxNLocator(4))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))


def _snap(v, step):
    return np.round(v / step) * step if step else v


def _points_and_star(ax, Xf, Y, i, j, marks):
    """Measured runs: one marker per distinct (knob_i, knob_j) setting, sized by the number of runs stacked
    there, coloured by their mean value, with the count printed when more than one run shares the spot."""
    y = np.asarray(Y, float).ravel()
    # group runs that fall within the same instrument step (e.g. 10.6 and 11 min plasma time at a 1-min step)
    key = np.column_stack([_snap(Xf[:, k], PARAMS[FREE_IDX[k]].step) for k in (i, j)])
    pts, inv, cnt = np.unique(np.round(key, 6), axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    ymean = np.array([y[inv == k].mean() for k in range(len(pts))])
    ax.scatter(pts[:, 0], pts[:, 1], c=ymean, cmap="viridis", s=30 + 9 * np.minimum(cnt, 30), edgecolor="k", zorder=4)
    for (px, py), n in zip(pts, cnt):
        if n > 1:
            ax.annotate(str(n), (px, py), xytext=(5, 4), textcoords="offset points", fontsize=7, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.75))
    if marks is not None:   # clip_on=False: a suggestion on a bound is drawn whole, not cut by the panel frame
        ax.scatter(marks[:, i], marks[:, j], marker="*", s=260, c="red", edgecolor="k", zorder=6, clip_on=False)


def _title(fig, panels, text):
    fig.canvas.draw()
    top = max(ax.get_position().y1 for ax in panels)
    fig.suptitle(text, fontsize=11, y=min(top + 0.06, 0.995))


def _upper_triangle(values, grids, Xf, Y, names, marks, title, path, cell=4.5):
    """Upper-triangle GridSpec of square panels, one array (mean or variance)."""
    D = len(grids)
    if D < 2:
        return None
    n = D - 1
    fig = plt.figure(figsize=(cell * n, cell * n))
    gs = GridSpec(n, n, figure=fig, wspace=0.75, hspace=0.15, top=0.93)
    panels = []
    for i in range(D - 1):
        for j in range(i + 1, D):
            ax = fig.add_subplot(gs[i, j - i - 1])
            proj = _project(values, grids, i, j)
            gi, gj = grids[i], grids[j]
            im = ax.imshow(proj.T, origin="lower", aspect="auto", extent=(gi[0], gi[-1], gj[0], gj[-1]))
            _points_and_star(ax, Xf, Y, i, j, marks)
            _panel_finish(ax, gi, gj, names, i, j)
            _colourbar(fig, im, ax)
            panels.append(ax)
    _title(fig, panels, title)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def _three_columns(values, grids, Xf, Y, names, marks, title, path):
    """All pairs in 3 columns (acquisition)."""
    D = len(grids)
    combos = list(itertools.combinations(range(D), 2))
    if not combos:
        return None
    ncol = 3
    nrow = math.ceil(len(combos) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.7 * nrow), squeeze=False)
    panels = []
    for idx, (i, j) in enumerate(combos):
        ax = axes[idx // ncol][idx % ncol]
        proj = _project(values, grids, i, j)
        Gi, Gj = np.meshgrid(grids[i], grids[j], indexing="ij")
        im = ax.pcolormesh(Gi, Gj, proj, shading="auto")
        _points_and_star(ax, Xf, Y, i, j, marks)
        ax.set_xlabel(names[i]); ax.set_ylabel(names[j]); ax.set_title(f"{names[i]} vs {names[j]}")
        _colourbar(fig, im, ax)
        panels.append(ax)
    for idx in range(len(combos), nrow * ncol):
        fig.delaxes(axes[idx // ncol][idx % ncol])
    fig.subplots_adjust(wspace=0.5, hspace=0.45, top=0.93)
    _title(fig, panels, title)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def surrogate_figures(mean, var, grids, X, Y, names, label: str, out: str | Path, marks=None,
                      target: str = "", n_runs: Optional[int] = None) -> dict:
    """Mean and variance projections of one surrogate."""
    out = Path(out)
    Xf = np.asarray(X)[:, FREE_IDX]
    n_runs = len(Xf) if n_runs is None else n_runs
    paths = {}
    for kind, vals in (("mean", mean), ("variance", var)):
        p = out / f"{label}_{kind}.png"
        _upper_triangle(vals, grids, Xf, Y, names, marks,
                        f"{label} {kind} of {target}, {n_runs} runs (projection over the other free knobs, context fixed)"
                        + ("; red star = suggested next run" if marks is not None else ""), p)
        paths[kind] = p
    return paths


def acquisition_figure(acq, grids, X, Y, names, label: str, path: str | Path, marks=None, target: str = "",
                       n_runs: Optional[int] = None):
    """Acquisition projection with the suggestion starred."""
    Xf = np.asarray(X)[:, FREE_IDX]
    n_runs = len(Xf) if n_runs is None else n_runs
    return _three_columns(acq, grids, Xf, Y, names, marks,
                          f"acquisition {label} of {target} (larger = more worth trying), {n_runs} runs; "
                          f"red star = suggested next run", path)
