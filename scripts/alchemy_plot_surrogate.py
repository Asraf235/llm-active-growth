"""Projection figures drawn by alchemy's own plotting functions (data_utils.plot_surrogate and
plot_gp_mean_projections) for the GP, the BNN and the random forest, for a side-by-side check
against the package figures (aln_bo/alchemy_plots.py).

    wsl: python scripts/alchemy_plot_surrogate.py "AlN Sputter Log.xlsx" --runs data/run_selection.csv

Surrogates:
    GP   the campaign's BoTorch SingleTaskGP (built as alchemy.gp_models.create_single_task_gp)
    BNN  alchemy.surrogates.NeuroBayesBNNSurrogate (NeuroBayes MLP, NUTS)
    RF   alchemy.surrogates.RandomForestSurrogate (bootstrap ensemble of random forests)
All three are fitted to the same runs and evaluated on the same full grid: one 1-D grid per FREE
knob, all combinations, context knobs at their fixed values. alchemy's plotting functions average
the resulting array over the other axes. They call plt.show(); with the Agg backend that is a no-op,
so the figures they create are saved afterwards. Needs the WSL environment (torch, botorch, alchemy,
neurobayes).
"""
import argparse
import pathlib
import sys
import time
import warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from aln_bo import config as cfg
from aln_bo.config import PARAMS, FREE_IDX, apply_fixed
from aln_bo.data import load_sputter_log, load_dataset, parse_run_selection, select_runs
from aln_bo.campaign import MainCampaign


def tidy(fig, title, wspace=1.1, hspace=0.55):
    """Layout fixes applied AFTER alchemy has drawn, leaving its drawing code untouched:
    * more room between panels so y-labels do not run into the neighbouring colourbar (the
      colourbars re-parent each panel into a sub-grid, so the TOP-level gridspec must be updated);
    * four colourbar ticks printed with 4 significant digits (alchemy turns the offset notation off
      for the axes but not for the colourbars, whose '1e-10+3.4e-1' text hit the panel titles);
      a panel that is flat to within 0.01 % is labelled as such;
    * the figure title placed above the top row of panels."""
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    try:
        panels[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().update(wspace=wspace, hspace=hspace, top=0.93)
    except Exception:
        fig.subplots_adjust(wspace=wspace, hspace=hspace, top=0.93)
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            lo, hi = ax.get_ylim()
            if hi - lo < 1e-4 * max(abs(lo), abs(hi), 1e-12):
                ax.set_yticks([lo, hi])
                ax.set_yticklabels([f"{lo:.4g}", f"{hi:.4g}"])
                ax.set_title(f"flat\n(range {hi - lo:.0e})", fontsize=7)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))
    fig.canvas.draw()                                       # positions are only final after a draw
    top = max(ax.get_position().y1 for ax in panels)
    fig.suptitle(title, fontsize=11, y=min(top + 0.06, 0.995))
    return fig


def posterior_on_grid(model_posterior, X, chunk=4096):
    """mean, variance (numpy) of a BoTorch-style model.posterior(X) over a large grid, in chunks."""
    means, variances = [], []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            post = model_posterior(torch.as_tensor(X[s:s + chunk], dtype=torch.float64))
            means.append(np.asarray(post.mean, float).reshape(-1))
            variances.append(np.asarray(post.variance, float).reshape(-1))
    return np.concatenate(means), np.concatenate(variances)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="runs CSV/XLSX (one row per run), or the two-sheet AlN sputter log")
    ap.add_argument("--target", choices=cfg.objective_names(), default=None)
    ap.add_argument("--kernel", choices=["rbf", "matern"], default="rbf")
    ap.add_argument("--surrogates", default="GP,BNN,RF", help="comma list from GP, BNN, RF")
    ap.add_argument("--n", type=int, default=12, help="grid points per free knob (full grid = n**5 points)")
    ap.add_argument("--bnn-samples", type=int, default=500, help="NUTS warmup and samples for the BNN")
    ap.add_argument("--rf-bootstrap", type=int, default=10)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--runs-keep", default="INCLUDE")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="alchemy_plots")
    ap.add_argument("--suffix", default="", help="appended to the file names (e.g. _v2 when the old PNGs are open)")
    a = ap.parse_args()
    from alchemy.data_utils import plot_surrogate, plot_gp_mean_projections

    out = pathlib.Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    ds = load_sputter_log(a.data) if a.data.lower().endswith((".xlsx", ".xls")) and "sputter" in a.data.lower() else load_dataset(a.data)
    if a.runs:
        ds = select_runs(ds, parse_run_selection(a.runs, a.runs_keep))
    target = cfg.target(a.target)
    c = MainCampaign(ds, target=target, backend="botorch", kernel=a.kernel).fit()
    Xt, Yt = torch.as_tensor(c.X, dtype=torch.float64), torch.as_tensor(c.Y, dtype=torch.float64)
    print(f"{len(c.X)} runs; BoTorch GP ({a.kernel}) lengthscales {c.importance()}")

    # ---- full grid over the free knobs, context fixed
    free = [(i, PARAMS[i]) for i in FREE_IDX]
    grids = [np.linspace(p.low, p.high, a.n) for _, p in free]
    mesh = np.meshgrid(*grids, indexing="ij")                    # alchemy reshapes mean to (n0, n1, ...): "ij" order
    G = np.column_stack([m.ravel() for m in mesh])
    X = np.repeat(apply_fixed(np.zeros((1, len(PARAMS)))), len(G), axis=0)
    for k, (i, _) in enumerate(free):
        X[:, i] = G[:, k]
    print(f"grid: {a.n}^{len(free)} = {len(G)} recipes")
    Xtr = torch.as_tensor(c.X[:, FREE_IDX])
    names = [f"{p.name} [{p.unit}]" if p.unit else p.name for _, p in free]
    tgrids = [torch.as_tensor(g) for g in grids]

    # ---- the surrogates: GP (ours, alchemy-built), BNN and RF (alchemy's own classes)
    wanted = [s.strip().upper() for s in a.surrogates.split(",")]
    models = {}
    if "GP" in wanted:
        models["GP"] = c.model.model.posterior
    if "BNN" in wanted:
        from alchemy.surrogates import NeuroBayesBNNSurrogate
        import neurobayes as nb
        t0 = time.time()
        bnn = NeuroBayesBNNSurrogate(architecture=nb.FlaxMLP(hidden_dims=[32, 32], target_dim=1),
                                     num_warmup=a.bnn_samples, num_samples=a.bnn_samples, progress_bar=False).fit(Xt, Yt)
        bnn.eval()                                             # freeze the Normalize bounds learned in fit
        models["BNN"] = bnn.posterior
        print(f"alchemy NeuroBayesBNNSurrogate fitted in {time.time() - t0:.0f} s")
    if "RF" in wanted:
        from alchemy.surrogates import RandomForestSurrogate
        rf = RandomForestSurrogate(n_bootstrap=a.rf_bootstrap, n_trees=100).fit(Xt, Yt)
        rf.eval()
        models["RF"] = rf.posterior
        print("alchemy RandomForestSurrogate fitted")

    for label, posterior in models.items():
        t0 = time.time()
        mean, var = posterior_on_grid(posterior, X)
        print(f"{label}: grid evaluated in {time.time() - t0:.0f} s; mean range {mean.min():.3f}..{mean.max():.3f}, "
              f"std range {np.sqrt(var).min():.3f}..{np.sqrt(var).max():.3f}")
        mean_t, var_t = torch.as_tensor(mean), torch.as_tensor(var)
        plt.close("all")
        plot_surrogate(mean_t, tgrids, Xtr, Yt, var=var_t, param_names=names, figsize=(18, 18))
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig, kind in zip(figs, ("mean", "variance")):
            tidy(fig, f"alchemy plot_surrogate: {label} {kind} of {target.name}, full-grid projection over the other "
                      f"free knobs (context fixed), {len(c.X)} runs", wspace=0.75, hspace=0.15)
            fig.savefig(out / f"alchemy_plot_surrogate_{label}_{kind}{a.suffix}.png", dpi=120, bbox_inches="tight")
        plt.close("all")
        plot_gp_mean_projections(mean_t, tgrids, Xtr, Yt, param_names=names, figsize=(15, 14))
        fig = plt.figure(plt.get_fignums()[-1])
        tidy(fig, f"alchemy plot_gp_mean_projections: {label} mean of {target.name}, {len(c.X)} runs", wspace=0.5, hspace=0.45)
        fig.savefig(out / f"alchemy_plot_gp_mean_projections_{label}{a.suffix}.png", dpi=120, bbox_inches="tight")
        print(f"wrote alchemy_plot_surrogate_{label}_mean/variance{a.suffix}.png, alchemy_plot_gp_mean_projections_{label}{a.suffix}.png")
    print("->", out)


if __name__ == "__main__":
    main()
