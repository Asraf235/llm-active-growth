"""One active-learning step done with alchemy's own functions end to end (WSL / Linux: needs alchemy).

    python scripts/alchemy_step.py "<runs.csv>" --surrogate bnn --target RC_FWHM --out "<folder>"

Uses alchemy.active_learning.step_surrogate: it fits alchemy's surrogate (NeuroBayesBNNSurrogate,
RandomForestSurrogate or GPSurrogate), builds BoTorch's qLogExpectedImprovement with best_f = the best
observed value, and scores a discrete candidate set (optimize_acqf_discrete). The candidate set is the
package's Sobol pool snapped to the instrument grid with the context knobs at their fixed values, so the
proposal is a recipe you can grow. Everything else (model, acquisition, optimiser) is alchemy's.

Known alchemy quirk, reported as a note in the summary: step_surrogate calls model.posterior() while the
model is still in training mode, so BoTorch's Normalize transform re-learns its bounds from the candidate
set instead of using the training bounds. The package's own scripts call .eval() to avoid this.
"""
import argparse
import datetime as dt
import pathlib
import sys
import time
import warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import torch

from aln_bo import config as cfg
from aln_bo.config import PARAM_NAMES, add_physical
from aln_bo.data import load_dataset, parse_run_selection, select_runs
from aln_bo.acquisition import sobol_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--target", default=None, help="target name (default: the spec's main target)")
    ap.add_argument("--surrogate", choices=["bnn", "gp", "rf"], default="bnn")
    ap.add_argument("--bnn-samples", type=int, default=500)
    ap.add_argument("--n-pool", type=int, default=4096, help="candidate recipes (Sobol, snapped, context fixed)")
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--runs", default=None); ap.add_argument("--runs-keep", default="INCLUDE")
    ap.add_argument("--fix-normalize", action="store_true",
                    help="put alchemy's model in eval mode before scoring, so the input normalisation uses the training bounds")
    ap.add_argument("--seed", type=int, default=0, help="random seed for the BNN sampler (and torch / numpy)")
    ap.add_argument("--figures", action="store_true", help="draw mean / variance / EI projections with alchemy's plotting functions")
    ap.add_argument("--n-grid", type=int, default=12, help="grid points per free knob for the figures")
    ap.add_argument("--out", default="results"); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    from alchemy.active_learning import step_surrogate
    import neurobayes as nb

    if a.spec:
        cfg.configure(cfg.load_spec(a.spec))
    tgt = cfg.target(a.target)
    ds = load_dataset(a.data)
    if a.runs:
        ds = select_runs(ds, parse_run_selection(a.runs, a.runs_keep))
    X, Y = ds.xy([tgt.name])
    out = pathlib.Path(a.out) / (a.name or f"alchemy_{a.surrogate}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"alchemy step_surrogate on {len(X)} runs, target {tgt.name} ({'minimise' if tgt.minimize else 'maximise'}), "
             f"surrogate {a.surrogate}, acquisition qLogExpectedImprovement (discrete, {a.n_pool} candidates)"]

    pool = sobol_pool(a.n_pool, seed=0)                       # grid recipes at the fixed context
    surrogate = {"gp": {"name": "gp"},
                 "rf": {"name": "rf", "init_kwargs": {"n_bootstrap": 10, "n_trees": 100}},
                 "bnn": {"name": "bnn", "init_kwargs": {"architecture": nb.FlaxMLP(hidden_dims=[32, 32], target_dim=1),
                                                        "num_warmup": a.bnn_samples, "num_samples": a.bnn_samples,
                                                        "progress_bar": False,
                                                        # alchemy forwards extra kwargs to NeuroBayes' fit: a fixed
                                                        # random seed makes the NUTS samples, and so the suggestion, repeatable
                                                        "rng_key": __import__("jax").random.PRNGKey(a.seed)}}}[a.surrogate]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    Xt, Yt, Pt = (torch.as_tensor(v, dtype=torch.float64) for v in (X, Y, pool))
    if not a.fix_normalize:
        cand, mean, var, acq = step_surrogate(Xt, Yt, Pt, surrogate=surrogate,
                                              acquisition={"name": "EI", "discrete": True, "kwargs": {"q": a.q}},
                                              return_acq_vals=True, maximize=not tgt.minimize)
    else:
        # the same alchemy surrogate class and the same BoTorch acquisition, but the model is switched to
        # evaluation mode after fitting so Normalize keeps the bounds learned from the training runs
        from alchemy.surrogates import GPSurrogate, RandomForestSurrogate, NeuroBayesBNNSurrogate
        from botorch.acquisition import qLogExpectedImprovement
        from botorch.optim import optimize_acqf_discrete
        Ytr = Yt if not tgt.minimize else -Yt
        cls = {"gp": GPSurrogate, "rf": RandomForestSurrogate, "bnn": NeuroBayesBNNSurrogate}[a.surrogate]
        model = cls(**surrogate.get("init_kwargs", {})).fit(Xt, Ytr)
        model.eval()
        post = model.posterior(Pt)
        mean, var = post.mean.detach(), post.variance.detach()
        if tgt.minimize:
            mean = -mean
        acqf = qLogExpectedImprovement(model, Ytr.max().item())
        acqf.set_X_pending(Xt)
        cand, _ = optimize_acqf_discrete(acq_function=acqf, choices=Pt, q=a.q)
        with torch.no_grad():
            acq = torch.cat([acqf(Pt[i:i + 1000].unsqueeze(-2)).squeeze(-1) for i in range(0, len(Pt), 1000)])
    lines.append(f"fitted and scored in {time.time() - t0:.0f} s" + ("  [model.eval() before posterior]" if a.fix_normalize else "  [alchemy as shipped]"))
    cand = cand.detach().numpy(); mean = mean.numpy().ravel(); std = np.sqrt(var.numpy().ravel())
    acq = acq.detach().numpy().ravel() if acq.numel() else np.full(len(cand), np.nan)

    # posterior at the chosen candidates (same quirky posterior as alchemy returns for the pool)
    idx = [int(np.argmin(np.abs(pool - c).sum(1))) for c in cand]
    sug = add_physical(pd.DataFrame(cand, columns=PARAM_NAMES))
    sug[f"pred_{tgt.name}"] = mean[idx]; sug[f"pred_{tgt.name}_std"] = std[idx]; sug["logEI"] = acq[:len(cand)]
    best = float(Y.min() if tgt.minimize else Y.max())
    lines.append(f"best observed {tgt.name}: {best:.4f} (alchemy uses it as best_f)")
    lines.append("suggested next sample:\n" + sug.round(4).T.to_string())
    top = np.argsort(-acq)[:5] if np.isfinite(acq).all() and len(acq) == len(pool) else []
    if len(top):
        alt = add_physical(pd.DataFrame(pool[top], columns=PARAM_NAMES))
        alt[f"pred_{tgt.name}"] = mean[top]; alt[f"pred_{tgt.name}_std"] = std[top]; alt["logEI"] = acq[top]
        lines.append("top candidates by acquisition over the pool:\n" + alt.round(4).to_string(index=False))
        alt.to_csv(out / "alternatives.csv", index=False)
    lines.append(f"posterior over the pool: mean {mean.min():.3f}..{mean.max():.3f}, std {std.min():.3f}..{std.max():.3f}")
    lines.append("note: alchemy evaluates the posterior with the model in training mode, so BoTorch's Normalize "
                 "re-learns its input bounds from the candidate pool rather than the training runs.")
    sug.to_csv(out / "suggestion.csv", index=False)

    if a.figures:
        # alchemy's own plotting functions on alchemy's fitted model: mean / variance projections and the EI projection
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from alchemy.data_utils import plot_surrogate, plot_gp_mean_projections
        from alchemy_plot_surrogate import tidy
        from aln_bo.config import PARAMS, FREE_IDX, apply_fixed
        if not a.fix_normalize:
            lines.append("figures need the fitted model: rerun with --fix-normalize to draw them")
        else:
            free = [(i, PARAMS[i]) for i in FREE_IDX]
            grids = [np.linspace(p.low, p.high, a.n_grid) for _, p in free]
            mesh = np.meshgrid(*grids, indexing="ij")
            G = np.column_stack([m.ravel() for m in mesh])
            XG = np.repeat(apply_fixed(np.zeros((1, len(PARAMS)))), len(G), axis=0)
            for k, (i, _) in enumerate(free):
                XG[:, i] = G[:, k]
            XGt = torch.as_tensor(XG, dtype=torch.float64)
            ms, vs, es = [], [], []
            with torch.no_grad():
                for s in range(0, len(XGt), 2048):
                    p = model.posterior(XGt[s:s + 2048])
                    ms.append(p.mean.reshape(-1)); vs.append(p.variance.reshape(-1))
                    es.append(acqf(XGt[s:s + 2048].unsqueeze(-2)).reshape(-1))
            gmean = torch.cat(ms); gvar = torch.cat(vs); gei = torch.cat(es)
            if tgt.minimize:
                gmean = -gmean
            names = [f"{p.name} [{p.unit}]" if p.unit else p.name for _, p in free]
            tg = [torch.as_tensor(g) for g in grids]
            Xtr = torch.as_tensor(X[:, FREE_IDX]); Ytr_plot = torch.as_tensor(Y)
            plt.close("all")
            plot_surrogate(gmean, tg, Xtr, Ytr_plot, var=gvar, param_names=names, figsize=(18, 18))
            for fig, kind in zip([plt.figure(n) for n in plt.get_fignums()], ("mean", "variance")):
                tidy(fig, f"alchemy plot_surrogate: {a.surrogate.upper()} {kind} of {tgt.name}, {len(X)} runs "
                          f"(projection over the other free knobs, context fixed)", wspace=0.75, hspace=0.15)
                fig.savefig(out / f"alchemy_plot_surrogate_{a.surrogate.upper()}_{kind}.png", dpi=120, bbox_inches="tight")
            plt.close("all")
            plot_gp_mean_projections(gei, tg, Xtr, Ytr_plot, param_names=names, figsize=(15, 14))
            fig = plt.figure(plt.get_fignums()[-1])
            # alchemy's function has no marker option: add the suggested run as a red star on every pair panel
            import itertools
            panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
            for ax, (i, j) in zip(panels, itertools.combinations(range(len(free)), 2)):
                ax.scatter(cand[:, free[i][0]], cand[:, free[j][0]], marker="*", s=260, c="red", edgecolor="k", zorder=6)
            tidy(fig, f"alchemy plot_gp_mean_projections of the acquisition: qLogEI of {tgt.name} from the "
                      f"{a.surrogate.upper()} (larger = more worth trying), {len(X)} runs; red star = suggested next run",
                 wspace=0.5, hspace=0.45)
            fig.savefig(out / f"alchemy_projection_logEI_{a.surrogate.upper()}.png", dpi=120, bbox_inches="tight")
            lines.append(f"figures: alchemy_plot_surrogate_{a.surrogate.upper()}_mean/variance.png, "
                         f"alchemy_projection_logEI_{a.surrogate.upper()}.png ({a.n_grid}^{len(free)} grid)")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print("wrote", out)


if __name__ == "__main__":
    main()
