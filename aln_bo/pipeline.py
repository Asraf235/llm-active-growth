"""One BO step on a Dataset: surrogate + acquisition -> suggestion, auxiliary GPs, figures, LLM packet / review.

This is the body of the step made callable, so the CLI, the scripts and an agent all run the same code.
Everything is written to `out`; the return value carries the objects.

Figures (alchemy's projection layouts, see alchemy_plots.py):
    <LABEL>_mean.png, <LABEL>_variance.png      the driving surrogate (BNN by default; GP or RF with surrogate=...)
    acquisition_<ACQ>_<LABEL>.png               its acquisition with the suggested next run as a red star (the only starred figure)
    <OTHER>_mean.png, <OTHER>_variance.png      the other two of GP / RF / BNN fitted to the same runs, as cross-checks
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd

from . import config as cfg
from .config import PARAMS, PARAM_NAMES, FIXED
from .campaign import MainCampaign, AuxModels
from .backends import resolve_backend_name, available_backends, get_backend
from . import alchemy_plots, llm_review

SURROGATE_LABEL = {"gp": "GP", "bnn": "BNN", "rf": "RF"}


def run_step(ds, out: str | Path, *, target: Optional[str] = None, q: int = 1, n_alternatives: int = 5,
             backend: str = "auto", acq: str = "ei", beta: float = 1.0, kernel: str = "rbf", surrogate: str = "bnn",
             n_grid: int = 12, cross_check: bool = True,
             llm: bool = False, model: Optional[str] = None, provider: str = "anthropic", label: str = "",
             log: Optional[Callable[[str], None]] = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s=""):
        lines.append(str(s))
        (log or print)(s)

    rep = ds.report or {}
    say(f"{label}usable runs: {rep.get('usable_runs', len(ds.df))}   rows per target: {rep.get('rows_per_target')}")
    if rep.get("dropped"):
        by_reason: dict = {}
        for r, why in rep["dropped"]:
            by_reason.setdefault(why, []).append(r)
        for why, runs in by_reason.items():
            named = [f"{r:g}" for r in runs if r is not None]
            say(f"dropped {len(runs):2d} rows - {why}" + (f": runs {', '.join(named)}" if named else ""))
    say(f"backend: {resolve_backend_name(backend)}  (available: {available_backends()});  kernel: {kernel};  surrogate: {surrogate}")
    ds.df.to_csv(out / "cleaned_runs.csv", index=False)

    # ---- main surrogate + acquisition
    tgt = cfg.target(target) if target else cfg.MAIN
    main_c = MainCampaign(ds, target=tgt, backend=backend, acq=acq, beta=beta, kernel=kernel, surrogate=surrogate).fit()
    sug = main_c.suggest(q=q)
    alt = main_c.alternatives(n=n_alternatives)
    say(f"\n=== main surrogate {main_c.backend_name}: {tgt.name} ({'minimise' if tgt.minimize else 'maximise'}) on {len(main_c.X)} runs ===")
    bic = main_c.best_in_context()
    say(f"best run at the fixed context ({', '.join(f'{PARAMS[i].name}={v:g}' for i, v in FIXED.items())}):\n"
        + (bic.to_string() if bic is not None else "  (none)"))
    say("best run over any context:\n" + main_c.best_so_far().to_string())
    say(f"EI incumbent (best surrogate mean at the fixed context): {main_c.incumbent():.4f}")
    say(f"learned noise std: {main_c.noise():.4f}" if main_c.noise() is not None else "learned noise std: n/a (not a GP)")
    say(f"importance ({'ARD lengthscales, small = influential' if surrogate == 'gp' else 'surrogate importance'}): {main_c.importance() or 'n/a'}")
    say(f"irrelevant knobs pinned to the best in-context run: {main_c.irrelevant()}")

    # ---- auxiliary GPs
    aux = AuxModels(ds, backend=backend, kernel=kernel).fit()
    say("\n=== auxiliary GPs (prediction only) ===")
    for t in cfg.AUX:
        say(f"{t.name}: {aux.n_rows.get(t.name)} runs" + (f", lengthscales {aux.importance()[t.name]}" if t.name in aux.models else "  (too few runs, skipped)"))
    aux_sug = aux.predict(sug[PARAM_NAMES].to_numpy())
    aux_alt = aux.predict(alt[PARAM_NAMES].to_numpy())
    sug_full = pd.concat([sug, aux_sug], axis=1)
    alt_full = pd.concat([alt, aux_alt], axis=1)
    say("\nSUGGESTED NEXT SAMPLE (acquisition maximum):\n" + sug_full.round(4).T.to_string())
    say("\nnext-best alternatives:\n" + alt_full.round(4).to_string(index=False))
    sug_full.to_csv(out / "suggestion.csv", index=False)
    alt_full.to_csv(out / "alternatives.csv", index=False)

    # ---- figures: alchemy's projection layouts
    Xs = sug[PARAM_NAMES].to_numpy()
    grids, XG = alchemy_plots.full_grid(n_grid)
    names = alchemy_plots.free_names()
    lab = SURROGATE_LABEL.get(surrogate, surrogate.upper())
    mean = alchemy_plots.evaluate(lambda X: main_c.predict(X)[0], XG)
    std = alchemy_plots.evaluate(lambda X: main_c.predict(X)[1], XG)
    acqv = alchemy_plots.evaluate(main_c.acquisition, XG)
    alchemy_plots.surrogate_figures(mean, std ** 2, grids, main_c.X, main_c.Y, names, lab, out, target=tgt.name)
    acq_label = f"{acq.upper()} ({lab})"
    acq_path = alchemy_plots.acquisition_figure(acqv, grids, main_c.X, main_c.Y, names, acq_label,
                                                out / f"acquisition_{acq.upper()}_{lab}.png", marks=Xs, target=tgt.name)
    say(f"\nfigures ({len(grids[0])}^{len(grids)} grid): {lab}_mean.png, {lab}_variance.png, acquisition_{acq.upper()}_{lab}.png")
    checks = {}
    if cross_check:
        from . import validation as V
        makers = {"GP": lambda: get_backend(backend).GP(kernel=kernel), "RF": V.RandomForest, "BNN": V.BNN}
        for other, make in makers.items():
            if other == lab:
                continue
            if other == "BNN" and not V.bnn_available():
                say("cross-check BNN skipped: JAX + NeuroBayes not installed (pip install 'aln-bo[bnn]')")
                continue
            m = make().fit(main_c.X, main_c.Y)
            mo = alchemy_plots.evaluate(lambda X, m=m: m.predict(X)[0], XG)
            so = alchemy_plots.evaluate(lambda X, m=m: m.predict(X)[1], XG)
            alchemy_plots.surrogate_figures(mo, so ** 2, grids, main_c.X, main_c.Y, names, other, out, target=tgt.name)
            ms, ss = m.predict(Xs)
            checks[other] = {"mean": float(ms[0]), "std": float(ss[0])}
            say(f"cross-check {other}: at the suggestion {ms[0]:.4f} +/- {ss[0]:.4f}  (figures {other}_mean.png, {other}_variance.png)")

    # ---- LLM packet / review
    packet = llm_review.build_packet(main_c, aux, sug, alt, ds, aux_sug, aux_alt)
    packet["cross_check_at_suggestion"] = checks
    (out / "packet.json").write_text(json.dumps(packet, indent=1, default=float), encoding="utf-8")
    llm_review.write_prompt_file(packet, out / "llm_prompt.md")
    review = None
    if llm:
        try:
            review = llm_review.review(packet, acq_path, model=model, provider=provider)
            (out / "llm_review.json").write_text(json.dumps(review, indent=1), encoding="utf-8")
            md = llm_review.review_to_markdown(review, packet)
            (out / "llm_review.md").write_text(md, encoding="utf-8")
            say("\n=== LLM review ===\n" + md)
        except Exception as e:                       # no credentials, network, refusal ...
            say(f"\nLLM review skipped: {type(e).__name__}: {e}\n"
                f"-> paste {out / 'llm_prompt.md'} into a chat model, or set the provider API key and rerun with --llm")
    else:
        say(f"\n(LLM review not requested; prompt written to {out / 'llm_prompt.md'})")

    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"\nwrote results to {out}")
    return {"campaign": main_c, "aux": aux, "suggestion": sug_full, "alternatives": alt_full,
            "packet": packet, "review": review, "out": out, "target": tgt, "cross_check": checks}
