"""Acquisition functions: turning the GP into a decision.

Expected Improvement (EI) at a recipe x, for a target we MINIMISE:
    EI(x) = E[ max(f_best - f(x), 0) ]
          = (f_best - mu) * Phi(z) + sigma * phi(z),   z = (f_best - mu) / sigma
It is large where the GP predicts a better value than anything seen
(exploitation) OR where sigma is large (exploration). Nothing to tune.

Context parameters (config.FIXED) are never moved: the candidate pool, the
local polish and the final snap all hold them at their fixed values.

Optimisation of the acquisition:
    1. score a large quasi-random pool of recipes (Sobol) snapped to the
       instrument grid,
    2. polish the best few with L-BFGS-B (continuous), re-snap,
    3. for a batch of q > 1: kriging believer - commit the winner with its
       predicted outcome, refit (frozen hyperparameters), repeat.
"""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from .config import PARAMS, bounds_array, normalize, denormalize, snap_to_grid, apply_fixed


# ----------------------------------------------------------------------------- EI
def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float, minimize: bool = True) -> np.ndarray:
    imp = (best - mean) if minimize else (mean - best)
    std = np.maximum(std, 1e-12)
    z = imp / std
    return imp * norm.cdf(z) + std * norm.pdf(z)


def confidence_bound(mean: np.ndarray, std: np.ndarray, beta: float = 1.0, minimize: bool = True) -> np.ndarray:
    """Lower (minimise) / upper (maximise) confidence bound, returned as a score to MAXIMISE.
    beta ~ 0.5-1 exploits; beta ~ 2-3 explores. Less exploratory than EI when only one good run exists."""
    return -(mean - beta * std) if minimize else (mean + beta * std)


def make_acq(name: str, best: float, minimize: bool = True, beta: float = 1.0):
    """Return f(mean, std) -> score for 'ei' or 'lcb'/'ucb'."""
    name = name.lower()
    if name == "ei":
        return lambda m, s: expected_improvement(m, s, best, minimize)
    if name in ("lcb", "ucb", "cb"):
        return lambda m, s: confidence_bound(m, s, beta, minimize)
    raise ValueError(f"unknown acquisition {name!r}; use 'ei' or 'lcb'")


# --------------------------------------------------------------------------- EHVI
def is_non_dominated(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows for a MAXIMISATION frame (all columns up)."""
    Y = np.asarray(Y, float)
    n = len(Y)
    mask = np.ones(n, bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by_i = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)
        mask[dominated_by_i] = False
    return mask


def hv_improvement_2d(y: np.ndarray, front: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Hypervolume gained by adding point(s) y to a 2-D front (maximisation frame)."""
    y = np.asarray(y, float)
    front = np.asarray(front, float)
    front = front[np.argsort(front[:, 0])]
    y1 = y[..., 0:1]; y2 = y[..., 1:2]
    q1 = np.clip(np.minimum(front[:, 0], y1) - ref[0], 0, None)
    q2 = np.clip(np.minimum(front[:, 1], y2) - ref[1], 0, None)
    widths = np.diff(np.concatenate([np.zeros_like(q1[..., :1]), q1], axis=-1), axis=-1)
    union = np.sum(widths * q2, axis=-1)
    box = np.clip(y1[..., 0] - ref[0], 0, None) * np.clip(y2[..., 0] - ref[1], 0, None)
    return np.clip(box - union, 0, None)


def ehvi_mc(mean, std, front, ref, n_samples: int = 128, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng(0)
    eps = rng.standard_normal((n_samples, *mean.shape))
    return hv_improvement_2d(mean[None] + std[None] * eps, front, ref).mean(axis=0)


# --------------------------------------------------------------- optimiser utils
def sobol_pool(n: int = 4096, seed: int = 0) -> np.ndarray:
    """Quasi-random candidate recipes over the FREE parameters, context held fixed, snapped to the grid."""
    U = qmc.Sobol(d=len(PARAMS), scramble=True, seed=seed).random(n)
    return np.unique(snap_to_grid(denormalize(U)), axis=0)


def _polish(acq: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> np.ndarray:
    """Local refinement in normalised space (L-BFGS-B, finite differences); context dims pinned."""
    u0 = normalize(x0[None])[0]
    lo, hi = normalize(bounds_array())            # context dims: lo == hi
    res = minimize(lambda u: -float(acq(denormalize(u[None]))[0]), u0,
                   method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": 60})
    return denormalize(res.x[None])[0]


def _is_in(x: np.ndarray, X: Optional[np.ndarray]) -> bool:
    return X is not None and len(X) > 0 and bool(np.any(np.all(np.isclose(X, x), axis=1)))


def maximize_acq(acq: Callable[[np.ndarray], np.ndarray], candidates: Optional[np.ndarray] = None,
                 n_pool: int = 4096, n_polish: int = 8, snap: bool = True, seed: int = 0,
                 exclude: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """Return (x_best, acq_best). With `candidates`, choose only among those recipes.
    `exclude`: recipes that must not be returned (already measured or already suggested)."""
    pool = apply_fixed(candidates) if candidates is not None else sobol_pool(n_pool, seed)
    if exclude is not None and len(exclude):
        pool = pool[~np.array([_is_in(x, exclude) for x in pool])]
    vals = acq(pool)
    order = np.argsort(vals)[::-1]
    if candidates is not None:
        i = int(order[0])
        return pool[i], float(vals[i])
    best_x, best_v = pool[order[0]], float(vals[order[0]])
    for i in order[:n_polish]:
        x = _polish(acq, pool[i])
        x = snap_to_grid(x[None])[0] if snap else apply_fixed(x[None])[0]
        if _is_in(x, exclude):
            continue
        v = float(acq(x[None])[0])
        if v > best_v:
            best_x, best_v = x, v
    return best_x, best_v


# ------------------------------------------------------------------ public API
def incumbent(gp, minimize: bool = True) -> float:
    """The value EI must beat, defined INSIDE the fixed context.

    With context parameters (e.g. growth time) the raw best observation may come
    from a run at a different context (a thicker film) that no recipe at the fixed
    context can match; EI would then be ~0 everywhere. So the incumbent is the
    GP posterior mean at the observed recipes with their context columns reset
    to the fixed values (the "plug-in" incumbent used in noisy BO)."""
    Xc = apply_fixed(gp.X)
    m, _ = gp.predict(Xc)
    return float(m.min() if minimize else m.max())


def suggest_ei(gp, q: int = 3, *, minimize: bool = True, candidates: Optional[np.ndarray] = None,
               snap: bool = True, seed: int = 0, best: Optional[float] = None,
               acq_name: str = "ei", beta: float = 1.0, pool_transform=None):
    """Propose q experiments by EI (or LCB) on a single-output GP (kriging believer for q > 1).

    pool_transform(X) -> X: applied to every candidate before scoring (used to pin
    knobs the GP finds irrelevant, so the search runs over the knobs that matter)."""
    X_out, v_out = [], []
    g = gp
    best = incumbent(gp, minimize) if best is None else float(best)
    exclude = gp.X.copy()
    tf = pool_transform or (lambda X: X)

    def acq(X):
        X = tf(np.atleast_2d(X))
        m, s = g.predict(X)
        return make_acq(acq_name, best, minimize, beta)(m, s)

    for k in range(q):
        x, v = maximize_acq(acq, candidates, snap=snap, seed=seed + k, exclude=exclude)
        x = tf(x[None])[0]
        X_out.append(x); v_out.append(v)
        exclude = np.vstack([exclude, x])
        if k < q - 1:                                   # kriging believer for the rest of the batch (GP only)
            believed = float(g.predict(x[None])[0][0])
            g = g.fantasize(x, believed)
            best = min(best, believed) if minimize else max(best, believed)
        if candidates is not None:
            candidates = candidates[~np.all(np.isclose(candidates, x), axis=1)]
    return np.array(X_out), np.array(v_out)


def top_pool_candidates(gp, n: int = 5, *, minimize: bool = True, n_pool: int = 4096, seed: int = 1,
                        exclude: Optional[np.ndarray] = None, min_separation: float = 0.15,
                        best: Optional[float] = None, acq_name: str = "ei", beta: float = 1.0,
                        pool_transform=None):
    """The n best DISTINCT grid recipes by the acquisition from a Sobol pool (alternatives for the LLM).
    Two recipes count as distinct when they differ by > min_separation (normalised) in some dimension."""
    pool = sobol_pool(n_pool, seed)
    if pool_transform is not None:
        pool = np.unique(pool_transform(pool), axis=0)
    if exclude is not None and len(exclude):
        pool = pool[~np.array([_is_in(x, exclude) for x in pool])]
    m, s = gp.predict(pool)
    best = incumbent(gp, minimize) if best is None else float(best)
    ei = make_acq(acq_name, best, minimize, beta)(m, s)
    order = np.argsort(ei)[::-1]
    chosen, idx = [], []
    for i in order:
        u = normalize(pool[i][None])[0]
        if all(np.max(np.abs(u - normalize(pool[j][None])[0])) > min_separation for j in idx):
            idx.append(i); chosen.append(pool[i])
        if len(idx) == n:
            break
    return np.array(chosen), ei[idx], m[idx], s[idx]


def suggest_ehvi(gplist, q: int = 3, *, maximize_mask=(True, False), ref_point=None,
                 candidates=None, snap: bool = True, seed: int = 0, n_samples: int = 128):
    """Propose q experiments by Monte-Carlo EHVI on independent GPs (kept for multi-objective use)."""
    w = np.array([1.0 if m else -1.0 for m in maximize_mask])
    Y = np.column_stack([g.Y[:, 0] for g in gplist.models]) * w
    if ref_point is None:
        span = Y.max(0) - Y.min(0)
        ref = Y.min(0) - 0.1 * np.maximum(span, 1e-9)
    else:
        ref = np.asarray(ref_point, float) * w
    front = Y[is_non_dominated(Y)]
    X_out, v_out = [], []
    g = gplist
    exclude = gplist.models[0].X.copy()

    def acq(X):
        m, s = g.predict(X)
        return ehvi_mc(m * w, s, front, ref, n_samples, np.random.default_rng(seed))

    for k in range(q):
        x, v = maximize_acq(acq, candidates, n_pool=2048, n_polish=4, snap=snap, seed=seed + k, exclude=exclude)
        X_out.append(x); v_out.append(v)
        m_new = g.predict(x[None])[0]
        g = g.fantasize(x, m_new)
        all_y = np.vstack([front, m_new * w])
        front = all_y[is_non_dominated(all_y)]
        exclude = np.vstack([exclude, x])
        if candidates is not None:
            candidates = candidates[~np.all(np.isclose(candidates, x), axis=1)]
    return np.array(X_out), np.array(v_out)


def scalarize(Y: np.ndarray, weights) -> np.ndarray:
    """Weighted z-score combination of several targets into one number to MINIMISE."""
    Y = np.asarray(Y, float)
    z = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    return (z @ np.asarray(weights, float))[:, None]
