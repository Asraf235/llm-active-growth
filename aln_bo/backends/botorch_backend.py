"""BoTorch backend - the same stack as sumner-harris/alchemy.

Surrogate : SingleTaskGP (RBF ARD as in the paper / alchemy; Matern-5/2 optional) with Normalize / Standardize,
            hyperparameters by marginal likelihood (fit_gpytorch_mll).
            Several targets -> ModelListGP of independent SingleTaskGPs.
Acquisition: qLogExpectedImprovement (single objective) and qLogExpectedHypervolumeImprovement
            (two objectives, kept for multi-objective use), optimised with optimize_acqf (multi-start L-BFGS, joint q-batch),
            then snapped to the instrument grid.

Numpy in, numpy out, so campaign.py is identical for both backends.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
import warnings
import numpy as np
import torch
warnings.filterwarnings("ignore", message=".*fused qLogEHVI C\\+\\+ extension.*")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*deprecated.*")
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from ..config import PARAM_NAMES, data_bounds_array, snap_to_grid, apply_fixed, FIXED
from ..acquisition import sobol_pool

NAME = "botorch"
DT = torch.float64

# GP matrices here are tiny (tens of rows). Letting torch fan them out over all
# cores makes every step 10-50x SLOWER through thread thrashing, so cap it unless
# the user set OMP_NUM_THREADS themselves.
if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.array(a, dtype=float), dtype=DT)


def _make_gp(X: torch.Tensor, Y: torch.Tensor, kernel: str = "rbf") -> SingleTaskGP:
    d = X.shape[-1]
    if kernel == "rbf":
        # exactly alchemy.gp_models.create_single_task_gp (the paper's GP): ScaleKernel(RBF, ARD),
        # ConstantMean, Normalize / Standardize, no extra hyper-priors
        covar = ScaleKernel(RBFKernel(ard_num_dims=d))
    elif kernel == "matern":
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=GammaPrior(3.0, 6.0)),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
    else:
        raise ValueError(f"kernel must be 'rbf' or 'matern', got {kernel!r}")
    return SingleTaskGP(X, Y, covar_module=covar, mean_module=ConstantMean(),
                        input_transform=Normalize(d=d), outcome_transform=Standardize(m=1))


def _lengthscales(gp: SingleTaskGP) -> dict:
    ls = gp.covar_module.base_kernel.lengthscale.detach().flatten().numpy()
    return dict(zip(PARAM_NAMES, np.round(ls, 3).tolist()))


@dataclass
class GP:
    """Single-output GP; .X (N x d) and .Y (N x 1) are kept as numpy for the acquisition code."""
    kernel: str = "rbf"
    model: Optional[SingleTaskGP] = field(default=None, init=False)
    X: Optional[np.ndarray] = field(default=None, init=False)
    Y: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, X, Y) -> "GP":
        self.X = np.asarray(X, float)
        self.Y = np.asarray(Y, float).reshape(len(self.X), -1)
        if self.Y.shape[1] != 1:
            raise ValueError("GP is single-output; use GPList for several targets")
        self.model = _make_gp(_t(self.X), _t(self.Y), self.kernel)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood, self.model))
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        post = self.model.posterior(_t(np.atleast_2d(X)))
        return post.mean[:, 0].numpy(), post.variance.sqrt()[:, 0].numpy()

    def lengthscales(self) -> dict:
        return _lengthscales(self.model)

    def noise(self) -> float:
        std_y = float(self.model.outcome_transform.stdvs.flatten()[0])
        return float(self.model.likelihood.noise.sqrt().item()) * std_y


@dataclass
class GPList:
    """One independent SingleTaskGP per column of Y, bundled as a ModelListGP."""
    models: list = field(default_factory=list)
    kernel: str = "rbf"
    model: Optional[ModelListGP] = field(default=None, init=False)

    def fit(self, X, Y) -> "GPList":
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        self.models = []
        gps = []
        for i in range(Y.shape[1]):
            g = GP(kernel=self.kernel)
            g.X, g.Y = X, Y[:, i:i + 1]
            g.model = _make_gp(_t(X), _t(g.Y), self.kernel)
            self.models.append(g)
            gps.append(g.model)
        self.model = ModelListGP(*gps)
        fit_gpytorch_mll(SumMarginalLogLikelihood(self.model.likelihood, self.model))
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        post = self.model.posterior(_t(np.atleast_2d(X)))
        return post.mean.numpy(), post.variance.sqrt().numpy()

    def lengthscales(self) -> list[dict]:
        return [g.lengthscales() for g in self.models]


# ----------------------------------------------------------------- acquisition
def _optimize(acqf, q: int, candidates, num_restarts: int, raw_samples: int) -> torch.Tensor:
    if candidates is not None:
        X, _ = optimize_acqf_discrete(acqf, q=q, choices=_t(apply_fixed(candidates)))
    else:
        # context parameters are pinned through fixed_features; bounds span the data range
        X, _ = optimize_acqf(acqf, bounds=_t(data_bounds_array()), q=q, num_restarts=num_restarts,
                             raw_samples=raw_samples, sequential=q > 1,
                             fixed_features={int(i): float(v) for i, v in FIXED.items()} or None)
    return X.detach()


def _per_point(acqf, X: np.ndarray) -> np.ndarray:
    """Acquisition value of each suggested recipe on its own (for reporting)."""
    with torch.no_grad():
        return np.array([float(acqf(_t(x)[None, None, :])) for x in X])


def _dedupe(acqf, X: np.ndarray, exclude: np.ndarray, snap: bool) -> np.ndarray:
    """After snapping, replace rows that duplicate each other or a measured recipe
    by the best grid recipes from a Sobol pool (discrete search)."""
    if not snap:
        return X
    keep, seen = [], [tuple(r) for r in np.round(exclude, 9)]
    for r in X:
        key = tuple(np.round(r, 9))
        if key in seen:
            keep.append(False)
        else:
            keep.append(True); seen.append(key)
    keep = np.array(keep)
    n_missing = int((~keep).sum())
    if n_missing == 0:
        return X
    pool = sobol_pool(4096)
    pool = np.array([p for p in pool if tuple(np.round(p, 9)) not in seen])
    repl, _ = optimize_acqf_discrete(acqf, q=n_missing, choices=_t(pool))
    X = X.copy()
    X[~keep] = repl.detach().numpy()
    return X


def suggest_ei(gp: GP, q: int = 3, *, minimize: bool = True, candidates=None, snap: bool = True,
               seed: int = 0, num_restarts: int = 10, raw_samples: int = 512, best=None):
    from ..acquisition import incumbent
    torch.manual_seed(seed)
    sign = -1.0 if minimize else 1.0
    objective = GenericMCObjective(lambda samples, X=None: sign * samples[..., 0])
    best_f = float(sign * (incumbent(gp, minimize) if best is None else best))
    acqf = qLogExpectedImprovement(model=gp.model, best_f=best_f, objective=objective)
    X = _optimize(acqf, q, candidates, num_restarts, raw_samples).numpy()
    X = snap_to_grid(X) if snap else X
    X = _dedupe(acqf, X, gp.X, snap)
    return X, np.exp(_per_point(acqf, X))            # EI (not log) for readability


def suggest_ehvi(gpl: GPList, q: int = 3, *, maximize_mask=(True, False), ref_point=None,
                 candidates=None, snap: bool = True, seed: int = 0, num_restarts: int = 5, raw_samples: int = 128):
    torch.manual_seed(seed)
    w = torch.tensor([1.0 if m else -1.0 for m in maximize_mask], dtype=DT)
    Y = _t(np.column_stack([g.Y[:, 0] for g in gpl.models])) * w     # maximisation frame
    if ref_point is None:
        span = Y.max(0).values - Y.min(0).values
        ref = Y.min(0).values - 0.1 * span.clamp_min(1e-9)
    else:
        ref = _t(ref_point) * w
    front = Y[is_non_dominated(Y)]
    acqf = qLogExpectedHypervolumeImprovement(
        model=gpl.model, ref_point=ref.tolist(),
        partitioning=FastNondominatedPartitioning(ref_point=ref, Y=front),
        objective=WeightedMCMultiOutputObjective(w),
    )
    X = _optimize(acqf, q, candidates, num_restarts, raw_samples).numpy()
    X = snap_to_grid(X) if snap else X
    X = _dedupe(acqf, X, gpl.models[0].X, snap)
    return X, np.exp(_per_point(acqf, X))            # EHVI (not log)
