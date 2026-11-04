"""The main BO loop and the auxiliary (prediction-only) surrogates.

MainCampaign  : GP on the objective + Expected Improvement -> the suggested next sample.
                The objective is the spec's main target by default (the rocking-curve FWHM in the
                AlN example) or any target passed as `target` (e.g. the 2theta FWHM).
                One .suggest() call is one active-learning step:
                runs -> X, Y -> fit GP -> acquisition -> recipe.

AuxModels     : one GP per auxiliary target (in the AlN example: SiOxNy thickness and
                roughness, AlN density). They never drive the acquisition; they predict what
                those properties would be at any recipe, so the LLM / the user can judge the
                main suggestion with more than the objective in view.

Two details that matter with context parameters (growth time etc.):
  * the EI incumbent is evaluated INSIDE the fixed context (acquisition.incumbent),
    otherwise a run at another context that nobody can match (a thicker film) kills EI everywhere;
  * knobs the GP finds irrelevant (ARD lengthscale at the upper bound) are pinned
    to the best in-context run instead of being set to arbitrary values by the
    optimiser - "do not move what does not matter".

Nothing is cached between steps on purpose: the campaign database (or the run file) is the state.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from . import config as cfg
from .config import PARAMS, PARAM_NAMES, FREE_NAMES, FREE_IDX, FIXED, AUX, PHYSICAL, Target, apply_fixed, in_context_mask, add_physical
from .data import Dataset
from .acquisition import expected_improvement, make_acq, top_pool_candidates, incumbent
from .backends import get_backend, resolve_backend_name

IRRELEVANT_LENGTHSCALE = 5.0      # sklearn backend bound is 10; BoTorch priors keep values well below this


def _to_df(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.atleast_2d(np.asarray(X, float)), columns=PARAM_NAMES)


@dataclass
class MainCampaign:
    dataset: Dataset
    target: Optional[Target] = None          # default: the spec's main target
    backend: Optional[str] = None            # "sklearn", "botorch" or None -> auto / $ALN_BO_BACKEND
    pin_irrelevant: bool = True
    acq: str = "ei"                          # "ei" (default) or "lcb" (sklearn backend; beta below)
    beta: float = 1.0
    kernel: str = "rbf"                      # "rbf" (paper / alchemy) or "matern"
    surrogate: str = "gp"                    # "gp" (backend GP) | "bnn" | "rf": the model that drives EI
    model: Optional[object] = field(default=None, init=False)
    X: Optional[np.ndarray] = field(default=None, init=False)
    Y: Optional[np.ndarray] = field(default=None, init=False)
    rows: Optional[pd.DataFrame] = field(default=None, init=False)

    @property
    def backend_name(self) -> str:
        return resolve_backend_name(self.backend) if self.surrogate == "gp" else f"{self.surrogate} + numpy EI"

    def fit(self):
        if self.target is None:
            self.target = cfg.MAIN
        self.rows = self.dataset.rows_for([self.target.name]).reset_index(drop=True)
        self.X, self.Y = self.dataset.xy([self.target.name])
        if len(self.X) < 3:
            raise ValueError(f"Need at least 3 runs with {self.target.name}; have {len(self.X)}")
        if self.surrogate == "gp":
            self.be = get_backend(self.backend)
            self.model = self.be.GP(kernel=self.kernel).fit(self.X, self.Y)
        else:                                  # a validation surrogate drives the acquisition (numpy EI path)
            from . import validation as V
            cls = {"bnn": V.BNN, "rf": V.RandomForest}
            if self.surrogate == "bnn" and not V.bnn_available():
                raise RuntimeError("the BNN needs JAX + NeuroBayes: pip install 'aln-bo[bnn]' (on a Windows machine with "
                                   "Smart App Control, run inside WSL), or pass surrogate='gp'")
            if self.surrogate not in cls:
                raise ValueError(f"surrogate must be one of gp, {', '.join(cls)}; got {self.surrogate!r}")
            self.be = get_backend("sklearn")
            self.model = cls[self.surrogate]().fit(self.X, self.Y)
        return self

    # ---- predictions / acquisition on arbitrary recipes
    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        return self.model.predict(np.atleast_2d(X))

    def incumbent(self) -> float:
        """Value EI must beat: best GP mean over observed recipes moved to the fixed context."""
        return incumbent(self.model, self.target.minimize)

    def acquisition(self, X) -> np.ndarray:
        """Acquisition score at recipes X (EI by default; closed form)."""
        m, s = self.predict(X)
        return make_acq(self.acq, self.incumbent(), self.target.minimize, self.beta)(m, s)

    # ---- the decision
    def suggest(self, q: int = 1, candidates: Optional[np.ndarray] = None, snap: bool = True) -> pd.DataFrame:
        if self.model is None:
            self.fit()
        extra = {} if self.backend_name == "botorch" and self.surrogate == "gp" else {"acq_name": self.acq, "beta": self.beta,
                                                            "pool_transform": self._pin}
        Xnew, _ = self.be.suggest_ei(self.model, q=q, minimize=self.target.minimize,
                                     candidates=candidates, snap=snap, best=self.incumbent(), **extra)
        Xnew = self._fill_duplicates(self._pin(np.atleast_2d(Xnew)))
        return self._table(Xnew)

    def _fill_duplicates(self, X: np.ndarray) -> np.ndarray:
        """Pinning can make two batch members identical (BoTorch path pins after its own
        de-duplication); replace repeats by the next-best distinct pinned grid recipes."""
        keep, seen = [], set()
        for r in np.round(X, 9):
            key = tuple(r)
            keep.append(key not in seen)
            seen.add(key)
        keep = np.array(keep, bool)
        n_missing = int((~keep).sum())
        if n_missing == 0:
            return X
        Xalt, *_ = top_pool_candidates(self.model, n=n_missing, minimize=self.target.minimize,
                                       exclude=np.vstack([self.X, X[keep]]), best=self.incumbent(),
                                       acq_name=self.acq, beta=self.beta, pool_transform=self._pin)
        X = np.array(X, float, copy=True)
        X[~keep] = Xalt[:n_missing]
        return X

    def alternatives(self, n: int = 5) -> pd.DataFrame:
        """Top-n distinct grid recipes by EI (for the LLM to compare against the suggestion)."""
        Xalt, *_ = top_pool_candidates(self.model, n=n, minimize=self.target.minimize, exclude=self.X,
                                       best=self.incumbent(), acq_name=self.acq, beta=self.beta,
                                       pool_transform=self._pin)
        return self._table(Xalt)

    def irrelevant(self) -> list[str]:
        ls = self.importance()
        return [n for n in FREE_NAMES if ls.get(n, 0) >= IRRELEVANT_LENGTHSCALE]

    def _pin(self, X: np.ndarray) -> np.ndarray:
        """Set knobs the GP calls irrelevant to the best in-context run's values."""
        if not self.pin_irrelevant:
            return X
        ref = self.best_in_context()
        X = np.array(X, float, copy=True)
        for name in self.irrelevant():
            if ref is not None and name in ref:
                X[:, PARAM_NAMES.index(name)] = float(ref[name])
        return X

    def _table(self, X) -> pd.DataFrame:
        X = apply_fixed(np.atleast_2d(X))
        m, s = self.predict(X)
        out = add_physical(_to_df(X))                 # + OxyAr / OxyN2 in sccm for the recipe sheet
        out[f"pred_{self.target.name}"] = m
        out[f"pred_{self.target.name}_std"] = s
        out[self.acq.upper()] = self.acquisition(X)
        return out

    # ---- bookkeeping
    def _row(self, i: int) -> pd.Series:
        cols = (["run"] if "run" in self.rows else []) + PARAM_NAMES + [c for c in PHYSICAL if c in self.rows] + [self.target.name]
        return self.rows.iloc[i][cols]

    def best_so_far(self) -> pd.Series:
        """Best observed value over ALL runs (any context)."""
        i = int(np.argmin(self.Y[:, 0]) if self.target.minimize else np.argmax(self.Y[:, 0]))
        return self._row(i)

    def in_context_mask(self) -> np.ndarray:
        """Runs whose context parameters equal the fixed values (within one instrument step)."""
        return in_context_mask(self.X)

    def best_in_context(self) -> Optional[pd.Series]:
        """Best observed run at the standard (fixed) context, e.g. the 13-min growth runs."""
        mask = self.in_context_mask()
        if not mask.any():
            return None
        y = np.where(mask, self.Y[:, 0], np.inf if self.target.minimize else -np.inf)
        i = int(np.argmin(y) if self.target.minimize else np.argmax(y))
        return self._row(i)

    def importance(self) -> dict:
        """GP: ARD lengthscales (SMALL = influential). RF: impurity importance (LARGE = influential).
        BNN: none ({}), so no knob is pinned."""
        if hasattr(self.model, "lengthscales"):
            return self.model.lengthscales()
        if hasattr(self.model, "feature_importance"):
            return self.model.feature_importance()
        return {}

    def noise(self) -> Optional[float]:
        return self.model.noise() if hasattr(self.model, "noise") else None


@dataclass
class AuxModels:
    dataset: Dataset
    targets: list = field(default_factory=lambda: list(AUX))
    backend: Optional[str] = None
    kernel: str = "rbf"
    models: dict = field(default_factory=dict, init=False)
    n_rows: dict = field(default_factory=dict, init=False)
    data: dict = field(default_factory=dict, init=False)      # name -> (X, Y)

    def fit(self):
        be = get_backend(self.backend)
        for t in self.targets:
            X, Y = self.dataset.xy([t.name])
            self.n_rows[t.name] = len(X)
            if len(X) < 3:
                continue
            self.models[t.name] = be.GP(kernel=self.kernel).fit(X, Y)
            self.data[t.name] = (X, Y)
        return self

    def predict(self, X) -> pd.DataFrame:
        X = np.atleast_2d(np.asarray(X, float))
        out = pd.DataFrame(index=range(len(X)))
        for name, g in self.models.items():
            m, s = g.predict(X)
            out[f"pred_{name}"] = m
            out[f"pred_{name}_std"] = s
        return out

    def importance(self) -> dict:
        return {name: g.lengthscales() for name, g in self.models.items()}

    def best_observed(self) -> dict:
        out = {}
        for t in self.targets:
            if t.name in self.data:
                Y = self.data[t.name][1][:, 0]
                out[t.name] = {"min": float(Y.min()), "max": float(Y.max()), "mean": float(Y.mean())}
        return out
