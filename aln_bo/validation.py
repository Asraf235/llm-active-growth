"""Kernel-free surrogates that cross-check the GP (paper SI Fig. S2, alchemy.surrogates).

    RandomForest - bootstrap ensemble of scikit-learn random forests
                   (alchemy.RandomForestSurrogate): mean and std across the ensemble.
                   Runs anywhere.
    BNN          - NeuroBayes Bayesian neural network sampled with NUTS
                   (alchemy.NeuroBayesBNNSurrogate). Needs jax + neurobayes, which on
                   this laptop only run inside WSL next to BoTorch.

Both expose the GP interface (.fit(X, Y) -> self, .predict(X) -> (mean, std), .X, .Y)
so alchemy_plots draws them the same way. They can also drive the acquisition
(MainCampaign(surrogate=...)) and are fitted as cross-checks in every step: if three
different model families agree on where the minimum is, the conclusion is a property
of the data, not of the GP kernel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .config import PARAMS, PARAM_NAMES, normalize


def _prep(X, Y):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float).reshape(len(X), -1)
    if Y.shape[1] != 1:
        raise ValueError("validation surrogates are single-output")
    return X, Y


@dataclass
class RandomForest:
    """alchemy RandomForestSurrogate: n_bootstrap forests, each on a bootstrap resample of the runs."""
    n_bootstrap: int = 10
    n_trees: int = 100
    min_samples_leaf: int = 1
    random_state: int = 0
    models: list = field(default_factory=list, init=False)
    X: np.ndarray | None = field(default=None, init=False)
    Y: np.ndarray | None = field(default=None, init=False)
    y_mu: float = field(default=0.0, init=False)
    y_sd: float = field(default=1.0, init=False)

    def fit(self, X, Y) -> "RandomForest":
        self.X, self.Y = _prep(X, Y)
        self.y_mu, self.y_sd = float(self.Y.mean()), float(self.Y.std() + 1e-12)
        Xn, yn = normalize(self.X), ((self.Y - self.y_mu) / self.y_sd).ravel()
        rng = np.random.default_rng(self.random_state)
        self.models = []
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, len(Xn), len(Xn))
            rf = RandomForestRegressor(n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
                                       random_state=self.random_state + b)
            self.models.append(rf.fit(Xn[idx], yn[idx]))
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        Xn = normalize(np.atleast_2d(X))
        P = np.stack([m.predict(Xn) for m in self.models])
        return P.mean(0) * self.y_sd + self.y_mu, (P.std(0) + 1e-6) * self.y_sd

    def feature_importance(self) -> dict:
        """Mean impurity importance per knob (LARGE = influential; the RF analogue of a short lengthscale)."""
        imp = np.mean([m.feature_importances_ for m in self.models], axis=0)
        return dict(zip(PARAM_NAMES, np.round(imp, 3).tolist()))


def bnn_available() -> bool:
    try:
        import neurobayes  # noqa: F401
        import jax         # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class BNN:
    """alchemy NeuroBayesBNNSurrogate: a fully Bayesian MLP (NUTS samples of every weight)."""
    hidden: tuple = (32, 32)
    num_warmup: int = 500
    num_samples: int = 500
    seed: int = 0
    model: object = field(default=None, init=False)
    X: np.ndarray | None = field(default=None, init=False)
    Y: np.ndarray | None = field(default=None, init=False)
    y_mu: float = field(default=0.0, init=False)
    y_sd: float = field(default=1.0, init=False)

    def fit(self, X, Y) -> "BNN":
        import jax
        import jax.numpy as jnp
        import neurobayes as nb
        self.X, self.Y = _prep(X, Y)
        self.y_mu, self.y_sd = float(self.Y.mean()), float(self.Y.std() + 1e-12)
        Xn, yn = normalize(self.X), (self.Y - self.y_mu) / self.y_sd
        self.model = nb.BNN(nb.FlaxMLP(hidden_dims=list(self.hidden), target_dim=1))
        self.model.fit(jnp.asarray(Xn), jnp.asarray(yn), num_warmup=self.num_warmup, num_samples=self.num_samples,
                       progress_bar=False, rng_key=jax.random.PRNGKey(self.seed))
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        import jax.numpy as jnp
        m, v = self.model.predict(jnp.asarray(normalize(np.atleast_2d(X))))
        m = np.asarray(m, float).ravel()
        s = np.sqrt(np.clip(np.asarray(v, float).ravel(), 0, None)) + 1e-6
        return m * self.y_sd + self.y_mu, s * self.y_sd



