"""The Gaussian-process surrogate (scikit-learn backend).

A GP is a distribution over functions. Given the runs we have done, it gives
us at ANY untried recipe x:
    mean(x)  - the predicted target: the objective for the main GP (in the AlN example the
               rocking-curve FWHM, or the 2theta FWHM as the alternative), or an auxiliary
               property such as interlayer thickness, rms roughness or film density
    std(x)   - how unsure the model is there

Design choices (similar to BoTorch GP used in sumner-harris/alchemy package):
  * RBF (squared-exponential) kernel with ARD - the kernel of the paper and of
    alchemy - one lengthscale per parameter, so the GP learns that e.g.
    temperature matters more than time. kernel="matern" gives Matern-5/2.
  * Inputs mapped to [0, 1] (config.normalize), targets z-scored, so every
    parameter / objective is on a comparable scale inside the model.
  * WhiteKernel = learned measurement noise (XRD / XRR repeatability).
  * Hyperparameters are fitted by maximising the marginal log-likelihood,
    with several random restarts to avoid a bad local optimum.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

from .config import PARAM_NAMES, normalize


@dataclass
class GP:
    """Single-output GP in original units; X is N x d, Y is N x 1."""
    n_restarts: int = 8
    random_state: int = 0
    kernel: str = "rbf"          # "rbf" (squared exponential: the paper / alchemy) or "matern" (5/2)
    gpr: GaussianProcessRegressor | None = field(default=None, init=False)
    X: np.ndarray | None = field(default=None, init=False)
    Y: np.ndarray | None = field(default=None, init=False)
    y_mu: float = field(default=0.0, init=False)
    y_sd: float = field(default=1.0, init=False)

    def _kernel(self, d):
        # Lengthscales live in normalised [0,1] units: 10 means "flat across the
        # whole range" (parameter irrelevant), 0.05 means "wiggles every 5 %".
        ls, bnd = np.full(d, 0.5), (5e-2, 10.0)
        if self.kernel == "rbf":
            base = RBF(length_scale=ls, length_scale_bounds=bnd)
        elif self.kernel == "matern":
            base = Matern(length_scale=ls, length_scale_bounds=bnd, nu=2.5)
        else:
            raise ValueError(f"kernel must be 'rbf' or 'matern', got {self.kernel!r}")
        return (ConstantKernel(1.0, (1e-2, 1e2)) * base
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0)))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GP":
        X = np.asarray(X, float)
        Y = np.asarray(Y, float).reshape(len(X), -1)
        if Y.shape[1] != 1:
            raise ValueError("GP is single-output; use GPList for several targets")
        self.X, self.Y = X, Y
        self.y_mu, self.y_sd = float(Y.mean()), float(Y.std() + 1e-12)
        yn = ((Y - self.y_mu) / self.y_sd).ravel()
        with warnings.catch_warnings():
            # A lengthscale pinned at its upper bound just means "this parameter
            # does not matter for this target"; it is information, not an error.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gpr = GaussianProcessRegressor(
                kernel=self._kernel(X.shape[1]), normalize_y=False,
                n_restarts_optimizer=self.n_restarts, random_state=self.random_state,
            ).fit(normalize(X), yn)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and std of the *latent* function (noise removed), original units."""
        m, s = self.gpr.predict(normalize(np.atleast_2d(X)), return_std=True)
        noise = self.gpr.kernel_.k2.noise_level            # WhiteKernel term
        s_latent = np.sqrt(np.clip(s ** 2 - noise, 1e-12, None))
        return m * self.y_sd + self.y_mu, s_latent * self.y_sd

    def fantasize(self, X_new: np.ndarray, Y_new: np.ndarray) -> "GP":
        """Copy of this GP with (X_new, Y_new) added and hyperparameters FROZEN.

        Used for batch suggestions (kriging believer): after choosing one
        experiment we pretend its outcome equals the GP mean, so the next
        suggestion is pushed elsewhere.
        """
        g = copy.deepcopy(self)
        Xa = np.vstack([self.X, np.atleast_2d(X_new)])
        Ya = np.vstack([self.Y, np.asarray(Y_new, float).reshape(-1, 1)])
        yn = ((Ya - self.y_mu) / self.y_sd).ravel()
        g.X, g.Y = Xa, Ya
        g.gpr = GaussianProcessRegressor(kernel=self.gpr.kernel_, optimizer=None).fit(normalize(Xa), yn)
        return g

    def lengthscales(self) -> dict:
        """ARD lengthscales in normalised [0,1] units. SMALL = objective varies quickly along it."""
        ls = self.gpr.kernel_.k1.k2.length_scale
        return dict(zip(PARAM_NAMES, np.round(np.atleast_1d(ls), 3).tolist()))

    def noise(self) -> float:
        """Learned observation noise std, in original units."""
        return float(np.sqrt(self.gpr.kernel_.k2.noise_level) * self.y_sd)


@dataclass
class GPList:
    """One independent GP per column of Y (N x m): several targets fitted side by side."""
    models: list = field(default_factory=list)
    kernel: str = "rbf"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GPList":
        Y = np.asarray(Y, float)
        self.models = [GP(kernel=self.kernel).fit(X, Y[:, i:i + 1]) for i in range(Y.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mean, std each of shape (M, m)."""
        ms, ss = zip(*(g.predict(X) for g in self.models))
        return np.column_stack(ms), np.column_stack(ss)

    def fantasize(self, X_new, Y_new) -> "GPList":
        Y_new = np.atleast_2d(Y_new)
        return GPList([g.fantasize(X_new, Y_new[:, i]) for i, g in enumerate(self.models)])

    def lengthscales(self) -> list[dict]:
        return [g.lengthscales() for g in self.models]
