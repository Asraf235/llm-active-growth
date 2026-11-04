"""Backend selection: the same GP-BO maths on two libraries.

    sklearn  - scikit-learn GaussianProcessRegressor + numpy/scipy EI (and MC-EHVI, unused).
               Works wherever scikit-learn runs; the light tier of the package.
    botorch  - BoTorch SingleTaskGP / ModelListGP + qLogEI (and qLogEHVI, unused),
               the stack used by sumner-harris/alchemy. Needs torch (the [botorch] extra).

Pick with  get_backend("botorch")  or the environment variable  ALN_BO_BACKEND.
"auto" (default) uses botorch when it imports, else sklearn.

Every backend exposes the same four names:
    GP, GPList           - .fit(X, Y) -> self ; .predict(X) -> (mean, std) ; .lengthscales()
    suggest_ei(gp, q, minimize=..., candidates=..., snap=...)         -> (X_new, acq_values)
    suggest_ehvi(gplist, q, maximize_mask=..., candidates=..., snap=...) -> (X_new, acq_values)
with numpy arrays in and out, so campaign.py never sees torch.
"""
import importlib
import os

_AVAILABLE = ("sklearn", "botorch")


def available_backends() -> list[str]:
    out = ["sklearn"]
    try:
        import botorch  # noqa: F401
        import torch    # noqa: F401
        out.append("botorch")
    except Exception:
        pass
    return out


def resolve_backend_name(name: str | None = None) -> str:
    name = (name or os.environ.get("ALN_BO_BACKEND", "auto")).lower()
    if name == "auto":
        return "botorch" if "botorch" in available_backends() else "sklearn"
    if name not in _AVAILABLE:
        raise ValueError(f"unknown backend {name!r}; choose from {_AVAILABLE} or 'auto'")
    return name


def get_backend(name: str | None = None):
    """Return the backend module (aln_bo.backends.sklearn_backend or .botorch_backend)."""
    return importlib.import_module(f"aln_bo.backends.{resolve_backend_name(name)}_backend")
