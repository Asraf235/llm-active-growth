"""The parameter space and the objectives, loaded from an experiment SPEC.

Everything else in the package reads the module-level containers below (PARAMS, FIXED, MAIN ...).
They are filled by `configure(spec)`, which the campaign database calls when an experiment is
activated (Store.activate / Store.load), or which you call yourself with a dict or a JSON file.
At import time the AlN sputter example (DEFAULT_SPEC) is loaded so the examples and tests run
without a database.

Two kinds of GP inputs:
  * FREE parameters   - the knobs the optimiser may move.
  * CONTEXT parameters - conditions that varied in the data and affect the measurements (e.g. the
                         AlN growth time sets the film thickness), so the GP must see them, but the
                         optimiser holds them at `fixed` for suggestions and maps.

A spec is a dict (JSON on disk):
    name, display_name, description
    params:  [{name, column, low, high, unit, step, fixed, encode, derive}, ...]
             derive = expression over the raw knob columns, evaluated with pandas (e.g. "OxyN2 / (OxyAr + OxyN2)")
    physical:              {raw knob name: data column}   raw knobs that derived parameters are built from
    physical_from_derived: {raw knob name: expression over the parameters}   to map a recipe back to the tool
    targets: [{name, column, unit, minimize, role: main | alt | aux}, ...]
    import:  {process_sheet, data_sheet, run_col, exclude, substrate_must_contain}   for the two-sheet log reader
    llm_context: free text with the physics the LLM reviewer may use
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Param:
    name: str
    column: str
    low: float
    high: float
    unit: str = ""
    step: Optional[float] = None
    fixed: Optional[float] = None
    encode: Optional[dict] = None
    derive: Optional[Union[Callable, str]] = None

    @property
    def is_free(self) -> bool:
        return self.fixed is None


@dataclass(frozen=True)
class Target:
    name: str
    column: str
    unit: str = ""
    minimize: bool = True
    role: str = "aux"            # main | alt | aux


# ---- the AlN sputter example (the campaign this package was built on) -------------------------
DEFAULT_SPEC = {
    "name": "aln_sputter_sioxny",
    "display_name": "SiOxNy plasma step before AlN sputtering on SiO2",
    "description": "Plasma oxynitridation of SiO2 (temperature, bias, gas ratio, pressure, time) followed by reactive "
                   "sputtering of an AlN seed layer; objective = rocking-curve width of the AlN (0002) peak (texture), "
                   "with the 2theta peak width as the alternative objective.",
    "params": [
        {"name": "OxyT", "column": "Oxy temp (C)", "low": 25.0, "high": 700.0, "unit": "C", "step": 25.0},
        {"name": "OxyBias", "column": "Oxy bias (W)", "low": 0.0, "high": 60.0, "unit": "W", "step": 5.0},
        # Ar and N2 flows enter as ONE chemistry knob (N2 fraction) plus the total flow held as context
        {"name": "N2frac", "column": "N2 / (Ar + N2)", "low": 0.25, "high": 0.75, "unit": "", "step": 0.125,
         "derive": "OxyN2 / (OxyAr + OxyN2)"},
        {"name": "OxyP", "column": "Oxy pressure (mTorr)", "low": 5.0, "high": 40.0, "unit": "mTorr", "step": 5.0},
        {"name": "OxyTime", "column": "Oxynitride time (min)", "low": 1.0, "high": 30.0, "unit": "min", "step": 1.0},
        # context: the AlN sputter step (seen by the GP, held fixed for suggestions). Growth time enters as 1/t:
        # the 2theta width scales with 1/thickness (Scherrer) and thickness ~ t (docs/growth_time_vs_inverse.png).
        {"name": "InvGrowth", "column": "1 / growth time (1/min)", "low": 1.0 / 200.0, "high": 1.0 / 10.0, "unit": "1/min",
         "step": None, "fixed": 1.0 / 13.0, "derive": "1 / GrowthT"},
        {"name": "WorkP", "column": "Working pressure (mTorr)", "low": 1.5, "high": 10.0, "unit": "mTorr", "step": 0.1, "fixed": 1.7},
        {"name": "StageMax", "column": "Oxy stage height", "low": 0.0, "high": 1.0, "unit": "1=Max", "step": 1.0, "fixed": 1.0,
         "encode": {"max": 1.0, "min": 0.0}},
        {"name": "OxyFlow", "column": "Ar + N2 (sccm)", "low": 30.0, "high": 60.0, "unit": "sccm", "step": 5.0, "fixed": 40.0,
         "derive": "OxyAr + OxyN2"},
    ],
    "physical": {"OxyAr": "Oxy Ar (sccm)", "OxyN2": "Oxy N2 (sccm)", "GrowthT": "Growth time (mins)"},
    "physical_from_derived": {"OxyAr": "OxyFlow * (1 - N2frac)", "OxyN2": "OxyFlow * N2frac", "GrowthT": "1 / InvGrowth"},
    "targets": [
        {"name": "FWHM", "column": "XRD FWHM at 36° 2θ", "unit": "deg 2theta", "minimize": True, "role": "alt"},
        {"name": "RC_FWHM", "column": "Rocking Curve FWHM at 36° 2θ", "unit": "deg omega", "minimize": True, "role": "main"},
        {"name": "SiOxNy_thick", "column": "SiOxNy XRR thickness (nm)", "unit": "nm", "minimize": True, "role": "aux"},
        {"name": "SiOxNy_rough", "column": "SiOxNy XRR roughness (nm)", "unit": "nm", "minimize": True, "role": "aux"},
        {"name": "AlN_density", "column": "AlN XRR density (g/cm3)", "unit": "g/cm3", "minimize": False, "role": "aux"},
    ],
    "import": {"process_sheet": "AlN Oxy", "data_sheet": "AlN Oxy SiO2 Data", "run_col": "Run (#)",
               "exclude": {"Stage Grounded": [1, 1.0, True]}, "substrate_must_contain": "SiO2"},
    "llm_context": (
        "- The MAIN objective is the rocking-curve FWHM of the AlN (0002) reflection: SMALLER is better (tighter c-axis "
        "alignment / less mosaic spread). The 2theta peak FWHM is the alternative objective: smaller = larger coherent "
        "domains, less strain; it is strongly thickness dependent, which is why growth time is a context parameter.\n"
        "- Auxiliary GPs predict SiOxNy XRR thickness and roughness and AlN XRR density. A dense AlN (~3.2-3.3 g/cm3), "
        "a thin and smooth SiOxNy interlayer are usually what a good seed layer looks like.\n"
        "- Oxy bias (RF substrate bias during plasma treatment) controls ion energy: more bias = more nitrogen incorporation / "
        "sputter-cleaning but also more damage and roughening of the interlayer.\n"
        "- Oxy pressure sets the plasma density and ion mean free path; high pressure = softer, more thermalised plasma; "
        "very low pressure = energetic ions.\n"
        "- N2 / Ar ratio sets reactive-species supply; temperature and time set diffusion-limited growth of the oxynitride "
        "(thickness ~ sqrt(time), strongly thermally activated).\n"
        "- A very thick or rough SiOxNy interlayer tends to degrade AlN texture; so does an untreated SiO2 surface. "
        "There is usually a sweet spot."),
}

# ---- live configuration (filled by configure; mutated IN PLACE so `from .config import PARAMS` stays valid) ----
SPEC: dict = {}
PARAMS: list = []
PARAM_NAMES: list = []
FREE_NAMES: list = []
FREE_IDX: list = []
FIXED: dict = {}
BOUNDS: dict = {}
PHYSICAL: dict = {}
PHYSICAL_FROM_DERIVED: dict = {}
TARGETS: list = []
AUX: list = []
IMPORT: dict = {}
EXCLUDE: dict = {}
MAIN: Target = None            # rebound by configure(): read it as `config.MAIN` at call time, not via `from ... import MAIN`
MAIN_ALT: Optional[Target] = None
SUBSTRATE_MUST_CONTAIN: Optional[str] = None


def evaluate(expr, df):
    """Evaluate a derive / physical expression: a callable gets the frame, a string goes to pandas eval."""
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if callable(expr):
        return expr(df)
    return df.eval(str(expr))


def configure(spec: dict) -> dict:
    """Make `spec` the live configuration. Returns the spec."""
    global MAIN, MAIN_ALT, SUBSTRATE_MUST_CONTAIN
    SPEC.clear()
    SPEC.update(json.loads(json.dumps(spec)))
    params = [Param(name=p["name"], column=p.get("column", p["name"]), low=float(p["low"]), high=float(p["high"]),
                    unit=p.get("unit", "") or "", step=(None if p.get("step") in (None, "", 0) else float(p["step"])),
                    fixed=(None if p.get("fixed") is None else float(p["fixed"])), encode=p.get("encode"),
                    derive=p.get("derive")) for p in spec["params"]]
    PARAMS[:] = params
    PARAM_NAMES[:] = [p.name for p in params]
    FREE_NAMES[:] = [p.name for p in params if p.is_free]
    FREE_IDX[:] = [i for i, p in enumerate(params) if p.is_free]
    FIXED.clear(); FIXED.update({i: p.fixed for i, p in enumerate(params) if not p.is_free})
    BOUNDS.clear(); BOUNDS.update({p.name: (p.low, p.high) for p in params})
    PHYSICAL.clear(); PHYSICAL.update(spec.get("physical", {}))
    PHYSICAL_FROM_DERIVED.clear(); PHYSICAL_FROM_DERIVED.update(spec.get("physical_from_derived", {}))
    targets = [Target(t["name"], t.get("column", t["name"]), t.get("unit", "") or "", bool(t.get("minimize", True)),
                      t.get("role", "aux")) for t in spec["targets"]]
    TARGETS[:] = targets
    mains = [t for t in targets if t.role == "main"]
    MAIN = mains[0] if mains else targets[0]
    alts = [t for t in targets if t.role == "alt"]
    MAIN_ALT = alts[0] if alts else None
    AUX[:] = [t for t in targets if t.role == "aux"]
    imp = spec.get("import", {}) or {}
    IMPORT.clear(); IMPORT.update(imp)
    EXCLUDE.clear(); EXCLUDE.update(imp.get("exclude", {}) or {})
    SUBSTRATE_MUST_CONTAIN = imp.get("substrate_must_contain")
    return spec


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def current_spec() -> dict:
    return json.loads(json.dumps(SPEC))


def target(name: Optional[str] = None) -> Target:
    """Target by name (default: the main target)."""
    if name is None:
        return MAIN
    for t in TARGETS:
        if t.name == name:
            return t
    raise KeyError(f"unknown target {name!r}; targets are {target_names()}")


def target_names() -> list[str]:
    return [t.name for t in TARGETS]


def objective_names() -> list[str]:
    """Targets a campaign may optimise: the main one and the alternative(s)."""
    return [t.name for t in TARGETS if t.role in ("main", "alt")]


configure(DEFAULT_SPEC)


def bounds_array() -> np.ndarray:
    """2 x d: row 0 lower, row 1 upper. Context dims get [fixed, fixed] so optimisers cannot move them."""
    lo = [p.low if p.is_free else p.fixed for p in PARAMS]
    hi = [p.high if p.is_free else p.fixed for p in PARAMS]
    return np.array([lo, hi], dtype=float)


def data_bounds_array() -> np.ndarray:
    """2 x d over the full observed range (used to normalise GP inputs)."""
    return np.array([[p.low for p in PARAMS], [p.high for p in PARAMS]], dtype=float)


def normalize(X: np.ndarray) -> np.ndarray:
    lo, hi = data_bounds_array()
    return (np.asarray(X, dtype=float) - lo) / (hi - lo)


def denormalize(U: np.ndarray) -> np.ndarray:
    lo, hi = data_bounds_array()
    return lo + np.asarray(U, dtype=float) * (hi - lo)


def apply_fixed(X: np.ndarray) -> np.ndarray:
    """Overwrite context columns with their fixed values."""
    X = np.array(X, dtype=float, copy=True)
    for i, v in FIXED.items():
        X[:, i] = v
    return X


def add_physical(df):
    """Append the raw tool knobs computed from the derived parameters in a recipe table."""
    df = df.copy()
    for name, expr in PHYSICAL_FROM_DERIVED.items():
        df[name] = np.round(np.asarray(evaluate(expr, df), float), 6)
    return df


def in_context_mask(X: np.ndarray) -> np.ndarray:
    """True for rows whose context columns equal the fixed values (within half an instrument step)."""
    X = np.atleast_2d(np.asarray(X, float))
    mask = np.ones(len(X), bool)
    for i, v in FIXED.items():
        tol = PARAMS[i].step * 0.5 if PARAMS[i].step else 1e-6 * max(1.0, abs(v))   # no grid: relative tolerance
        mask &= np.abs(X[:, i] - v) <= tol
    return mask


def snap_to_grid(X: np.ndarray) -> np.ndarray:
    """Round each column to its instrument resolution and clip to bounds; context dims stay fixed."""
    X = np.array(X, dtype=float, copy=True)
    for j, p in enumerate(PARAMS):
        if p.step:
            X[:, j] = np.round(X[:, j] / p.step) * p.step
        X[:, j] = np.clip(X[:, j], p.low, p.high)
    return apply_fixed(X)
