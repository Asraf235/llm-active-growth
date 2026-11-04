"""Runs in, Dataset out: the bridge between stored runs and the modelling code.

The modelling code (campaign, surrogates, plots) never reads files or databases; it reads a
`Dataset`: a frame with one row per run, one column per parameter of the live spec (derived
parameters already computed) and one column per target, plus a `report` of what was dropped and why.

How a Dataset is made
    dataset_from_frame(df)    from any flat frame with the spec's raw knob and target columns:
                              numeric coercion, derived parameters from the raw knobs, missing
                              targets left as NaN (a run is used by every GP whose target it has).
                              Store.load() builds its Dataset this way from the database.
    load_dataset(path)        the same from a flat CSV / XLSX (the synthetic example, the tests).
    load_sputter_log(path)    the two-sheet AlN workbook of the original campaign: joins the
                              process sheet to the measurement sheet by run number, cleans text
                              cells ("13??", "Too thick"), averages repeat measurements, drops
                              rows the spec's `import` block says to drop. An example of a
                              lab-specific importer; the CLI uses it with `aln-bo import --sputter-log`.

Run selections
    parse_run_selection(spec) / select_runs(ds, runs)   keep a named list of runs (a CSV with
    run, GPBO, reason, or a comma list); the report records the runs left out.

Nothing disappears silently: every dropped row is listed in `report["dropped"]` with its reason.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import re
import numpy as np
import pandas as pd

from . import config as cfg
from .config import PARAMS, PARAM_NAMES, PHYSICAL


@dataclass
class Dataset:
    df: pd.DataFrame
    report: dict = field(default_factory=dict)

    def xy(self, targets: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """X (N x d, all PARAMS incl. context) and Y (N x len(targets)) for rows where all exist."""
        cols = list(PARAM_NAMES) + list(targets)
        sub = self.df.dropna(subset=cols)
        return sub[PARAM_NAMES].to_numpy(dtype=float), sub[list(targets)].to_numpy(dtype=float)

    def rows_for(self, targets: Sequence[str]) -> pd.DataFrame:
        return self.df.dropna(subset=list(PARAM_NAMES) + list(targets))

    def __len__(self):
        return len(self.df)


# ------------------------------------------------------------------ cleaning
def _num(series: pd.Series) -> pd.Series:
    """Numeric coercion that also rescues '19.9', ' 13 ' and leaves '13??', '?', 'Too thick' as NaN."""
    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        s = str(v).strip()
        if re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", s, flags=re.I):
            return float(s)
        return np.nan
    return series.map(conv).astype(float)


def _encode(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(lambda v: mapping.get(str(v).strip().lower(), np.nan) if not pd.isna(v) else np.nan).astype(float)


def _read_bytes_shared(path: Path) -> bytes:
    """Read a file Excel currently has open. Python's open() asks for a share mode Excel
    refuses; opening the handle ourselves with FILE_SHARE_READ|WRITE|DELETE works."""
    import ctypes, msvcrt, os
    from ctypes import wintypes
    GENERIC_READ, SHARE_ALL, OPEN_EXISTING = 0x80000000, 0x1 | 0x2 | 0x4, 3
    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.restype = wintypes.HANDLE
    h = CreateFileW(str(path), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, 0x80, None)
    if h == wintypes.HANDLE(-1).value:
        raise PermissionError(f"cannot open {path} even with shared access (error {ctypes.GetLastError()})")
    fd = msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(fd, "rb") as f:
        return f.read()


def _read_sheets(path: Path, sheets: list[str]) -> list[pd.DataFrame]:
    """Read sheets; if Excel holds the file locked (PermissionError), read through a shared handle."""
    try:
        return [pd.read_excel(path, sheet_name=s) for s in sheets]
    except PermissionError:
        import io, sys
        if sys.platform != "win32":
            raise
        raw = _read_bytes_shared(path)
        return [pd.read_excel(io.BytesIO(raw), sheet_name=s) for s in sheets]


def load_sputter_log(path: str | Path, main: Optional[str] = None) -> Dataset:
    """The two-sheet AlN log of this project (sheet names etc. come from the spec's `import` block)."""
    main = main or cfg.MAIN.name
    imp = cfg.IMPORT
    PROCESS_SHEET, DATA_SHEET, RUN_COL = imp.get("process_sheet", "AlN Oxy"), imp.get("data_sheet", "AlN Oxy SiO2 Data"), imp.get("run_col", "Run (#)")
    EXCLUDE, SUBSTRATE_MUST_CONTAIN = cfg.EXCLUDE, cfg.SUBSTRATE_MUST_CONTAIN
    TARGETS, TARGET_NAMES = cfg.TARGETS, cfg.target_names()
    path = Path(path)
    proc, meas = _read_sheets(path, [PROCESS_SHEET, DATA_SHEET])
    proc.columns = [str(c).strip() for c in proc.columns]
    meas.columns = [str(c).strip() for c in meas.columns]
    report = {"process_rows": len(proc), "measurement_rows": len(meas), "dropped": []}

    # --- process sheet -> one row per run with short parameter names
    P = pd.DataFrame({"run": _num(proc[RUN_COL])})
    for name, col in PHYSICAL.items():                  # raw knobs that derived parameters are built from
        if col not in proc.columns:
            raise KeyError(f"column {col!r} not found in sheet {PROCESS_SHEET!r}; have {list(proc.columns)}")
        P[name] = _num(proc[col])
    for p in PARAMS:
        if p.derive is not None:
            P[p.name] = cfg.evaluate(p.derive, P)
            continue
        if p.column not in proc.columns:
            raise KeyError(f"column {p.column!r} not found in sheet {PROCESS_SHEET!r}; have {list(proc.columns)}")
        P[p.name] = _encode(proc[p.column], p.encode) if p.encode else _num(proc[p.column])
    P["substrate"] = proc.get("Substrate", "")
    P["notes"] = proc.get("Notes", "").astype(str)
    for col, bad in EXCLUDE.items():
        if col in proc.columns:
            mask = proc[col].isin(bad)
            for r in P.loc[mask, "run"]:
                report["dropped"].append((r, f"{col} in {bad}"))
            P = P[~mask]
    no_run = P["run"].isna()
    report["dropped"] += [(None, "process row without run number")] * int(no_run.sum())
    P = P[~no_run]
    dup = P["run"].duplicated(keep="last")
    report["dropped"] += [(r, "duplicate run number in process sheet, kept last") for r in P.loc[dup, "run"]]
    P = P[~dup]

    # --- measurement sheet -> numeric targets, averaged over repeat rows of the same run
    M = pd.DataFrame({"run": _num(meas[RUN_COL])})
    for t in TARGETS:
        M[t.name] = _num(meas[t.column]) if t.column in meas.columns else np.nan
    M["m_substrate"] = meas.get("Substrate", "")
    no_run_m = M["run"].isna() & M[TARGET_NAMES].notna().any(axis=1)
    report["dropped"] += [(None, "measurement row with data but no run number")] * int(no_run_m.sum())
    M = M[M["run"].notna()]
    reps = M.groupby("run")[TARGET_NAMES].agg(["mean", "std", "count"])
    Mavg = reps.xs("mean", axis=1, level=1).reset_index()
    Mcount = reps.xs("count", axis=1, level=1)[main].rename("n_rep").reset_index()
    Mstd = reps.xs("std", axis=1, level=1)[main].rename(f"{main}_rep_std").reset_index()
    sub = M.groupby("run")["m_substrate"].first().reset_index()

    df = P.merge(Mavg, on="run", how="inner").merge(Mcount, on="run").merge(Mstd, on="run").merge(sub, on="run")
    if SUBSTRATE_MUST_CONTAIN:
        s = df["substrate"].astype(str).str.contains(SUBSTRATE_MUST_CONTAIN, case=False, na=False) | \
            df["m_substrate"].astype(str).str.contains(SUBSTRATE_MUST_CONTAIN, case=False, na=False)
        report["dropped"] += [(r, f"substrate without {SUBSTRATE_MUST_CONTAIN}") for r in df.loc[~s, "run"]]
        df = df[s]
    incomplete = df[PARAM_NAMES].isna().any(axis=1)
    report["dropped"] += [(r, "missing/unparseable process parameter") for r in df.loc[incomplete, "run"]]
    df = df[~incomplete]
    df = df.sort_values("run").reset_index(drop=True)
    report["usable_runs"] = len(df)
    report["rows_per_target"] = {t: int(df[t].notna().sum()) for t in TARGET_NAMES}
    return Dataset(df=df, report=report)


def load_dataset(path: str | Path, sheet: Optional[str] = None) -> Dataset:
    """Flat CSV/XLSX with one row per run and the short column names of the spec."""
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet or 0) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    return dataset_from_frame(df)


def dataset_from_frame(df: pd.DataFrame, report: Optional[dict] = None) -> Dataset:
    """Build a Dataset from a flat frame: numeric coercion, derived parameters from the raw knobs."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in PHYSICAL:
        if c in df.columns:
            df[c] = _num(df[c])
    for p in PARAMS:                                     # derived parameters from the raw knobs if absent
        if p.name not in df.columns and p.derive is not None and all(c in df.columns for c in PHYSICAL):
            df[p.name] = cfg.evaluate(p.derive, df)
    missing = [c for c in PARAM_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"missing parameter columns {missing}; found {list(df.columns)}")
    for c in PARAM_NAMES + cfg.target_names():
        if c in df.columns:
            df[c] = _num(df[c])
        else:
            df[c] = np.nan
    rep = dict(report or {})
    rep.setdefault("usable_runs", len(df))
    rep["rows_per_target"] = {t: int(df[t].notna().sum()) for t in cfg.target_names()}
    return Dataset(df=df.reset_index(drop=True), report=rep)


def parse_run_selection(spec: str, keep: str = "INCLUDE") -> list[float]:
    """`spec` is either a CSV with columns run + GPBO (rows whose GPBO is in `keep`, comma-separated,
    are used) or a comma-separated list of run numbers."""
    p = Path(spec)
    if p.suffix.lower() == ".csv" and p.exists():
        t = pd.read_csv(p)
        labels = {k.strip().upper() for k in keep.split(",")}
        return [float(r) for r in t.loc[t["GPBO"].str.upper().isin(labels), "run"]]
    return [float(x) for x in spec.split(",") if x.strip()]


def select_runs(ds: Dataset, runs: list[float]) -> Dataset:
    """Keep only the listed run numbers; the report records what was left out."""
    keep = ds.df["run"].isin(runs)
    rep = dict(ds.report)
    rep["dropped"] = list(rep.get("dropped", [])) + [(r, "not in the selected run set") for r in ds.df.loc[~keep, "run"]]
    rep["selected_runs"] = sorted(float(r) for r in ds.df.loc[keep, "run"])
    df = ds.df[keep].reset_index(drop=True)
    rep["usable_runs"] = len(df)
    rep["rows_per_target"] = {t: int(df[t].notna().sum()) for t in cfg.target_names() if t in df.columns}
    return Dataset(df=df, report=rep)
