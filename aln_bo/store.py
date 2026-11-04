"""The campaign database: one SQLite file, one runs table per experiment.

    experiments          registry: name, display name, description, the parameter/target spec (JSON)
    <name>_runs          one row per growth: run id, timestamps, source, status, raw knobs, targets, notes
    <name>_measurements  every individual measurement (repeat XRD rows, re-measurements); their mean is
                         written to the runs table when they are imported or recorded
    selections           named run selections (INCLUDE / OPTIONAL / EXCLUDE with reasons) per experiment
    suggestions          every BO step: model, selection, recipe, prediction, EI, alternatives, review, decision

Rules
    * The user's spreadsheet is never written to. It is imported; exports are generated from the database.
    * Knobs are stored RAW (the values set on the tool). Derived parameters (N2 fraction, 1/time ...)
      are computed when a Dataset is loaded, from the spec's `derive` expressions.
    * Runs and experiments carry created_at / updated_at (UTC); measurements, selections and suggestions
      are append-only and carry created_at, so "what was added since the last step" is a query.

Typical use
    st = Store("campaign.sqlite")
    st.register(spec)                                  # spec: dict as in config.DEFAULT_SPEC or a JSON file
    st.import_frame("aln_sputter", df)                 # df: one row per run, raw knob + target columns
    ds = st.load("aln_sputter", selection="thin_films")  # -> aln_bo.data.Dataset, config activated
    ... MainCampaign(ds).fit().suggest() ...
    st.record_suggestion("aln_sputter", ...)
    st.export("aln_sputter", "aln_sputter.xlsx")
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from . import config as cfg

META_COLS = ["run", "created_at", "updated_at", "source", "status", "date", "notes"]
STATUSES = ("imported", "planned", "grown", "measured", "excluded")


def _q(name: str) -> str:
    """Quote an identifier for SQLite."""
    return '"' + name.replace('"', '""') + '"'


def _now() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")


def raw_knob_names(spec: dict) -> list[str]:
    """Columns the runs table stores for the knobs: physical knobs + every non-derived parameter."""
    names = list(spec.get("physical", {}).keys())
    for p in spec["params"]:
        if not p.get("derive") and p["name"] not in names:
            names.append(p["name"])
    return names


def target_names(spec: dict) -> list[str]:
    return [t["name"] for t in spec["targets"]]


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.con = sqlite3.connect(str(self.path))
        self.con.row_factory = sqlite3.Row
        self._init()

    # ------------------------------------------------------------------ schema
    def _init(self):
        c = self.con
        c.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            name TEXT PRIMARY KEY, display_name TEXT, description TEXT, spec_json TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS selections (
            experiment TEXT NOT NULL, name TEXT NOT NULL, run REAL NOT NULL, label TEXT NOT NULL, reason TEXT,
            created_at TEXT NOT NULL, PRIMARY KEY (experiment, name, run));
        CREATE TABLE IF NOT EXISTS suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, experiment TEXT NOT NULL, created_at TEXT NOT NULL,
            target TEXT, backend TEXT, kernel TEXT, selection TEXT, n_runs INTEGER,
            recipe_json TEXT, pred REAL, pred_std REAL, acq REAL, alternatives_json TEXT,
            review_json TEXT, decision TEXT, planned_run REAL, note TEXT);
        """)
        c.commit()

    def close(self):
        self.con.close()

    # ------------------------------------------------------------------ registry
    def experiments(self) -> pd.DataFrame:
        return pd.read_sql("SELECT name, display_name, description, created_at, updated_at FROM experiments", self.con)

    def spec(self, name: str) -> dict:
        row = self.con.execute("SELECT spec_json FROM experiments WHERE name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"experiment {name!r} is not registered; have {list(self.experiments()['name'])}")
        return json.loads(row["spec_json"])

    def register(self, spec: dict, replace: bool = False) -> None:
        """Add an experiment (parameter + target spec) and create its runs / measurements tables."""
        name = spec["name"]
        exists = self.con.execute("SELECT 1 FROM experiments WHERE name=?", (name,)).fetchone()
        if exists and not replace:
            raise ValueError(f"experiment {name!r} already registered (use replace=True to update the spec)")
        now = _now()
        self.con.execute(
            "INSERT INTO experiments(name, display_name, description, spec_json, created_at, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET display_name=excluded.display_name, description=excluded.description, "
            "spec_json=excluded.spec_json, updated_at=excluded.updated_at",
            (name, spec.get("display_name", name), spec.get("description", ""), json.dumps(spec), now, now))
        cols = [f"{_q(k)} REAL" for k in raw_knob_names(spec)] + [f"{_q(t)} REAL" for t in target_names(spec)]
        self.con.execute(f"""CREATE TABLE IF NOT EXISTS {_q(name + '_runs')} (
            run REAL PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            source TEXT, status TEXT, date TEXT, notes TEXT, {', '.join(cols)})""")
        # add any columns that a replaced spec introduced
        have = {r["name"] for r in self.con.execute(f"PRAGMA table_info({_q(name + '_runs')})")}
        for k in raw_knob_names(spec) + target_names(spec):
            if k not in have:
                self.con.execute(f"ALTER TABLE {_q(name + '_runs')} ADD COLUMN {_q(k)} REAL")
        self.con.execute(f"""CREATE TABLE IF NOT EXISTS {_q(name + '_measurements')} (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run REAL NOT NULL, target TEXT NOT NULL, value REAL,
            created_at TEXT NOT NULL, note TEXT)""")
        self.con.commit()

    def activate(self, name: str) -> dict:
        """Make this experiment's spec the live configuration (PARAMS, targets ...) of aln_bo."""
        spec = self.spec(name)
        cfg.configure(spec)
        return spec

    # ------------------------------------------------------------------ runs
    def _runs_table(self, name: str) -> str:
        return _q(name + "_runs")

    def runs(self, name: str) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {self._runs_table(name)} ORDER BY run", self.con)

    def import_frame(self, name: str, df: pd.DataFrame, source: str = "import", status: str = "measured") -> dict:
        """Upsert runs from a flat frame (one row per run, raw knob and target columns; extra columns ignored).
        Rows that repeat a run number are treated as repeat measurements: each value goes to the
        measurements table and the runs table gets the mean. Returns counts."""
        spec = self.spec(name)
        knobs, targets = raw_knob_names(spec), target_names(spec)
        df = df.copy()
        if "run" not in df.columns:
            raise ValueError("the frame needs a 'run' column")
        df["run"] = pd.to_numeric(df["run"], errors="coerce")
        df = df[df["run"].notna()]
        now = _now()
        n_new = n_upd = n_meas = 0
        tbl = self._runs_table(name)
        for run, g in df.groupby("run", sort=True):
            vals = {}
            for k in knobs:
                if k in g.columns:
                    v = pd.to_numeric(g[k], errors="coerce").dropna()
                    vals[k] = float(v.iloc[0]) if len(v) else None
            for t in targets:
                if t in g.columns:
                    v = pd.to_numeric(g[t], errors="coerce").dropna()
                    vals[t] = float(v.mean()) if len(v) else None
                    if len(v) > 1 or (len(v) == 1 and len(g) > 1):
                        for x in v:
                            self.con.execute(f"INSERT INTO {_q(name + '_measurements')}(run, target, value, created_at) VALUES (?,?,?,?)",
                                             (float(run), t, float(x), now))
                            n_meas += 1
            for extra in ("date", "notes"):
                if extra in g.columns:
                    s = g[extra].dropna().astype(str)
                    vals[extra] = s.iloc[0] if len(s) else None
            exists = self.con.execute(f"SELECT 1 FROM {tbl} WHERE run=?", (float(run),)).fetchone()
            if exists:
                sets = ", ".join(f"{_q(k)}=?" for k in vals) + ", updated_at=?"
                self.con.execute(f"UPDATE {tbl} SET {sets} WHERE run=?", (*vals.values(), now, float(run)))
                n_upd += 1
            else:
                cols = ["run", "created_at", "updated_at", "source", "status"] + list(vals)
                self.con.execute(f"INSERT INTO {tbl}({', '.join(_q(c) for c in cols)}) VALUES ({', '.join('?' * len(cols))})",
                                 (float(run), now, now, source, status, *vals.values()))
                n_new += 1
        self.con.commit()
        return {"new": n_new, "updated": n_upd, "measurements": n_meas}

    def add_run(self, name: str, recipe: dict, status: str = "planned", source: str = "suggestion",
                run: Optional[float] = None, notes: Optional[str] = None) -> float:
        """Insert a planned run (e.g. a BO suggestion). Recipe values may be derived parameters; the
        raw knobs are computed through the spec's physical_from_derived expressions."""
        spec = self.spec(name)
        tbl = self._runs_table(name)
        if run is None:
            m = self.con.execute(f"SELECT MAX(run) FROM {tbl}").fetchone()[0]
            run = float(np.floor(m) + 1) if m is not None else 1.0
        row = pd.DataFrame([recipe])
        for raw, expr in spec.get("physical_from_derived", {}).items():
            if raw not in row.columns:
                try:
                    row[raw] = cfg.evaluate(expr, row)
                except Exception:
                    pass
        vals = {k: float(row[k].iloc[0]) for k in raw_knob_names(spec) if k in row.columns and pd.notna(row[k].iloc[0])}
        now = _now()
        cols = ["run", "created_at", "updated_at", "source", "status", "notes"] + list(vals)
        self.con.execute(f"INSERT INTO {tbl}({', '.join(_q(c) for c in cols)}) VALUES ({', '.join('?' * len(cols))})",
                         (float(run), now, now, source, status, notes, *vals.values()))
        self.con.commit()
        return float(run)

    def add_result(self, name: str, run: float, values: dict, note: Optional[str] = None, status: str = "measured") -> None:
        """Record measured target values for a run (each one also goes to the measurements table)."""
        spec = self.spec(name)
        tbl = self._runs_table(name)
        now = _now()
        if self.con.execute(f"SELECT 1 FROM {tbl} WHERE run=?", (float(run),)).fetchone() is None:
            raise KeyError(f"run {run} not in {name}")
        for t, v in values.items():
            if t not in target_names(spec):
                raise KeyError(f"{t!r} is not a target of {name}; targets are {target_names(spec)}")
            self.con.execute(f"INSERT INTO {_q(name + '_measurements')}(run, target, value, created_at, note) VALUES (?,?,?,?,?)",
                             (float(run), t, float(v), now, note))
            mean = self.con.execute(f"SELECT AVG(value) FROM {_q(name + '_measurements')} WHERE run=? AND target=?",
                                    (float(run), t)).fetchone()[0]
            self.con.execute(f"UPDATE {tbl} SET {_q(t)}=?, updated_at=?, status=? WHERE run=?", (mean, now, status, float(run)))
        self.con.commit()

    def set_status(self, name: str, run: float, status: str, notes: Optional[str] = None) -> None:
        self.con.execute(f"UPDATE {self._runs_table(name)} SET status=?, updated_at=?, notes=COALESCE(?, notes) WHERE run=?",
                         (status, _now(), notes, float(run)))
        self.con.commit()

    # ------------------------------------------------------------------ selections
    def save_selection(self, name: str, sel_name: str, table: pd.DataFrame, replace: bool = True) -> int:
        """table columns: run, GPBO (INCLUDE / OPTIONAL / EXCLUDE), reason (optional)."""
        now = _now()
        if replace:
            self.con.execute("DELETE FROM selections WHERE experiment=? AND name=?", (name, sel_name))
        n = 0
        for _, r in table.iterrows():
            self.con.execute("INSERT OR REPLACE INTO selections(experiment, name, run, label, reason, created_at) VALUES (?,?,?,?,?,?)",
                             (name, sel_name, float(r["run"]), str(r["GPBO"]).upper(), str(r.get("reason", "")), now))
            n += 1
        self.con.commit()
        return n

    def selections(self, name: str) -> pd.DataFrame:
        return pd.read_sql("SELECT name, label, COUNT(*) AS runs, MIN(created_at) AS created_at FROM selections "
                           "WHERE experiment=? GROUP BY name, label ORDER BY name, label", self.con, params=(name,))

    def selection(self, name: str, sel_name: str, keep=("INCLUDE",)) -> list[float]:
        labels = [k.upper() for k in keep]
        rows = self.con.execute(f"SELECT run FROM selections WHERE experiment=? AND name=? AND label IN ({','.join('?' * len(labels))})",
                                (name, sel_name, *labels)).fetchall()
        if not rows:
            raise KeyError(f"selection {sel_name!r} with labels {labels} has no runs in {name}")
        return [float(r["run"]) for r in rows]

    # ------------------------------------------------------------------ load into the modelling code
    def load(self, name: str, selection: Optional[str] = None, keep=("INCLUDE",),
             statuses=("imported", "measured", "grown")):
        """Dataset for the modelling code: activates the experiment's spec, averages measurements, derives
        the derived parameters, applies a named selection, and records what was left out."""
        from .data import dataset_from_frame, select_runs
        self.activate(name)
        df = self.runs(name)
        df = df[df["status"].isin(statuses)].reset_index(drop=True)
        ds = dataset_from_frame(df, report={"source": str(self.path), "experiment": name, "usable_runs": len(df)})
        if selection:
            # runs the selection labels with another tag are dropped; runs it does not mention at all (added
            # after the selection was saved) are KEPT, so a fresh result is never silently left out
            labelled = pd.read_sql("SELECT run, label FROM selections WHERE experiment=? AND name=?", self.con,
                                   params=(name, selection))
            if labelled.empty:
                raise KeyError(f"selection {selection!r} has no runs in {name}")
            labels = {k.upper() for k in keep}
            drop = set(labelled.loc[~labelled["label"].str.upper().isin(labels), "run"].astype(float))
            keep_runs = [float(r) for r in ds.df["run"] if float(r) not in drop]
            ds = select_runs(ds, keep_runs)
            new = sorted(set(keep_runs) - set(labelled["run"].astype(float)))
            ds.report["selection"] = {"name": selection, "keep": list(keep), "unlabelled_runs_kept": new}
        return ds

    def plan_suggestion(self, name: str, suggestion_id: int, notes: Optional[str] = None) -> float:
        """Turn a logged suggestion into a planned run (what `aln-bo step --plan` does at step time)."""
        row = self.con.execute("SELECT recipe_json, planned_run FROM suggestions WHERE id=? AND experiment=?",
                               (int(suggestion_id), name)).fetchone()
        if row is None:
            raise KeyError(f"suggestion {suggestion_id} not found for {name}")
        if row["planned_run"] is not None:
            return float(row["planned_run"])
        recipe = {k: v for k, v in json.loads(row["recipe_json"]).items() if k in cfg.PARAM_NAMES}
        run = self.add_run(name, recipe, status="planned", notes=notes or f"suggestion {suggestion_id}")
        self.set_decision(int(suggestion_id), "accepted", planned_run=run)
        return run

    # ------------------------------------------------------------------ suggestions
    def record_suggestion(self, name: str, target: str, backend: str, kernel: str, selection: Optional[str],
                          n_runs: int, suggestion: pd.DataFrame, alternatives: pd.DataFrame,
                          review: Optional[dict] = None, decision: Optional[str] = None,
                          planned_run: Optional[float] = None, note: Optional[str] = None) -> int:
        s = suggestion.iloc[0]
        acq_col = next((c for c in suggestion.columns if c.upper() in ("EI", "LCB", "UCB")), None)
        cur = self.con.execute(
            "INSERT INTO suggestions(experiment, created_at, target, backend, kernel, selection, n_runs, recipe_json, pred, "
            "pred_std, acq, alternatives_json, review_json, decision, planned_run, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, _now(), target, backend, kernel, selection, int(n_runs),
             suggestion.round(6).iloc[0].to_json(), float(s.get(f"pred_{target}", np.nan)), float(s.get(f"pred_{target}_std", np.nan)),
             float(s[acq_col]) if acq_col else None, alternatives.round(6).to_json(orient="records"),
             json.dumps(review) if review is not None else None, decision, planned_run, note))
        self.con.commit()
        return int(cur.lastrowid)

    def suggestions(self, name: str) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM suggestions WHERE experiment=? ORDER BY id", self.con, params=(name,))

    def set_decision(self, suggestion_id: int, decision: str, planned_run: Optional[float] = None) -> None:
        self.con.execute("UPDATE suggestions SET decision=?, planned_run=COALESCE(?, planned_run) WHERE id=?",
                         (decision, planned_run, suggestion_id))
        self.con.commit()

    # ------------------------------------------------------------------ export
    def export(self, name: str, path: str | Path) -> Path:
        """Write the experiment to an .xlsx (or .csv of the runs table): runs, measurements, selections, suggestions."""
        path = Path(path)
        runs = self.runs(name)
        if path.suffix.lower() == ".csv":
            runs.to_csv(path, index=False)
            return path
        meas = pd.read_sql(f"SELECT * FROM {_q(name + '_measurements')} ORDER BY run, target, id", self.con)
        sel = pd.read_sql("SELECT name, run, label, reason, created_at FROM selections WHERE experiment=? ORDER BY name, run",
                          self.con, params=(name,))
        sug = self.suggestions(name)
        with pd.ExcelWriter(path) as xw:
            runs.to_excel(xw, sheet_name="runs", index=False)
            meas.to_excel(xw, sheet_name="measurements", index=False)
            sel.to_excel(xw, sheet_name="selections", index=False)
            sug.to_excel(xw, sheet_name="suggestions", index=False)
            pd.DataFrame([{"key": k, "value": json.dumps(v) if not isinstance(v, str) else v}
                          for k, v in self.spec(name).items()]).to_excel(xw, sheet_name="spec", index=False)
        return path
