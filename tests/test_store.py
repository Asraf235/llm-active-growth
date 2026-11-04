"""Round trip through the campaign database: register, import, select, load, step, record, export."""
import pathlib
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aln_bo import config as cfg
from aln_bo.store import Store
from aln_bo.campaign import MainCampaign
from aln_bo.pipeline import run_step


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    out = tmp_path_factory.mktemp("d") / "runs.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_data.py"), "--n", "20", "--out", str(out)],
                   check=True, capture_output=True)
    return pd.read_csv(out)


def test_store_round_trip(tmp_path, flat):
    db = tmp_path / "c.sqlite"
    st = Store(db)
    st.register(cfg.DEFAULT_SPEC)
    assert list(st.experiments()["name"]) == [cfg.DEFAULT_SPEC["name"]]
    name = cfg.DEFAULT_SPEC["name"]
    # import, with one run repeated -> a measurement pair
    df = pd.concat([flat, flat.iloc[[0]].assign(FWHM=flat.FWHM.iloc[0] + 0.02)], ignore_index=True)
    n = st.import_frame(name, df)
    assert n["new"] == len(flat) and n["measurements"] >= 2      # the repeated run's targets go to the measurements table
    runs = st.runs(name)
    assert len(runs) == len(flat) and runs["created_at"].notna().all()
    assert abs(runs.loc[runs.run == flat.run.iloc[0], "FWHM"].iloc[0] - (flat.FWHM.iloc[0] + 0.01)) < 1e-9
    # selection: drop two runs
    sel = pd.DataFrame({"run": flat.run, "GPBO": "INCLUDE", "reason": ""})
    sel.loc[:1, "GPBO"] = "EXCLUDE"
    st.save_selection(name, "test", sel)
    ds = st.load(name, selection="test")
    assert len(ds.df) == len(flat) - 2 and set(cfg.PARAM_NAMES) <= set(ds.df.columns)
    # a step through the pipeline, logged, planned, measured
    res = run_step(ds, tmp_path / "out", backend="sklearn", surrogate="gp", n_alternatives=2, n_grid=5, log=lambda s: None)
    sid = st.record_suggestion(name, res["target"].name, "sklearn", "rbf", "test", len(res["campaign"].X),
                               res["suggestion"], res["alternatives"])
    recipe = res["suggestion"].iloc[0][cfg.PARAM_NAMES].to_dict()
    run = st.add_run(name, recipe, notes=f"suggestion {sid}")
    st.set_decision(sid, "accepted", planned_run=run)
    planned = st.runs(name).set_index("run").loc[run]
    assert planned["status"] == "planned" and abs(planned["OxyAr"] + planned["OxyN2"] - 40.0) < 1e-6
    st.add_result(name, run, {"FWHM": 0.3})
    assert st.runs(name).set_index("run").loc[run, "status"] == "measured"
    assert st.suggestions(name).iloc[0]["decision"] == "accepted"
    # a run added after the selection was saved is kept by that selection; a logged suggestion can be planned later
    ds2 = st.load(name, selection="test")
    assert run in set(ds2.df["run"]) and run in ds2.report["selection"]["unlabelled_runs_kept"]
    sid2 = st.record_suggestion(name, res["target"].name, "sklearn", "rbf", "test", len(ds2.df), res["suggestion"], res["alternatives"])
    run2 = st.plan_suggestion(name, sid2)
    assert st.runs(name).set_index("run").loc[run2, "status"] == "planned" and st.plan_suggestion(name, sid2) == run2
    # export
    p = st.export(name, tmp_path / "e.xlsx")
    sheets = pd.read_excel(p, sheet_name=None)
    assert {"runs", "measurements", "selections", "suggestions", "spec"} <= set(sheets)
    assert len(sheets["runs"]) == len(flat) + 2          # the measured planned run and the second planned run


def test_configure_switches_experiment():
    spec = {"name": "toy", "params": [{"name": "a", "column": "a", "low": 0, "high": 1, "step": 0.1},
                                      {"name": "b", "column": "b", "low": 0, "high": 10, "fixed": 5}],
            "targets": [{"name": "y", "column": "y", "minimize": False, "role": "main"}]}
    try:
        cfg.configure(spec)
        assert cfg.PARAM_NAMES == ["a", "b"] and cfg.FREE_NAMES == ["a"] and cfg.MAIN.name == "y" and not cfg.MAIN.minimize
        assert cfg.AUX == [] and cfg.MAIN_ALT is None
        df = pd.DataFrame({"run": [1, 2, 3, 4], "a": [0.1, 0.5, 0.9, 0.3], "b": [5, 5, 5, 5], "y": [1.0, 2.0, 1.5, 1.8]})
        from aln_bo.data import dataset_from_frame
        c = MainCampaign(dataset_from_frame(df), backend="sklearn").fit()
        sug = c.suggest()
        assert list(sug.columns[:2]) == ["a", "b"] and np.isclose(sug["b"].iloc[0], 5)
    finally:
        cfg.configure(cfg.DEFAULT_SPEC)
