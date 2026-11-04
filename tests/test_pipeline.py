"""Smoke tests: synthetic flat data -> main campaign + auxiliary models on every available backend and surrogate,
one full step with its figures, and the real sputter log loader when the workbook is present."""
import pathlib
import subprocess
import sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aln_bo import load_dataset, load_sputter_log, MainCampaign, AuxModels, available_backends, PARAMS, PARAM_NAMES, FREE_NAMES
from aln_bo.config import FIXED, AUX
from aln_bo.validation import RandomForest, BNN, bnn_available
from aln_bo.pipeline import run_step

BACKENDS = available_backends()
LOG = ROOT / "AlN Sputter Log.xlsx"


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    out = tmp_path_factory.mktemp("d") / "runs.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_data.py"), "--n", "25", "--out", str(out)],
                   check=True, capture_output=True)
    return load_dataset(out)


def _check_recipe_table(df, n):
    assert len(df) == n and list(df.columns[:len(PARAM_NAMES)]) == PARAM_NAMES
    for p in PARAMS:
        assert (df[p.name] >= p.low - 1e-9).all() and (df[p.name] <= p.high + 1e-9).all()
        if p.step and p.is_free:                   # context values are fixed, not snapped
            assert np.allclose((df[p.name] / p.step).round(), df[p.name] / p.step)
    for i, v in FIXED.items():                     # context parameters never move
        assert np.allclose(df[PARAMS[i].name], v)
    assert not df[FREE_NAMES].duplicated().any()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kernel", ["rbf", "matern"])
def test_main_campaign(ds, backend, kernel):
    c = MainCampaign(ds, backend=backend, kernel=kernel).fit()
    sug = c.suggest(q=2)
    _check_recipe_table(sug, 2)
    alt = c.alternatives(n=4)
    _check_recipe_table(alt, 4)
    assert (sug["EI"] >= 0).all() and set(c.importance()) == set(PARAM_NAMES)
    assert np.all(c.acquisition(sug[PARAM_NAMES].to_numpy()) >= 0)
    assert np.allclose(sug["OxyAr"] + sug["OxyN2"], sug["OxyFlow"])      # derived -> physical knobs
    assert np.allclose(sug["GrowthT"], 13.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_aux_models(ds, backend):
    aux = AuxModels(ds, backend=backend).fit()
    assert set(aux.models) == {t.name for t in AUX}
    pred = aux.predict(ds.df[PARAM_NAMES].to_numpy()[:3])
    assert pred.shape == (3, 2 * len(AUX)) and np.isfinite(pred.to_numpy()).all()


@pytest.mark.parametrize("surrogate", ["rf"] + (["bnn"] if bnn_available() else []))
def test_non_gp_surrogates_drive_ei(ds, surrogate):
    c = MainCampaign(ds, surrogate=surrogate).fit()
    sug = c.suggest(q=1)
    _check_recipe_table(sug, 1)
    assert (sug["EI"] >= 0).all() and c.backend_name.startswith(surrogate)


def test_validation_surrogates(ds):
    c = MainCampaign(ds, backend="sklearn").fit()
    rf = RandomForest(n_bootstrap=3, n_trees=20).fit(c.X, c.Y)
    m, s = rf.predict(c.X[:4])
    assert m.shape == (4,) and (s > 0).all() and set(rf.feature_importance()) == set(PARAM_NAMES)


@pytest.mark.skipif(not bnn_available(), reason="neurobayes / jax not installed")
def test_bnn(ds):
    c = MainCampaign(ds, backend="sklearn").fit()
    b = BNN(hidden=(8,), num_warmup=20, num_samples=20).fit(c.X, c.Y)
    m, s = b.predict(c.X[:3])
    assert m.shape == (3,) and np.isfinite(m).all() and (s > 0).all()


def test_run_step_figures(ds, tmp_path):
    res = run_step(ds, tmp_path, backend="sklearn", surrogate="gp", n_alternatives=2, n_grid=5, log=lambda s: None)
    names = {p.name for p in tmp_path.glob("*.png")}
    assert {"GP_mean.png", "GP_variance.png", "acquisition_EI_GP.png", "RF_mean.png", "RF_variance.png"} <= names
    assert ("BNN_mean.png" in names) == bnn_available()
    assert (tmp_path / "suggestion.csv").exists() and (tmp_path / "llm_prompt.md").exists()
    assert set(res["cross_check"]) >= {"RF"}


@pytest.mark.skipif(not LOG.exists(), reason="real sputter log not present")
def test_real_log_loads():
    d = load_sputter_log(LOG)
    assert d.report["usable_runs"] >= 20
    X, Y = d.xy(["FWHM"])
    assert X.shape[1] == len(PARAMS) and len(Y) >= 20
    assert d.df["run"].is_unique
