"""Fake flat dataset with the SHORT column names from config.py, for tests and dry runs.

A hidden ground truth is baked in: FWHM has a minimum near OxyBias~35 W, OxyP~20 mTorr,
OxyTime~10 min and gets narrower with longer growth time (thicker film).

    python tests/make_synthetic_data.py --n 30 --out runs.csv
"""
import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aln_bo import config as cfg
from aln_bo.config import PARAMS, PARAM_NAMES, AUX


def truth(X: dict, rng=None):
    T, B, Ar, N2, P, t = (X[k] for k in ["OxyT", "OxyBias", "OxyAr", "OxyN2", "OxyP", "OxyTime"])
    tg, Pw, sm = X["GrowthT"], X["WorkP"], X["StageMax"]
    fwhm = (0.25 + 1.0e-4 * (B - 35) ** 2 + 2.0e-4 * (P - 20) ** 2 + 1.0e-3 * (t - 10) ** 2
            + 1.0e-7 * (T - 600) ** 2 + 0.12 * np.exp(-tg / 40) + 0.02 * (Pw - 1.7) ** 2 + 0.01 * (1 - sm))
    rc = 1.0 + 6.0 * (fwhm - 0.25)
    thick = 1.5 + 0.05 * B * np.sqrt(t / 10) * np.exp(-(600 - T) / 400) * (N2 / 20)
    rough = 0.1 + 0.015 * B + 0.004 * (P - 20) ** 2 / 10 + 0.3 * (t > 30)
    dens = 3.3 - 0.01 * (B - 40) ** 2 / 100 - 0.05 * (rough - 0.5).clip(0) - 0.002 * (P - 20) ** 2 / 10
    if rng is not None:
        fwhm = fwhm + rng.normal(0, 0.015, fwhm.shape)
        rc = rc + rng.normal(0, 0.2, rc.shape)
        thick = thick + rng.normal(0, 0.2, thick.shape)
        rough = rough + rng.normal(0, 0.1, rough.shape)
        dens = dens + rng.normal(0, 0.03, dens.shape)
    return fwhm, rc, thick, rough, dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="synthetic_runs.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    X = {}
    for p in PARAMS:
        v = rng.uniform(p.low, p.high, a.n) if p.is_free else np.full(a.n, p.fixed, float)
        if p.is_free and p.name == "GrowthT":
            pass
        if p.step:
            v = np.round(v / p.step) * p.step
        X[p.name] = v
    # let growth time vary a bit like the real log does
    X["GrowthT"] = rng.choice([10.0, 13.0, 13.0, 25.0, 60.0], a.n)
    X["OxyAr"] = X["OxyFlow"] * (1 - X["N2frac"])        # the tool knobs behind the derived parameters
    X["OxyN2"] = X["OxyFlow"] * X["N2frac"]
    X["InvGrowth"] = 1.0 / X["GrowthT"]
    fwhm, rc, thick, rough, dens = truth(X, rng)
    df = pd.DataFrame(X)
    df.insert(0, "run", np.arange(1, a.n + 1))
    df["FWHM"] = np.round(fwhm, 4)
    df["RC_FWHM"] = np.round(rc, 3)
    df[AUX[0].name] = np.round(thick, 2)
    df[AUX[1].name] = np.round(rough, 2)
    df[AUX[2].name] = np.round(dens, 3)
    # like the real log: XRR of the thin interlayer only exists for some runs
    miss = rng.random(a.n) < 0.4
    df.loc[miss, [AUX[0].name, AUX[1].name]] = np.nan
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} with {len(df)} rows")


if __name__ == "__main__":
    main()
