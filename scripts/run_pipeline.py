"""One BO step on a flat file or on the AlN sputter log (no database):

    python scripts/run_pipeline.py "AlN Sputter Log.xlsx" [--target RC_FWHM] [--backend botorch] [--runs data/run_selection.csv] [--llm]

With a campaign database use `aln-bo step DB EXPERIMENT ...` instead (same step, logged in the database).
Everything lands in results/<timestamp>/ (or --out/--name).
"""
import argparse
import datetime as dt
import pathlib
import sys
import warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

from aln_bo import config as cfg
from aln_bo.data import load_sputter_log, load_dataset, parse_run_selection, select_runs
from aln_bo.pipeline import run_step
from aln_bo import llm_review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="AlN Sputter Log.xlsx")
    ap.add_argument("--spec", default=None, help="experiment spec JSON (default: the built-in AlN example)")
    ap.add_argument("--target", choices=cfg.objective_names(), default=None)
    ap.add_argument("--q", type=int, default=1, help="samples to suggest")
    ap.add_argument("--alternatives", type=int, default=5)
    ap.add_argument("--backend", choices=["auto", "sklearn", "botorch"], default="auto")
    ap.add_argument("--acq", choices=["ei", "lcb"], default="ei", help="acquisition (lcb = mean - beta*std, less exploratory)")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kernel", choices=["rbf", "matern"], default="rbf", help="GP kernel (rbf = paper / alchemy)")
    ap.add_argument("--surrogate", choices=["bnn", "gp", "rf"], default="bnn", help="model that drives the acquisition (BNN needs the [bnn] extra)")
    ap.add_argument("--n-grid", type=int, default=12, help="grid points per free knob for the projection figures")
    ap.add_argument("--no-cross-check", action="store_true", help="skip the GP / RF / BNN cross-check figures")
    ap.add_argument("--llm", action="store_true", help="call a model API for the review (needs that provider key)")
    ap.add_argument("--llm-provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None, help="model name (default: the provider default)")
    ap.add_argument("--runs", default=None, help="CSV with run + GPBO columns (keeps --runs-keep labels) or a comma list of run numbers")
    ap.add_argument("--runs-keep", default="INCLUDE", help="GPBO labels to keep from the CSV, comma-separated")
    ap.add_argument("--name", default=None, help="output folder name instead of a timestamp")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.spec:
        cfg.configure(cfg.load_spec(a.spec))

    out = pathlib.Path(a.out) / (a.name or dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if a.data.lower().endswith((".xlsx", ".xls")) and "sputter" in a.data.lower():
        ds = load_sputter_log(a.data)
    else:
        ds = load_dataset(a.data)
    if a.runs:
        ds = select_runs(ds, parse_run_selection(a.runs, a.runs_keep))
    label = f"data: {a.data}" + (f"   run selection: {a.runs} [{a.runs_keep}] -> {len(ds.df)} runs" if a.runs else "") + "\n"
    run_step(ds, out, target=a.target, q=a.q, n_alternatives=a.alternatives, backend=a.backend, acq=a.acq, beta=a.beta,
             kernel=a.kernel, surrogate=a.surrogate, n_grid=a.n_grid, cross_check=not a.no_cross_check,
             llm=a.llm, model=a.model, provider=a.llm_provider, label=label)


if __name__ == "__main__":
    main()
