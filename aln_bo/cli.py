"""Command line: `aln-bo <command> ...` (also `python -m aln_bo.cli`).

    aln-bo init      DB                                    create the database file
    aln-bo register  DB SPEC.json [--replace]              add an experiment (parameters, targets, physics notes)
    aln-bo show      DB [EXPERIMENT]                       list experiments, or one experiment's runs / selections / suggestions
    aln-bo import    DB EXPERIMENT FILE [--sputter-log]    import a flat CSV/XLSX (one row per run); --sputter-log uses the
                                                           two-sheet AlN log reader from the example
    aln-bo select    DB EXPERIMENT NAME FILE.csv           save a named run selection (columns run, GPBO, reason)
    aln-bo step      DB EXPERIMENT [options]               fit, suggest, draw maps, write the LLM packet; log the suggestion
    aln-bo record    DB EXPERIMENT RUN target=value ...    enter measured results for a run
    aln-bo export    DB EXPERIMENT OUT.xlsx|.csv           write the experiment back out as a spreadsheet
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path
import pandas as pd

from .store import Store
from . import config as cfg


def _load_spec(path: str) -> dict:
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional dependency
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_init(a):
    Store(a.db).close()
    print(f"database ready: {a.db}")


def cmd_register(a):
    st = Store(a.db)
    spec = _load_spec(a.spec)
    st.register(spec, replace=a.replace)
    print(f"registered {spec['name']!r} ({spec.get('display_name', '')}) in {a.db}")


def cmd_show(a):
    st = Store(a.db)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    if not a.experiment:
        print(st.experiments().to_string(index=False)); return
    spec = st.spec(a.experiment)
    print(f"{spec.get('display_name', a.experiment)}: {spec.get('description', '')}")
    print("knobs:", ", ".join(f"{p['name']}" + (" (context)" if p.get("fixed") is not None else "") for p in spec["params"]))
    print("targets:", ", ".join(f"{t['name']} ({t.get('role', 'aux')})" for t in spec["targets"]))
    runs = st.runs(a.experiment)
    print(f"\nruns: {len(runs)}  by status: {runs['status'].value_counts().to_dict()}")
    print(runs.drop(columns=["created_at", "updated_at"]).tail(a.tail).round(4).to_string(index=False))
    sel = st.selections(a.experiment)
    if len(sel):
        print("\nselections:\n" + sel.to_string(index=False))
    sug = st.suggestions(a.experiment)
    if len(sug):
        print("\nsuggestions:\n" + sug[["id", "created_at", "target", "backend", "selection", "n_runs", "pred", "pred_std", "acq", "decision"]].round(4).to_string(index=False))


def cmd_import(a):
    st = Store(a.db)
    st.activate(a.experiment)
    if a.sputter_log:
        from .data import load_sputter_log
        df = load_sputter_log(a.file).df
    else:
        p = Path(a.file)
        df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    n = st.import_frame(a.experiment, df, source=a.source, status=a.status)
    print(f"imported into {a.experiment}: {n}")


def cmd_select(a):
    st = Store(a.db)
    t = pd.read_csv(a.file)
    n = st.save_selection(a.experiment, a.name, t)
    print(f"saved selection {a.name!r}: {n} rows; labels {t['GPBO'].str.upper().value_counts().to_dict()}")


def cmd_step(a):
    from .pipeline import run_step
    st = Store(a.db)
    ds = st.load(a.experiment, selection=a.selection, keep=tuple(k.strip() for k in a.keep.split(",")))
    out = Path(a.out) / a.experiment / (a.name or dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    res = run_step(ds, out, target=a.target, q=a.q, n_alternatives=a.alternatives, backend=a.backend, acq=a.acq,
                   beta=a.beta, kernel=a.kernel, surrogate=a.surrogate, n_grid=a.n_grid, cross_check=not a.no_cross_check,
                   llm=a.llm, model=a.model, provider=a.llm_provider,
                   label=f"experiment {a.experiment}" + (f", selection {a.selection}" if a.selection else "") + "\n")
    sid = st.record_suggestion(a.experiment, res["target"].name, res["campaign"].backend_name, a.kernel if a.surrogate == "gp" else a.surrogate, a.selection,
                               len(res["campaign"].X), res["suggestion"], res["alternatives"], review=res["review"],
                               note=str(out))
    print(f"\nsuggestion logged as id {sid} in {a.db}")
    if a.plan:
        recipe = res["suggestion"].iloc[0][cfg.PARAM_NAMES].to_dict()
        run = st.add_run(a.experiment, recipe, status="planned", notes=f"suggestion {sid}")
        st.set_decision(sid, "accepted", planned_run=run)
        print(f"planned run {run:g} added to {a.experiment}; enter its results with: aln-bo record {a.db} {a.experiment} {run:g} <target>=<value>")


def cmd_plan(a):
    st = Store(a.db)
    st.activate(a.experiment)
    run = st.plan_suggestion(a.experiment, a.suggestion, notes=a.note)
    print(f"suggestion {a.suggestion} is planned run {run:g} in {a.experiment}; after growth: "
          f"aln-bo record {a.db} {a.experiment} {run:g} <target>=<value>")


def cmd_record(a):
    st = Store(a.db)
    vals = {}
    for kv in a.values:
        k, v = kv.split("=", 1)
        vals[k.strip()] = float(v)
    st.add_result(a.experiment, float(a.run), vals, note=a.note)
    print(f"recorded {vals} for run {a.run} in {a.experiment}")


def cmd_export(a):
    st = Store(a.db)
    p = st.export(a.experiment, a.out)
    print(f"wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aln-bo", description="Gaussian-process Bayesian optimisation of growth recipes "
                                                            "with a local SQLite campaign database.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init"); p.add_argument("db"); p.set_defaults(fn=cmd_init)
    p = sub.add_parser("register"); p.add_argument("db"); p.add_argument("spec"); p.add_argument("--replace", action="store_true"); p.set_defaults(fn=cmd_register)
    p = sub.add_parser("show"); p.add_argument("db"); p.add_argument("experiment", nargs="?"); p.add_argument("--tail", type=int, default=15); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("import"); p.add_argument("db"); p.add_argument("experiment"); p.add_argument("file")
    p.add_argument("--sputter-log", action="store_true", help="read the two-sheet AlN sputter log instead of a flat file")
    p.add_argument("--source", default="import"); p.add_argument("--status", default="measured"); p.set_defaults(fn=cmd_import)
    p = sub.add_parser("select"); p.add_argument("db"); p.add_argument("experiment"); p.add_argument("name"); p.add_argument("file"); p.set_defaults(fn=cmd_select)
    p = sub.add_parser("step"); p.add_argument("db"); p.add_argument("experiment")
    p.add_argument("--target", default=None, help="target name (default: the experiment's main target)")
    p.add_argument("--selection", default=None, help="named run selection to fit on"); p.add_argument("--keep", default="INCLUDE")
    p.add_argument("--backend", choices=["auto", "sklearn", "botorch"], default="auto")
    p.add_argument("--kernel", choices=["rbf", "matern"], default="rbf")
    p.add_argument("--surrogate", choices=["bnn", "gp", "rf"], default="bnn",
                   help="model that drives the acquisition: NeuroBayes BNN (default, needs the [bnn] extra), GP or random forest")
    p.add_argument("--acq", choices=["ei", "lcb"], default="ei"); p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--q", type=int, default=1); p.add_argument("--alternatives", type=int, default=5)
    p.add_argument("--n-grid", type=int, default=12, help="grid points per free knob for the projection figures")
    p.add_argument("--no-cross-check", action="store_true", help="skip the GP / RF / BNN cross-check figures")
    p.add_argument("--llm", action="store_true", help="call a model API for the review (needs that provider key)")
    p.add_argument("--llm-provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default=None, help="model name (default: the provider default)")
    p.add_argument("--plan", action="store_true", help="also insert the suggestion as a planned run")
    p.add_argument("--out", default="results"); p.add_argument("--name", default=None); p.set_defaults(fn=cmd_step)
    p = sub.add_parser("plan"); p.add_argument("db"); p.add_argument("experiment")
    p.add_argument("--suggestion", type=int, required=True, help="id from `aln-bo show`"); p.add_argument("--note", default=None); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("record"); p.add_argument("db"); p.add_argument("experiment"); p.add_argument("run")
    p.add_argument("values", nargs="+", help="target=value pairs"); p.add_argument("--note", default=None); p.set_defaults(fn=cmd_record)
    p = sub.add_parser("export"); p.add_argument("db"); p.add_argument("experiment"); p.add_argument("out"); p.set_defaults(fn=cmd_export)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
