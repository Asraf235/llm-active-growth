"""Teaching walkthrough on the real sputter log: every stage printed and explained.

    python scripts/walkthrough.py "AlN Sputter Log.xlsx"

Read alongside README.md. Same functions as scripts/run_pipeline.py, but it stops
at each stage to show what the numbers mean.
"""
import pathlib
import sys
import warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd

from aln_bo import config as cfg
from aln_bo.config import PARAMS, PARAM_NAMES, FREE_NAMES, FIXED, AUX, normalize, apply_fixed
MAIN = cfg.target("FWHM")        # the walkthrough follows the 2theta width
from aln_bo.data import load_sputter_log
from aln_bo.campaign import MainCampaign, AuxModels
from aln_bo.acquisition import sobol_pool, expected_improvement, incumbent

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
path = sys.argv[1] if len(sys.argv) > 1 else "AlN Sputter Log.xlsx"


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- 1
banner("STEP 1  Parameter space (aln_bo/config.py)")
for p in PARAMS:
    role = "FREE   " if p.is_free else f"CONTEXT fixed at {p.fixed:g}"
    print(f"  {p.name:<9} {role:<24} [{p.low:g}, {p.high:g}] {p.unit:<12} step={p.step}   <- '{p.column}'")
print("\nFree = the optimiser may move it. Context = the GP sees it (it varied in the log and")
print("matters), but suggestions are made at the standard value, e.g. 13 min growth.")

# --------------------------------------------------------------------------- 2
banner("STEP 2  Data (aln_bo/data.py): join 'AlN Oxy' <- run # -> 'AlN Oxy SiO2 Data'")
ds = load_sputter_log(path)
print(f"usable runs: {ds.report['usable_runs']}   rows per target: {ds.report['rows_per_target']}")
print("\nfirst rows after cleaning:")
print(ds.df[["run"] + PARAM_NAMES + [MAIN.name] + [t.name for t in AUX]].head(6).round(3).to_string(index=False))
print("\nText cells ('13??', 'Too thick', '600C for 30 min...') become NaN; repeat measurements of one")
print("run are averaged; grounded-stage runs and non-SiO2 substrates are dropped (see report['dropped']).")

# --------------------------------------------------------------------------- 3
banner("STEP 3  Main surrogate: GP on FWHM (aln_bo/surrogate.py, backends/)")
main = MainCampaign(ds, target=MAIN).fit()
print(f"backend: {main.backend_name};  runs with FWHM: {len(main.X)};  learned noise std: {main.noise():.4f}")
print("ARD lengthscales (normalised units; SMALL = parameter matters, 10 = flat):")
for k, v in main.importance().items():
    print(f"  {k:<9} {v:6.3f} {'#' * int(min(40, 4 / max(v, 0.1)))}")
print(f"\n-> knobs the GP calls irrelevant and will not move: {main.irrelevant()}")
x = apply_fixed(np.array([[650, 50, 0.5, 20, 3, 0, 0, 0, 0]], float))   # OxyT, OxyBias, N2frac, OxyP, OxyTime, context...
m, s = main.predict(x)
print(f"\nPrediction at an untried recipe OxyT=650, OxyTime=3 (others standard): FWHM = {m[0]:.3f} +/- {s[0]:.3f}")

# --------------------------------------------------------------------------- 4
banner("STEP 4  The incumbent: what EI has to beat (aln_bo/acquisition.py)")
print(f"best FWHM in the whole log         : {main.Y.min():.4f}  (run {main.best_so_far()['run']:g}, GrowthT={main.best_so_far()['GrowthT']:g} min)")
bic = main.best_in_context()
print(f"best FWHM at the fixed context     : {bic[MAIN.name]:.4f}  (run {bic['run']:g}, OxyTime={bic['OxyTime']:g} min)")
print(f"EI incumbent (GP mean, in context) : {main.incumbent():.4f}")
print("\nA 166-min film is thicker and gives a narrower peak; no 13-min recipe can beat it, so if EI")
print("used it as the target, EI would be ~0 everywhere. The incumbent is therefore the best GP")
print("mean among the observed recipes moved to the standard context (a 'plug-in' incumbent).")

# --------------------------------------------------------------------------- 5
banner("STEP 5  Acquisition: Expected Improvement over a candidate pool")
pool = np.unique(main._pin(sobol_pool(4096)), axis=0)
mp, sp = main.predict(pool)
ei = expected_improvement(mp, sp, main.incumbent(), True)
top = np.argsort(ei)[::-1][:6]
tbl = pd.DataFrame(pool[top], columns=PARAM_NAMES)[FREE_NAMES]
tbl["pred_FWHM"] = mp[top]; tbl["std"] = sp[top]; tbl["EI"] = ei[top]
print(tbl.round(4).to_string(index=False))
print("\nEI = (incumbent - mean) * Phi(z) + std * phi(z): rewards a low predicted FWHM AND a large std.")
print("The best pool points are then polished with L-BFGS-B and snapped to the instrument grid.")

# --------------------------------------------------------------------------- 6
banner("STEP 6  The suggestion and the alternatives (aln_bo/campaign.py)")
sug = main.suggest(q=1)
alt = main.alternatives(n=4)
print("suggested next sample:\n" + sug[FREE_NAMES + ["pred_FWHM", "pred_FWHM_std", "EI"]].round(4).to_string(index=False))
print("\ndistinct alternatives:\n" + alt[FREE_NAMES + ["pred_FWHM", "pred_FWHM_std", "EI"]].round(4).to_string(index=False))

# --------------------------------------------------------------------------- 7
banner("STEP 7  Auxiliary GPs (prediction only): what else would that recipe give?")
aux = AuxModels(ds).fit()
for t in AUX:
    print(f"  {t.name:<13} {aux.n_rows[t.name]:3d} runs   observed {aux.best_observed().get(t.name)}")
print("\nat the suggested recipe:")
print(aux.predict(sug[PARAM_NAMES].to_numpy()).round(3).T.to_string(header=False))
print("\nThese never drive the acquisition. They go to the LLM so it can judge the suggestion with the")
print("interlayer thickness/roughness and AlN density in view (e.g. reject a recipe predicted to roughen).")

# --------------------------------------------------------------------------- 8
banner("STEP 8  LLM review (aln_bo/llm_review.py)")
print("run_pipeline.py bundles: suggestion + alternatives (+ aux predictions), lengthscales, the best runs,")
print("the whole cleaned table and dashboard.png into a packet -> Claude answers accept/reject + recipe + reason.")
print("Without credentials the same packet is written to results/<ts>/llm_prompt.md to paste into any chat model.")
banner("DONE")
