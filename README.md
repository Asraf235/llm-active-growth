# aln-bo: Bayesian optimisation of growth recipes, with a campaign database and surrogate cross-checks

`aln-bo` proposes the next growth run from the runs you have already made. It fits a Gaussian process (GP)
to your measured objective, maximises Expected Improvement over the knobs you allow it to move, holds the
rest of the recipe fixed ("context"), draws the maps, and writes a packet an LLM can review. Everything
lives in one local SQLite file that can hold several experiments (sputtering, PLD, CVD ...), each with its
own parameter definition, so a coding agent (Claude Code, Codex) can operate the loop from plain requests.

It was built on an AlN seed-layer campaign (plasma oxynitridation of SiO2 before AlN sputtering), which
ships as the worked example, and follows the GP-BO stack of [alchemy](https://github.com/sumner-harris/alchemy)
(BoTorch GP, qLogEI) with a scikit-learn implementation of the same model that runs anywhere.
The campaign-specific notes that used to be this README are in [docs/aln_campaign_notes.md](docs/aln_campaign_notes.md).

## Install

```bash
pip install -e .                      # light tier: GP-BO with scikit-learn, random-forest cross-check, maps
pip install -e ".[botorch]"           # + BoTorch backend (torch)
pip install -e ".[bnn]"               # + NeuroBayes Bayesian neural network (JAX)
pip install -e ".[alchemy]"           # + alchemy's own surrogates and plotting functions (from GitHub)
pip install -e ".[llm]"               # + the API review step via Anthropic (ANTHROPIC_API_KEY); ".[llm-openai]" for OpenAI
```

Python 3.11 or newer. By default the Bayesian neural network drives the acquisition with the GP and the random
forest as cross-checks, so install the `[bnn]` extra; on the light tier alone, pass `--surrogate gp` and the BNN
figures are skipped. Footnote for Windows machines with Smart App Control switched on: it blocks torch's and
JAX's unsigned DLLs, so run those tiers in WSL.

## Five minutes on synthetic data

```bash
python scripts/make_synthetic_data.py --n 30 --out examples/synthetic/runs.csv
aln-bo init     demo.sqlite
aln-bo register demo.sqlite examples/aln_sputter/spec.json
aln-bo import   demo.sqlite aln_sputter_sioxny examples/synthetic/runs.csv
aln-bo step     demo.sqlite aln_sputter_sioxny --plan --name step1
aln-bo record   demo.sqlite aln_sputter_sioxny 31 FWHM=0.29
aln-bo show     demo.sqlite aln_sputter_sioxny
aln-bo export   demo.sqlite aln_sputter_sioxny demo.xlsx
```

`step` prints the suggested recipe (in the units you set on the tool), its predicted value and uncertainty,
the best run so far, and the knobs the GP finds irrelevant; it writes the figures and the LLM packet to
`results/<experiment>/step1/` and logs the suggestion in the database. `--plan` inserts it as a planned run
so the result can be recorded against it. (Without `pip install`, replace `aln-bo` by `python -m aln_bo.cli`.)

## Your own process

1. **Write a spec** (JSON; copy `examples/aln_sputter/spec.json`). For every knob: name, bounds, unit, instrument
   step, and `fixed` if it is a context knob the optimiser must not move. Optional `derive` expressions build
   parameters from raw knobs (a gas ratio, 1/time) and `physical_from_derived` maps them back. For every target:
   name, unit, minimise or maximise, and its role (`main`, `alt`, `aux`). `llm_context` holds the physics notes
   the reviewer may use.
2. **Register and import.** `aln-bo register DB spec.json`, then `aln-bo import DB EXPERIMENT runs.csv` with one
   row per run (columns = raw knob names and target names of the spec; repeated run numbers are repeat measurements).
   Your spreadsheet is never written to; `aln-bo export` produces spreadsheets from the database.
3. **Select runs deliberately.** `aln-bo select DB EXPERIMENT NAME selection.csv` saves a named list with a reason per
   run (`GPBO` = INCLUDE / OPTIONAL / EXCLUDE). Fits use a named selection, so every suggestion is reproducible.
4. **Loop.** `step` → grow → `record` → `step` ...

### Why context parameters

A knob that varied in your history and affects the measurement, but that you do not want to optimise (film
thickness, a substrate batch, a working pressure), is declared `fixed`. The GP learns its effect from the
history, and suggestions, maps and the "best run so far" are all evaluated at the fixed value. Without this,
a run that happened to be thicker would win every comparison. Two details that follow from it: the EI
incumbent is the best GP mean among the observed recipes moved to the fixed context, and knobs whose
lengthscale sits at the bound are pinned to the best in-context run instead of being set arbitrarily.

### Physics-informed parameters

Use `derive` when the raw knobs are not the natural variables: in the example the Ar and N2 flows enter as
one chemistry knob (N2 fraction) plus the total flow as context, and the AlN growth time enters as 1/t because
the peak width scales with 1/thickness ([docs/growth_time_vs_inverse.png](docs/growth_time_vs_inverse.png)).
Both removed lengthscale pathologies that the raw variables caused.

## What is in the package

| module | role |
|---|---|
| `config.py` | parameter space and targets, loaded from a spec (`configure`); the AlN example is the default |
| `store.py` | SQLite: experiment registry, per-experiment runs and measurements, named selections, suggestions, timestamps, import/export |
| `data.py` | `Dataset`, flat-file loader, run selection; the two-sheet AlN log reader as an example importer |
| `surrogate.py`, `backends/` | the GP (scikit-learn, BoTorch), `--kernel rbf|matern`, ARD lengthscales, learned noise |
| `acquisition.py` | EI and LCB, the in-context incumbent, Sobol pool + L-BFGS-B polish + grid snapping, kriging believer |
| `campaign.py` | `MainCampaign` (fit, suggest, alternatives, pinning), `AuxModels` (prediction-only GPs) |
| `pipeline.py` | `run_step()`: one BO step with maps and the LLM packet (used by the CLI and the scripts) |
| `validation.py` | NeuroBayes BNN and random forest: drivers of the acquisition or cross-checks |
| `alchemy_plots.py` | the figures: alchemy's projection layouts (mean, variance, acquisition with the suggested run starred), for any surrogate |
| `llm_review.py` | the Claude review: packet, prompt built from the spec, JSON verdict |
| `cli.py` | the `aln-bo` command |

Scripts: `run_pipeline.py` (a step on a file, no database), `alchemy_step.py` and `alchemy_plot_surrogate.py` (the same
step and figures made with alchemy's own surrogate classes and functions; need the `[alchemy]` extra),
`walkthrough.py` (a printed teaching run), `make_synthetic_data.py`. Retired figure code is in `attic/`.

Agents: `AGENTS.md` / `CLAUDE.md` describe the commands, the spec and the rules for an agent operating the loop.

## Reading the output

- **Figures.** Every step draws, in alchemy's projection layouts, the driving surrogate's mean and variance over every
  pair of free knobs (averaged over the others, context fixed), its acquisition with the suggested run as a red star
  (the only starred figure), and the mean and variance of the other two of GP / random forest / BNN on the same runs.
  Agreement between the three model families means the shape is a property of the data, not of a kernel.
- **Lengthscales.** Small = influential; at the bound = irrelevant (and pinned).
- **A flat GP** (noise equal to the data spread, tiny std everywhere) means the data do not support a smooth
  trend: look for outlier runs or a one-sided target distribution before trusting it. The AlN example shows
  this on the rocking curve: three outlier runs collapse the GP, excluding them restores it.
- **The LLM review.** When the package is driven from Claude Code or Codex, the agent reads `llm_prompt.md` and is
  the reviewer; no API key is needed. `--llm --llm-provider anthropic|openai` instead calls a model API directly
  (default models `claude-opus-5-5` / `gpt-5`), for use outside an agent session; other providers plug in through
  `llm_review.review(call=...)`.

## The AlN example in numbers

60 usable runs in the log, 53 with a 2theta FWHM; 34 at a consistent thin-film AlN recipe form the default
selection, 31 for the rocking curve. GP, BNN and random forest agree that short plasma time gives the narrowest
2theta peak and that 600 to 700 C with low pressure gives the narrowest rocking curve; the two objectives rank the
recipes differently. Details, figures and decks: `results/on INCLUDED runs only/`, `examples/aln_sputter/README.md`,
and the campaign notes in `docs/aln_campaign_notes.md`.

## Tests

```bash
python -m pytest tests -q
```

Synthetic data through the campaign on every available backend, the store round trip (register, import,
select, step, record, export), spec switching, and the real log loader when the workbook is present.

## Layout

```
aln_bo/        the package (see table above)
scripts/       command-line scripts for files and figures
examples/      aln_sputter (spec, importer notes), synthetic (five-minute loop)
data/          run selections for the example, TEMPLATE_runs.csv
docs/          workflow schematic, slice-vs-projection and 1/t figures, campaign notes
tests/
AGENTS.md, CLAUDE.md   instructions for coding agents
```
