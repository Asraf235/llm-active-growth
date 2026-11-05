# llm-active-growth

Active learning for thin-film growth recipes. Each step fits a Bayesian neural network, a Gaussian process and a
random forest to the runs so far, proposes the next recipe by expected improvement, and asks an LLM to review it.
All runs, selections and suggestions live in one local SQLite database. The database can hold several growth
tools, each with its own parameter spec. The shipped example is an AlN seed-layer campaign: plasma oxynitridation
of SiO2 before AlN sputtering. A coding agent can run the loop from plain requests; `AGENTS.md` tells it how
(`CLAUDE.md` imports it for Claude Code).

The GP stack follows [alchemy](https://github.com/sumner-harris/alchemy) by Sumner Harris (BoTorch GP, qLogEI); the package name
inside Python is `aln_bo`.    

## Install

```bash
pip install -e .                 # scikit-learn GP, random forest, figures (runs anywhere)
pip install -e ".[botorch]"      # + BoTorch backend (torch)
pip install -e ".[bnn]"          # + NeuroBayes BNN (JAX), the default driver of the acquisition
pip install -e ".[llm]"          # + direct API review (Anthropic; ".[llm-openai]" for OpenAI)
```

Python 3.11 or newer. By default `aln-bo step` lets the Bayesian neural network choose the next recipe, which needs
the `[bnn]` extra (JAX and NeuroBayes). With only the base install, add `--surrogate gp` to the command so the
Gaussian process chooses instead.

## How to use

1. **Spec.** Copy `examples/aln_sputter/spec.json`: for each knob its bounds, unit, instrument step and `fixed` if it
   is context the optimiser must not move; `derive` expressions for physics-informed parameters (a gas ratio, 1/time);
   for each target whether to minimise and its role (`main`, `alt`, `aux`); `llm_context` for the reviewer.
2. **Register and import.** `aln-bo register DB spec.json`, then `aln-bo import DB EXPERIMENT runs.csv`: one row per
   run, columns named as the spec's knobs and targets (`examples/TEMPLATE_runs.csv` has the header), repeated run
   numbers are repeat measurements. Your spreadsheet is never written to; `aln-bo export` makes one.
3. **Select runs.** `aln-bo select DB EXPERIMENT NAME selection.csv` saves a named list with a reason per run
   (INCLUDE / OPTIONAL / EXCLUDE). Fits use a named selection, so every suggestion is reproducible.
4. **Loop.** Suggest, grow, record, suggest again:

   ```bash
   aln-bo step   DB EXPERIMENT --selection NAME --plan          # fit, suggest, draw; the suggestion becomes a planned run
   aln-bo record DB EXPERIMENT RUN RC_FWHM=2.1                  # enter what you measured for that run
   aln-bo step   DB EXPERIMENT --selection NAME --plan          # next suggestion, now including the new run
   ```

   `step` prints the suggested recipe in tool units with its predicted value and uncertainty, writes its outputs to
   `results/<experiment>/<timestamp>/` and logs the suggestion in the database. Options: `--target` (default: the main
   target), `--surrogate bnn|gp|rf`, `--acq ei|lcb`, `--q N` for a batch. A suggestion made without `--plan` can be turned
   into a planned run later with `aln-bo plan DB EXPERIMENT --suggestion ID`. Without `pip install`, use
   `python -m aln_bo.cli` in place of `aln-bo`.

## Output of a step

Seven projection maps: mean and variance of the driving surrogate (BNN by default) and of the two cross-check
surrogates (GP, random forest), and the acquisition with the suggested run as a red star. Markers grow with the
number of runs stacked at a setting. Files: `summary.txt` (what was printed), `suggestion.csv` and
`alternatives.csv` (recipes in tool units with predictions), `cleaned_runs.csv` (the runs the models were fitted on),
`packet.json` and `llm_prompt.md` (the review prompt; a coding agent answers it directly, `--llm` sends it to a model
API instead and adds `llm_review.md`).

## Layout

```
aln_bo/       config (spec), store (SQLite), data, surrogate + backends (sklearn, BoTorch), acquisition,
              campaign, validation (BNN, random forest), plots, llm_review, pipeline, cli
examples/     TEMPLATE_runs.csv (import columns), aln_sputter (spec, notes, figures of one real step)
tests/        python -m pytest tests -q  (synthetic data, every available backend)
AGENTS.md     instructions for a coding agent operating the loop (CLAUDE.md imports it for Claude Code)
```

