# llm-active-growth

Active learning for thin-film growth recipes: Bayesian optimisation with a Bayesian neural network, a Gaussian
process and a random forest as surrogates, a local SQLite campaign database, and an LLM review of every suggestion.
Built for an AlN seed-layer campaign (plasma oxynitridation of SiO2 before AlN sputtering, the shipped example) and
written so that one database can hold several growth tools, each with its own parameter spec. A coding agent
can run the loop from plain requests; `AGENTS.md` tells it how (`CLAUDE.md` imports it for Claude Code).

The GP stack follows [alchemy](https://github.com/sumner-harris/alchemy) (BoTorch GP, qLogEI); the package name
inside Python is `aln_bo`.

## Install

```bash
pip install -e .                 # scikit-learn GP, random forest, figures (runs anywhere)
pip install -e ".[botorch]"      # + BoTorch backend (torch)
pip install -e ".[bnn]"          # + NeuroBayes BNN (JAX), the default driver of the acquisition
pip install -e ".[alchemy]"      # + alchemy's own surrogates and plots
pip install -e ".[llm]"          # + direct API review (Anthropic; ".[llm-openai]" for OpenAI)
```

Python 3.11 or newer. Without the `[bnn]` extra pass `--surrogate gp`. On Windows with Smart App Control on, torch
and JAX do not load; run those tiers in WSL.

## Your own process

1. **Spec.** Copy `examples/aln_sputter/spec.json`: for each knob its bounds, unit, instrument step and `fixed` if it
   is context the optimiser must not move; `derive` expressions for physics-informed parameters (a gas ratio, 1/time);
   for each target whether to minimise and its role (`main`, `alt`, `aux`); `llm_context` for the reviewer.
2. **Register and import.** `aln-bo register DB spec.json`, then `aln-bo import DB EXPERIMENT runs.csv` (one row per
   run; repeated run numbers are repeat measurements). Your spreadsheet is never written to; `aln-bo export` makes one.
3. **Select runs.** `aln-bo select DB EXPERIMENT NAME selection.csv` saves a named list with a reason per run
   (INCLUDE / OPTIONAL / EXCLUDE). Fits use a named selection, so every suggestion is reproducible.
4. **Loop.** `step` → grow → `record` → `step`. `step` prints the suggested recipe in tool units with its predicted
   value and uncertainty, writes the figures and the LLM packet to `results/<experiment>/<timestamp>/`, and logs the
   suggestion in the database. `--plan` inserts it as a planned run so the measured result can be recorded against it.
   Without `pip install`, use `python -m aln_bo.cli`.

## Output of a step

Seven projection maps: mean and variance of the driving surrogate (BNN by default) and of the
two cross-check surrogates (GP, random forest), and the acquisition with the suggested run as a red star. Markers
grow with the number of runs stacked at a setting. Also `suggestion.csv`, `alternatives.csv`, `packet.json`,
`llm_prompt.md` (the review prompt an agent answers; no API key needed) and `summary.txt`.

## Layout

```
aln_bo/       config (spec), store (SQLite), data, surrogate + backends (sklearn, BoTorch), acquisition,
              campaign, validation (BNN, random forest), plots, llm_review, pipeline, cli
scripts/      run_pipeline (a step on a file, no database), alchemy_step, alchemy_plot_surrogate (alchemy's own classes)
examples/     TEMPLATE_runs.csv (import columns), aln_sputter (spec, notes, figures of one real step)
tests/        python -m pytest tests -q  (synthetic data, every available backend)
```

