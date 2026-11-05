# Working with aln-bo (instructions for coding agents)

`aln-bo` runs Gaussian-process Bayesian optimisation (GP-BO) of a thin-film growth recipe from a local
SQLite campaign database. One database holds many experimental growth tools  (sputtering, auto-PLD-RHEED, agentic-CVD, auto-PLD-Raman ...), each with its
own runs table and its own parameter spec. The agent's job is usually: import the user's data once,
run a BO step when asked, record results, and keep the database as the single record.

## Ground rules

- Never write to the user's spreadsheet. Import it (`aln-bo import`); produce spreadsheets with `aln-bo export`.
- Suggestions are made at the experiment's fixed context (e.g. the standard AlN growth time). Do not change
  context values unless the user asks; to make a context knob free, edit the spec (`fixed: null`) and re-register with `--replace`.
- Run selections are named and saved (`aln-bo select`). Never filter runs ad hoc; save the selection with a reason per run.
  A run added after a selection was saved is still used by that selection (only runs it labels EXCLUDE / OPTIONAL are left out).
- Derived parameters (ratios, 1/time) are computed from raw knobs at load time. Store raw knob values only.
- Check the tier before promising a backend: scikit-learn always works; BoTorch needs torch; the BNN (the default driver of
  the acquisition) needs JAX + NeuroBayes, otherwise pass `--surrogate gp`.
  `python -c "import aln_bo; print(aln_bo.available_backends())"` tells you.

## Commands

```
aln-bo init      DB                                     create the database
aln-bo register  DB spec.json [--replace]               add / update an experiment (parameters, targets, physics notes)
aln-bo show      DB [EXPERIMENT]                        list experiments, or runs / selections / suggestions of one
aln-bo import    DB EXPERIMENT runs.csv|.xlsx [--sputter-log]   one row per run; repeated run numbers = repeat measurements
aln-bo select    DB EXPERIMENT NAME selection.csv       columns: run, GPBO (INCLUDE|OPTIONAL|EXCLUDE), reason
aln-bo step      DB EXPERIMENT [--target T] [--selection NAME] [--backend sklearn|botorch] [--kernel rbf|matern]
                 [--surrogate bnn|gp|rf] [--acq ei|lcb] [--q N] [--llm --llm-provider anthropic|openai --model M]
                 [--plan] [--out results] [--name FOLDER]
aln-bo plan      DB EXPERIMENT --suggestion ID         turn a logged suggestion into a planned run (if --plan was not used)
aln-bo record    DB EXPERIMENT RUN target=value ...     enter measured results (also for a planned run)
aln-bo export    DB EXPERIMENT out.xlsx|.csv            runs, measurements, selections, suggestions, spec
```

`step` writes `results/<experiment>/<name>/`: `suggestion.csv`, `alternatives.csv`, the figures (projection
maps: `<SURROGATE>_mean.png` and `<SURROGATE>_variance.png` for the driving surrogate (BNN by default) and for the
two cross-check surrogates among GP / RF / BNN, and `acquisition_<ACQ>_<SURROGATE>.png`, the only figure with the
suggested run as a red star),
`packet.json`, `llm_prompt.md` (and `llm_review.md` with `--llm`), `summary.txt`; and logs the suggestion in the database.
`--plan` also inserts the suggestion as a planned run so results can be recorded against it.

You are the reviewer. After a `step`, read `llm_prompt.md` (system prompt + packet) and answer it yourself:
accept or reject the suggestion, with a recipe and reasons, in the JSON shape the prompt asks for. The `--llm`
flag (a direct API call: `--llm-provider anthropic|openai`, needs that provider's API key) is only for use
outside an agent session.

## The spec (what an experiment is)

JSON with: `name`, `display_name`, `description`; `params` (each: `name`, `column` (heading in the tool's own log; flat
CSV imports use `name`), `low`, `high`, `unit`, `step`,
`fixed` for context knobs, optional `encode` for text categories, optional `derive` expression over raw knobs);
`physical` (raw knob columns that derived parameters use); `physical_from_derived` (how to map a recipe back to the tool);
`targets` (each: `name`, `column`, `unit`, `minimize`, `role` = main | alt | aux); optional `import` block for the
two-sheet AlN log reader; `llm_context` (physics notes the LLM reviewer may use). `examples/aln_sputter/spec.json` is a full example.

## Typical requests and what to do

- "Set up BO for my <process>": write a spec from the user's description (ask for bounds and steps you do not know),
  `register`, `import` their data, `show` it back and confirm the column mapping before any step.
- "Suggest the next run": `step` with the right `--target` and `--selection`; report the recipe in raw tool units
  (the suggestion table carries them), the predicted value with its std, the best run so far, and which knobs the GP
  calls irrelevant. If the suggestion sits on a bound, say so.
- "I grew it, here are the numbers": `record` against the planned run, then `show`.
- "Which runs should the model use?": build a selection CSV with a reason per excluded run and save it; do not delete runs.
- "Compare surrogates": every `step` already draws GP, RF and BNN mean / variance figures for the same runs.

## Reading a result

- `pred_<target>` and `pred_<target>_std`: GP mean and std at the recipe. A std comparable to the data scatter means extrapolation.
- `EI`: expected improvement over the best GP mean at the fixed context. Near zero everywhere means the data do not support a trend.
- Lengthscales: small = influential; at the upper bound (10 in scikit-learn, very large in BoTorch) = irrelevant, and such knobs
  are pinned to the best in-context run instead of being moved.
- A GP whose noise equals the data spread (prediction flat, std tiny) has decided everything is noise; look for outlier runs
  or a one-sided target distribution (consider a log target) before trusting it.
