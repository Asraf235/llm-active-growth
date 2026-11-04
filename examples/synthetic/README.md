# Five-minute example on synthetic data

Runs the whole loop without any real data, using the built-in AlN example spec and a generated dataset
with a hidden ground truth.

```bash
python scripts/make_synthetic_data.py --n 30 --out examples/synthetic/runs.csv
aln-bo init examples/synthetic/demo.sqlite
aln-bo register examples/synthetic/demo.sqlite examples/aln_sputter/spec.json
aln-bo import examples/synthetic/demo.sqlite aln_sputter_sioxny examples/synthetic/runs.csv
aln-bo step   examples/synthetic/demo.sqlite aln_sputter_sioxny --backend sklearn --plan --name step1
aln-bo record examples/synthetic/demo.sqlite aln_sputter_sioxny 31 FWHM=0.29
aln-bo show   examples/synthetic/demo.sqlite aln_sputter_sioxny
aln-bo export examples/synthetic/demo.sqlite aln_sputter_sioxny examples/synthetic/demo.xlsx
```

`results/aln_sputter_sioxny/step1/` holds the suggestion, the maps and the LLM packet.

## Starting from a user-style spreadsheet with four runs

`four_runs.xlsx` is what a user's own sheet looks like: one row per run with the raw knobs
(OxyT, OxyBias, OxyAr, OxyN2, OxyP, OxyTime, GrowthT, WorkP, StageMax), the measured targets and notes.
Any surrogate can drive Expected Improvement; the BNN needs the `[bnn]` extra, GP and random forest run anywhere.

```bash
aln-bo init     four_runs.sqlite
aln-bo register four_runs.sqlite examples/aln_sputter/spec.json
aln-bo import   four_runs.sqlite aln_sputter_sioxny examples/synthetic/four_runs.xlsx
aln-bo step     four_runs.sqlite aln_sputter_sioxny --surrogate bnn --acq ei --plan --name bnn_step1   # or --surrogate gp / rf
```

With four runs every surrogate is mostly uncertainty, so the first suggestions explore the edges of the box;
that is expected behaviour of EI, not a fault. The maps narrow down as results are recorded.
The synthetic truth has its FWHM minimum near 35 W bias, 20 mTorr and a 10-minute plasma step, so after a
few recorded steps the suggestions should converge there.
