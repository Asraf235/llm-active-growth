# The AlN seed-layer campaign (the experiment this package was built on)

`spec.json` is the experiment definition: five free plasma knobs (temperature, bias, N2 fraction, pressure,
time), four context knobs held at the standard AlN recipe (1/growth time at 1/13 min, 1.7 mTorr, stage max,
40 sccm total flow), the main target (rocking-curve FWHM, texture), the alternative (2theta FWHM) and three auxiliary
targets (SiOxNy thickness and roughness, AlN density), plus the physics notes the LLM reviewer may use.

The data live in a two-sheet workbook (not part of this repository); `aln-bo import ... --sputter-log` reads it
with the project's own reader (`aln_bo.data.load_sputter_log`), which joins the process and measurement sheets,
cleans text cells, averages repeat XRD rows and drops the runs the log marks as scrapped. The run selections used
in the campaign (`thin_films`: 34 runs at one AlN recipe; `thin_films_rc`: 31 runs, three rocking-curve outliers
removed) are CSVs with `run, GPBO, reason` columns kept next to the workbook.
Growth time enters the model as 1/t: see [../../docs/growth_time_vs_inverse.png](../../docs/growth_time_vs_inverse.png).

```bash
aln-bo init campaign.sqlite
aln-bo register campaign.sqlite examples/aln_sputter/spec.json
aln-bo import  campaign.sqlite aln_sputter_sioxny "AlN Sputter Log.xlsx" --sputter-log
aln-bo select  campaign.sqlite aln_sputter_sioxny thin_films    data/run_selection.csv
aln-bo select  campaign.sqlite aln_sputter_sioxny thin_films_rc data/run_selection_RC.csv
aln-bo step    campaign.sqlite aln_sputter_sioxny --selection thin_films --name fwhm_step
aln-bo step    campaign.sqlite aln_sputter_sioxny --selection thin_films_rc --target RC_FWHM --name rc_step
```

The same step and figures made with alchemy's own surrogate classes and plotting functions (need the `[alchemy]` extra):

```bash
python scripts/alchemy_step.py runs.csv --runs run_selection_RC.csv --target RC_FWHM --figures --fix-normalize --seed 0
python scripts/alchemy_plot_surrogate.py "AlN Sputter Log.xlsx" --runs run_selection_RC.csv --target RC_FWHM
```
