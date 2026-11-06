# The AlN seed-layer campaign (the experiment this package was built on)

`spec.json` is the experiment definition: five free plasma knobs (temperature, bias, N2 fraction, pressure,
time), four context knobs held at the standard AlN recipe (1/growth time at 1/13 min, 1.7 mTorr, stage max,
40 sccm total flow), the main target (rocking-curve FWHM, texture), the alternative (2theta FWHM) and three secondary
metrics (SiOxNy thickness and roughness, AlN density), plus the physics notes the LLM reviewer may use.

The data live in a two-sheet workbook (not part of this repository); `aln-bo import ... --sputter-log` reads it
with the project's own reader (`aln_bo.data.load_sputter_log`), which joins the process and measurement sheets,
cleans text cells, averages repeat XRD rows and drops the runs the log marks as scrapped. The run selections used
in the campaign (`thin_films`: 34 runs at one AlN recipe; `thin_films_rc`: 31 runs, three rocking-curve outliers
removed) are CSVs with `run, GPBO, reason` columns kept next to the workbook.
Growth time enters the model as 1/t (against t the data are a cliff then a plateau, which broke the RBF lengthscales).

```bash
aln-bo init campaign.sqlite
aln-bo register campaign.sqlite examples/aln_sputter/spec.json
aln-bo import  campaign.sqlite aln_sputter_sioxny "AlN Sputter Log.xlsx" --sputter-log
aln-bo select  campaign.sqlite aln_sputter_sioxny thin_films    run_selection.csv
aln-bo select  campaign.sqlite aln_sputter_sioxny thin_films_rc run_selection_RC.csv
aln-bo step    campaign.sqlite aln_sputter_sioxny --selection thin_films --name fwhm_step
aln-bo step    campaign.sqlite aln_sputter_sioxny --selection thin_films_rc --target RC_FWHM --name rc_step
```

## What one step looks like

`example_step/` is the output of a real step on the 31-run rocking-curve selection, BNN + EI, with the GP and random
forest as cross-checks: the objective maps (`BNN_mean.png`, `BNN_variance.png`, `GP_*.png`, `RF_*.png`), the
acquisition map `acquisition_EI_BNN.png` with the suggested run as the red star, one GP mean / variance pair per
secondary metric (`SiOxNy_thick_GP_*.png`, `SiOxNy_rough_GP_*.png`, `AlN_density_GP_*.png`, fitted on the runs
that have that measurement), and `summary.txt`, `suggestion.csv`, `alternatives.csv`. The run table and the LLM
packet are left out, since they hold the campaign data.
