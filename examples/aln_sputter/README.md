# The AlN seed-layer campaign (the experiment this package was built on)

`spec.json` is the experiment definition: five free plasma knobs (temperature, bias, N2 fraction, pressure,
time), four context knobs held at the standard AlN recipe (1/growth time at 1/13 min, 1.7 mTorr, stage max,
40 sccm total flow), the main target (rocking-curve FWHM, texture), the alternative (2theta FWHM) and three auxiliary
targets (SiOxNy thickness and roughness, AlN density), plus the physics notes the LLM reviewer may use.

The data live in a two-sheet workbook; `aln-bo import ... --sputter-log` reads it with the project's own reader
(`aln_bo.data.load_sputter_log`), which joins the process and measurement sheets, cleans text cells, averages
repeat XRD rows and drops the runs the log marks as scrapped. Run selections with reasons are in
`data/run_selection.csv` (2theta FWHM) and `data/run_selection_RC.csv` (rocking curve, three outlier runs removed).

```bash
aln-bo init campaign.sqlite
aln-bo register campaign.sqlite examples/aln_sputter/spec.json
aln-bo import  campaign.sqlite aln_sputter_sioxny "AlN Sputter Log.xlsx" --sputter-log
aln-bo select  campaign.sqlite aln_sputter_sioxny thin_films    data/run_selection.csv
aln-bo select  campaign.sqlite aln_sputter_sioxny thin_films_rc data/run_selection_RC.csv
aln-bo step    campaign.sqlite aln_sputter_sioxny --selection thin_films --name fwhm_step
aln-bo step    campaign.sqlite aln_sputter_sioxny --selection thin_films_rc --target RC_FWHM --name rc_step
```

Paper-style figures (projections, GP / BNN / RF cross-check, regret) and the alchemy-drawn figures:

```bash
python scripts/run_paper_figures.py "AlN Sputter Log.xlsx" --runs data/run_selection.csv
python scripts/alchemy_plot_surrogate.py "AlN Sputter Log.xlsx" --runs data/run_selection_RC.csv --target RC_FWHM   # needs alchemy
```
