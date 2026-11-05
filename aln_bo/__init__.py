"""aln_bo: Gaussian-process Bayesian optimisation of growth recipes.

    config      the parameter space and targets, loaded from an experiment spec (config.configure)
    store       the SQLite campaign database: experiments, runs, measurements, selections, suggestions
    data        Dataset + loaders (flat files; the two-sheet AlN log as an example)
    campaign    MainCampaign (GP + acquisition -> suggestion), AuxModels (prediction-only GPs)
    pipeline    run_step(): one BO step with maps and the LLM packet
    validation  NeuroBayes BNN and random forest (cross-checks, or drivers of the acquisition)
    plots       the figures: projection maps (mean, variance, acquisition with the star)
    llm_review  the LLM review of a suggestion (agent-as-reviewer, or Anthropic / OpenAI API)
    cli         `aln-bo` command line

Reading order for newcomers: config -> data -> surrogate -> acquisition -> campaign -> plots -> pipeline.

Backends (aln_bo/backends/): scikit-learn (runs anywhere) and BoTorch (needs torch).
"""
from . import config
from .config import PARAMS, PARAM_NAMES, FREE_NAMES, bounds_array, snap_to_grid, apply_fixed, configure, load_spec
from .data import load_sputter_log, load_dataset, dataset_from_frame, select_runs, parse_run_selection, Dataset
from .surrogate import GP, GPList
from .acquisition import expected_improvement, suggest_ei, suggest_ehvi, top_pool_candidates, is_non_dominated
from .campaign import MainCampaign, AuxModels
from .backends import get_backend, available_backends, resolve_backend_name
from .validation import RandomForest, BNN, bnn_available
from .store import Store
from .pipeline import run_step
from . import plots

__all__ = [
    "config", "PARAMS", "PARAM_NAMES", "FREE_NAMES", "bounds_array", "snap_to_grid", "apply_fixed", "configure", "load_spec",
    "load_sputter_log", "load_dataset", "dataset_from_frame", "select_runs", "parse_run_selection", "Dataset",
    "GP", "GPList", "expected_improvement", "suggest_ei", "suggest_ehvi", "top_pool_candidates", "is_non_dominated",
    "MainCampaign", "AuxModels", "get_backend", "available_backends", "resolve_backend_name",
    "RandomForest", "BNN", "bnn_available", "Store", "run_step", "plots",
]
