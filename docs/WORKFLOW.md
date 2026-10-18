# Workflows and scientific exceptions

This guide follows the inspected source, stored outputs and Polish thesis. It
does not claim end-to-end reproduction. The 14 retained notebooks preserve the
stored output records and execution counts of their retained cells. Obsolete
random/popularity baseline and grid-search cells were removed from the tuning
notebook along with the separate legacy notebooks. Old paths
and errors in those outputs document earlier execution, not active references.

## Paths and modules

`steam_recommender.paths.project_root()` resolves the package's checkout.
`project_path("data/processed/...")` locates data; paths starting `results/` honor
`STEAM_RESULTS_DIR`. Relative overrides are relative to the checkout. Notebook
kernels may start at the root or anywhere inside it. Install editable for scripts
outside the repository. There is no fallback to the historical machine directory.

Imports use `steam_recommender.als`, `bpr`, `confidence`, `splitting`, `metrics`,
`baselines`, and `statistics` directly. The import-only facade was removed.
`tuning_als` retains the earlier delta-GAP/diversity behavior. The tuning notebook
still switches to the final class at the original point in its cell sequence.

## Experiments

Notebook paths in this table start with `notebooks/experiments/`.

| Notebook | Inputs | Role / outputs |
| --- | --- | --- |
| `data_preprocessing.ipynb` | `data/raw/users_items.csv` | Clip at 120,000 minutes, item then user filters at five, seed-42 test split; three CSVs in `data/processed/` |
| `implicit_ALS_models_refactored_optuna.ipynb` | Four processed CSVs | Five studies in `results/optuna/`, later refined log-offset search |
| `final_test_evaluation.ipynb` | Train/validation, test, best configurations | Original four-variant ALS/popularity evaluation in `results/final/`; constant row unsupported |
| `validation_stability.ipynb` | Train/validation, configurations | Split seeds 0–4, model seed 0; `results/validation/` and `results/figures/validation/`; same constant-row limitation |
| `constant_confidence_full.ipynb` | Existing results, data, configurations | Constant rows, summaries/statistics, gamma sensitivity, bootstrap; final/validation/sensitivity folders and optional `results/podsumowanie_wynikow.txt` |
| `sensitivity_analysis.ipynb` | Train/validation, configurations | Alpha/beta sweeps in `results/sensitivity/`, plots in `results/figures/` |
| `bpr_experiment.ipynb` | Train/validation, test | Study/trials in `results/bpr/optuna/`; test outputs in `results/bpr/final/` |

The raw-data reconstruction command is now
`python -m steam_recommender.prepare_data --output-dir .runs/reconstructed-data`.
It flattens the supplied Steam Version 1 user/item records, calls the shared
`preprocessing.prepare_interactions` filter and the original seed-42 splitter,
and exports all four processed CSVs. The notebook retains its historical
three-file export cell; the command also writes `users_items_filtered.csv`.
See [dataset requirements and validation](DATA.md).

The full experiment chain still is not a single verified fresh “Run All” workflow.
Four-variant evaluators precede the constant row in `best_als_configs.csv` and
raise on it; the constant supplement assumes earlier tables. No automatic
configuration-row filtering was introduced. Evaluation reruns still require
execution/configuration history or a separately reviewed scientific repair.

## Supporting analysis

| Location | Role |
| --- | --- |
| `notebooks/analysis/statistical_tests.ipynb` | Wilcoxon and shared Holm correction; requires excluded `results/final/per_user_ndcg.csv` |
| `notebooks/analysis/final_test_figures.ipynb` | Earlier summaries and plots in `results/figures/test/` |
| `notebooks/analysis/figures_after_fix.ipynb` | Revised figures/tables in `results/figures_revised/`; some cells retrain for co-occurrence ILD |
| `notebooks/eda/` | Raw/filtered summaries and metadata enrichment; `results/eda/` |
| `notebooks/literature/cheuque_als_replication.ipynb` | Article parameters and Optuna; `results/literature/results/` and `figures/` |
| `notebooks/literature/bayram2021_implicit_als_replication.ipynb` | Paid-game ALS/popularity; `results/literature/results_bayram/` |

Literature protocols retain their own filtering, relevance, splitting and metric
functions. They were not merged with the main experiment. The tuning ALS variant
remains because the four initial Optuna studies use it; later studies use the
final class. Co-occurrence ILD and plotting
helpers stay notebook-local because they depend on notebook-built matrices and
report state; runtime verification was unavailable. The preprocessing filter now
lives in `preprocessing.py`, shared by the notebook and reconstruction command.

Early experiments, their results, the superseded configuration exporter and
refactoring tests/tools are excluded from the publication tree. Exact duplicate
figure copies were consolidated under `results/`; thesis and presentation sources
reference those same files. Unique thesis figures remain under `thesis/figures/`.
Treat published result files as research artifacts: use `STEAM_RESULTS_DIR` for
reruns to avoid changing the figures used by the thesis.

## Known scientific and execution limitations

- `diversity_at_k` uses model-specific latent factors; later cross-model ILD uses
  co-occurrence. The thesis describes the latter single-run measure, while older
  tables retain repeated-run latent diversity. They are not interchangeable.
- Test and validation popularity evaluators keep separate signatures/returns and
  aggregation code. The test split preserves shuffled indices; validation resets
  them. Beta dispatchers differ in validation/coercion and remain separate.
- `test_metrics_summary.csv` and `test_summary.csv` describe different result
  generations; the latter includes constant ALS. Several notebooks overwrite the
  same files. Running older reporting after the supplement can replace newer data.
- Optuna uses `load_if_exists=True`, normally 150 extra trials per invocation;
  the refined log-offset search uses 300. Parallel `n_jobs=6` and numerical threads
  mean seed-based reproducibility asserted in the thesis is not independently
  established by this audit. Settings and randomness were preserved.
- The 5,000 × 62,944 int32 bootstrap array alone needs about 1.26 GB. No full
  notebook or expensive research run was executed.
- The figure notebook's historical BPR `NameError` remains in stored output.
  Replacing notebook `exec` with a module import supplies the class dependencies
  explicitly, removing that import-order failure without changing its mathematics.
  The subsequent historical recovery cell remains.
- Mixed-type ID warnings, broad mapping `dropna()`, metric empty-input behavior,
  ranking/ties and random-number usage were preserved. Missing refined log-offset
  result files were not manufactured from notebook output.

For reruns copy results and set `STEAM_RESULTS_DIR` as in the
[README](../README.md). The new preparation command uses its own `--output-dir`.
The original preparation notebook still requires a separate checkout with copied
inputs to protect historical data; `STEAM_RESULTS_DIR` does not redirect its writes.
