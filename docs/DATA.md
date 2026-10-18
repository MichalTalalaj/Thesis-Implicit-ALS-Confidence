# Dataset and artifact inventory

Inventory updated on 2026-10-18, including the newly supplied raw dataset. The Git publication
excludes the data CSVs and two original-ID per-user result exports; their absence
from a clone is intentional. They have not been deleted from the working copy.
Sizes are decimal MB, rounded from observed byte sizes.

## Source and acquisition

The thesis's data chapter attributes the Steam user libraries to Julian McAuley's
UCSD collection. The [upstream Steam section](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)
lists **Version 1: User and Item Data**, as well as separate review and metadata
downloads (source page checked on 2026-10-18). Obtain source data through that
page and follow its citation guidance.

On 2026-10-18 the user supplied the decompressed
`australian_users_items.json`. The official Steam section links this file as
**Version 1: User and Item Data**. It contains one Python dictionary literal per
line with a nested `items` list. The importer uses `ast.literal_eval`, supports
plain/gzip inputs, and writes `user_id,item_id,item_name,playtime_forever` in
record/item order, keeping duplicates and zero-playtime observations.

[Dataset requirements](../data/requirements.json) record the download URL, local
path, byte size and SHA-256 of the supplied decompressed file. This checksum is
local evidence, not an upstream-published authenticity checksum. The raw source
and CSVs remain excluded from Git. The game-metadata conversion is still missing;
the user/item importer does not reconstruct `steam_games_final.csv` or reviews.

Use `python -m steam_recommender.prepare_data --output-dir .runs/reconstructed-data`
to write separate `raw/` and `processed/` outputs. The command refuses existing
CSV destinations. On a fresh checkout omit the override to populate `data/`.
Filtering uses the original pandas CSV dtype inference, and the train/test split
reuses the original fraction 0.1, seed 42 implementation. CSV line endings may
differ by platform; verification compares decoded fields and row order.

## Inputs and prepared data

| Relative path | MB | Observed header / purpose | Producer or consumer |
| --- | ---: | --- | --- |
| `data/raw/australian_users_items.json` | 553.09 | Nested user/item dictionary literals | Required raw source; optional `.json.gz` input supported |
| `data/raw/users_items.csv` | 230.24 | `user_id,item_id,item_name,playtime_forever`; playtime in minutes | Reconstructed by `steam_recommender.prepare_data`; input to preparation, raw EDA, and literature comparisons |
| `data/raw/steam_games_final.csv` | 21.13 | `genres,app_name,tags,price,id,About the game` | Game labels in filtered EDA and prices in Bayram comparison; original converter missing |
| `data/raw/steam_reviews_filtered_columns.csv` | 121.64 | `username,hours,product_id,user_id` | Auxiliary data retained; no reader found in the inspected notebooks/module |
| `data/processed/users_items_filtered.csv` | 229.33 | Same four columns as `users_items.csv`; full filtered interactions | Read by ALS tuning/EDA; now exported by the reconstruction command |
| `data/processed/users_items_filtered_train_val.csv` | 205.13 | Same four columns | Written by preparation; input to tuning, evaluation, validation, sensitivity, BPR, revised diversity |
| `data/processed/users_items_filtered_test.csv` | 24.20 | Same four columns | Written by preparation; final ALS/BPR and revised diversity evaluation |
| `data/processed/cold_start_users.csv` | 0.75 | Same four columns | Written by preparation; retained excluded-user interactions, also read by ALS tuning |

The historical filter clips playtime at 120,000 minutes, retains games with
at least five interaction rows, then users with at least five retained rows.
This is sequential, not iterative k-core filtering. It does not count distinct
users/games or deduplicate. The supplied raw CSV has 59,127 repeated user–game
rows. Counting distinct users per game and distinct games per user instead would
retain 5,132,520 rows rather than 5,132,610 (90 fewer, even when duplicate output
rows remain).
Zero-playtime ownership records remain positive observations. The test split uses per-user
shuffle with seed 42 and `ceil(0.1 * n)` test records. Validation is split later
from the remaining interactions. Preserve pandas' existing inferred types and
row ordering; stored outputs show mixed-type warnings for `user_id`.

The thesis and stored preparation output report 5,153,209 raw interaction rows,
5,132,610 filtered interactions, 4,591,379 train/validation records, 541,231 test
records, and 16,902 cold-start records. The filtered population is reported as
62,944 users and 9,198 games. These counts were read from the thesis/artifacts,
not recomputed during the initial cleanup. The raw-data follow-up independently
checks raw/filtered/cold-start row contents and counts; it does not execute the
historical pandas train/test split in this environment.

The notebook still exports train/validation, test and cold-start CSVs. The new
reconstruction command also exports the full filtered CSV, before shuffling for
the test split. The function shared with the notebook preserves every original
filtering expression, its order and indices. Python runtime checks remain blocked
by interpreter access denial; full reconstruction/split comparison still needs
an accessible Python environment.

The independent full-file comparison passed: all 5,153,209 raw rows, 5,132,610
filtered rows and 16,902 cold-start rows match the existing CSV fields and order.
This confirms the input
conversion and historical cap/item/user rule; it does not claim the newly written
Python command or train/test split was executed here.

## Results and other artifacts

| Location | Meaning and publication treatment |
| --- | --- |
| `results/optuna/als_studies.db`, `results/bpr/optuna/bpr_studies.db` | Existing optimization study state, retained; these are not trained model weights |
| `results/optuna/*.csv`, `results/bpr/optuna/*.csv` | Trial exports and chosen ALS settings, retained |
| `results/final/per_user_ndcg.csv` | 127.92 MB, `model,function,function_label,seed,user_id,ndcg@10`; ignored because of size and original identifiers; needed for paired statistics |
| `results/bpr/final/bpr_per_user_ndcg.csv` | 16.26 MB, same header; ignored because it retains original identifiers |
| Other `results/final/` and `results/bpr/final/` CSVs | Aggregate/per-seed results and statistical tables, retained |
| `results/validation/`, `results/sensitivity/` | Aggregate validation splits and confidence-parameter sweeps, retained |
| `results/literature/results*/` | Comparisons and per-user metrics using mapped `user_idx`; retained as research artifacts, not claimed anonymous |
| `results/figures/`, `results/figures_revised/`, `results/eda/figures*/` | Deliberate plots, retained |
| `thesis/figures/`, `thesis/tables/` | Unique thesis assets; byte-identical figures are shared from `results/` through relative LaTeX paths |
| `thesis/` and `presentation/` PDFs | Finished thesis, included title page and slides; exact duplicate PDF copies removed |

Early experiments and their associated results were removed from the publication
tree and remain in the owner's excluded local archive. No retained input dataset
or scientific result was regenerated during cleanup.

No separate trained-weight or prediction archive was found. The later refined
log-offset experiment refers to additional output CSVs that are not present in
the supplied tree. Do not treat those absent files as published results.

Original user identifiers also occur in stored notebook previews. Their stored outputs remain intact; source paths and reusable code were refactored. Before a public
upload, the owner must review notebook outputs, per-user artifacts, and document
personal details. `.gitignore` does not sanitize embedded output or untrack files
that were already committed elsewhere.
