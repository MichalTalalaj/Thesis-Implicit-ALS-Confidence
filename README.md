# Confidence functions for Steam game recommendations

Research code and Polish bachelor's thesis on how confidence functions affect
implicit-feedback ALS recommendations. The study compares five ALS confidence
functions, Bayesian Personalized Ranking (BPR), and popularity on Steam libraries,
with ranking, coverage, popularity-bias, diversity and sensitivity analyses.

Start with the [thesis PDF](thesis/thesis-pl.pdf) and
[workflow and known issues](docs/WORKFLOW.md). Saved outputs are historical results,
not fresh executions of the refactored notebooks.

## Layout

```text
src/steam_recommender/  ALS/BPR, confidence, splits, metrics, baselines, statistics
notebooks/
  experiments/         preparation, tuning, evaluation, validation, sensitivity
  analysis/            figures, tables, paired statistics
  eda/                 raw and filtered exploration
  literature/          distinct Bayram and Cheuque comparison protocols
data/{raw,processed}/  local CSVs (excluded from Git)
results/               studies, tables and figures; generations kept distinct
thesis/                Polish sources, unique assets, PDFs and template notices
presentation/          defence slides
docs/                  data, environment and workflow guides
```

## Setup

Pins come from readable metadata in the original `recommender` environment; a
clean installation has not been tested. Create a separate environment:

```console
conda env create -f environment.yml
conda activate steam-thesis
python -m pip install --no-deps -e .
python -m pip check
```

Alternatively, in a Python 3.11 virtual environment:

```console
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

These specifications are not transitive or platform-independent locks. See
[environment evidence](docs/ENVIRONMENT.md).

## Data and notebooks

The required dataset is **Steam Version 1: User and Item Data** from the
[UCSD Steam collection](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data).
Its URL, local path and supplied-file checksum are in
[data/requirements.json](data/requirements.json); `requirements.txt` points to this
dataset requirement separately from pip dependencies. Put the decompressed
`australian_users_items.json` in `data/raw/`, or pass the downloaded `.json.gz`
with `--source`. The source file and existing CSVs remain local and ignored.

Reconstruct into a fresh directory to preserve the existing data:

```console
python -m steam_recommender.prepare_data --output-dir .runs/reconstructed-data
```

This writes `raw/users_items.csv` and all four `processed/` CSVs, including the
full filtered dataset. On a fresh checkout, omit `--output-dir` to use `data/`.
Existing CSVs are always refused. The code caps playtime at 120,000 minutes,
filters games below five interaction rows, then users below five remaining rows,
exactly once. Duplicate and zero-playtime rows remain. These historical row counts
differ from distinct-user/game counts; see [data verification](docs/DATA.md).

The test split uses the existing per-user splitter with fraction 0.1 and seed 42.
No new Python dependency is needed. Game metadata for supporting EDA/Bayram
notebooks remains a separate prepared input.

From the repository root, launch `python -m jupyterlab` and use that environment's
kernel. The kernel working directory may be anywhere inside this checkout.
Run the setup cells first: they locate `src/`, verify the package's origin and
reject a cached different checkout. Restart the kernel when switching checkouts.
Library imports run no experiments and write no artifacts.

Read the [workflow guide](docs/WORKFLOW.md) before running experiments. This
research record has execution-history gaps: older evaluators reject the constant
row now present in the configuration table, and tuning reruns append trials.

To rerun stages that consume existing results, copy `results/` and set
`STEAM_RESULTS_DIR` before launching Jupyter. PowerShell example from the root:

```powershell
New-Item -ItemType Directory -Force .runs | Out-Null
Copy-Item -LiteralPath results -Destination .runs/review-results -Recurse
$env:STEAM_RESULTS_DIR = '.runs/review-results'
python -m jupyterlab
```

Choose a new destination per run. The override redirects result reads and writes;
it does not copy dependencies. It does not redirect `data/processed/`. Run data
preparation in a separate checkout with copied inputs to protect original splits.

## Validation status

Source comparisons, artifact hashes, notebook output preservation and file
references were checked during refactoring. An independent full-file comparison
confirmed the raw, filtered and cold-start CSV contents and order. Python
execution was unavailable in the cleanup environment, so imports, numerical
equivalence, the reconstruction command and full experiments remain unverified.

## Thesis and slides

Build from `thesis/` using pdfLaTeX and Biber, with separate build outputs:

```console
mkdir build
pdflatex -interaction=nonstopmode -output-directory=build thesis-pl.tex
biber --input-directory build --output-directory build thesis-pl
pdflatex -interaction=nonstopmode -output-directory=build thesis-pl.tex
pdflatex -interaction=nonstopmode -output-directory=build thesis-pl.tex
```

For slides, create `build/` inside `presentation/` and run pdfLaTeX twice there
with `-output-directory=build prezentacja.tex`. Slide figures use relative thesis
paths. Thesis and slides share figure files under `results/`; keep that directory
alongside the sources when building. Builds were not verified here.
[Thesis notes](thesis/README.md) explain the shared assets and template notices.

No repository-wide license was selected; the original template notices in
`thesis/template/` do not establish a license for the research code or datasets.
