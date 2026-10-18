# Environment evidence

Inspected on 2026-10-17. The original `D:/Anaconda/envs/recommender/conda-meta/`
and distribution metadata under `Lib/site-packages/` were readable. Both base
and recommender Python failed to launch; Conda execution was denied. These
records establish installed versions, not runtime health or the environment of
every historical result. Notebook metadata records Python 3.11.15; the thesis
names implicit 0.7.2 and SciPy 1.17.1.

| Dependency | Version | Observed owner / use |
| --- | --- | --- |
| Python | 3.11.15 | Conda defaults |
| pip | 26.0.1 | Conda defaults |
| numpy | 1.26.4 | Conda defaults; numerics/randomness |
| pandas | 3.0.2 | Conda defaults; dataframes/CSV |
| scipy | 1.17.1 | Conda defaults; sparse matrices, Wilcoxon |
| implicit | 0.7.2 | pip; ALS/BPR |
| optuna | 4.8.0 | pip; optimization/storage |
| matplotlib | 3.10.9 | pip; figures |
| scikit-learn | 1.8.0 | Conda defaults; co-occurrence normalization |
| tqdm | 4.67.3 | pip; progress |
| threadpoolctl | 3.5.0 | Conda defaults; thread limits |
| jupyterlab | 4.5.6 | Conda defaults; notebook interface |
| ipykernel | 7.2.0 | Conda-forge; kernel |

`requirements.txt` records direct imports with observed versions.
`environment.yml` preserves observed Conda/pip ownership and names a separate
`steam-thesis` environment. Neither is a solved transitive lock. The original
environment records MKL; pip pins do not encode its BLAS/OpenMP runtime. Platform
differences and parallel training can affect numerical results.

`pyproject.toml` adds editable packaging. Its unpinned dependency list describes
library imports; use the pinned files for research setup. Setuptools is a new
build-tool dependency without an invented historical version. Standard-library
imports and unrelated installed packages such as scikit-surprise were omitted.

Installation, solving, `pip check`, Python syntax checks, imports, and numerical
tests could not run here. No equivalent-environment guarantee is made. On a host
where the original environment runs, inspect it before changing anything:

```console
conda list -n recommender
conda run -n recommender python --version
conda run -n recommender python -m pip check
```

Then follow the [README](../README.md) to create a separate environment and run
`python -m pip check`. The local refactoring test suite and audit tools are not
part of the publication tree.

The 2026-10-18 raw-data importer adds only standard-library `ast`, `csv`, `gzip`
and `argparse` usage alongside the existing pandas/NumPy split dependency.
The dataset requirement is recorded separately in `data/requirements.json` and
referenced from `requirements.txt`; it is not a pip-installable package.
