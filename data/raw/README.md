# Local raw inputs

Place `australian_users_items.json` here. See [dataset requirements](../requirements.json)
for its download URL, compressed alternative and supplied-file checksum. Run
`python -m steam_recommender.prepare_data --output-dir .runs/reconstructed-data`
from the root to reconstruct interaction CSVs without overwriting historical data.
On a fresh checkout omit the output override to write under `data/`.

`users_items.csv` is the flattened input to preparation and literature comparisons.
`steam_games_final.csv` is additionally needed for filtered EDA and the paid-game
comparison; its converter is still unavailable. The reviews CSV remains auxiliary.

The source JSON and CSVs are deliberately excluded from Git. See the
[data inventory](../../docs/DATA.md) for schemas, source, sizes, and acquisition gaps.
