# Local prepared interactions

The historical workflows expect `users_items_filtered.csv`,
`users_items_filtered_train_val.csv`, `users_items_filtered_test.csv`, and
`cold_start_users.csv` here. These CSVs are deliberately excluded from Git.

`python -m steam_recommender.prepare_data` reconstructs all four files from the
raw dataset and refuses existing outputs. Use `--output-dir .runs/reconstructed-data`
to compare against historical files. The notebook uses the same extracted filter
but retains its original three-file export cell. See the
[data inventory](../../docs/DATA.md) and [workflow caveats](../../docs/WORKFLOW.md).
