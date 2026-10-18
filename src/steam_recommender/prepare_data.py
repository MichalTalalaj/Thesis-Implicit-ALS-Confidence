"""Reconstruct the thesis interaction CSVs from Steam Version 1 user/item data.

Usage from an editable installation:
    python -m steam_recommender.prepare_data --output-dir .runs/reconstructed-data

No files are downloaded automatically; no existing CSV is overwritten.
"""
import argparse
import ast
import csv
import gzip
from pathlib import Path

from .paths import project_path, project_root

CSV_COLUMNS = ("user_id", "item_id", "item_name", "playtime_forever")


def iter_interactions(source):
    """Yield every item in source order, including duplicates and zero playtime.

    The upstream .json is one Python dictionary literal per line, not a JSON
    array. literal_eval reads literals without executing source code. Both the
    original .json.gz download and the decompressed .json are supported.
    """
    source = Path(source)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Invalid user record on line {line_number}") from error
            if not isinstance(record, dict) or not isinstance(record.get("items"), list):
                raise ValueError(f"Expected a user dictionary with items on line {line_number}")
            user_id = record["user_id"]
            for item in record["items"]:
                yield (user_id, item["item_id"], item["item_name"], item["playtime_forever"])


def flatten_to_csv(source, destination):
    """Write the four historical columns; refuse to overwrite an existing file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for row in iter_interactions(source):
            writer.writerow(row)
            count += 1
    return count


def reconstruct(source, output_dir):
    """Write raw, filtered, cold-start, train/validation and test CSVs.

    Read the flattened CSV with the original pandas defaults before filtering;
    forcing string identifiers here would change historical dtype inference.
    The full filtered export is newly supported; its row order precedes splitting.
    """
    import pandas as pd

    from .preprocessing import prepare_interactions
    from .splitting import train_test_split_first

    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {source}. See data/requirements.json")
    output_dir = Path(output_dir)
    raw_csv = output_dir / "raw/users_items.csv"
    processed_dir = output_dir / "processed"
    names = (
        "users_items_filtered.csv",
        "cold_start_users.csv",
        "users_items_filtered_train_val.csv",
        "users_items_filtered_test.csv",
    )
    targets = [raw_csv, *(processed_dir / name for name in names)]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Choose a fresh --output-dir; existing files: " + ", ".join(existing))

    raw_count = flatten_to_csv(source, raw_csv)
    users_items = pd.read_csv(raw_csv)
    filtered, cold_start = prepare_interactions(users_items)
    train_val, test = train_test_split_first(filtered, test_frac=0.1, seed=42)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(names, (filtered, cold_start, train_val, test)):
        frame.to_csv(processed_dir / name, index=False, mode="x")
    return dict(zip(("raw", "filtered", "cold_start", "train_val", "test"),
                    (raw_count, len(filtered), len(cold_start), len(train_val), len(test))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project_path("data/raw/australian_users_items.json"),
                        help="Downloaded .json.gz or decompressed .json file")
    parser.add_argument("--output-dir", type=Path, default=project_path("data"),
                        help="Parent directory for raw/ and processed/; existing CSVs are refused")
    args = parser.parse_args(argv)
    # Relative command-line paths consistently refer to the project root.
    source = args.source if args.source.is_absolute() else project_root() / args.source
    output = args.output_dir if args.output_dir.is_absolute() else project_root() / args.output_dir
    counts = reconstruct(source, output)
    print(f"Wrote data to {output}")
    for name, count in counts.items():
        print(f"{name}: {count:,} rows")


if __name__ == "__main__":
    main()
