#!/usr/bin/env python3
"""Aggregate the TD1 plantar and event data into unified CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from textwrap import dedent

import pandas as pd


def collect_dataset(
    root: Path, dataset_name: str, filename: str, verbose: bool
) -> tuple[pd.DataFrame, dict[str, int], list[Path]]:
    dataset_root = root / dataset_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"{dataset_root} does not exist")

    records: list[pd.DataFrame] = []
    missing_sequences: list[Path] = []
    first_columns: list[str] | None = None
    sequences_seen = 0

    subject_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir())
    for subject_dir in subject_dirs:
        sequence_dirs = sorted(d for d in subject_dir.iterdir() if d.is_dir())
        for sequence_dir in sequence_dirs:
            sequences_seen += 1
            csv_path = sequence_dir / filename
            if not csv_path.exists():
                missing_sequences.append(csv_path)
                continue

            if verbose:
                print(f"Reading {csv_path}")

            df = pd.read_csv(csv_path, sep=";", engine="python")
            if first_columns is None:
                first_columns = df.columns.tolist()
            elif df.columns.tolist() != first_columns:
                print(
                    f"Column mismatch in {csv_path},\n" f"  expected {first_columns}\n" f"  got      {df.columns.tolist()}",
                    file=sys.stderr,
                )

            metadata = (
                ("dataset", dataset_name),
                ("sequence", sequence_dir.name),
                ("subject", subject_dir.name),
            )
            for key, value in reversed(metadata):
                df.insert(0, key, value)

            records.append(df)

    if not records:
        raise ValueError(f"No {filename} files were found under {dataset_root}")

    combined = pd.concat(records, ignore_index=True)
    stats = {
        "subjects": len(subject_dirs),
        "sequences_total": sequences_seen,
        "sequences_with_data": len(records),
        "rows": combined.shape[0],
    }
    return combined, stats, missing_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=dedent(
            """\
Aggregate the insoles and classification chunks spread across the TD1 archive
into two flat CSV files that are easier to open in pandas or similar tools.
"""
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="root of the TD1 dataset (contains Events/ and Plantar_activity/)",
    )
    parser.add_argument(
        "--insoles-output",
        type=Path,
        default=Path("insoles.csv"),
        help="path for the merged plantar_activity data",
    )
    parser.add_argument(
        "--classif-output",
        type=Path,
        default=Path("classif.csv"),
        help="path for the merged events annotations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log each file as it is read",
    )
    return parser.parse_args()


def dump_summary(dataset: str, stats: dict[str, int], missing: list[Path]) -> None:
    print(
        f"{dataset}: {stats['subjects']} subjects, {stats['sequences_with_data']}"
        f"/{stats['sequences_total']} sequences with data ({stats['rows']} rows)"
    )
    if missing:
        print(f"  missing {len(missing)} {dataset} files (first example: {missing[0]})")


def main() -> None:
    args = parse_args()

    insoles_df, insoles_stats, insoles_missing = collect_dataset(
        args.root, "Plantar_activity", "insoles.csv", args.verbose
    )
    classif_df, classif_stats, classif_missing = collect_dataset(
        args.root, "Events", "classif.csv", args.verbose
    )

    args.insoles_output.parent.mkdir(parents=True, exist_ok=True)
    args.classif_output.parent.mkdir(parents=True, exist_ok=True)

    insoles_df.to_csv(args.insoles_output, index=False)
    classif_df.to_csv(args.classif_output, index=False)

    print("Written:")
    print(f"  {args.insoles_output}: {insoles_df.shape[0]} rows, {insoles_df.shape[1]} columns")
    print(f"  {args.classif_output}: {classif_df.shape[0]} rows, {classif_df.shape[1]} columns")

    dump_summary("Plantar_activity", insoles_stats, insoles_missing)
    dump_summary("Events", classif_stats, classif_missing)


if __name__ == "__main__":
    main()
