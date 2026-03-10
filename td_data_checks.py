#!/usr/bin/env python3
"""Simple checks for the TD1 plantar dataset so beginners can spot issues."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import pandas as pd


def report_missing(df: pd.DataFrame, label: str) -> None:
    total = len(df)
    print(f"\nChecking missing values in {label} ({total} rows)")
    missing_info = []
    for column in df.columns:
        miss = df[column].isna().sum()
        if miss:
            missing_info.append((column, miss))
    if not missing_info:
        print("  no missing values detected")
        return
    for column, miss in missing_info:
        percent = 100 * miss / total if total else 0
        print(f"  {column}: {miss} missing ({percent:.1f}%)")


def describe_numeric(df: pd.DataFrame, label: str) -> list[str]:
    print(f"\nNumeric summary for {label}")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    shown = []
    for column in numeric_cols:
        values = df[column].dropna()
        if not values.size:
            continue
        mean = values.mean()
        std = values.std()
        mn = values.min()
        mx = values.max()
        print(f"  {column}: min={mn:.3f}, max={mx:.3f}, mean={mean:.3f}, std={std:.3f}")
        shown.append(column)
    return shown


def simple_outlier_check(df: pd.DataFrame, columns: list[str]) -> None:
    print("\nLooking for simple outliers using mean ± 4σ")
    for column in columns:
        values = df[column].dropna()
        if len(values) < 5:
            continue
        mean = values.mean()
        std = values.std()
        threshold = 4 * std
        low = mean - threshold
        high = mean + threshold
        outliers = values[(values < low) | (values > high)]
        if not outliers.empty:
            print(f"  {column}: {len(outliers)} points outside [{low:.3f}, {high:.3f}]")


def event_duration_checks(df: pd.DataFrame) -> None:
    print("\nEvent durations")
    start = df["Timestamp Start"].astype(float)
    end = df["Timestamp End"].astype(float)
    duration = end - start
    negative = (duration < 0).sum()
    print(f"  average duration {duration.mean():.2f}s over {len(duration)} events")
    if negative:
        print(f"  {negative} events have negative duration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple data health checks for TD1 datasets")
    parser.add_argument("--root", type=Path, default=Path("."), help="root folder that contains outputs/insoles.csv and outputs/classif.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    insoles_path = args.root / "outputs" / "insoles.csv"
    classif_path = args.root / "outputs" / "classif.csv"
    if not insoles_path.exists() or not classif_path.exists():
        print("Run aggregate_td_data.py first to create outputs/insoles.csv and outputs/classif.csv")
        return

    insoles = pd.read_csv(insoles_path)
    classif = pd.read_csv(classif_path)

    report_missing(insoles, "insoles.csv")
    report_missing(classif, "classif.csv")

    insoles_numeric = describe_numeric(insoles, "insoles.csv")
    simple_outlier_check(insoles, insoles_numeric)

    describe_numeric(classif, "classif.csv")
    event_duration_checks(classif)

    print("\nFinished basic checks")


if __name__ == "__main__":
    main()
