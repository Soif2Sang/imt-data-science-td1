#!/usr/bin/env python3
"""Train a K-fold ensemble activity classifier from plantar insole reference data.

The script reads:
  - Plantar_activity_reference/Sxx/Sequence_xx/insoles.csv
  - Events/Sxx/Sequence_xx/classif.csv

It aligns each annotated event with the matching insole signal interval, extracts
statistical features, trains scikit-learn models with K-fold validation, ensembles
their probabilities on a final test split, and reports accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_STATS = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q25",
    "q75",
    "iqr",
    "rms",
    "mad",
    "first",
    "last",
    "delta",
    "abs_delta",
)
DIFF_STATS = ("diff_mean", "diff_std", "diff_rms")
META_FEATURES = ("duration_seconds", "sample_count")


@dataclass(frozen=True)
class SampleMeta:
    subject: str
    sequence: str
    start: float
    end: float
    label: str
    class_id: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a K-fold ensemble plantar activity classifier from reference insole data."
    )
    parser.add_argument("--plantar-root", type=Path, default=Path("Plantar_activity_reference"))
    parser.add_argument("--events-root", type=Path, default=Path("Events"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/plantar_activity_kfold_ensemble_model.joblib"),
    )
    parser.add_argument(
        "--mode",
        choices=("event", "sliding", "sample"),
        default="event",
        help=(
            "event = one feature vector per annotation; sliding = fixed windows inside "
            "annotations; sample = sampled raw rows inside annotations"
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--max-samples-per-event",
        type=int,
        default=12,
        help="only used in sample mode to keep the dataset balanced and fast",
    )
    parser.add_argument(
        "--split",
        choices=("random", "subject", "sequence"),
        default="random",
        help="random is easier and usually higher; subject/sequence are stricter generalization checks",
    )
    parser.add_argument(
        "--model",
        choices=("mlp", "extra_trees", "random_forest", "hist_gradient_boosting", "knn"),
        default="mlp",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0001, help="MLP L2 regularization")
    parser.add_argument("--hidden-layers", default="256,128")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--overfit-gap", type=float, default=0.10)
    parser.add_argument("--val-drop", type=float, default=0.03)
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help='tree models only: "sqrt", "log2", "none", or a float such as 0.3',
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-files", type=int, default=None, help="debug option: limit paired files")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="if >1, train one model per fold on train+val and ensemble them on the final test set",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="turn data leakage / split warnings into hard errors",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def find_pairs(plantar_root: Path, events_root: Path) -> list[tuple[Path, Path, str, str]]:
    pairs: list[tuple[Path, Path, str, str]] = []
    for insoles_path in sorted(plantar_root.glob("S*/Sequence_*/insoles.csv")):
        subject = insoles_path.parts[-3]
        sequence = insoles_path.parts[-2]
        events_path = events_root / subject / sequence / "classif.csv"
        if events_path.exists():
            pairs.append((insoles_path, events_path, subject, sequence))
    return pairs


def make_feature_names(sensor_columns: list[str]) -> list[str]:
    names = [f"{column}__{stat}" for stat in BASE_STATS for column in sensor_columns]
    names.extend(f"{column}__{stat}" for stat in DIFF_STATS for column in sensor_columns)
    names.extend(META_FEATURES)
    return names


def make_sample_feature_names(sensor_columns: list[str]) -> list[str]:
    names = [f"{column}__raw" for column in sensor_columns]
    names.extend(f"{column}__diff_previous" for column in sensor_columns)
    names.extend(("phase_in_event", "duration_seconds", "event_sample_count"))
    return names


def summarize_window(values: np.ndarray, duration_seconds: float, sample_count: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        minimum = np.nanmin(values, axis=0)
        maximum = np.nanmax(values, axis=0)
        median = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        rms = np.sqrt(np.nanmean(values * values, axis=0))
        mad = np.nanmean(np.abs(values - mean), axis=0)

        if len(values) > 1:
            diff = np.diff(values, axis=0)
            diff_mean = np.nanmean(diff, axis=0)
            diff_std = np.nanstd(diff, axis=0)
            diff_rms = np.sqrt(np.nanmean(diff * diff, axis=0))
        else:
            diff_mean = np.full(values.shape[1], np.nan)
            diff_std = np.full(values.shape[1], np.nan)
            diff_rms = np.full(values.shape[1], np.nan)

    first = values[0]
    last = values[-1]
    delta = values[-1] - values[0]
    features = np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            median,
            q25,
            q75,
            q75 - q25,
            rms,
            mad,
            first,
            last,
            delta,
            np.abs(delta),
            diff_mean,
            diff_std,
            diff_rms,
            np.array([duration_seconds, sample_count], dtype=float),
        ]
    )
    return np.where(np.isfinite(features), features, np.nan)


def event_label(row: pd.Series) -> tuple[str, int | None]:
    label = str(row["Name"]).strip()
    class_value = pd.to_numeric(row.get("Class"), errors="coerce")
    class_id = None if pd.isna(class_value) else int(class_value)
    return label, class_id


def add_sample(
    X_rows: list[np.ndarray],
    metas: list[SampleMeta],
    values: np.ndarray,
    times: np.ndarray,
    start: float,
    end: float,
    subject: str,
    sequence: str,
    label: str,
    class_id: int | None,
    min_samples: int,
) -> None:
    left = np.searchsorted(times, start, side="left")
    right = np.searchsorted(times, end, side="right")
    if right - left < min_samples:
        return
    X_rows.append(summarize_window(values[left:right], end - start, right - left))
    metas.append(SampleMeta(subject, sequence, float(start), float(end), label, class_id))


def add_row_samples(
    X_rows: list[np.ndarray],
    metas: list[SampleMeta],
    values: np.ndarray,
    times: np.ndarray,
    start: float,
    end: float,
    subject: str,
    sequence: str,
    label: str,
    class_id: int | None,
    min_samples: int,
    max_samples_per_event: int,
) -> None:
    left = np.searchsorted(times, start, side="left")
    right = np.searchsorted(times, end, side="right")
    event_sample_count = right - left
    if event_sample_count < min_samples:
        return

    take = min(max_samples_per_event, event_sample_count)
    positions = np.linspace(left, right - 1, take, dtype=int)
    for position in np.unique(positions):
        previous = max(left, position - 1)
        raw = values[position]
        diff_previous = raw - values[previous]
        phase = (times[position] - start) / (end - start)
        features = np.concatenate(
            [
                raw,
                diff_previous,
                np.array([phase, end - start, event_sample_count], dtype=float),
            ]
        )
        X_rows.append(np.where(np.isfinite(features), features, np.nan))
        metas.append(
            SampleMeta(subject, sequence, float(times[position]), float(times[position]), label, class_id)
        )


def extract_pair_features(
    insoles_path: Path,
    events_path: Path,
    subject: str,
    sequence: str,
    mode: str,
    window_seconds: float,
    stride_seconds: float,
    min_samples: int,
    max_samples_per_event: int,
    expected_columns: list[str] | None,
) -> tuple[list[np.ndarray], list[SampleMeta], list[str]]:
    insoles = pd.read_csv(insoles_path, sep=";", low_memory=False)
    events = pd.read_csv(events_path, sep=";", low_memory=False)

    if "Time" not in insoles.columns:
        raise ValueError(f"{insoles_path} has no Time column")

    sensor_columns = [column for column in insoles.columns if column != "Time"]
    if expected_columns is None:
        expected_columns = sensor_columns
    else:
        missing = [column for column in expected_columns if column not in insoles.columns]
        if missing:
            raise ValueError(f"{insoles_path} is missing columns: {missing[:5]}")

    times = pd.to_numeric(insoles["Time"], errors="coerce").to_numpy(dtype=float)
    valid_time = np.isfinite(times)
    sensor_frame = insoles.reindex(columns=expected_columns).apply(pd.to_numeric, errors="coerce")
    values = sensor_frame.to_numpy(dtype=np.float32)

    times = times[valid_time]
    values = values[valid_time]
    order = np.argsort(times)
    times = times[order]
    values = values[order]

    events["Timestamp Start"] = pd.to_numeric(events["Timestamp Start"], errors="coerce")
    events["Timestamp End"] = pd.to_numeric(events["Timestamp End"], errors="coerce")
    events = events.dropna(subset=["Timestamp Start", "Timestamp End", "Name"])

    X_rows: list[np.ndarray] = []
    metas: list[SampleMeta] = []

    for _, row in events.iterrows():
        start = float(row["Timestamp Start"])
        end = float(row["Timestamp End"])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue

        label, class_id = event_label(row)
        if mode == "sample":
            add_row_samples(
                X_rows,
                metas,
                values,
                times,
                start,
                end,
                subject,
                sequence,
                label,
                class_id,
                min_samples,
                max_samples_per_event,
            )
            continue

        if mode == "event":
            add_sample(
                X_rows,
                metas,
                values,
                times,
                start,
                end,
                subject,
                sequence,
                label,
                class_id,
                min_samples,
            )
            continue

        if end - start < window_seconds:
            add_sample(
                X_rows,
                metas,
                values,
                times,
                start,
                end,
                subject,
                sequence,
                label,
                class_id,
                min_samples,
            )
            continue

        window_start = start
        while window_start + window_seconds <= end + 1e-9:
            add_sample(
                X_rows,
                metas,
                values,
                times,
                window_start,
                window_start + window_seconds,
                subject,
                sequence,
                label,
                class_id,
                min_samples,
            )
            window_start += stride_seconds

    return X_rows, metas, expected_columns


def build_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[SampleMeta], list[str]]:
    pairs = find_pairs(args.plantar_root, args.events_root)
    if args.max_files is not None:
        pairs = pairs[: args.max_files]
    if not pairs:
        raise FileNotFoundError("No paired insoles.csv/classif.csv files found")

    X_rows: list[np.ndarray] = []
    metas: list[SampleMeta] = []
    sensor_columns: list[str] | None = None

    print(f"Found {len(pairs)} paired files")
    for index, (insoles_path, events_path, subject, sequence) in enumerate(pairs, start=1):
        pair_X, pair_metas, sensor_columns = extract_pair_features(
            insoles_path,
            events_path,
            subject,
            sequence,
            args.mode,
            args.window_seconds,
            args.stride_seconds,
            args.min_samples,
            args.max_samples_per_event,
            sensor_columns,
        )
        X_rows.extend(pair_X)
        metas.extend(pair_metas)
        if index == 1 or index % 25 == 0 or index == len(pairs):
            print(f"  processed {index:>3}/{len(pairs)} files -> {len(X_rows):,} samples")

    if not X_rows or sensor_columns is None:
        raise ValueError("No training samples were extracted")

    X = np.vstack(X_rows)
    y = np.array([meta.label for meta in metas], dtype=object)
    if args.mode == "sample":
        feature_names = make_sample_feature_names(sensor_columns)
    else:
        feature_names = make_feature_names(sensor_columns)
    return X, y, metas, feature_names


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    metas: list[SampleMeta],
    split: str,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    indices = np.arange(len(y))
    if split == "random":
        counts = pd.Series(y).value_counts()
        n_test = int(np.ceil(test_size * len(y)))
        stratify = y if counts.min() >= 2 and n_test >= len(counts) else None
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_val_y = y[train_val_idx]
        val_relative_size = val_size / (1.0 - test_size)
        train_val_counts = pd.Series(train_val_y).value_counts()
        n_val = int(np.ceil(val_relative_size * len(train_val_y)))
        train_val_stratify = (
            train_val_y
            if train_val_counts.min() >= 2 and n_val >= len(train_val_counts)
            else None
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=train_val_stratify,
        )
    else:
        if split == "subject":
            groups = np.array([meta.subject for meta in metas], dtype=object)
        else:
            groups = np.array([f"{meta.subject}/{meta.sequence}" for meta in metas], dtype=object)
        test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(test_splitter.split(X, y, groups=groups))
        val_relative_size = val_size / (1.0 - test_size)
        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_relative_size,
            random_state=random_state + 1,
        )
        train_local_idx, val_local_idx = next(
            val_splitter.split(
                X[train_val_idx],
                y[train_val_idx],
                groups=groups[train_val_idx],
            )
        )
        train_idx = train_val_idx[train_local_idx]
        val_idx = train_val_idx[val_local_idx]

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
        train_idx,
        val_idx,
        test_idx,
    )


def labels_missing_from_train(y_train: np.ndarray, *others: np.ndarray) -> set[str]:
    train_labels = set(map(str, y_train))
    missing: set[str] = set()
    for labels in others:
        missing.update(label for label in map(str, labels) if label not in train_labels)
    return missing


def split_group_values(metas: list[SampleMeta], indices: np.ndarray, split: str) -> set[str]:
    if split == "subject":
        return {metas[index].subject for index in indices}
    if split == "sequence":
        return {f"{metas[index].subject}/{metas[index].sequence}" for index in indices}
    return set()


def validate_split(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    metas: list[SampleMeta],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split: str,
    strict: bool,
) -> None:
    problems: list[str] = []
    train_set = set(map(int, train_idx))
    val_set = set(map(int, val_idx))
    test_set = set(map(int, test_idx))

    if train_set & val_set:
        problems.append("Train and validation indices overlap.")
    if train_set & test_set:
        problems.append("Train and test indices overlap.")
    if val_set & test_set:
        problems.append("Validation and test indices overlap.")

    missing = labels_missing_from_train(y_train, y_val, y_test)
    if missing:
        problems.append(
            "Some validation/test labels are absent from train: "
            + ", ".join(sorted(missing)[:8])
        )

    if split in {"subject", "sequence"}:
        train_groups = split_group_values(metas, train_idx, split)
        val_groups = split_group_values(metas, val_idx, split)
        test_groups = split_group_values(metas, test_idx, split)
        if train_groups & val_groups:
            problems.append(f"{split} groups overlap between train and validation.")
        if train_groups & test_groups:
            problems.append(f"{split} groups overlap between train and test.")
        if val_groups & test_groups:
            problems.append(f"{split} groups overlap between validation and test.")

    print("\nSplit audit:")
    for name, labels in (("train", y_train), ("val", y_val), ("test", y_test)):
        counts = pd.Series(labels).value_counts()
        print(
            f"  {name}: {len(labels):,} samples, {len(counts)} classes, "
            f"min/class={counts.min()}, max/class={counts.max()}"
        )

    if not problems:
        print("  OK: no index leakage detected.")
        return

    for problem in problems:
        print(f"  WARNING: {problem}")
    if strict:
        raise ValueError("Strict split validation failed: " + " ".join(problems))


def make_model(args: argparse.Namespace) -> Pipeline:
    if args.max_features.lower() == "none":
        max_features: str | float | None = None
    elif args.max_features in {"sqrt", "log2"}:
        max_features = args.max_features
    else:
        max_features = float(args.max_features)

    if args.model == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight="balanced",
            max_features=max_features,
            n_jobs=args.n_jobs,
        )
    elif args.model == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight="balanced",
            max_features=max_features,
            n_jobs=args.n_jobs,
        )
    elif args.model == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=args.random_state,
            class_weight="balanced",
        )
    else:
        classifier = KNeighborsClassifier(
            n_neighbors=args.n_neighbors,
            weights="distance",
            n_jobs=args.n_jobs,
        )

    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if args.model == "knn":
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("--hidden-layers must contain at least one integer")
    return layers


def train_mlp_with_validation(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Pipeline, list[dict[str, float]]]:
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_val_prepared = preprocessor.transform(X_val)

    classifier = MLPClassifier(
        hidden_layer_sizes=parse_hidden_layers(args.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        learning_rate_init=args.learning_rate,
        random_state=args.random_state,
        shuffle=False,
        max_iter=1,
    )

    rng = np.random.default_rng(args.random_state)
    classes = np.array(sorted(set(y_train)), dtype=object)
    best_classifier: MLPClassifier | None = None
    best_val_accuracy = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    iteration = 0
    first_batch = True

    for epoch in range(1, args.epochs + 1):
        shuffled = rng.permutation(len(y_train))
        for batch_start in range(0, len(shuffled), args.batch_size):
            batch_idx = shuffled[batch_start : batch_start + args.batch_size]
            if first_batch:
                classifier.partial_fit(
                    X_train_prepared[batch_idx],
                    y_train[batch_idx],
                    classes=classes,
                )
                first_batch = False
            else:
                classifier.partial_fit(X_train_prepared[batch_idx], y_train[batch_idx])
            iteration += 1

        train_accuracy = classifier.score(X_train_prepared, y_train)
        val_accuracy = classifier.score(X_val_prepared, y_val)
        loss = float(getattr(classifier, "loss_", np.nan))
        history.append(
            {
                "epoch": float(epoch),
                "iteration": float(iteration),
                "loss": loss,
                "train_accuracy": float(train_accuracy),
                "val_accuracy": float(val_accuracy),
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | iteration {iteration:>5} | "
            f"loss={loss:.4f} | train_acc={train_accuracy:.3f} | val_acc={val_accuracy:.3f}",
            flush=True,
        )

        if val_accuracy > best_val_accuracy + 1e-4:
            best_val_accuracy = val_accuracy
            best_classifier = copy.deepcopy(classifier)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {epoch} epochs "
                    f"(best val_acc={best_val_accuracy:.3f})",
                    flush=True,
                )
                break

    if best_classifier is None:
        best_classifier = classifier

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", best_classifier),
        ]
    )
    return model, history


def print_top_confusions(y_true: np.ndarray, y_pred: np.ndarray, limit: int = 12) -> None:
    confusions = top_confusions(y_true, y_pred, limit)
    if not confusions:
        print("\nTop confusions: none")
        return
    print("\nTop confusions:")
    for item in confusions:
        print(f"  {item['count']:>4} x true={item['true']!r} predicted={item['predicted']!r}")


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, limit: int = 12) -> list[dict[str, object]]:
    labels = np.array(sorted(set(y_true) | set(y_pred)), dtype=object)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusions: list[dict[str, object]] = []
    for row_index, true_label in enumerate(labels):
        for col_index, pred_label in enumerate(labels):
            if row_index == col_index:
                continue
            count = int(matrix[row_index, col_index])
            if count:
                confusions.append(
                    {
                        "count": count,
                        "true": str(true_label),
                        "predicted": str(pred_label),
                    }
                )
    confusions.sort(key=lambda item: int(item["count"]), reverse=True)
    return confusions[:limit]


def diagnose_overfitting(
    history: list[dict[str, float]],
    test_accuracy: float,
    test_balanced_accuracy: float,
    overfit_gap: float,
    val_drop_threshold: float,
) -> dict[str, object]:
    if not history:
        return {"available": False, "messages": ["No epoch history available for this model."]}

    best = max(history, key=lambda row: row["val_accuracy"])
    final = history[-1]
    best_gap = best["train_accuracy"] - best["val_accuracy"]
    final_gap = final["train_accuracy"] - final["val_accuracy"]
    val_drop = best["val_accuracy"] - final["val_accuracy"]
    test_vs_val_gap = best["val_accuracy"] - test_accuracy

    messages: list[str] = []
    status = "ok"
    if best_gap >= overfit_gap or final_gap >= overfit_gap:
        status = "overfitting_risk"
        messages.append(
            "Overfitting risk: train accuracy is much higher than validation accuracy."
        )
    if val_drop >= val_drop_threshold:
        status = "overfitting_risk"
        messages.append(
            "Validation dropped after the best epoch; early stopping kept the best model."
        )
    if best["train_accuracy"] < 0.70 and best["val_accuracy"] < 0.70:
        status = "underfitting_risk"
        messages.append("Underfitting risk: both train and validation accuracy are low.")
    if abs(test_vs_val_gap) > 0.05:
        messages.append(
            "Test accuracy differs from best validation accuracy by more than 5 points."
        )
    if not messages:
        messages.append("No strong overfitting signal based on the configured thresholds.")

    report = {
        "available": True,
        "status": status,
        "best_epoch": int(best["epoch"]),
        "best_iteration": int(best["iteration"]),
        "best_train_accuracy": best["train_accuracy"],
        "best_val_accuracy": best["val_accuracy"],
        "final_epoch": int(final["epoch"]),
        "final_train_accuracy": final["train_accuracy"],
        "final_val_accuracy": final["val_accuracy"],
        "best_train_val_gap": best_gap,
        "final_train_val_gap": final_gap,
        "validation_drop_after_best": val_drop,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_vs_best_val_gap": test_vs_val_gap,
        "messages": messages,
    }
    return report


def print_diagnostic(report: dict[str, object]) -> None:
    print("\nOverfitting diagnostic:")
    if not report.get("available"):
        print("  " + str(report["messages"][0]))
        return
    print(
        f"  status={report['status']} | best_epoch={report['best_epoch']} | "
        f"best_val_acc={float(report['best_val_accuracy']):.3f} | "
        f"best_train_acc={float(report['best_train_accuracy']):.3f}"
    )
    print(
        f"  best_train_val_gap={float(report['best_train_val_gap']):.3f} | "
        f"final_train_val_gap={float(report['final_train_val_gap']):.3f} | "
        f"val_drop_after_best={float(report['validation_drop_after_best']):.3f}"
    )
    for message in report["messages"]:
        print(f"  - {message}")


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def sidecar_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_audit_files(
    output: Path,
    history: list[dict[str, float]],
    metrics: dict[str, object],
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    if history:
        pd.DataFrame(history).to_csv(sidecar_path(output, "_history.csv"), index=False)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        sidecar_path(output, "_test_predictions.csv"),
        index=False,
    )
    with sidecar_path(output, "_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2, ensure_ascii=False)


def train_estimator(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    label: str,
) -> tuple[Pipeline, list[dict[str, float]], float]:
    print(f"\nTraining {label}: {len(y_train):,} train samples, {len(y_val):,} val samples")
    if args.model == "mlp":
        model, history = train_mlp_with_validation(X_train, X_val, y_train, y_val, args)
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        return model, history, val_accuracy

    model = make_model(args)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    history = [
        {
            "epoch": 1.0,
            "iteration": 1.0,
            "loss": np.nan,
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "val_accuracy": float(val_accuracy),
        }
    ]
    print(f"{label} validation accuracy: {val_accuracy:.3f}")
    return model, history, val_accuracy


def estimator_classes(model: Pipeline) -> np.ndarray:
    classifier = model.named_steps.get("classifier")
    if classifier is not None and hasattr(classifier, "classes_"):
        return np.array(classifier.classes_, dtype=object)
    if hasattr(model, "classes_"):
        return np.array(model.classes_, dtype=object)
    raise AttributeError("Could not find classes_ on fitted estimator")


def predict_proba_aligned(model: Pipeline, X: np.ndarray, class_order: np.ndarray) -> np.ndarray:
    class_to_column = {label: idx for idx, label in enumerate(class_order)}
    aligned = np.zeros((len(X), len(class_order)), dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for source_idx, label in enumerate(estimator_classes(model)):
            if label in class_to_column:
                aligned[:, class_to_column[label]] = proba[:, source_idx]
        return aligned

    pred = model.predict(X)
    for row_idx, label in enumerate(pred):
        aligned[row_idx, class_to_column[label]] = 1.0
    return aligned


def cv_splitter(
    y_train_val: np.ndarray,
    metas: list[SampleMeta],
    train_val_idx: np.ndarray,
    split: str,
    requested_folds: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if requested_folds < 2:
        raise ValueError("--cv-folds must be >= 2")

    if split == "random":
        counts = pd.Series(y_train_val).value_counts()
        n_splits = min(requested_folds, int(counts.min()))
        if n_splits < 2:
            raise ValueError("Not enough samples per class for cross-validation")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(len(y_train_val)), y_train_val))

    if split == "subject":
        groups = np.array([metas[index].subject for index in train_val_idx], dtype=object)
    else:
        groups = np.array([f"{metas[index].subject}/{metas[index].sequence}" for index in train_val_idx], dtype=object)
    n_splits = min(requested_folds, len(set(groups)))
    if n_splits < 2:
        raise ValueError(f"Not enough {split} groups for cross-validation")
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(y_train_val)), y_train_val, groups=groups))


def train_cv_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    metas: list[SampleMeta],
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    (
        X_train_base,
        X_val_base,
        X_test,
        y_train_base,
        y_val_base,
        y_test,
        train_idx,
        val_idx,
        test_idx,
    ) = split_train_val_test(
        X,
        y,
        metas,
        args.split,
        args.val_size,
        args.test_size,
        args.random_state,
    )
    del X_train_base, X_val_base, y_train_base, y_val_base

    train_val_idx = np.concatenate([train_idx, val_idx])
    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    print(
        f"CV ensemble: {len(y_train_val):,} train+val samples; "
        f"{len(y_test):,} final test samples; requested folds={args.cv_folds}"
    )

    folds = cv_splitter(y_train_val, metas, train_val_idx, args.split, args.cv_folds, args.random_state)
    class_order = np.array(sorted(set(y_train_val)), dtype=object)
    test_probabilities: list[np.ndarray] = []
    fold_summaries: list[dict[str, object]] = []
    all_history: list[dict[str, float]] = []
    models: list[Pipeline] = []

    for fold_number, (fold_train_local, fold_val_local) in enumerate(folds, start=1):
        fold_args = copy.copy(args)
        fold_args.random_state = args.random_state + fold_number
        model, history, val_accuracy = train_estimator(
            X_train_val[fold_train_local],
            X_train_val[fold_val_local],
            y_train_val[fold_train_local],
            y_train_val[fold_val_local],
            fold_args,
            label=f"fold {fold_number}/{len(folds)}",
        )
        for row in history:
            row = dict(row)
            row["fold"] = float(fold_number)
            all_history.append(row)
        fold_pred = model.predict(X_train_val[fold_val_local])
        fold_summary = {
            "fold": fold_number,
            "train_samples": int(len(fold_train_local)),
            "val_samples": int(len(fold_val_local)),
            "val_accuracy": float(val_accuracy),
            "val_balanced_accuracy": float(balanced_accuracy_score(y_train_val[fold_val_local], fold_pred)),
        }
        fold_summaries.append(fold_summary)
        test_probabilities.append(predict_proba_aligned(model, X_test, class_order))
        models.append(model)
        print(
            f"Fold {fold_number}/{len(folds)} summary: "
            f"val_acc={fold_summary['val_accuracy']:.3f}, "
            f"val_balanced={fold_summary['val_balanced_accuracy']:.3f}"
        )

    mean_proba = np.mean(np.stack(test_probabilities, axis=0), axis=0)
    y_pred = class_order[mean_proba.argmax(axis=1)]
    accuracy = accuracy_score(y_test, y_pred)
    balanced = balanced_accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    diagnostic = diagnose_overfitting(all_history, accuracy, balanced, args.overfit_gap, args.val_drop)

    metrics: dict[str, object] = {
        "config": serializable_args(args),
        "dataset": {
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "classes": int(len(set(y))),
        },
        "split": {
            "train_val_samples": int(len(y_train_val)),
            "test_samples": int(len(y_test)),
            "train_val_classes": int(len(set(y_train_val))),
            "test_classes": int(len(set(y_test))),
        },
        "cv": {
            "folds": len(folds),
            "fold_summaries": fold_summaries,
            "ensemble_method": "mean_predict_proba",
        },
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "classification_report": report_dict,
        "top_confusions": top_confusions(y_test, y_pred),
        "diagnostic": diagnostic,
    }
    payload = {
        "models": models,
        "feature_names": feature_names,
        "mode": args.mode,
        "split": args.split,
        "train_val_indices": train_val_idx,
        "test_indices": test_idx,
        "metadata": metas,
        "history": all_history,
        "diagnostic": diagnostic,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "cv": metrics["cv"],
    }
    return y_test, y_pred, metrics, payload


def main() -> None:
    args = parse_args()

    X, y, metas, feature_names = build_dataset(args)
    print(f"\nDataset: {X.shape[0]:,} samples, {X.shape[1]:,} features, {len(set(y))} classes")
    print(f"Mode: {args.mode}; split: {args.split}; model: {args.model}")

    if args.val_size + args.test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be < 1")

    if args.cv_folds > 1:
        y_test, y_pred, metrics, payload = train_cv_ensemble(X, y, metas, feature_names, args)
        accuracy = float(metrics["accuracy"])
        balanced = float(metrics["balanced_accuracy"])
        diagnostic = metrics["diagnostic"]

        print(f"\nEnsemble accuracy: {accuracy:.3f}")
        print(f"Ensemble balanced accuracy: {balanced:.3f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print_top_confusions(y_test, y_pred)
        print_diagnostic(diagnostic)  # type: ignore[arg-type]

        if not args.no_save:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, args.output)
            save_audit_files(args.output, payload["history"], metrics, y_test, y_pred)  # type: ignore[arg-type]
            print(f"\nSaved ensemble to: {args.output}")
            print(f"Saved history to: {sidecar_path(args.output, '_history.csv')}")
            print(f"Saved metrics to: {sidecar_path(args.output, '_metrics.json')}")
            print(f"Saved test predictions to: {sidecar_path(args.output, '_test_predictions.csv')}")
        return

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_idx,
        val_idx,
        test_idx,
    ) = split_train_val_test(
        X,
        y,
        metas,
        args.split,
        args.val_size,
        args.test_size,
        args.random_state,
    )
    print(
        f"Train: {len(y_train):,} samples; val: {len(y_val):,} samples; "
        f"test: {len(y_test):,} samples"
    )
    validate_split(
        y_train,
        y_val,
        y_test,
        metas,
        train_idx,
        val_idx,
        test_idx,
        args.split,
        args.strict,
    )

    if args.model == "mlp":
        model, history = train_mlp_with_validation(X_train, X_val, y_train, y_val, args)
    else:
        history = []
        model = make_model(args)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.3f}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced = balanced_accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    diagnostic = diagnose_overfitting(
        history,
        accuracy,
        balanced,
        args.overfit_gap,
        args.val_drop,
    )

    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print_top_confusions(y_test, y_pred)
    print_diagnostic(diagnostic)

    metrics: dict[str, object] = {
        "config": serializable_args(args),
        "dataset": {
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "classes": int(len(set(y))),
        },
        "split": {
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "train_classes": int(len(set(y_train))),
            "val_classes": int(len(set(y_val))),
            "test_classes": int(len(set(y_test))),
        },
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "classification_report": report_dict,
        "top_confusions": top_confusions(y_test, y_pred),
        "diagnostic": diagnostic,
    }

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "feature_names": feature_names,
            "mode": args.mode,
            "split": args.split,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
            "metadata": metas,
            "history": history,
            "diagnostic": diagnostic,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
        }
        joblib.dump(payload, args.output)
        save_audit_files(args.output, history, metrics, y_test, y_pred)
        print(f"\nSaved model to: {args.output}")
        if history:
            print(f"Saved history to: {sidecar_path(args.output, '_history.csv')}")
        print(f"Saved metrics to: {sidecar_path(args.output, '_metrics.json')}")
        print(f"Saved test predictions to: {sidecar_path(args.output, '_test_predictions.csv')}")

    if args.split == "random":
        print(
            "\nNote: this is a unique random Train/Val/Test split. "
            "Use --split subject for the stricter question: generalize to unseen people."
        )


if __name__ == "__main__":
    main()
