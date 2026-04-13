#!/usr/bin/env python3
"""Train LSTM activity models with subject cross-validation and ensembling.

This script follows the project instructions in data_instruction_LSTM (1).ipynb:
  - windows of plantar signals are fed to a PyTorch LSTM
  - subjects S01-S24 are used for train/validation cross-validation
  - subjects S25-S32 are held out for the final test set
  - one model is trained per fold, then test logits are averaged as an ensemble

By default the repository uses Plantar_activity_reference/. If your project
statement requires Plantar_activity/, pass --plantar-root Plantar_activity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class WindowSample:
    plantar_path: Path
    start_idx: int
    window_len: int
    label: int
    subject: str
    sequence: str
    action_name: str


class WindowedPlantarDataset(Dataset):
    def __init__(
        self,
        samples: list[WindowSample],
        label_to_idx: dict[int, int],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.mean = mean
        self.std = std
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_values(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            df = pd.read_csv(path, sep=";", low_memory=False)
            values = df.drop(columns=["Time"]).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            self._cache[path] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return self._cache[path]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        values = self._load_values(sample.plantar_path)
        x = values[sample.start_idx : sample.start_idx + sample.window_len].astype(np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = self.label_to_idx[sample.label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.input_norm = nn.LayerNorm(n_features)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        lstm_features = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_features * 3),
            nn.Dropout(dropout),
            nn.Linear(lstm_features * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(output), dim=1)
        attention_pool = (output * attention_weights).sum(dim=1)
        mean_pool = output.mean(dim=1)
        max_pool = output.max(dim=1).values
        pooled = torch.cat([attention_pool, mean_pool, max_pool], dim=1)
        return self.classifier(pooled)


def parse_subjects(value: str) -> list[int]:
    subjects: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            subjects.extend(range(int(start), int(end) + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM cross-validation + ensemble for plantar activity recognition.")
    parser.add_argument("--events-root", type=Path, default=Path("Events"))
    parser.add_argument("--plantar-root", type=Path, default=Path("Plantar_activity_reference"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lstm_ensemble"))
    parser.add_argument("--trainval-subjects", default="1-24")
    parser.add_argument("--test-subjects", default="25-32")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=None, help="optional sliding-window stride; default keeps one window per annotation")
    parser.add_argument("--fps", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-files-per-subject", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=None, help="optional readable name for the output run folder")
    parser.add_argument("--flat-output", action="store_true", help="write directly in --output-dir instead of creating a run subfolder")
    parser.add_argument("--no-save-models", action="store_true")
    return parser.parse_args()


def jsonable_config(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{stamp}_lstm"
        f"_h{args.hidden_size}"
        f"_layers{args.num_layers}"
        f"_do{args.dropout:g}"
        f"_lr{args.learning_rate:g}"
        f"_wd{args.weight_decay:g}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_folds{args.folds}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def label_value(value: object) -> int:
    return int(float(value))


def collect_samples(
    subjects: list[int],
    events_root: Path,
    plantar_root: Path,
    duration_s: float,
    fps: int,
    threshold: float,
    stride_s: float | None,
    max_files_per_subject: int | None = None,
) -> tuple[list[WindowSample], dict[int, str], int]:
    samples: list[WindowSample] = []
    class_names: dict[int, str] = {}
    window_len = int(round(duration_s * fps))
    stride = duration_s if stride_s is None else stride_s
    n_features = 0

    for sid in tqdm(subjects, desc=f"Collecting samples from {plantar_root.name}"):
        subject = f"S{sid:02d}"
        ev_dir = events_root / subject
        pl_dir = plantar_root / subject
        if not ev_dir.exists() or not pl_dir.exists():
            continue

        event_files = sorted(ev_dir.rglob("classif.csv"))
        if max_files_per_subject is not None:
            event_files = event_files[:max_files_per_subject]

        for ev_file in event_files:
            sequence = ev_file.parent.name
            plantar_file = pl_dir / sequence / "insoles.csv"
            if not plantar_file.exists():
                continue

            insoles = pd.read_csv(plantar_file, sep=";", low_memory=False)
            times = pd.to_numeric(insoles["Time"], errors="coerce").to_numpy(float)
            if n_features == 0:
                n_features = insoles.drop(columns=["Time"]).shape[1]

            ann = pd.read_csv(ev_file, sep=";", low_memory=False)
            ann["Timestamp Start"] = pd.to_numeric(ann["Timestamp Start"], errors="coerce")
            ann["Timestamp End"] = pd.to_numeric(ann["Timestamp End"], errors="coerce")
            ann = ann.dropna(subset=["Timestamp Start", "Timestamp End", "Class", "Name"])
            if len(ann) <= 2:
                continue

            # The instruction notebook removes first/last T-pose boundary events.
            ann = ann.iloc[1:-1].reset_index(drop=True)
            for idx, row in ann.iterrows():
                start_t = float(row["Timestamp Start"])
                end_t = float(row["Timestamp End"])
                if end_t <= start_t:
                    continue

                label = label_value(row["Class"])
                action_name = str(row["Name"]).strip()
                class_names.setdefault(label, action_name)
                win_start = start_t
                while win_start + duration_s <= end_t + 1e-9:
                    win_end = win_start + duration_s
                    ratio = overlap_len(win_start, win_end, start_t, end_t) / duration_s
                    if ratio >= threshold:
                        start_idx = int(np.searchsorted(times, win_start, side="left"))
                        end_idx = start_idx + window_len
                        if end_idx <= len(times):
                            samples.append(WindowSample(plantar_file, start_idx, window_len, label, subject, sequence, action_name))
                    win_start += stride

    return samples, class_names, n_features


def make_label_mapping(train_samples: list[WindowSample], test_samples: list[WindowSample]) -> dict[int, int]:
    train_labels = {sample.label for sample in train_samples}
    test_labels = {sample.label for sample in test_samples}
    missing = sorted(test_labels - train_labels)
    if missing:
        raise ValueError(f"Test labels absent from train/validation subjects: {missing}")
    return {label: idx for idx, label in enumerate(sorted(train_labels))}


def compute_normalization(samples: list[WindowSample], n_features: int) -> tuple[np.ndarray, np.ndarray]:
    cache: dict[Path, np.ndarray] = {}
    total = 0
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)
    for sample in samples:
        if sample.plantar_path not in cache:
            df = pd.read_csv(sample.plantar_path, sep=";", low_memory=False)
            values = df.drop(columns=["Time"]).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            cache[sample.plantar_path] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        window = cache[sample.plantar_path][sample.start_idx : sample.start_idx + sample.window_len]
        sum_x += window.sum(axis=0)
        sum_x2 += (window * window).sum(axis=0)
        total += window.shape[0]
    mean = sum_x / max(total, 1)
    var = (sum_x2 / max(total, 1)) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def make_loader(
    samples: list[WindowSample],
    label_to_idx: dict[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = WindowedPlantarDataset(samples, label_to_idx, mean, std)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probability = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            loss_sum += loss.item() * y.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            probs.append(probability.cpu().numpy())
    return (
        loss_sum / max(total, 1),
        correct / max(total, 1),
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(probs),
    )


def train_fold(
    fold: int,
    train_samples: list[WindowSample],
    val_samples: list[WindowSample],
    test_samples: list[WindowSample],
    label_to_idx: dict[int, int],
    n_features: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    mean, std = compute_normalization(train_samples, n_features)
    train_loader = make_loader(train_samples, label_to_idx, mean, std, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_samples, label_to_idx, mean, std, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_samples, label_to_idx, mean, std, args.batch_size, False, args.num_workers)

    model = LSTMClassifier(
        n_features=n_features,
        n_classes=len(label_to_idx),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -np.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.max_grad_norm)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"Fold {fold} | epoch {epoch:03d}/{args.epochs} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | val_loss={val_loss:.4f}",
            flush=True,
        )
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Fold {fold} early stopping at epoch {epoch} (best val_acc={best_val_acc:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, y_val, pred_val, _ = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, y_test, pred_test, test_probs = evaluate(model, test_loader, criterion, device)

    return {
        "fold": fold,
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "mean": mean,
        "std": std,
        "history": history,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "y_val": y_val,
        "pred_val": pred_val,
        "y_test": y_test,
        "pred_test": pred_test,
        "test_probs": test_probs,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    base_output_dir = args.output_dir
    if not args.flat_output:
        args.output_dir = base_output_dir / make_run_name(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Output dir: {args.output_dir}")
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable_config(args), handle, indent=2, ensure_ascii=False)

    trainval_subjects = parse_subjects(args.trainval_subjects)
    test_subjects = parse_subjects(args.test_subjects)
    trainval_samples, class_names, n_features = collect_samples(
        trainval_subjects,
        args.events_root,
        args.plantar_root,
        args.duration_s,
        args.fps,
        args.threshold,
        args.stride_s,
        args.max_files_per_subject,
    )
    test_samples, test_class_names, test_n_features = collect_samples(
        test_subjects,
        args.events_root,
        args.plantar_root,
        args.duration_s,
        args.fps,
        args.threshold,
        args.stride_s,
        args.max_files_per_subject,
    )
    class_names.update(test_class_names)
    if test_n_features and test_n_features != n_features:
        raise ValueError(f"Feature mismatch: train={n_features}, test={test_n_features}")
    label_to_idx = make_label_mapping(trainval_samples, test_samples)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    class_display = [class_names.get(idx_to_label[idx], str(idx_to_label[idx])) for idx in range(len(idx_to_label))]

    print(f"Train/val samples: {len(trainval_samples):,}; test samples: {len(test_samples):,}")
    print(f"Features per timestep: {n_features}; classes: {len(label_to_idx)}")

    groups = np.array([sample.subject for sample in trainval_samples])
    splitter = GroupKFold(n_splits=args.folds)
    fold_results: list[dict[str, object]] = []
    ensemble_probs: list[np.ndarray] = []
    y_test_reference: np.ndarray | None = None

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(trainval_samples)), groups=groups), start=1):
        train_samples = [trainval_samples[i] for i in train_idx]
        val_samples = [trainval_samples[i] for i in val_idx]
        print(
            f"\nFold {fold}/{args.folds}: "
            f"train subjects={sorted({s.subject for s in train_samples})}; "
            f"val subjects={sorted({s.subject for s in val_samples})}"
        )
        result = train_fold(fold, train_samples, val_samples, test_samples, label_to_idx, n_features, args, device)
        fold_results.append(result)
        ensemble_probs.append(result["test_probs"])  # type: ignore[arg-type]
        y_test_reference = result["y_test"]  # type: ignore[assignment]
        print(
            f"Fold {fold} summary | val_acc={result['val_accuracy']:.3f} | "
            f"test_acc={result['test_accuracy']:.3f}"
        )

        if not args.no_save_models:
            torch.save(
                {
                    "model_state": result["model_state"],
                    "mean": result["mean"],
                    "std": result["std"],
                    "label_to_idx": label_to_idx,
                    "n_features": n_features,
                    "args": vars(args),
                },
                args.output_dir / f"lstm_fold_{fold}.pt",
            )

    if y_test_reference is None:
        raise RuntimeError("No fold was trained")

    mean_probs = np.mean(np.stack(ensemble_probs, axis=0), axis=0)
    ensemble_pred = mean_probs.argmax(axis=1)
    ensemble_acc = accuracy_score(y_test_reference, ensemble_pred)
    ensemble_balanced = balanced_accuracy_score(y_test_reference, ensemble_pred)
    report = classification_report(y_test_reference, ensemble_pred, target_names=class_display, zero_division=0, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    cm = confusion_matrix(y_test_reference, ensemble_pred)

    metrics = {
        "config": jsonable_config(args),
        "trainval_subjects": trainval_subjects,
        "test_subjects": test_subjects,
        "n_features": n_features,
        "n_classes": len(label_to_idx),
        "trainval_samples": len(trainval_samples),
        "test_samples": len(test_samples),
        "folds": [
            {
                "fold": result["fold"],
                "val_accuracy": result["val_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "history": result["history"],
            }
            for result in fold_results
        ],
        "ensemble_accuracy": ensemble_acc,
        "ensemble_balanced_accuracy": ensemble_balanced,
        "ensemble_macro_f1": macro_f1,
        "ensemble_weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": {str(idx): {"class_id": idx_to_label[idx], "name": class_display[idx]} for idx in range(len(idx_to_label))},
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    pd.DataFrame({"y_true": y_test_reference, "y_pred": ensemble_pred}).to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([row for result in fold_results for row in result["history"]]).to_csv(args.output_dir / "history.csv", index=False)
    joblib.dump({"label_to_idx": label_to_idx, "class_names": class_display}, args.output_dir / "label_mapping.joblib")

    run_summary = {
        "run_dir": str(args.output_dir),
        "accuracy": ensemble_acc,
        "balanced_accuracy": ensemble_balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "folds": args.folds,
        "duration_s": args.duration_s,
        "threshold": args.threshold,
        "device": str(device),
    }
    runs_index = base_output_dir.parent / "lstm_runs.csv"
    index_row = pd.DataFrame([run_summary])
    index_row.to_csv(runs_index, mode="a", index=False, header=not runs_index.exists())

    print("\nEnsemble results:")
    print(f"  accuracy: {ensemble_acc:.3f}")
    print(f"  balanced accuracy: {ensemble_balanced:.3f}")
    print(f"  macro F1: {macro_f1:.3f}")
    print(f"  weighted F1: {weighted_f1:.3f}")
    print(f"  metrics: {args.output_dir / 'metrics.json'}")
    print(f"  config: {args.output_dir / 'config.json'}")
    print(f"  runs index: {runs_index}")


if __name__ == "__main__":
    main()
