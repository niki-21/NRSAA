"""A reproducible multilabel random-forest baseline with group-held-out testing.

This maintained baseline is not the neural network that produced the historical
2024 results. It provides a small, inspectable end-to-end reference pipeline.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from .dataset import read_dataset
from .labels import LABELS


def positive_probabilities(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Handle labels absent (or always positive) in a training partition."""
    probabilities = pipeline.predict_proba(X)
    classes = pipeline.named_steps["classifier"].classes_
    result = np.zeros((len(X), len(classes)))
    for i, (values, labels) in enumerate(zip(probabilities, classes)):
        positive = np.flatnonzero(labels == 1)
        if positive.size:
            result[:, i] = values[:, positive[0]]
    return result


def evaluate(y: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    """Separate exact-match and per-label accuracy; report absent-class support."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one")
    predicted = (probabilities >= threshold).astype(int)
    scores = {
        "threshold": threshold,
        "exact_match_accuracy": accuracy_score(y, predicted),
        "hamming_loss": hamming_loss(y, predicted),
    }
    for average in ("micro", "macro"):
        for name, metric in [
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]:
            scores[f"{name}_{average}"] = metric(
                y, predicted, average=average, zero_division=0
            )
    scores["per_label"] = {}
    for i, label in enumerate(LABELS):
        tn, fp, fn, tp = multilabel_confusion_matrix(y, predicted)[i].ravel()
        scores["per_label"][label] = {
            "support": int(y[:, i].sum()),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "average_precision": float(
                average_precision_score(y[:, i], probabilities[:, i])
            )
            if y[:, i].any()
            else None,
        }
    return scores


def train(
    dataset_path: Path,
    output: Path,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    trees: int = 200,
    allow_record_split: bool = False,
) -> dict:
    data = read_dataset(dataset_path)
    if data["metadata"]["grouping"] != "patient" and not allow_record_split:
        raise ValueError(
            "Supply patient groups during prepare, or explicitly use --allow-record-split for exploratory evaluation"
        )
    if len(np.unique(data["groups"])) < 5:
        raise ValueError("At least five independent groups are required")
    if not 0 < test_size < 1 or trees < 1:
        raise ValueError("test_size must be between 0 and 1 and trees must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve existing runs")
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(split.split(data["X"], data["y"], data["groups"]))
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            (
                "classifier",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=trees,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )
    pipeline.fit(data["X"][train_idx], data["y"][train_idx])
    probabilities = positive_probabilities(pipeline, data["X"][test_idx])
    metrics = evaluate(data["y"][test_idx], probabilities)
    metrics.update(
        {
            "model": "random-forest-baseline",
            "seed": seed,
            "trees": trees,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "grouping": data["metadata"]["grouping"],
            "train_support": dict(
                zip(LABELS, map(int, data["y"][train_idx].sum(axis=0)))
            ),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": data["feature_names"].tolist(),
            "label_names": list(LABELS),
            "schema_version": 1,
        },
        output / "model.joblib",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    (output / "split.json").write_text(
        json.dumps(
            {
                "train": data["record_ids"][train_idx].tolist(),
                "test": data["record_ids"][test_idx].tolist(),
                "train_groups": np.unique(data["groups"][train_idx]).tolist(),
                "test_groups": np.unique(data["groups"][test_idx]).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "predictions.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            record_ids=data["record_ids"][test_idx],
            y_true=data["y"][test_idx],
            probabilities=probabilities,
            label_names=np.asarray(LABELS),
        )
    return metrics
