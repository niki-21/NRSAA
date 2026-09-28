"""Maintained adaptation of the recovered lstm2.py experiment.

The LSTM(32) -> Dropout(0.1) -> Dense(32) -> Dense(16) trunk is preserved.
Input now has a true temporal axis and three channels. Nine sigmoid outputs
replace the original four-class softmax to support the final report's labels.
This is not the missing final SE-ResNet implementation.

Upstream MIT notice: third_party/atrial_fibrillation_MIT.txt.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .dataset import read_dataset
from .labels import LABELS
from .model import evaluate


def build_lstm(seed: int = 42):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ValueError('Install the neural extra: pip install ".[neural]"') from exc
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(5000, 3)),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(len(LABELS), activation="sigmoid"),
        ],
        name="three_lead_lstm",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def train_lstm(
    dataset_path: Path,
    output: Path,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    allow_record_split: bool = False,
) -> dict:
    data = read_dataset(dataset_path)
    if "signals" not in data:
        raise ValueError(
            "Prepare the dataset with --include-signals for neural training"
        )
    if data["metadata"]["grouping"] != "patient" and not allow_record_split:
        raise ValueError(
            "Patient groups are required unless --allow-record-split is explicit"
        )
    if len(np.unique(data["groups"])) < 10:
        raise ValueError(
            "At least ten independent groups are required for train/validation/test splits"
        )
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    # Build only after validation so malformed input cannot initiate training.
    model = build_lstm(seed)
    from tensorflow import keras

    train_val, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            data["signals"], groups=data["groups"]
        )
    )
    train_local, val_local = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            train_val, groups=data["groups"][train_val]
        )
    )
    train, val = train_val[train_local], train_val[val_local]
    # Per-lead normalization fitted on the training partition only.
    mean = data["signals"][train].mean(axis=(0, 1), keepdims=True)
    scale = data["signals"][train].std(axis=(0, 1), keepdims=True)
    scale = np.maximum(scale, 1e-6)
    X = (data["signals"] - mean) / scale
    output.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        X[train],
        data["y"][train],
        validation_data=(X[val], data["y"][val]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
    )
    probabilities = model.predict(X[test], verbose=0)
    metrics = evaluate(data["y"][test], probabilities)
    metrics.update(
        {
            "model": "recovered-lstm-adaptation",
            "seed": seed,
            "train_records": len(train),
            "validation_records": len(val),
            "test_records": len(test),
            "grouping": data["metadata"]["grouping"],
            "epochs_run": len(history.history["loss"]),
        }
    )
    model.save(output / "model.keras")
    np.savez_compressed(
        output / "normalization.npz",
        mean=mean,
        scale=scale,
        label_names=np.asarray(LABELS),
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    (output / "history.json").write_text(json.dumps(history.history, indent=2) + "\n")
    (output / "split.json").write_text(
        json.dumps(
            {
                name: {
                    "records": data["record_ids"][indices].tolist(),
                    "groups": np.unique(data["groups"][indices]).tolist(),
                }
                for name, indices in [
                    ("train", train),
                    ("validation", val),
                    ("test", test),
                ]
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        output / "predictions.npz",
        record_ids=data["record_ids"][test],
        y_true=data["y"][test],
        probabilities=probabilities,
        label_names=np.asarray(LABELS),
    )
    return metrics


def predict_lstm(run: Path, header_path: Path) -> dict:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ValueError('Install the neural extra: pip install ".[neural]"') from exc
    from .data import load_recording
    from .signal import prepare_signal

    with np.load(run / "normalization.npz", allow_pickle=False) as archive:
        mean, scale = archive["mean"], archive["scale"]
        if tuple(archive["label_names"]) != LABELS:
            raise ValueError("Unexpected label order")
    if (
        mean.shape != (1, 1, 3)
        or scale.shape != (1, 1, 3)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Invalid normalization parameters")
    model = keras.models.load_model(run / "model.keras", compile=False, safe_mode=True)
    header, raw = load_recording(header_path)
    signal = (prepare_signal(raw, header.sample_rate)[None] - mean) / scale
    scores = model(signal, training=False).numpy()[0]
    if scores.shape != (len(LABELS),) or not np.isfinite(scores).all():
        raise ValueError("Invalid model output")
    return {
        "record_id": header.record_id,
        "threshold": 0.5,
        "probabilities": dict(zip(LABELS, map(float, scores))),
        "labels": [label for label, value in zip(LABELS, scores) if value >= 0.5],
    }
