"""Build compact, versioned feature datasets with explicit record provenance."""

import json
from pathlib import Path

import numpy as np

from .data import load_recording
from .features import extract_features
from .labels import LABELS, encode_labels
from .signal import prepare_signal

SCHEMA_VERSION = 1


def build_dataset(
    input_dir: Path,
    output: Path,
    group_file: Path | None = None,
    *,
    include_signals: bool = False,
) -> dict:
    """Process all matched records recursively; fail on corrupt data.

    A group file maps relative recording paths (without .hea) to patient IDs.
    If absent, record IDs are retained but training requires explicit consent
    to a record-level split. Records without any target label are counted/skipped.
    """
    paths = sorted(input_dir.rglob("*.hea"))
    if not paths:
        raise ValueError(f"No .hea files found in {input_dir}")
    group_map = json.loads(group_file.read_text()) if group_file else None
    if group_map is not None and not isinstance(group_map, dict):
        raise ValueError(
            "Patient groups must be a JSON object mapping record paths to IDs"
        )
    features, labels, ids, groups, names = [], [], [], [], None
    signals = []
    skipped = 0
    for path in paths:
        header, raw = load_recording(path)
        target = encode_labels(header.codes)
        if not any(target):
            skipped += 1
            continue
        record_id = path.relative_to(input_dir).with_suffix("").as_posix()
        if group_map is not None:
            group = group_map.get(record_id)
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"Missing nonempty patient group for {record_id}")
        else:
            group = record_id
        prepared = prepare_signal(raw, header.sample_rate)
        values = extract_features(prepared, header.age, header.sex)
        if include_signals:
            signals.append(prepared)
        names = tuple(values)
        features.append(list(values.values()))
        labels.append(target)
        ids.append(record_id)
        groups.append(group)
    if not features:
        raise ValueError("No recordings contain a supported diagnosis label")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "records": len(ids),
        "skipped_no_target": skipped,
        "grouping": "patient" if group_map is not None else "record",
        "sample_rate": 500,
        "samples": 5000,
        "denoise": "sym5-detail-soft",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(
            stream,
            X=np.asarray(features),
            y=np.asarray(labels, dtype=np.uint8),
            record_ids=np.asarray(ids),
            groups=np.asarray(groups),
            feature_names=np.asarray(names),
            label_names=np.asarray(LABELS),
            metadata=np.asarray(json.dumps(metadata)),
            **({"signals": np.asarray(signals)} if include_signals else {}),
        )
    return metadata


def read_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "X",
        "y",
        "record_ids",
        "groups",
        "feature_names",
        "label_names",
        "metadata",
    }
    if not required <= data.keys():
        raise ValueError("Incomplete feature dataset")
    metadata = json.loads(str(data["metadata"]))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema")
    X, y = data["X"], data["y"]
    n = len(X)
    if X.ndim != 2 or X.shape[1] != len(data["feature_names"]) or np.isinf(X).any():
        raise ValueError("Invalid feature matrix")
    if y.shape != (n, len(LABELS)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid multilabel target matrix")
    if tuple(data["label_names"]) != LABELS:
        raise ValueError("Label order does not match the current schema")
    if len(set(data["feature_names"])) != len(data["feature_names"]):
        raise ValueError("Duplicate feature names")
    if (
        len(data["record_ids"]) != n
        or len(set(data["record_ids"])) != n
        or len(data["groups"]) != n
    ):
        raise ValueError(
            "Record IDs must be unique and group IDs must align with records"
        )
    if metadata.get("grouping") not in {"patient", "record"} or any(
        not str(g).strip() for g in data["groups"]
    ):
        raise ValueError("Invalid grouping metadata")
    if "signals" in data and (
        data["signals"].shape != (n, 5000, 3) or not np.isfinite(data["signals"]).all()
    ):
        raise ValueError("Invalid prepared signal tensor")
    data["metadata"] = metadata
    return data
