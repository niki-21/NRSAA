"""Command-line entry points for the maintained ECG pipeline."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_recording
from .dataset import build_dataset
from .features import extract_features
from .labels import LABELS
from .model import positive_probabilities, train
from .signal import prepare_signal


def predict(model_path: Path, header_path: Path) -> dict:
    """Use a trusted local joblib model on one paired recording."""
    artifact = joblib.load(model_path)
    if artifact.get("schema_version") != 1 or artifact.get("label_names") != list(
        LABELS
    ):
        raise ValueError("Unsupported model schema or label ordering")
    header, raw = load_recording(header_path)
    values = extract_features(
        prepare_signal(raw, header.sample_rate), header.age, header.sex
    )
    if list(values) != artifact["feature_names"]:
        raise ValueError("Model feature schema differs from the current pipeline")
    scores = positive_probabilities(
        artifact["pipeline"], np.asarray([list(values.values())])
    )[0]
    return {
        "record_id": header.record_id,
        "threshold": 0.5,
        "probabilities": dict(zip(LABELS, map(float, scores))),
        "labels": [label for label, value in zip(LABELS, scores) if value >= 0.5],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="NRSAA: three-lead ECG research pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a synthetic end-to-end demonstration")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    prepare = commands.add_parser(
        "prepare", help="Convert paired .hea/.mat recordings to features"
    )
    prepare.add_argument("input", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument(
        "--include-signals",
        action="store_true",
        help="Store waveforms for optional LSTM training; uses more memory",
    )
    prepare.add_argument(
        "--groups", type=Path, help="JSON mapping relative record paths to patient IDs"
    )
    training = commands.add_parser(
        "train", help="Train and evaluate a multilabel random-forest baseline"
    )
    training.add_argument("dataset", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--trees", type=int, default=200)
    training.add_argument("--test-size", type=float, default=0.2)
    training.add_argument("--allow-record-split", action="store_true")
    inference = commands.add_parser(
        "predict", help="Predict a paired recording using a trusted local model"
    )
    inference.add_argument(
        "model",
        type=Path,
        help="Trusted .joblib file; loading executes serialized Python",
    )
    inference.add_argument("header", type=Path)
    neural = commands.add_parser(
        "train-lstm", help="Train the three-lead LSTM classifier"
    )
    neural.add_argument("dataset", type=Path)
    neural.add_argument("--output", type=Path, required=True)
    neural.add_argument("--epochs", type=int, default=50)
    neural.add_argument("--batch-size", type=int, default=32)
    neural.add_argument("--seed", type=int, default=42)
    neural.add_argument("--allow-record-split", action="store_true")
    neural_predict = commands.add_parser(
        "predict-lstm", help="Predict using a saved LSTM run directory"
    )
    neural_predict.add_argument("run", type=Path)
    neural_predict.add_argument("header", type=Path)
    capture = commands.add_parser(
        "capture", help="Capture the single-channel AD8232 prototype"
    )
    capture.add_argument("--port", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--max-lines", type=int, default=15000)
    capture.add_argument("--baud", type=int, default=9600)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo

            result = run_demo(args.output)
        elif args.command == "prepare":
            if args.output.exists():
                raise ValueError("Output already exists; choose a new dataset path")
            result = build_dataset(
                args.input,
                args.output,
                args.groups,
                include_signals=args.include_signals,
            )
        elif args.command == "train":
            result = train(
                args.dataset,
                args.output,
                seed=args.seed,
                trees=args.trees,
                test_size=args.test_size,
                allow_record_split=args.allow_record_split,
            )
        elif args.command == "train-lstm":
            from .neural import train_lstm

            result = train_lstm(
                args.dataset,
                args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                allow_record_split=args.allow_record_split,
            )
        elif args.command == "predict-lstm":
            from .neural import predict_lstm

            result = predict_lstm(args.run, args.header)
        elif args.command == "capture":
            from .capture import capture_port

            if args.max_lines < 1 or args.baud < 1 or args.output.exists():
                raise ValueError(
                    "Require positive max-lines/baud and a new output file"
                )
            result = capture_port(args.port, args.output, args.max_lines, args.baud)
        else:
            result = predict(args.model, args.header)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
