import json

import numpy as np
import pytest
from ecg_monitor.cli import main, predict
from ecg_monitor.data import load_recording
from ecg_monitor.dataset import build_dataset, read_dataset
from ecg_monitor.demo import write_synthetic_records
from ecg_monitor.features import extract_features
from ecg_monitor.labels import LABELS, encode_labels
from ecg_monitor.model import evaluate, train
from ecg_monitor.signal import prepare_signal
from scipy.io import savemat


def test_multilabel_encoding():
    encoded = encode_labels(("164889003", "426783006", "unknown", "164889003"))
    assert sum(encoded) == 2
    assert encoded[LABELS.index("AF")] == encoded[LABELS.index("NSR")] == 1


def test_named_leads_and_adc_calibration(tmp_path):
    header = tmp_path / "sample.hea"
    header.write_text(
        "sample 3 250 2\nsample.mat 16 100(10)/mV 16 0 0 0 0 III\n"
        "sample.mat 16 100/mV 16 10 0 0 0 I\n"
        "sample.mat 16 100/mV 16 10 0 0 0 II\n#Age: NaN\n# Sex: Unknown\n#Dx: 164889003\n"
    )
    savemat(tmp_path / "sample.mat", {"val": [[310, 310], [110, 110], [210, 210]]})
    metadata, values = load_recording(header)
    np.testing.assert_allclose(values, [[1, 2, 3], [1, 2, 3]])
    assert metadata.codes == ("164889003",)
    assert np.isnan(metadata.age) and np.isnan(metadata.sex)


@pytest.mark.parametrize("sample_rate,seconds", [(250, 6), (500, 10), (1000, 12)])
def test_resample_pad_truncate(sample_rate, seconds):
    raw = np.ones((sample_rate * seconds, 3))
    prepared = prepare_signal(raw, sample_rate, denoise=False)
    assert prepared.shape == (5000, 3)
    assert np.isfinite(prepared).all()
    if seconds < 10:
        assert not prepared[seconds * 500 :].any()
    np.testing.assert_allclose(prepared[500:2000].mean(), 1, atol=0.01)


def test_flat_signal_and_feature_missingness():
    signal = prepare_signal(np.zeros((5000, 3)), 500)
    assert np.isfinite(signal).all()
    features = extract_features(signal, np.nan, np.nan)
    assert np.isnan(features["I_heart_rate_bpm"])
    assert len(features) == 53


def test_peak_features_use_seconds():
    signal = np.zeros((5000, 3))
    signal[np.arange(250, 5000, 500), :] = 1
    features = extract_features(signal, 40, 1)
    assert features["I_heart_rate_bpm"] == pytest.approx(60)
    assert features["I_rr_mean_s"] == pytest.approx(1)


@pytest.mark.parametrize("rate", [0, -1, float("nan")])
def test_invalid_sampling_rate(rate):
    with pytest.raises(ValueError):
        prepare_signal(np.ones((5000, 3)), rate)


def test_missing_lead_is_rejected(tmp_path):
    write_synthetic_records(tmp_path, count=1)
    path = tmp_path / "synthetic_000.hea"
    path.write_text(path.read_text().replace("0 III", "0 V1"))
    with pytest.raises(ValueError, match="leads I, II and III"):
        load_recording(path)


def test_dataset_accumulates_nested_directories(tmp_path):
    root = tmp_path / "raw"
    write_synthetic_records(root / "a", count=2)
    write_synthetic_records(root / "b", count=2)
    output = tmp_path / "features.npz"
    metadata = build_dataset(root, output)
    data = read_dataset(output)
    assert metadata["records"] == 4
    assert set(data["record_ids"]) == {
        "a/synthetic_000",
        "a/synthetic_001",
        "b/synthetic_000",
        "b/synthetic_001",
    }
    with pytest.raises(ValueError, match="patient groups"):
        train(output, tmp_path / "run")


def test_incomplete_group_mapping_rejected(tmp_path):
    group_file = write_synthetic_records(tmp_path / "raw", count=2)
    group_file.write_text("{}")
    with pytest.raises(ValueError, match="Missing nonempty patient group"):
        build_dataset(tmp_path / "raw", tmp_path / "dataset.npz", group_file)


def test_end_to_end_group_split_and_prediction(tmp_path):
    root = tmp_path / "raw"
    groups = write_synthetic_records(root, count=30)
    path = tmp_path / "features.npz"
    build_dataset(root, path, groups)
    metrics = train(path, tmp_path / "run", trees=5)
    split = json.loads((tmp_path / "run/split.json").read_text())
    assert not set(split["train_groups"]) & set(split["test_groups"])
    assert not set(split["train"]) & set(split["test"])
    assert metrics["train_records"] + metrics["test_records"] == 30
    result = predict(tmp_path / "run/model.joblib", root / "synthetic_000.hea")
    assert list(result["probabilities"]) == list(LABELS)
    assert all(0 <= p <= 1 for p in result["probabilities"].values())
    assert result["probabilities"]["AF"] == 0  # Class absent from synthetic training.
    assert (tmp_path / "run/predictions.npz").exists()
    with pytest.raises(ValueError, match="empty output"):
        train(path, tmp_path / "run", trees=5)


def test_metrics_distinguish_exact_match_and_hamming_loss():
    y = np.zeros((2, len(LABELS)), dtype=int)
    scores = np.zeros_like(y, dtype=float)
    scores[0, 0] = 0.9
    metrics = evaluate(y, scores)
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["hamming_loss"] == pytest.approx(1 / (2 * len(LABELS)))
    assert metrics["per_label"]["AF"]["average_precision"] is None


def test_cli_errors_are_actionable(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["prepare", str(tmp_path / "missing"), str(tmp_path / "features.npz")])
    assert exc.value.code == 2
    assert "No .hea files" in capsys.readouterr().err


def test_imputation_is_fitted_on_training_records_only(tmp_path):
    import joblib
    from sklearn.model_selection import GroupShuffleSplit

    root = tmp_path / "raw"
    groups = write_synthetic_records(root, count=12)
    path = tmp_path / "features.npz"
    build_dataset(root, path, groups)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    training, testing = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            arrays["X"], groups=arrays["groups"]
        )
    )
    arrays["X"][training, 0] = 35
    arrays["X"][testing, 0] = 110
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    train(path, tmp_path / "run", trees=2)
    artifact = joblib.load(tmp_path / "run/model.joblib")
    assert artifact["pipeline"].named_steps["imputer"].statistics_[0] == 35
