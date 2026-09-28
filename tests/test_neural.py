import numpy as np
import pytest


def test_lstm_temporal_axis_multilabel_and_training_step():
    pytest.importorskip("tensorflow")
    from ecg_monitor.neural import build_lstm

    model = build_lstm()
    assert model.input_shape == (None, 5000, 3)
    assert model.output_shape == (None, 9)
    X = np.zeros((2, 5000, 3), dtype=np.float32)
    y = np.zeros((2, 9), dtype=np.float32)
    y[:, :2] = 1  # Multiple simultaneous target labels.
    loss = model.train_on_batch(X, y)
    assert np.isfinite(loss)
    probabilities = model(X, training=False).numpy()
    assert probabilities.shape == (2, 9)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert model.layers[-1].activation.__name__ == "sigmoid"


def test_saved_lstm_inference_round_trip(tmp_path):
    pytest.importorskip("tensorflow")
    from ecg_monitor.data import load_recording
    from ecg_monitor.demo import write_synthetic_records
    from ecg_monitor.labels import LABELS
    from ecg_monitor.neural import build_lstm, predict_lstm
    from ecg_monitor.signal import prepare_signal

    write_synthetic_records(tmp_path / "raw", count=1)
    header = tmp_path / "raw/synthetic_000.hea"
    metadata, raw = load_recording(header)
    model = build_lstm()
    expected = model(
        prepare_signal(raw, metadata.sample_rate)[None], training=False
    ).numpy()[0]
    model.save(tmp_path / "model.keras")
    np.savez_compressed(
        tmp_path / "normalization.npz",
        mean=np.zeros((1, 1, 3)),
        scale=np.ones((1, 1, 3)),
        label_names=np.asarray(LABELS),
    )
    actual = predict_lstm(tmp_path, header)
    np.testing.assert_allclose(
        list(actual["probabilities"].values()), expected, atol=1e-6
    )
