"""Synthetic fixtures for testing software behavior, never clinical performance."""

import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .dataset import build_dataset
from .model import train
from .signal import prepare_signal


def synthetic_signal(
    rate: float, *, sample_rate: int = 500, seconds: int = 10, seed: int = 42
) -> np.ndarray:
    """Generate a stylized repeating waveform with drift and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(sample_rate * seconds) / sample_rate
    wave = np.zeros_like(t)
    for beat in np.arange(0.4, seconds, 60 / rate):
        for offset, amplitude, width in [
            (-0.16, 0.12, 0.03),
            (-0.03, -0.15, 0.01),
            (0, 1, 0.012),
            (0.03, -0.25, 0.012),
            (0.23, 0.25, 0.06),
        ]:
            wave += amplitude * np.exp(-0.5 * ((t - beat - offset) / width) ** 2)
    return np.column_stack(
        [
            scale * wave
            + 0.06 * np.sin(2 * np.pi * 0.3 * t)
            + rng.normal(0, 0.035, len(t))
            for scale in (1.0, 0.8, 0.6)
        ]
    )


def write_synthetic_records(root: Path, count: int = 36, seed: int = 42) -> Path:
    """Write valid paired fixtures, with two recordings per synthetic subject."""
    root.mkdir(parents=True, exist_ok=True)
    groups = {}
    for i in range(count):
        record_id = f"synthetic_{i:03d}"
        rate, code = [(70, "426783006"), (48, "426177001"), (120, "427084000")][
            (i // 2) % 3
        ]
        signal = synthetic_signal(rate + (i % 2), seed=seed + i)
        savemat(root / f"{record_id}.mat", {"val": (signal.T * 1000).astype(np.int16)})
        lines = [f"{record_id} 3 500 5000"]
        lines += [
            f"{record_id}.mat 16+24 1000/mV 16 0 0 0 0 {lead}"
            for lead in ("I", "II", "III")
        ]
        lines += ["# Age: NaN", "# Sex: Unknown", f"# Dx: {code}"]
        (root / f"{record_id}.hea").write_text("\n".join(lines) + "\n")
        groups[record_id] = f"synthetic_subject_{i // 2:03d}"
    group_file = root / "groups.json"
    group_file.write_text(json.dumps(groups, indent=2) + "\n")
    return group_file


def plot_signals(raw: np.ndarray, prepared: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    for i, ax in enumerate(axes):
        time = np.arange(2500) / 500
        ax.plot(time, raw[:2500, i], color="#bac8d3", linewidth=0.7, label="Input")
        ax.plot(
            time,
            prepared[:2500, i],
            color="#087e8b",
            linewidth=1,
            label="Wavelet denoised",
        )
        ax.set_ylabel(f"Lead {('I', 'II', 'III')[i]} / mV")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("Time / seconds")
    fig.suptitle(
        "Three-lead signal pipeline · synthetic demonstration",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_demo(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Choose an empty demo directory to avoid overwriting an earlier run"
        )
    output.mkdir(parents=True, exist_ok=True)
    groups = write_synthetic_records(output / "recordings")
    build_dataset(output / "recordings", output / "features.npz", groups)
    train(output / "features.npz", output / "run", trees=40)
    raw = synthetic_signal(70)
    plot_signals(raw, prepare_signal(raw, 500), output / "signals.png")
    (output / "README.txt").write_text(
        "SYNTHETIC SOFTWARE DEMONSTRATION\n"
        "These generated waveforms are not patient data. Metrics test the software only.\n"
        "They do not reproduce or validate the results in the 2024 capstone report.\n"
        "The demo covers three labels; the other six intentionally test absent-class handling.\n"
    )
    return {
        "status": "complete",
        "data": "synthetic",
        "records": 36,
        "signal_plot": str(output / "signals.png"),
        "metrics": str(output / "run/metrics.json"),
    }
