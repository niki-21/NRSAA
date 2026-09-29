"""Regenerate documentation figures from cited values and synthetic signals."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from ecg_monitor.demo import plot_signals, synthetic_signal
from ecg_monitor.signal import prepare_signal


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    report = json.loads((root / "docs/reported_results.json").read_text())
    models = report["models"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "nrsaa",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
    y = np.arange(len(models))
    ax.barh(
        y - 0.17,
        [m["precision"] * 100 for m in models],
        height=0.3,
        color="#087e8b",
        label="Precision",
    )
    ax.barh(
        y + 0.17,
        [m["recall"] * 100 for m in models],
        height=0.3,
        color="#e3a544",
        label="Recall",
    )
    for i, model in enumerate(models):
        for offset, key in [(-0.17, "precision"), (0.17, "recall")]:
            ax.text(
                model[key] * 100 + 1,
                i + offset,
                f"{model[key] * 100:.2f}%",
                va="center",
                fontsize=10,
            )
    ax.set_yticks(
        y,
        [
            "Signal baseline",
            "Wavelet",
            "Wavelet + demographics + CV",
            "Features + demographics",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Reported metric / %")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    fig.suptitle(
        "2024 capstone results", x=0.02, ha="left", fontsize=20, fontweight="bold"
    )
    fig.text(
        0.02,
        -0.035,
        "Source: final report pp. 98, 100, 103, 105. Historical values; not reproduced by this repository.",
        fontsize=9,
    )
    fig.savefig(
        assets / "reported-results.svg", bbox_inches="tight", metadata={"Date": None}
    )
    svg = assets / "reported-results.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    raw = synthetic_signal(70)
    plot_signals(raw, prepare_signal(raw, 500), assets / "signals.png")
    wave = raw[:900, 0]
    points = " ".join(
        f"{710 + i * 0.47:.1f},{190 - float(value) * 65:.1f}"
        for i, value in enumerate(wave)
    )
    (
        assets / "hero.svg"
    ).write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="330" viewBox="0 0 1200 330" role="img" aria-labelledby="title desc">
    <title id="title">NRSAA — Wireless Health Monitoring</title>
    <desc id="desc">AUS capstone 2024: three-lead ECG, wireless sensing and classification research.</desc>
    <rect width="1200" height="330" rx="18" fill="#102b38"/>
    <text x="48" y="57" fill="#75d5c4" font-family="sans-serif" font-size="14" letter-spacing="3">AMERICAN UNIVERSITY OF SHARJAH · CAPSTONE 2024</text>
    <text x="48" y="128" fill="#ffffff" font-family="sans-serif" font-size="43" font-weight="700">NRSAA</text>
    <text x="48" y="180" fill="#ffffff" font-family="sans-serif" font-size="34" font-weight="700">Wireless Health Monitoring</text>
    <text x="48" y="225" fill="#c0d3dc" font-family="sans-serif" font-size="17">Three-lead ECG · Wireless sensing · Classification research</text>
    <text x="48" y="282" fill="#75d5c4" font-family="monospace" font-size="15">ACQUIRE → PROCESS → CLASSIFY → EVALUATE</text>
    <path d="M700 90H1145 M700 140H1145 M700 190H1145 M700 240H1145" stroke="#244452" stroke-width="1"/>
    <polyline points="{points}" fill="none" stroke="#75d5c4" stroke-width="2"/>
    <text x="710" y="280" fill="#a8c2ce" font-family="sans-serif" font-size="13">Synthetic signal illustration</text>
    </svg>\n''')
    print(f"Generated figures in {assets}")


if __name__ == "__main__":
    main()
