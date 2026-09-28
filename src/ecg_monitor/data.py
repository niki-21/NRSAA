"""Load paired PhysioNet headers and MATLAB recordings without CSV eval."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

LEADS = ("I", "II", "III")


@dataclass(frozen=True)
class Header:
    record_id: str
    sample_rate: float
    sample_count: int
    lead_names: tuple[str, ...]
    gains: tuple[float, ...]
    baselines: tuple[float, ...]
    codes: tuple[str, ...]
    age: float
    sex: float


def read_header(path: Path) -> Header:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty header: {path.name}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Incomplete header: {path.name}")
    n_leads = int(first[1])
    fs = float(first[2].split("/")[0])
    count = int(first[3])
    if not np.isfinite(fs) or fs <= 0 or count < 1 or n_leads < 1:
        raise ValueError(
            "Header requires positive sampling rate, sample count and lead count"
        )
    names, gains, baselines = [], [], []
    for line in lines[1 : n_leads + 1]:
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Incomplete signal specification: {path.name}")
        match = re.fullmatch(r"([0-9.eE+\-]+)(?:\(([0-9.eE+\-]+)\))?/mV", fields[2])
        if match is None:
            raise ValueError(
                "Expected an ADC gain in /mV; other physical units are unsupported"
            )
        gain = float(match[1])
        baseline = float(match[2]) if match[2] is not None else float(fields[4])
        if not np.isfinite(gain) or gain <= 0 or not np.isfinite(baseline):
            raise ValueError("ADC gain must be positive and calibration values finite")
        names.append(fields[-1])
        gains.append(gain)
        baselines.append(baseline)
    if len(names) != n_leads or len(set(names)) != n_leads:
        raise ValueError("Missing or duplicate lead specifications")
    metadata = {}
    for line in lines[n_leads + 1 :]:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            metadata[key.strip().lower()] = value.strip()
    try:
        age = float(metadata.get("age", "nan"))
    except ValueError:
        age = np.nan
    if not 0 <= age <= 120:
        age = np.nan
    sex = {"male": 1.0, "female": 0.0}.get(metadata.get("sex", "").lower(), np.nan)
    codes = tuple(c.strip() for c in metadata.get("dx", "").split(",") if c.strip())
    return Header(
        first[0],
        fs,
        count,
        tuple(names),
        tuple(gains),
        tuple(baselines),
        codes,
        age,
        sex,
    )


def load_recording(path: Path) -> tuple[Header, np.ndarray]:
    """Return named leads I/II/III as [samples, leads], calibrated to mV."""
    header = read_header(path)
    raw = loadmat(path.with_suffix(".mat")).get("val")
    if raw is None or raw.ndim != 2:
        raise ValueError(f"Expected a 2D 'val' matrix for {path.name}")
    if raw.shape != (len(header.lead_names), header.sample_count):
        raise ValueError(f"Recording shape disagrees with header: {path.name}")
    try:
        indices = [header.lead_names.index(lead) for lead in LEADS]
    except ValueError as exc:
        raise ValueError(
            f"Recording must contain leads I, II and III: {path.name}"
        ) from exc
    signal = raw[indices].astype(np.float64).T
    if not np.isfinite(signal).all():
        raise ValueError(f"Recording contains non-finite samples: {path.name}")
    signal = (signal - np.asarray(header.baselines)[indices]) / np.asarray(
        header.gains
    )[indices]
    return header, signal
