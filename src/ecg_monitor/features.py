"""Descriptive signal features; peak-based values are research heuristics."""

import numpy as np
import pywt
from scipy.signal import find_peaks, welch

from .data import LEADS
from .signal import TARGET_RATE


def extract_features(signal: np.ndarray, age: float, sex: float) -> dict[str, float]:
    """Compute a fixed schema with time, spectrum, wavelet and RR features.

    RR intervals use simple absolute-amplitude peak detection, not a validated
    QRS detector. Missing RR statistics remain NaN for training-only imputation.
    """
    signal = np.asarray(signal)
    if signal.shape != (5000, 3) or not np.isfinite(signal).all():
        raise ValueError(
            "Feature extraction expects a finite [5000, 3] prepared recording"
        )
    result = {"age": float(age), "sex": float(sex)}
    for i, name in enumerate(LEADS):
        x = signal[:, i].astype(np.float64)
        frequencies, psd = welch(x, fs=TARGET_RATE, nperseg=1024)
        peak_indices, _ = find_peaks(
            np.abs(x - np.median(x)),
            distance=int(0.25 * TARGET_RATE),
            prominence=max(0.5 * np.std(x), 1e-8),
        )
        rr = np.diff(peak_indices) / TARGET_RATE
        coeffs = pywt.wavedec(x, "sym5", level=5)
        values = {
            "mean_mv": np.mean(x),
            "std_mv": np.std(x),
            "rms_mv": np.sqrt(np.mean(x**2)),
            "range_mv": np.ptp(x),
            "dominant_hz": frequencies[np.argmax(psd)],
            "rr_mean_s": np.mean(rr) if len(rr) else np.nan,
            "rr_std_s": np.std(rr) if len(rr) else np.nan,
            "rr_rmssd_s": np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan,
            "heart_rate_bpm": 60 / np.mean(rr) if len(rr) else np.nan,
        }
        for low, high in [(0.5, 5), (5, 15), (15, 40)]:
            mask = (frequencies >= low) & (frequencies < high)
            values[f"power_{low:g}_{high:g}hz"] = np.sum(psd[mask]) * (
                frequencies[1] - frequencies[0]
            )
        for level, detail in enumerate(reversed(coeffs[1:]), 1):
            values[f"wavelet_d{level}_energy"] = np.mean(detail**2)
        result.update({f"{name}_{key}": float(value) for key, value in values.items()})
    return result
