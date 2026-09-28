"""Deterministic resampling, wavelet denoising, and fixed-duration windows."""

from fractions import Fraction

import numpy as np
import pywt
from scipy.signal import resample_poly

TARGET_RATE = 500
DURATION = 10


def prepare_signal(
    signal: np.ndarray, sample_rate: float, *, denoise: bool = True
) -> np.ndarray:
    """Convert to 500 Hz, retain the first ten seconds and pad short recordings.

    Denoising uses sym5 soft thresholding of detail coefficients. Approximation
    coefficients are retained to avoid suppressing the low-frequency waveform.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 3 or signal.shape[0] < 2:
        raise ValueError("Expected at least two samples and exactly three leads")
    if (
        not np.isfinite(signal).all()
        or not np.isfinite(sample_rate)
        or sample_rate <= 0
    ):
        raise ValueError("Samples must be finite and sampling rate positive")
    if sample_rate != TARGET_RATE:
        ratio = (Fraction(TARGET_RATE) / Fraction(str(sample_rate))).limit_denominator(
            10000
        )
        signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=0)
    signal = signal[: TARGET_RATE * DURATION].copy()
    if denoise:
        wavelet = pywt.Wavelet("sym5")
        level = min(5, pywt.dwt_max_level(len(signal), wavelet.dec_len))
        if level:
            for lead in range(3):
                coeffs = pywt.wavedec(signal[:, lead], wavelet, level=level)
                threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
                if threshold > 0:
                    coeffs[1:] = [
                        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
                    ]
                signal[:, lead] = pywt.waverec(coeffs, wavelet)[: len(signal)]
    return np.pad(signal, ((0, TARGET_RATE * DURATION - len(signal)), (0, 0))).astype(
        np.float32
    )
