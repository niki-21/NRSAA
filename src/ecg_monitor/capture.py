"""Bounded capture of the AD8232 sketch's single-channel serial output.

Adapted from the AF prototype's reader.py; upstream MIT notice is retained
in third_party/atrial_fibrillation_MIT.txt. Host timestamps are arrival times,
not a calibrated acquisition clock. This data is not three-lead model input.
"""

import csv
import time
from pathlib import Path
from typing import BinaryIO


def capture_lines(stream: BinaryIO, output: Path, max_lines: int = 15000) -> dict:
    if max_lines < 1:
        raise ValueError("max_lines must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"samples": 0, "lead_off": 0, "invalid": 0, "timeout": False}
    started = time.monotonic()
    with output.open("x", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["arrival_seconds", "adc_count", "status"])
        for _ in range(max_lines):
            line = stream.readline()
            if line == b"":
                counts["timeout"] = True
                break
            value = line.strip()
            if value == b"!":
                status, adc = "lead_off", ""
                counts["lead_off"] += 1
            elif value.isdigit() and 0 <= int(value) <= 1023:
                status, adc = "ok", int(value)
                counts["samples"] += 1
            else:
                status, adc = "invalid", ""
                counts["invalid"] += 1
            writer.writerow([f"{time.monotonic() - started:.6f}", adc, status])
    return counts


def capture_port(
    port: str, output: Path, max_lines: int = 15000, baud: int = 9600
) -> dict:
    try:
        import serial
    except ImportError as exc:
        raise ValueError(
            'Install the hardware extra: pip install ".[hardware]"'
        ) from exc
    with serial.Serial(port, baudrate=baud, timeout=5) as stream:
        return capture_lines(stream, output, max_lines)
