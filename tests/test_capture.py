import csv
from io import BytesIO

import pytest
from ecg_monitor.capture import capture_lines


def test_capture_timeout_and_events(tmp_path):
    stream = BytesIO(b"512\r\n!\r\nnot-a-sample\n1024\n0\n")
    output = tmp_path / "capture.csv"
    counts = capture_lines(stream, output, max_lines=10)
    assert counts == {"samples": 2, "lead_off": 1, "invalid": 2, "timeout": True}
    with output.open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 5
    assert rows[1]["status"] == "lead_off" and rows[1]["adc_count"] == ""
    assert float(rows[-1]["arrival_seconds"]) >= float(rows[0]["arrival_seconds"])


def test_capture_is_bounded_and_preserves_existing_files(tmp_path):
    output = tmp_path / "capture.csv"
    result = capture_lines(BytesIO(b"1\n2\n3\n"), output, max_lines=2)
    assert result["samples"] == 2 and not result["timeout"]
    with pytest.raises(FileExistsError):
        capture_lines(BytesIO(b"1\n"), output)
