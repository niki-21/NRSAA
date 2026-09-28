"""The nine diagnosis labels used in the 2024 final report.

Codes follow the PhysioNet 2020 SNOMED CT mappings. PAC and SVPB remain
separate to match the report; this is not the official Challenge scoring policy.
"""

LABELS = ("NSR", "AF", "STach", "SA", "SB", "PAC", "SVPB", "AFL", "SVT")
CODES = {
    "426783006": "NSR",
    "164889003": "AF",
    "427084000": "STach",
    "427393009": "SA",
    "426177001": "SB",
    "284470004": "PAC",
    "63593006": "SVPB",
    "164890007": "AFL",
    "426761007": "SVT",
}


def encode_labels(codes: tuple[str, ...]) -> list[int]:
    """Preserve co-occurring target labels and ignore out-of-scope codes."""
    found = {CODES[code] for code in codes if code in CODES}
    return [int(label in found) for label in LABELS]
