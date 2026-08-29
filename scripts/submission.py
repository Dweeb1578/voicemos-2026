"""Shared CodaBench submission helpers.

The CodaBench scorer reads a file named exactly `predictions.csv` at the zip
root. Any other arcname scores as a silent NA (empty scores.json, exit 0, no
error) -- this cost a full submission round-trip on 2026-06-09.
"""

import zipfile


def write_submission_zip(csv_path: str, zip_path: str) -> None:
    """Zip csv_path under the only arcname CodaBench accepts."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")


def rescale_to_range(values, lo, hi):
    """Linearly map values from [min,max] to [lo,hi]. Strictly monotonic, so
    SRCC is preserved exactly; used to land arbitrary-scale scores (mean
    ranks, OVRL diffs) inside the submission's legal range without the
    rank-destroying hard clamp. All-equal input returns all `lo`."""
    values = list(values)
    mn, mx = min(values), max(values)
    if mx == mn:
        return [lo] * len(values)
    span = mx - mn
    return [lo + (v - mn) / span * (hi - lo) for v in values]
