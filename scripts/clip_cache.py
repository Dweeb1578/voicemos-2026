"""Resumable per-clip score cache shared by zero-shot scorers.

Append-only CSV keyed by clip path. Crash-hardened the same way as the cache
in scripts/zero_shot_dnsmos.py (which predates this module and is kept as-is
because it is validated end-to-end): torn last rows parse as NaN and are
dropped on load (the clip is simply rescored), and appending after a torn
line with no trailing newline first terminates it so the CSV stays parseable.
"""

import csv
import os

import pandas as pd
from tqdm import tqdm


def load_cached(csv_path, value_cols):
    """{path: tuple(value_cols)} for clips already scored; torn/NaN rows are
    dropped so they get rescored on resume."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return {}
    df = pd.read_csv(csv_path).dropna(subset=list(value_cols))
    return {r["path"]: tuple(r[c] for c in value_cols) for _, r in df.iterrows()}


def score_paths(scorer, clip_paths, csv_path, value_cols, desc="scoring"):
    """Run scorer(path) -> tuple matching value_cols for every uncached path,
    appending row-by-row (flushed) so progress survives a crash. Returns
    {path: tuple} for ALL requested paths, cached and new."""
    done = load_cached(csv_path, value_cols)
    todo = [p for p in clip_paths if p not in done]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if not new_file:
        with open(csv_path, "rb+") as fb:
            fb.seek(-1, os.SEEK_END)
            if fb.read(1) != b"\n":
                fb.write(b"\n")
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["path", *value_cols])
        for p in tqdm(todo, desc=desc):
            vals = scorer(p)
            writer.writerow([p, *vals])
            f.flush()
            done[p] = tuple(vals)
    return {p: done[p] for p in clip_paths}
