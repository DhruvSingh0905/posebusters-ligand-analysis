"""Filesystem layout for the analysis."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
EXTRACTED = ROOT / "data" / "extracted"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RESULTS_CSV = RAW / "posebusters_paper_results.csv"

# The two benchmark sets shipped in the archive, keyed by the `dataset` value
# used in the results CSV.
SET_DIRS = {
    "posebuster": EXTRACTED / "posebusters_benchmark_set",
    "astex": EXTRACTED / "astex_diverse_set",
}

DESCRIPTORS_PARQUET = PROCESSED / "ligand_descriptors.parquet"
JOINED_PARQUET = PROCESSED / "poses_joined.parquet"
JOINED_CSV = PROCESSED / "poses_joined.csv"


def ensure_dirs() -> None:
    for d in (PROCESSED, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
