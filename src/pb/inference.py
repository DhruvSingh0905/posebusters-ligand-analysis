"""Interval estimates and multiplicity control.

Every ligand is docked by every method, so rows are not independent: 2,140
deep-learning rows carry only 428 independent ligands. Resampling rows would
report intervals about sqrt(5) times too narrow, so the bootstrap resamples
*clusters* - whole ligands, with all their method rows attached.

Because the association grid is 18 checks x 18 descriptors, reporting the
largest effects without correction guarantees false positives. Bootstrap
p-values feed a Benjamini-Hochberg step-up at the reporting boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a cluster-bootstrap interval."""

    point: float
    lo: float
    hi: float
    p_value: float
    n_fail: int
    n_eligible: int
    n_clusters: int

    def significant(self) -> bool:
        return not (self.lo <= 0.0 <= self.hi)


def cohens_d(failing: pd.Series, passing: pd.Series) -> float:
    """Standardised mean difference, positive when failing poses score higher."""
    n_f, n_p = len(failing), len(passing)
    if n_f < 2 or n_p < 2:
        return float("nan")
    var_f, var_p = failing.var(ddof=1), passing.var(ddof=1)
    pooled = ((n_f - 1) * var_f + (n_p - 1) * var_p) / (n_f + n_p - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return float("nan")
    return float((failing.mean() - passing.mean()) / np.sqrt(pooled))


def _d_from_frame(frame: pd.DataFrame, check: str, descriptor: str) -> float:
    failed = frame[check] == False  # noqa: E712
    failing = pd.Series(frame.loc[failed, descriptor]).dropna()
    passing = pd.Series(frame.loc[~failed, descriptor]).dropna()
    return cohens_d(failing, passing)


def cluster_bootstrap_d(
    df: pd.DataFrame,
    check: str,
    descriptor: str,
    cluster: str = "pdb_id",
    n_boot: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Cohen's d with a percentile interval from resampling whole ligands.

    The p-value is the two-sided proportion of bootstrap replicates on the
    opposite side of zero from the point estimate, floored at 1/n_boot since a
    finite bootstrap cannot resolve smaller.
    """
    frame = df[[check, descriptor, cluster]].copy()
    point = _d_from_frame(frame, check, descriptor)

    groups = {key: sub for key, sub in frame.groupby(cluster, observed=True)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)

    replicates: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[k] for k in picked], ignore_index=True)
        value = _d_from_frame(sample, check, descriptor)
        if np.isfinite(value):
            replicates.append(value)

    failed_mask = frame[check] == False  # noqa: E712
    n_fail = int(failed_mask.sum())

    if len(replicates) < 20 or not np.isfinite(point):
        return Estimate(point, float("nan"), float("nan"), float("nan"),
                        n_fail, len(frame), len(keys))

    draws = np.asarray(replicates)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tail = float(np.mean(draws <= 0) if point > 0 else np.mean(draws >= 0))
    p_value = float(min(1.0, max(2 * tail, 1.0 / len(draws))))

    return Estimate(float(point), float(lo), float(hi), p_value,
                    n_fail, len(frame), len(keys))


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Step-up FDR control. Returns a boolean array; True means reject the null."""
    p = np.asarray(p_values, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)

    finite = np.isfinite(p)
    if not finite.any():
        return rejected

    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    thresholds = alpha * np.arange(1, m + 1) / m

    passing = np.nonzero(ranked <= thresholds)[0]
    if len(passing) == 0:
        return rejected

    cutoff = ranked[passing.max()]
    rejected[finite] = values <= cutoff
    return rejected
