"""Per-method, cofactor-conditioned strata.

Two pooling mistakes are corrected here.

Pooling methods: DiffDock supplies 39% of the pooled deep-learning poses that
pass RMSD and EquiBind 3%, so a pooled effect is mostly a statement about models
that barely function. Effects are estimated per method; a pooled number is only
ever shown beside the breakdown.

Pooling cofactor presence: the cofactor checks pass almost automatically on
complexes with no cofactors (3.4% failure vs 31.3%). Since cofactor-bearing
complexes also carry systematically different ligands (1.56 vs 2.11 aromatic
rings), a pooled comparison measures the receptor, not the ligand. Those checks
are conditioned on `has_cofactors`.
"""

from __future__ import annotations

import pandas as pd

from .eligibility import eligible
from .inference import benjamini_hochberg, cluster_bootstrap_d

COFACTOR_CHECKS = {
    "no_clashes_with_organic_cofactors",
    "no_clashes_with_inorganic_cofactors",
    "no_volume_clash_with_organic_cofactors",
    "no_volume_clash_with_inorganic_cofactors",
}


def stratum_frame(df: pd.DataFrame, check: str, method: str) -> pd.DataFrame:
    """Eligible rows for one check within one method's stratum."""
    frame = df[df["method"] == method]
    frame = eligible(frame, check)
    if check in COFACTOR_CHECKS:
        frame = frame[frame["has_cofactors"].fillna(False).astype(bool)]
    return frame


def association_grid(
    df: pd.DataFrame,
    methods: list[str],
    descriptors: list[str],
    checks: list[str] | None = None,
    min_failures: int = 15,
    min_clusters: int = 30,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Every (method, check, descriptor) association with an interval and an FDR flag.

    Strata too thin to estimate are skipped rather than reported with a wide
    interval, and the skip is visible in the returned frame's absence - callers
    log the count so nothing is dropped silently.
    """
    from .eligibility import ELIGIBILITY

    checks = checks or list(ELIGIBILITY)
    rows = []

    for method in methods:
        for check in checks:
            frame = stratum_frame(df, check, method)
            n_fail = int((frame[check] == False).sum())  # noqa: E712
            n_clusters = frame["pdb_id"].nunique()
            if n_fail < min_failures or n_clusters < min_clusters:
                continue
            if n_fail == len(frame):  # no passing poses to contrast against
                continue

            for descriptor in descriptors:
                est = cluster_bootstrap_d(
                    frame, check, descriptor, n_boot=n_boot, seed=seed
                )
                rows.append({
                    "method": method,
                    "check": check,
                    "descriptor": descriptor,
                    "d": est.point,
                    "lo": est.lo,
                    "hi": est.hi,
                    "p_value": est.p_value,
                    "n_fail": est.n_fail,
                    "n_eligible": est.n_eligible,
                    "n_clusters": est.n_clusters,
                })

    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid

    grid["fdr_reject"] = benjamini_hochberg(grid["p_value"].to_numpy(), alpha=0.05)
    grid["abs_d"] = grid["d"].abs()
    return grid.sort_values("abs_d", ascending=False).reset_index(drop=True)
