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

import logging

import pandas as pd

from .eligibility import eligible
from .inference import benjamini_hochberg, cluster_bootstrap_d

log = logging.getLogger(__name__)

COFACTOR_CHECKS = {
    "no_clashes_with_organic_cofactors",
    "no_clashes_with_inorganic_cofactors",
    "no_volume_clash_with_organic_cofactors",
    "no_volume_clash_with_inorganic_cofactors",
}

# Columns produced by `association_grid`, kept as a module constant so an
# all-skipped grid still returns a frame with the documented schema rather
# than an empty frame with none of these columns.
_GRID_COLUMNS = [
    "method", "check", "descriptor", "d", "lo", "hi", "p_value",
    "n_fail", "n_eligible", "n_clusters", "fdr_reject", "abs_d",
]


def stratum_frame(df: pd.DataFrame, check: str, method: str) -> pd.DataFrame:
    """Eligible rows for one check within one method's stratum."""
    frame = df[df["method"] == method]
    frame = eligible(frame, check)
    if check in COFACTOR_CHECKS:
        has_cofactors = frame["has_cofactors"]
        unknown = int(has_cofactors.isna().sum())
        if unknown:
            log.warning(
                "%s/%s: %d rows have an unknown has_cofactors and are treated "
                "as cofactor-free - excluded from the cofactor-conditioned "
                "stratum",
                method, check, unknown,
            )
        frame = frame[has_cofactors.fillna(False).astype(bool)]
    return frame


def stratum_coverage(
    df: pd.DataFrame,
    methods: list[str],
    checks: list[str] | None = None,
    min_failures: int = 15,
    min_clusters: int = 30,
) -> pd.DataFrame:
    """Why every method x check combination was or was not estimated.

    `association_grid` cannot report on strata it skips, and a skipped stratum
    is indistinguishable in its output from one with no effect. This records
    the fate of all of them so a reader can tell "no association" from "never
    measured". Note in particular that `all_failed` is a scientifically
    interesting outcome - every pose failing a check - not a data shortage,
    even though both end up unestimated.
    """
    from .eligibility import ELIGIBILITY

    checks = list(ELIGIBILITY) if checks is None else checks
    rows = []
    for method in methods:
        for check in checks:
            frame = stratum_frame(df, check, method)
            n_fail = int((frame[check] == False).sum())  # noqa: E712
            n_clusters = frame["pdb_id"].nunique()

            if n_fail == 0:
                reason = "no_failures"
            elif n_fail < min_failures:
                reason = "too_few_failures"
            elif n_clusters < min_clusters:
                reason = "too_few_ligands"
            elif n_fail == len(frame):
                reason = "all_failed"
            else:
                reason = None

            rows.append({
                "method": method,
                "check": check,
                "n_eligible": len(frame),
                "n_fail": n_fail,
                "n_clusters": n_clusters,
                "estimated": reason is None,
                "skip_reason": reason,
            })
    return pd.DataFrame(rows)


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
    interval, and the skip is counted and logged so nothing is dropped
    silently - call `stratum_coverage()` for the full per-stratum breakdown,
    including *why* each skipped stratum was skipped.

    Benjamini-Hochberg is applied once, across every estimate in the returned
    grid - the family is the whole method x check x descriptor grid, not each
    method separately. Correcting per method would let a method with fewer
    surviving checks buy more power for the same nominal p-value, which is
    exactly the kind of pooling-shaped distortion this module exists to avoid.
    """
    from .eligibility import ELIGIBILITY

    checks = list(ELIGIBILITY) if checks is None else checks
    rows = []
    estimated = 0
    skipped = 0

    for method in methods:
        for check in checks:
            frame = stratum_frame(df, check, method)
            n_fail = int((frame[check] == False).sum())  # noqa: E712
            n_clusters = frame["pdb_id"].nunique()
            if n_fail < min_failures or n_clusters < min_clusters:
                skipped += 1
                continue
            if n_fail == len(frame):  # no passing poses to contrast against
                skipped += 1
                continue
            estimated += 1

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

    if skipped:
        log.warning(
            "association_grid estimated %d of %d method x check strata; "
            "%d skipped as too thin. Call stratum_coverage() for the breakdown.",
            estimated, estimated + skipped, skipped,
        )

    grid = pd.DataFrame(rows)
    if grid.empty:
        return pd.DataFrame(columns=_GRID_COLUMNS)

    grid["fdr_reject"] = benjamini_hochberg(grid["p_value"].to_numpy(), alpha=0.05)
    grid["abs_d"] = grid["d"].abs()
    return grid.sort_values("abs_d", ascending=False).reset_index(drop=True)
