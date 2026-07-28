"""Find which ligand properties predict which PoseBusters check fails.

`main()` builds the gated, clustered, FDR-controlled estimates that back the
published report: `association_grid` (Task 4 strata + Task 3 cluster
bootstrap + BH-FDR), `check_models` (Task 6 clustered logistic models, one per
method so no claim is a pooled artifact of whichever method contributes the
most rows), and the crystal-contact sensitivity split (Task 5). `method_summary`,
`check_rate_by_bin` and `outcome_by_bin` are kept as descriptive companions -
`pb.figures` still uses `outcome_by_bin` and `primary` for the two figures that
were not retired, and `main()` still writes their tables for the headline and
secondary ("accurate but invalid") sections of the report - but none of them is
the source of an effect-size claim; those all come from `association_grid` and
`check_models`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import paths
from .build import CHECKS, CLASSICAL_METHODS, DL_METHODS
from .eligibility import eligible
from .inference import cluster_bootstrap_d
from .strata import stratum_frame

log = logging.getLogger(__name__)

TABLES = paths.REPORTS / "tables"

MIN_FAILURES = 15  # below this an association is too thin to report


def primary(df: pd.DataFrame, post: str = "none") -> pd.DataFrame:
    """The analysis population: PoseBusters set, real docking methods."""
    return df[
        (df["dataset"] == "posebuster")
        & (df["post-processing"] == post)
        & (df["method"].isin(DL_METHODS + CLASSICAL_METHODS))
    ].copy()


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy vs validity per method - the headline table."""
    rows = []
    for key, group in df.groupby(["method", "post-processing"]):
        method, post = key  # type: ignore[misc]
        n = len(group)
        accurate = pd.Series(group["accurate"])
        rows.append(
            {
                "method": method,
                "post_processing": post,
                "method_class": group["method_class"].iloc[0],
                "n": n,
                "produced": group["pose_produced"].mean(),
                "accurate": accurate.mean(),
                "valid": group["pb_valid"].mean(),
                "accurate_and_valid": group["accurate_and_valid"].mean(),
                "n_accurate": int(accurate.sum()),
                "invalid_given_accurate": (
                    group.loc[accurate, "pb_valid"].eq(False).mean()
                    if bool(accurate.any())
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["post_processing", "accurate"], ascending=[True, False]
    )


def check_rate_by_bin(df: pd.DataFrame, bin_column: str) -> pd.DataFrame:
    """Failure rate of each check across the levels of one ligand partition."""
    rows = []
    for check in CHECKS:
        ran = df[df[check].notna()]
        if (ran[check] == False).sum() < MIN_FAILURES:  # noqa: E712
            continue
        for level, group in ran.groupby(bin_column, observed=True):
            rows.append(
                {
                    "check": check,
                    bin_column: level,
                    "n": len(group),
                    "failure_rate": float((group[check] == False).mean()),  # noqa: E712
                }
            )
    return pd.DataFrame(rows)


def outcome_by_bin(df: pd.DataFrame, bin_column: str) -> pd.DataFrame:
    """Accuracy, validity and the gap between them, per ligand class."""
    rows = []
    for key, group in df.groupby(["method_class", bin_column], observed=True):
        method_class, level = key  # type: ignore[misc]
        accurate = pd.Series(group["accurate"])
        rows.append(
            {
                "method_class": method_class,
                bin_column: level,
                "n": len(group),
                "accurate": accurate.mean(),
                "valid": group["pb_valid"].mean(),
                "accurate_and_valid": group["accurate_and_valid"].mean(),
                "n_accurate": int(accurate.sum()),
                "invalid_given_accurate": (
                    group.loc[accurate, "pb_valid"].eq(False).mean()
                    if bool(accurate.any())
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# ── new for the gated/clustered/FDR-controlled report (Task 9) ────────────

REPORT_PARTITIONS = ["rotb_bin", "mw_bin", "rings_bin", "stereo_bin"]


def cofactor_retraction(
    df: pd.DataFrame,
    methods: list[str],
    check: str = "no_clashes_with_organic_cofactors",
    descriptor: str = "n_aromatic_rings",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """The cofactor finding, pooled (as originally published) vs conditioned.

    The original claim pooled every complex regardless of whether it carries a
    cofactor at all; the check passes almost automatically when there is no
    cofactor to clash with (Task 4), so the pooled contrast mostly measures
    receptor composition, not ligand chemistry. Restricting to
    `has_cofactors == True` - the only population in which this check can be
    informative - is what `stratum_frame` does for every check in
    `pb.strata.COFACTOR_CHECKS`. This table exists so the retraction in the
    report traces to a real, regenerable pair of numbers rather than a remark.
    """
    pooled_frame = pd.concat(
        [eligible(df[df["method"] == m], check) for m in methods], ignore_index=True
    )
    conditioned_frame = pd.concat(
        [stratum_frame(df, check, m) for m in methods], ignore_index=True
    )

    pooled = cluster_bootstrap_d(pooled_frame, check, descriptor, n_boot=n_boot, seed=seed)
    conditioned = cluster_bootstrap_d(
        conditioned_frame, check, descriptor, n_boot=n_boot, seed=seed
    )

    def _row(variant: str, est) -> dict:
        return {
            "variant": variant, "check": check, "descriptor": descriptor,
            "d": est.point, "lo": est.lo, "hi": est.hi, "p_value": est.p_value,
            "n_fail": est.n_fail, "n_eligible": est.n_eligible,
            "n_clusters": est.n_clusters,
        }

    return pd.DataFrame([
        _row("pooled_unconditioned", pooled),
        _row("cofactor_conditioned", conditioned),
    ])


def crystal_contact_distance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive symmetry-contact distances, filtered by the flag itself.

    `min_symmetry_distance_protein`/`_any` carry `float("inf")` for "the
    neighbour search ran and found nothing within its radius" (Task 5) - a
    sentinel, not a measurement. Averaging the raw column over every complex
    mixes that sentinel with genuine 4-5 A near-misses and silently returns
    inf/NaN. The contact-bearing group (flag == True) is safe by
    construction: `crystal_contact` is defined as `distance < cutoff`, so
    every True row's distance is finite by definition. Only that group is
    aggregated here.
    """
    complexes = df.drop_duplicates(subset=["pdb_id"])
    rows = []
    for flag_col, dist_col in [
        ("crystal_contact", "min_symmetry_distance_protein"),
        ("crystal_contact_any", "min_symmetry_distance_any"),
    ]:
        if flag_col not in complexes or complexes[flag_col].isna().all():
            continue
        flag = complexes[flag_col]
        contact = complexes.loc[flag.fillna(False).astype(bool), dist_col]
        rows.append({
            "measure": dist_col,
            "n_resolved": int(flag.notna().sum()),
            "n_contact": int(flag.fillna(False).sum()),
            "mean_distance_when_contact": float(contact.mean()) if len(contact) else float("nan"),
            "median_distance_when_contact": float(contact.median()) if len(contact) else float("nan"),
        })
    return pd.DataFrame(rows)


def mw_rotb_correlation(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Spearman r(mw, n_rotatable_bonds) within each method's own ligand set.

    The central "flexibility not size" claim is only unidentified if the two
    candidate descriptors are entangled everywhere, not just on average. This
    checks that per method, on exactly the ligands that reach that method's
    models (`analysis_population`, so already pose-produced).
    """
    rows = []
    for method in methods:
        sub = df[df["method"] == method].drop_duplicates(subset=["pdb_id"])
        pair = sub[["mw", "n_rotatable_bonds"]].dropna()
        rows.append({
            "method": method,
            "n_ligands": len(pair),
            "spearman_r": pair["mw"].corr(pair["n_rotatable_bonds"], method="spearman"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    TABLES.mkdir(parents=True, exist_ok=True)

    from .build import analysis_population
    from .models import DESCRIPTOR_BASIS, fit_check_model
    from .strata import association_grid, stratum_coverage

    joined = pd.read_parquet(paths.JOINED_PARQUET)
    df = analysis_population(joined)
    methods = DL_METHODS + CLASSICAL_METHODS

    log.info("population: %d gated pose rows over %d complexes, %d methods",
             len(df), df["pdb_id"].nunique(), df["method"].nunique())

    # ── headline table + data-hygiene tables (kept from the original run) ──
    raw = primary(joined, post="none")
    minimised = primary(joined, post="energy minimization")
    everything = pd.concat([raw, minimised])
    summary = method_summary(everything)
    summary.to_csv(TABLES / "method_summary.csv", index=False)

    for bin_column in REPORT_PARTITIONS:
        outcome_by_bin(raw, bin_column).to_csv(
            TABLES / f"outcome_by_{bin_column}.csv", index=False
        )
    check_rate_by_bin(raw[raw["method_class"] == "deep learning"], "rotb_bin").to_csv(
        TABLES / "check_failure_by_rotb_bin.csv", index=False
    )

    # ── coverage: which of the 126 method x check strata were even estimable ──
    coverage = stratum_coverage(df, methods)
    coverage.to_csv(TABLES / "stratum_coverage.csv", index=False)
    log.info("stratum coverage: %d estimated, %d skipped (of %d strata)",
             int(coverage["estimated"].sum()), int((~coverage["estimated"]).sum()),
             len(coverage))

    # ── the gated, clustered, FDR-controlled effect grid ──
    grid = association_grid(df, methods, DESCRIPTOR_BASIS, n_boot=2000, seed=0)
    grid.to_csv(TABLES / "association_grid.csv", index=False)
    log.info("association grid: %d estimates, %d survive BH-FDR",
             len(grid), int(grid["fdr_reject"].sum()))

    # ── the retraction, evidenced ──
    cofactor_retraction(df, methods).to_csv(TABLES / "cofactor_retraction.csv", index=False)

    # ── per-method clustered logistic models ──
    models = []
    for method in methods:
        for check in grid["check"].unique():
            fitted = fit_check_model(df, check, method)
            if not fitted.empty:
                fitted.insert(0, "check", check)
                fitted.insert(0, "method", method)
                models.append(fitted)
    model_table = pd.concat(models, ignore_index=True) if models else pd.DataFrame()
    model_table.to_csv(TABLES / "check_models.csv", index=False)

    mw_rotb_correlation(df, methods).to_csv(TABLES / "mw_rotb_correlation.csv", index=False)

    # ── crystal-contact sensitivity (Task 5's refuted-hypothesis check) ──
    if df["crystal_contact"].notna().any():
        rows = []
        for method in methods:
            for flag in (True, False):
                sub = df[(df["method"] == method) & (df["crystal_contact"] == flag)]
                sub = sub[sub["no_clashes_with_protein"].notna()]
                if len(sub):
                    rows.append({
                        "method": method,
                        "crystal_contact": flag,
                        "n": len(sub),
                        "protein_clash_failure": float(
                            (sub["no_clashes_with_protein"] == False).mean()  # noqa: E712
                        ),
                    })
        pd.DataFrame(rows).to_csv(
            TABLES / "crystal_contact_sensitivity.csv", index=False
        )
        crystal_contact_distance_summary(df).to_csv(
            TABLES / "crystal_contact_distances.csv", index=False
        )

    print(grid[grid["fdr_reject"]]
          .head(25)[["method", "check", "descriptor", "d", "lo", "hi",
                     "n_fail", "n_eligible", "n_clusters"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))


if __name__ == "__main__":
    main()
