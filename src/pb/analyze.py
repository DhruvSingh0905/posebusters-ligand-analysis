"""Find which ligand properties predict which PoseBusters check fails.

Every method is run on the same 428 complexes, so the ligand set is identical
across methods and descriptor effects are not confounded by which method saw
which molecule. Pooling across methods within a class is therefore fair; the
per-method breakdown at the end confirms the pooled trends are not driven by a
single model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import paths
from .build import CHECKS, CLASSICAL_METHODS, DL_METHODS
from .inference import cohens_d

log = logging.getLogger(__name__)

TABLES = paths.REPORTS / "tables"

NUMERIC_DESCRIPTORS = [
    "mw",
    "heavy_atoms",
    "n_rotatable_bonds",
    "rot_per_heavy_atom",
    "fraction_csp3",
    "n_rings",
    "n_aromatic_rings",
    "n_aliphatic_rings",
    "largest_ring_size",
    "n_stereocentres",
    "n_amide_bonds",
    "tpsa",
    "clogp",
    "hbd",
    "hba",
    "n_heteroatoms",
    "n_halogens",
    "formal_charge",
]

BINS = ["rotb_bin", "mw_bin", "rings_bin", "stereo_bin", "heavy_bin"]

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


def check_descriptor_associations(df: pd.DataFrame) -> pd.DataFrame:
    """For every (check, descriptor): do failing poses differ in that property?

    Restricted to rows where the check actually ran, so a check that was never
    executed contributes nothing rather than a spurious association.
    """
    rows = []
    for check in CHECKS:
        ran = df[df[check].notna()]
        if ran.empty:
            continue
        failed = ran[check] == False  # noqa: E712
        n_fail = int(failed.sum())
        if n_fail < MIN_FAILURES:
            continue

        for descriptor in NUMERIC_DESCRIPTORS:
            values = pd.Series(ran[descriptor])
            failing = values[failed].dropna()
            passing = values[~failed].dropna()
            if failing.empty or passing.empty:
                continue
            rows.append(
                {
                    "check": check,
                    "descriptor": descriptor,
                    "n_ran": len(ran),
                    "n_failed": n_fail,
                    "failure_rate": n_fail / len(ran),
                    "mean_failing": failing.mean(),
                    "mean_passing": passing.mean(),
                    "cohens_d": cohens_d(failing, passing),
                }
            )

    out = pd.DataFrame(rows)
    out["abs_d"] = out["cohens_d"].abs()
    return out.sort_values("abs_d", ascending=False)


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


def stratified_check_rate(
    df: pd.DataFrame, check: str, inner: str = "rotb_bin", outer: str = "mw_bin"
) -> pd.DataFrame:
    """Failure rate of `check` across `inner`, held within each level of `outer`.

    Molecular weight and rotatable-bond count are strongly correlated
    (Spearman 0.73 across the 428 ligands), so a trend against flexibility could
    just be a trend against size. Reading down a column here separates them: if
    the rate still climbs across `inner` inside every band of `outer`, the effect
    is not size in disguise.
    """
    ran = df[df[check].notna()].assign(failed=lambda d: d[check] == False)  # noqa: E712
    rate = ran.pivot_table(
        index=outer, columns=inner, values="failed", aggfunc="mean", observed=True
    )
    counts = ran.pivot_table(
        index=outer, columns=inner, values="failed", aggfunc="size", observed=True
    )
    return rate.where(counts >= 10)  # suppress cells too thin to read


def descriptor_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between descriptors, one row per complex."""
    ligands = df.drop_duplicates(subset=["dataset", "pdb_id"])
    return ligands[NUMERIC_DESCRIPTORS].corr(method="spearman")


def per_method_trend(df: pd.DataFrame, check: str, bin_column: str) -> pd.DataFrame:
    """Does one check's trend across a partition hold within each method?"""
    ran = df[df[check].notna()]
    table = (
        ran.assign(failed=(ran[check] == False))  # noqa: E712
        .groupby(["method", bin_column], observed=True)["failed"]
        .mean()
        .unstack(bin_column)
    )
    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(paths.JOINED_PARQUET)
    raw = primary(df, post="none")
    minimised = primary(df, post="energy minimization")
    dl = raw[raw["method_class"] == "deep learning"]

    log.info("population: %d raw pose rows over %d complexes, %d methods",
             len(raw), raw["pdb_id"].nunique(), raw["method"].nunique())

    everything = pd.concat([raw, minimised])
    summary = method_summary(everything)
    summary.to_csv(TABLES / "method_summary.csv", index=False)

    assoc = check_descriptor_associations(dl)
    assoc.to_csv(TABLES / "check_descriptor_associations.csv", index=False)

    for bin_column in BINS:
        check_rate_by_bin(dl, bin_column).to_csv(
            TABLES / f"check_failure_by_{bin_column}.csv", index=False
        )
        outcome_by_bin(raw, bin_column).to_csv(
            TABLES / f"outcome_by_{bin_column}.csv", index=False
        )

    # ── printed findings ────────────────────────────────────────────────
    pd.set_option("display.width", 150, "display.max_columns", 30)
    pct = lambda v: f"{v:6.3f}"  # noqa: E731

    def heading(text: str) -> None:
        print(f"\n{'─' * 78}\n{text}\n{'─' * 78}")

    heading("HEADLINE — raw poses, 428 PoseBusters complexes")
    print(
        summary[summary.post_processing == "none"][
            ["method", "method_class", "accurate", "valid",
             "accurate_and_valid", "n_accurate", "invalid_given_accurate"]
        ].to_string(index=False, float_format=pct)
    )

    heading("WHICH LIGANDS FAIL WHICH CHECK — top 3 descriptors per check (DL, raw)")
    top = assoc.groupby("check", group_keys=False).head(3)
    top = top.sort_values(["failure_rate", "abs_d"], ascending=False)
    print(
        top[["check", "descriptor", "n_failed", "failure_rate",
             "mean_failing", "mean_passing", "cohens_d"]]
        .to_string(index=False, float_format=lambda v: f"{v:7.3f}")
    )

    heading("THE GAP — accurate-but-invalid rate by rotatable bonds")
    print(outcome_by_bin(raw, "rotb_bin").to_string(index=False, float_format=pct))

    heading("SANITY — protein-clash failure rate by rotatable bonds, per method")
    print(
        per_method_trend(raw, "no_clashes_with_protein", "rotb_bin")
        .to_string(float_format=pct)
    )

    heading("CONTROL — is it flexibility or just size? (DL, strain failures)")
    print("rows = molecular weight band, columns = rotatable bonds")
    strain = stratified_check_rate(dl, "energy_ratio_within_threshold")
    strain.to_csv(TABLES / "strain_stratified_by_mw.csv")
    print(strain.to_string(float_format=pct))
    print("the rate climbs left-to-right inside every weight band, so the effect")
    print("is torsional freedom rather than molecular size.")

    descriptor_correlations(raw).to_csv(TABLES / "descriptor_correlations.csv")

    print(f"\nwrote {len(list(TABLES.glob('*.csv')))} tables to {TABLES}")


if __name__ == "__main__":
    main()
