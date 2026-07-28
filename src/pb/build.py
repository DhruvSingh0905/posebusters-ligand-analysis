"""Join the published PoseBusters results to per-ligand descriptors.

The results CSV encodes every check as the string "True", "False" or "" and the
empty string means *the check was not run*, not that it failed. Two columns are
blank for every energy-minimised row, so reading blanks as failures makes each
minimised pose look invalid - including the classical methods', which is
nonsense. Everything here goes through a nullable boolean so a missing check
never counts against a pose, and `n_checks_run` records how many actually
applied.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import paths

log = logging.getLogger(__name__)

# The 18 physical-validity checks, in the order the paper's ladder applies them.
# RMSD is deliberately not in this list: it measures accuracy, not validity.
CHECKS = [
    # tier 1 - is it a molecule
    "docked_ligand_successfully_loaded",
    "molecule_passes_rdkit_sanity_check",
    # tier 2 - is it the same molecule
    "molecular_formula_preserved",
    "molecular_bonds_preserved",
    "sp3_stereochemistry_preserved",
    "double_bond_stereochemistry_preserved",
    # tier 3 - is its own geometry physical
    "bond_lengths_within_bounds",
    "bond_angles_within_bounds",
    "no_internal_clashes",
    "aromatic_ring_flatness_passes",
    "double_bond_flatness_passes",
    "energy_ratio_within_threshold",
    # tier 4 - does it fit the protein
    "no_clashes_with_protein",
    "no_clashes_with_organic_cofactors",
    "no_clashes_with_inorganic_cofactors",
    "no_volume_clash_with_protein",
    "no_volume_clash_with_organic_cofactors",
    "no_volume_clash_with_inorganic_cofactors",
]

TIERS = {
    "loading": CHECKS[0:2],
    "identity": CHECKS[2:6],
    "internal_geometry": CHECKS[6:12],
    "protein_fit": CHECKS[12:18],
}

ACCURACY = "rmsd_within_threshold"

DL_METHODS = ["deepdock", "diffdock", "equibind", "tankbind", "unimol"]
CLASSICAL_METHODS = ["gold", "vina"]


def _to_nullable_bool(series: pd.Series) -> pd.Series:
    """Map "True"/"False"/"" onto True/False/<NA>."""
    return series.map({"True": True, "False": False}).astype("boolean")


def load_results() -> pd.DataFrame:
    """Read the published results table with blanks preserved as <NA>."""
    df = pd.read_csv(paths.RESULTS_CSV, dtype=str, keep_default_na=False)

    for column in [*CHECKS, ACCURACY, "has_cofactors"]:
        df[column] = _to_nullable_bool(pd.Series(df[column]))
    df["rmsd"] = pd.to_numeric(df["rmsd"], errors="coerce")
    df["sequence_identity"] = pd.to_numeric(df["sequence_identity"], errors="coerce")

    return df


def add_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-pose validity summaries from the individual checks."""
    checks = df[CHECKS]

    # A pose is invalid if any check that ran came back False. Checks that did
    # not run are silent - they can neither pass nor fail a pose.
    df["n_checks_run"] = checks.notna().sum(axis=1)
    df["n_checks_failed"] = (checks == False).sum(axis=1)  # noqa: E712

    # A row where nothing ran is a pose the method never produced (or whose
    # minimisation failed), not a flawless one. Without this guard "no output"
    # scores as valid and inflates every validity rate.
    df["pose_produced"] = df["n_checks_run"] > 0
    df["pb_valid"] = df["pose_produced"] & (df["n_checks_failed"] == 0)

    for tier, columns in TIERS.items():
        df[f"fails_{tier}"] = (df[columns] == False).any(axis=1)  # noqa: E712

    # The two outcomes the whole project turns on.
    df["accurate"] = df[ACCURACY].fillna(False).astype(bool)
    df["accurate_and_valid"] = df["accurate"] & df["pb_valid"]
    df["accurate_but_invalid"] = df["accurate"] & ~df["pb_valid"]

    df["method_class"] = "other"
    df.loc[df["method"].isin(DL_METHODS), "method_class"] = "deep learning"
    df.loc[df["method"].isin(CLASSICAL_METHODS), "method_class"] = "classical"
    df.loc[df["method"] == "crystal_structures", "method_class"] = "reference"

    return df


def join_descriptors(results: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Attach ligand descriptors to every pose row."""
    key = ["dataset", "pdb_id"]

    # ccd_id lives on both sides; keep the results' copy and check they agree.
    merged = results.merge(
        descriptors.drop(columns=["ccd_id"]),
        on=key,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        missing = merged.loc[unmatched, key].drop_duplicates()
        raise ValueError(f"{len(missing)} complexes had no descriptors:\n{missing}")
    merged = merged.drop(columns=["_merge"])

    ccd_lookup = descriptors.set_index(key)["ccd_id"]
    expected = merged.set_index(key).index.map(ccd_lookup)
    mismatched = merged["ccd_id"].to_numpy() != expected.to_numpy()
    if mismatched.any():
        raise ValueError(
            f"ccd_id disagrees between results and archive for "
            f"{merged.loc[mismatched, key].drop_duplicates().to_dict('records')}"
        )

    return merged


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse ligand classes for partitioned reporting.

    Edges are chosen so each bin holds a usable number of complexes rather than
    to round numbers; the counts are printed at build time so they can be
    checked.
    """
    df["mw_bin"] = pd.cut(
        df["mw"],
        bins=[0, 300, 400, 500, 10_000],
        labels=["<300", "300-400", "400-500", "500+"],
    )
    df["rotb_bin"] = pd.cut(
        df["n_rotatable_bonds"],
        bins=[-1, 2, 5, 9, 100],
        labels=["0-2", "3-5", "6-9", "10+"],
    )
    df["rings_bin"] = pd.cut(
        df["n_rings"],
        bins=[-1, 0, 1, 2, 3, 100],
        labels=["0", "1", "2", "3", "4+"],
    )
    df["stereo_bin"] = pd.cut(
        df["n_stereocentres"],
        bins=[-1, 0, 1, 2, 100],
        labels=["0", "1", "2", "3+"],
    )
    df["heavy_bin"] = pd.cut(
        df["heavy_atoms"],
        bins=[0, 20, 30, 40, 1000],
        labels=["<20", "20-29", "30-39", "40+"],
    )
    return df


def analysis_population(
    df: pd.DataFrame, post: str = "none", dataset: str = "posebuster"
) -> pd.DataFrame:
    """The rows every reported statistic is computed on.

    Excludes the crystal-structure reference rows and any pose the method never
    produced, so downstream code never has to remember to filter them.
    """
    return df[
        (df["dataset"] == dataset)
        & (df["post-processing"] == post)
        & (df["method"].isin(DL_METHODS + CLASSICAL_METHODS))
        & df["pose_produced"]
    ].copy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths.ensure_dirs()

    if not paths.DESCRIPTORS_PARQUET.exists():
        raise FileNotFoundError("run `python -m pb.descriptors` first")

    results = load_results()
    descriptors = pd.read_parquet(paths.DESCRIPTORS_PARQUET)

    df = join_descriptors(results, descriptors)
    df = add_validity(df)
    df = add_bins(df)

    if paths.CRYSTAL_CONTACTS_PARQUET.exists():
        flags = pd.read_parquet(paths.CRYSTAL_CONTACTS_PARQUET)
        df = df.merge(flags, on="pdb_id", how="left")
        log.info("crystal-contact flag joined: %d complexes flagged",
                 int(flags["crystal_contact"].sum()))
    else:
        df["crystal_contact"] = pd.array([None] * len(df), dtype="boolean")
        df["crystal_contact_any"] = pd.array([None] * len(df), dtype="boolean")
        df["min_symmetry_distance_protein"] = float("nan")
        df["min_symmetry_distance_any"] = float("nan")
        log.warning("no crystal-contact flags - run `python -m pb.crystal_contacts`")

    df.to_parquet(paths.JOINED_PARQUET, index=False)
    df.to_csv(paths.JOINED_CSV, index=False)

    log.info("joined %d pose rows -> %s", len(df), paths.JOINED_PARQUET)
    log.info("  complexes  : %d", df.groupby(["dataset", "pdb_id"]).ngroups)
    log.info("  methods    : %d", df["method"].nunique())
    log.info("  descriptors: %d columns", len(descriptors.columns) - 3)

    log.info("checks-run distribution (18 raw, 16 minimised, 0 = no pose produced):")
    counts = df.groupby(["post-processing", "n_checks_run"]).size()
    for key, count in counts.items():
        post, n_run = key  # type: ignore[misc]
        log.info("  %-22s %2d checks  n=%d", post, n_run, count)
    log.info("  poses never produced: %d", int((~df["pose_produced"]).sum()))

    log.info("bin occupancy on the %d PoseBusters complexes:", 428)
    unique = df[df.dataset == "posebuster"].drop_duplicates(subset=["pdb_id"])
    for column in ["mw_bin", "rotb_bin", "rings_bin", "stereo_bin", "heavy_bin"]:
        counts = unique[column].value_counts().sort_index().to_dict()
        log.info("  %-11s %s", column, counts)


if __name__ == "__main__":
    main()
