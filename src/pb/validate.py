"""Reproduce the paper's crystal-structure rows by running PoseBusters ourselves.

Nothing else in this project executes a single check - it consumes the authors'
published results. Re-running `bust` on the 513 crystal ligands against their own
receptors reproduces the `crystal_structures` rows of that CSV, which is the only
end-to-end evidence that the checks are being interpreted correctly. Any
systematic disagreement invalidates the interpretation in the report.
"""

from __future__ import annotations

import logging

import pandas as pd
from posebusters import PoseBusters

from . import paths
from .build import load_results

log = logging.getLogger(__name__)

# bust's own column names -> the published CSV's names
COLUMN_MAP = {
    "sanitization": "molecule_passes_rdkit_sanity_check",
    "molecular_formula": "molecular_formula_preserved",
    "molecular_bonds": "molecular_bonds_preserved",
    "tetrahedral_chirality": "sp3_stereochemistry_preserved",
    "double_bond_stereochemistry": "double_bond_stereochemistry_preserved",
    "bond_lengths": "bond_lengths_within_bounds",
    "bond_angles": "bond_angles_within_bounds",
    "internal_steric_clash": "no_internal_clashes",
    "aromatic_ring_flatness": "aromatic_ring_flatness_passes",
    "double_bond_flatness": "double_bond_flatness_passes",
    "internal_energy": "energy_ratio_within_threshold",
    "minimum_distance_to_protein": "no_clashes_with_protein",
    "minimum_distance_to_organic_cofactors": "no_clashes_with_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors": "no_clashes_with_inorganic_cofactors",
    "volume_overlap_with_protein": "no_volume_clash_with_protein",
    "volume_overlap_with_organic_cofactors": "no_volume_clash_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors": "no_volume_clash_with_inorganic_cofactors",
}


def run_bust(limit: int | None = None) -> pd.DataFrame:
    """Run the redock suite on every crystal ligand against its own receptor."""
    jobs = []
    for dataset, directory in paths.SET_DIRS.items():
        for complex_dir in sorted(directory.iterdir()):
            if not complex_dir.is_dir():
                continue
            name = complex_dir.name
            jobs.append({
                "dataset": dataset,
                "pdb_id": name.partition("_")[0],
                "mol_pred": complex_dir / f"{name}_ligand.sdf",
                "mol_true": complex_dir / f"{name}_ligand.sdf",
                "mol_cond": complex_dir / f"{name}_protein.pdb",
            })
    if limit is not None:
        jobs = jobs[:limit]

    table = pd.DataFrame(jobs)
    log.info("running bust on %d crystal structures", len(table))

    buster = PoseBusters(config="redock")
    results = buster.bust_table(
        table[["mol_pred", "mol_true", "mol_cond"]], full_report=False
    ).reset_index(drop=True)

    return pd.concat([table[["dataset", "pdb_id"]], results], axis=1)


def compare_to_published(ours: pd.DataFrame) -> pd.DataFrame:
    """Per-check agreement between our run and the paper's crystal rows."""
    published = load_results()
    published = published[published["method"] == "crystal_structures"]

    renamed = ours.rename(columns=COLUMN_MAP)
    merged = renamed.merge(
        published, on=["dataset", "pdb_id"], suffixes=("_ours", "_theirs")
    )

    rows = []
    for column in COLUMN_MAP.values():
        ours_col, theirs_col = f"{column}_ours", f"{column}_theirs"
        if ours_col not in merged or theirs_col not in merged:
            continue
        both = merged[[ours_col, theirs_col]].dropna()
        agree = (both[ours_col].astype(bool) == both[theirs_col].astype(bool)).sum()
        rows.append({
            "check": column,
            "n_compared": len(both),
            "n_agree": int(agree),
            "agreement": agree / len(both) if len(both) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("agreement")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths.ensure_dirs()
    (paths.REPORTS / "tables").mkdir(parents=True, exist_ok=True)

    ours = run_bust()
    ours.to_parquet(paths.PROCESSED / "our_bust_crystal.parquet", index=False)

    agreement = compare_to_published(ours)
    agreement.to_csv(paths.REPORTS / "tables" / "bust_reproduction.csv", index=False)
    print(agreement.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))

    worst = agreement["agreement"].min()
    log.info("worst per-check agreement: %.1f%%", worst * 100)
    if worst < 0.95:
        log.error("a check disagrees on >5%% of crystal structures - "
                  "the interpretation in reports/findings.md is not safe to publish")


if __name__ == "__main__":
    main()
