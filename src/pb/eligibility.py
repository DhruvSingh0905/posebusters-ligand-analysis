"""Which ligands are even capable of failing each check.

A ligand with no stereocentres cannot fail the chirality check; one with no
aromatic rings cannot fail aromatic-ring flatness. Counting those ligands in a
check's denominator makes the check look more reliable than it is, and inflates
any effect size measured against the gating descriptor - the comparison ends up
encoding "can this check fire at all" rather than "does this molecule induce
failure". Every statistic in this project is computed on eligible rows only.
"""

from __future__ import annotations

import pandas as pd

from .build import CHECKS

# check -> descriptor that must be non-zero for the check to be able to fail.
# None means every produced pose is eligible.
ELIGIBILITY: dict[str, str | None] = {
    "docked_ligand_successfully_loaded": None,
    "molecule_passes_rdkit_sanity_check": None,
    "molecular_formula_preserved": None,
    "molecular_bonds_preserved": None,
    "sp3_stereochemistry_preserved": "n_stereocentres",
    "double_bond_stereochemistry_preserved": "n_stereo_double_bonds",
    "bond_lengths_within_bounds": None,
    "bond_angles_within_bounds": None,
    "no_internal_clashes": None,
    "aromatic_ring_flatness_passes": "n_aromatic_rings",
    "double_bond_flatness_passes": "n_stereo_double_bonds",
    "energy_ratio_within_threshold": None,
    # cofactor checks are gated by the complex, not the ligand - handled by
    # stratification in pb.analyze rather than here, because `has_cofactors`
    # describes the receptor.
    "no_clashes_with_protein": None,
    "no_clashes_with_organic_cofactors": None,
    "no_clashes_with_inorganic_cofactors": None,
    "no_volume_clash_with_protein": None,
    "no_volume_clash_with_organic_cofactors": None,
    "no_volume_clash_with_inorganic_cofactors": None,
}

assert set(ELIGIBILITY) == set(CHECKS), "eligibility table out of sync with CHECKS"


def eligible_mask(df: pd.DataFrame, check: str) -> pd.Series:
    """Rows where `check` both ran and could have failed."""
    if check not in ELIGIBILITY:
        raise KeyError(f"unknown check: {check}")

    mask = df[check].notna()
    gate = ELIGIBILITY[check]
    if gate is not None:
        mask = mask & (df[gate].fillna(0) > 0)
    return pd.Series(mask, index=df.index).fillna(False).astype(bool)


def eligible(df: pd.DataFrame, check: str) -> pd.DataFrame:
    """Subframe of rows eligible for `check`."""
    return df[eligible_mask(df, check)]
