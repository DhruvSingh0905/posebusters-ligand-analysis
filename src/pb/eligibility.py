"""Which ligands are even capable of failing each check.

A ligand with no stereocentres cannot fail the chirality check; one with no
aromatic rings cannot fail aromatic-ring flatness. Counting those ligands in a
check's denominator makes the check look more reliable than it is, and inflates
any effect size measured against the gating descriptor - the comparison ends up
encoding "can this check fire at all" rather than "does this molecule induce
failure". Every statistic in this project is computed on eligible rows only.
"""

from __future__ import annotations

import logging

import pandas as pd

from .build import CHECKS

log = logging.getLogger(__name__)

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
    "aromatic_ring_flatness_passes": "n_pb_aromatic_rings",
    "double_bond_flatness_passes": "n_trigonal_double_bonds",
    "energy_ratio_within_threshold": None,
    # The cofactor checks pass trivially on complexes that have no cofactors -
    # 4,281 of 4,410 such rows - so `eligible()` alone does not make them
    # meaningful. They are gated by the receptor, not the ligand, so the
    # conditioning lives in pb.strata.COFACTOR_CHECKS (Task 4) instead. Do not
    # report these four checks from `eligible()` output alone.
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
        values = df[gate]
        unknown = int(values.isna().sum())
        if unknown:
            log.warning(
                "%s: %d rows have an unknown %s and are excluded from the "
                "eligible set - check the descriptor stage",
                check, unknown, gate,
            )
        mask = mask & (values > 0)
    return pd.Series(mask, index=df.index).fillna(False).astype(bool)


def eligible(df: pd.DataFrame, check: str) -> pd.DataFrame:
    """Subframe of rows eligible for `check`."""
    return df[eligible_mask(df, check)]
