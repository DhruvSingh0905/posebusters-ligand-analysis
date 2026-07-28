"""Locks the copied-in SMARTS to the installed posebusters package's own config.

`pb.descriptors` hand-copies three SMARTS patterns from
`posebusters/config/redock.yml` so eligibility gating (`n_pb_aromatic_rings`,
`n_trigonal_double_bonds`) can be computed without importing posebusters'
internal check machinery. If a posebusters upgrade changes those patterns,
this project's eligibility denominators for the two flatness checks would
silently drift out of sync with what `bust` itself actually examines. This
test reads the same config file installed in this environment and fails
loudly if the copies have diverged.
"""

from __future__ import annotations

from pathlib import Path

import posebusters
import yaml
from rdkit import Chem

from pb.descriptors import (
    _AROMATIC_RING_5,
    _AROMATIC_RING_6,
    _TRIGONAL_DOUBLE_BOND,
)

_CONFIG = Path(posebusters.__file__).parent / "config" / "redock.yml"


def _flat_systems() -> dict[str, str]:
    """Every SMARTS pattern named under any "flatness" check's flat_systems."""
    config = yaml.safe_load(_CONFIG.read_text())
    patterns: dict[str, str] = {}
    for module in config["modules"]:
        params = module.get("parameters", {})
        patterns.update(params.get("flat_systems", {}))
    return patterns


def _canonical(smarts_or_mol) -> str:
    """Canonical SMARTS text, from either a raw pattern string or a compiled Mol."""
    mol = smarts_or_mol if isinstance(smarts_or_mol, Chem.Mol) else Chem.MolFromSmarts(smarts_or_mol)
    assert mol is not None, f"SMARTS failed to parse: {smarts_or_mol}"
    return Chem.MolToSmarts(mol)


def test_config_file_exists_and_has_the_expected_keys():
    patterns = _flat_systems()
    assert "aromatic_5_membered_rings_sp2" in patterns
    assert "aromatic_6_membered_rings_sp2" in patterns
    assert "trigonal_planar_double_bonds" in patterns


def test_trigonal_double_bond_matches_installed_posebusters():
    patterns = _flat_systems()
    assert _canonical(_TRIGONAL_DOUBLE_BOND) == _canonical(
        patterns["trigonal_planar_double_bonds"]
    )


def test_aromatic_ring_5_matches_installed_posebusters():
    patterns = _flat_systems()
    assert _canonical(_AROMATIC_RING_5) == _canonical(
        patterns["aromatic_5_membered_rings_sp2"]
    )


def test_aromatic_ring_6_matches_installed_posebusters():
    patterns = _flat_systems()
    assert _canonical(_AROMATIC_RING_6) == _canonical(
        patterns["aromatic_6_membered_rings_sp2"]
    )
