"""Compute per-ligand RDKit descriptors for the benchmark crystal ligands.

Descriptors are taken from the *crystal* ligand, not from any prediction: they
are intrinsic properties of the molecule that was docked, so they partition the
complexes independently of which method produced a pose.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from . import paths

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}
HALOGENS = {"F", "Cl", "Br", "I"}

# Counts of the substructures PoseBusters' flatness checks actually examine,
# using that tool's own SMARTS (posebusters/config/redock.yml). Gating the
# flatness checks on these makes eligibility mean "contains something the
# check looks at" rather than relying on a chemically adjacent proxy.
_TRIGONAL_DOUBLE_BOND = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
_AROMATIC_RING_5 = Chem.MolFromSmarts("[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1")
_AROMATIC_RING_6 = Chem.MolFromSmarts("[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1")


@dataclass
class LigandDescriptors:
    """One row per benchmark complex."""

    dataset: str
    pdb_id: str
    ccd_id: str
    smiles: str | None = None
    parse_ok: bool = False
    sanitize_ok: bool = False

    # size
    mw: float | None = None
    heavy_atoms: int | None = None
    n_frags: int | None = None

    # flexibility
    n_rotatable_bonds: int | None = None
    rot_per_heavy_atom: float | None = None
    fraction_csp3: float | None = None
    n_amide_bonds: int | None = None

    # rings
    n_rings: int | None = None
    n_aromatic_rings: int | None = None
    n_aliphatic_rings: int | None = None
    largest_ring_size: int | None = None
    is_macrocycle: bool | None = None
    n_spiro_atoms: int | None = None
    n_bridgehead_atoms: int | None = None

    # stereo
    n_stereocentres: int | None = None
    n_unassigned_stereocentres: int | None = None
    n_stereo_double_bonds: int | None = None
    n_trigonal_double_bonds: int | None = None
    n_pb_aromatic_rings: int | None = None

    # polarity / composition
    tpsa: float | None = None
    clogp: float | None = None
    hbd: int | None = None
    hba: int | None = None
    formal_charge: int | None = None
    n_heteroatoms: int | None = None
    n_halogens: int | None = None
    has_metal: bool | None = None


def _load_mol(sdf: Path) -> tuple[Chem.Mol | None, bool]:
    """Load the first record of an SDF.

    Returns (mol, sanitize_ok). PDB-derived ligands occasionally fail RDKit's
    full sanitization (unusual valences, metal coordination written as bonds),
    so fall back to an unsanitized load and repair what we can. A molecule that
    only loads unsanitized still yields the composition descriptors; the ones
    that need ring perception and hybridisation are left as None.
    """
    supplier = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True)
    mol = next(iter(supplier), None)
    if mol is not None:
        return mol, True

    supplier = Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
    mol = next(iter(supplier), None)
    if mol is None:
        return None, False

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            catchErrors=True,
        )
    except Exception:  # noqa: BLE001 - keep whatever partially-sane mol we have
        pass
    return mol, False


def _stereo_counts(mol: Chem.Mol) -> tuple[int, int, int]:
    """(tetrahedral centres, unassigned centres, stereogenic double bonds)."""
    try:
        elements = Chem.FindPotentialStereo(mol)
    except Exception:  # noqa: BLE001
        return (0, 0, 0)

    centres = unassigned = double_bonds = 0
    for element in elements:
        kind = str(element.type)
        if "Atom_Tetrahedral" in kind:
            centres += 1
            if str(element.specified) != "Specified":
                unassigned += 1
        elif "Bond_Double" in kind:
            double_bonds += 1
    return centres, unassigned, double_bonds


def describe(sdf: Path, dataset: str, pdb_id: str, ccd_id: str) -> LigandDescriptors:
    """Compute every descriptor we can for one crystal ligand."""
    row = LigandDescriptors(dataset=dataset, pdb_id=pdb_id, ccd_id=ccd_id)

    mol, sanitize_ok = _load_mol(sdf)
    if mol is None:
        log.warning("could not parse %s", sdf)
        return row

    row.parse_ok = True
    row.sanitize_ok = sanitize_ok

    # Composition — available even on a partially sanitized molecule.
    symbols = [a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    row.heavy_atoms = len(symbols)
    row.formal_charge = Chem.GetFormalCharge(mol)
    row.n_heteroatoms = sum(1 for s in symbols if s not in ("C",))
    row.n_halogens = sum(1 for s in symbols if s in HALOGENS)
    row.has_metal = any(s in METALS for s in symbols)

    try:
        row.smiles = Chem.MolToSmiles(mol)
        row.n_frags = len(Chem.GetMolFrags(mol))
    except Exception:  # noqa: BLE001
        pass

    if not sanitize_ok:
        return row

    row.mw = Descriptors.MolWt(mol)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True)
    row.n_rotatable_bonds = n_rot
    if row.heavy_atoms:
        row.rot_per_heavy_atom = n_rot / row.heavy_atoms
    row.fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    row.n_amide_bonds = rdMolDescriptors.CalcNumAmideBonds(mol)

    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    row.n_rings = len(ring_sizes)
    row.largest_ring_size = max(ring_sizes) if ring_sizes else 0
    row.is_macrocycle = row.largest_ring_size >= 12
    row.n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    row.n_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    row.n_spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    row.n_bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    centres, unassigned, stereo_bonds = _stereo_counts(mol)
    row.n_stereocentres = centres
    row.n_unassigned_stereocentres = unassigned
    row.n_stereo_double_bonds = stereo_bonds

    row.n_trigonal_double_bonds = len(
        mol.GetSubstructMatches(_TRIGONAL_DOUBLE_BOND, uniquify=True)
    )
    row.n_pb_aromatic_rings = len(
        mol.GetSubstructMatches(_AROMATIC_RING_5, uniquify=True)
    ) + len(mol.GetSubstructMatches(_AROMATIC_RING_6, uniquify=True))

    row.tpsa = rdMolDescriptors.CalcTPSA(mol)
    row.clogp = Crippen.MolLogP(mol)
    row.hbd = rdMolDescriptors.CalcNumHBD(mol)
    row.hba = rdMolDescriptors.CalcNumHBA(mol)

    return row


def build_table() -> pd.DataFrame:
    """Walk both benchmark sets and describe every crystal ligand."""
    rows: list[dict] = []

    for dataset, directory in paths.SET_DIRS.items():
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} missing - run `python -m pb.acquire` first"
            )
        for complex_dir in sorted(directory.iterdir()):
            if not complex_dir.is_dir():
                continue
            pdb_id, _, ccd_id = complex_dir.name.partition("_")
            sdf = complex_dir / f"{complex_dir.name}_ligand.sdf"
            if not sdf.exists():
                log.warning("no ligand sdf in %s", complex_dir)
                continue
            rows.append(asdict(describe(sdf, dataset, pdb_id, ccd_id)))

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths.ensure_dirs()

    table = build_table()
    table.to_parquet(paths.DESCRIPTORS_PARQUET, index=False)

    log.info("described %d ligands -> %s", len(table), paths.DESCRIPTORS_PARQUET)
    log.info("  parse failures    : %d", (~table.parse_ok).sum())
    log.info("  sanitize failures : %d", (~table.sanitize_ok).sum())
    for dataset, group in table.groupby("dataset"):
        log.info("  %-11s n=%d", dataset, len(group))


if __name__ == "__main__":
    main()
