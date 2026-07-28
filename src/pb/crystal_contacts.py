"""Flag complexes where the ligand touches a neighbouring copy in the crystal.

The published benchmark ships 428 complexes; the journal version reports on 308
after removing those whose ligand contacts a crystallisation artifact. That
308-ID list lives in supplementary Table S2 and is not distributed with the
data, so the flag is recomputed from first principles: read the deposited entry,
expand the unit cell by the spacegroup, and find the closest approach between
the ligand of interest and any symmetry image of the protein.

This matters because protein clash is by far the most-failed check (84% of
deep-learning poses). If clash failures concentrate in contact-bearing
complexes, the headline number is partly an artifact of the benchmark.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

import gemmi
import pandas as pd

from . import paths

log = logging.getLogger(__name__)

RCSB = "https://files.rcsb.org/download"


def fetch_entry(pdb_id: str) -> Path | None:
    """Download one mmCIF entry to the cache, or return the cached copy."""
    target = paths.PDB_CACHE / f"{pdb_id.upper()}.cif.gz"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{RCSB}/{pdb_id.upper()}.cif.gz"
    try:
        urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https RCSB URL
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        log.warning("could not fetch %s: %s", pdb_id, exc)
        target.unlink(missing_ok=True)
        return None
    return target


SEARCH_RADIUS = 5.0  # A - comfortably above the 4.0 A contact cutoff


def min_symmetry_distance(path: Path, ccd_id: str) -> float | None:
    """Closest approach between the named ligand and any symmetry image.

    Returns None when the entry has no ligand of that name, or no cell and
    spacegroup to expand - i.e. the check could not be run at all. A large
    value means the ligand sits well inside its own asymmetric unit; a small
    one means it is packed against a neighbour. When no symmetry-mate atom
    falls within SEARCH_RADIUS, the check *did* run and found no contact, so
    this reports SEARCH_RADIUS itself as an honest lower bound on the true
    distance rather than None - conflating "no contact" with "unresolved"
    would misclassify the (expected) majority of true negatives as NA.
    """
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_hydrogens()

    if structure.spacegroup_hm in ("", "P 1") and structure.cell.volume <= 0:
        return None

    ligand_atoms = [
        atom
        for chain in structure[0]
        for residue in chain
        if residue.name == ccd_id
        for atom in residue
    ]
    if not ligand_atoms:
        return None

    # NOTE: installed gemmi (0.7.5) does not accept `NeighborSearch(model,
    # structure, radius)` as its brief-form call suggested - the second
    # positional argument must be a gemmi.UnitCell, not the Structure itself.
    # Passing structure.cell resolves to the (model, cell, max_radius) overload.
    search = gemmi.NeighborSearch(structure[0], structure.cell, SEARCH_RADIUS).populate()

    closest = float("inf")
    for atom in ligand_atoms:
        for mark in search.find_atoms(atom.pos, "\0", radius=SEARCH_RADIUS):
            # image_idx 0 is the original copy; anything else is a symmetry mate
            if mark.image_idx == 0:
                continue
            cra = mark.to_cra(structure[0])
            if cra.residue.name == ccd_id:
                continue
            position = structure.cell.find_nearest_pbc_position(
                atom.pos, cra.atom.pos, mark.image_idx
            )
            closest = min(closest, atom.pos.dist(position))

    return SEARCH_RADIUS if closest == float("inf") else float(closest)


def build_flags(complexes: pd.DataFrame, cutoff: float = 4.0) -> pd.DataFrame:
    """One row per complex with its closest symmetry contact.

    `cutoff` of 4.0 A is roughly a van der Waals contact between heavy atoms;
    below it the ligand is genuinely touching a neighbouring copy.
    """
    paths.ensure_dirs()
    rows = []
    for record in complexes.itertuples():
        entry = fetch_entry(record.pdb_id)
        distance = (
            min_symmetry_distance(entry, record.ccd_id) if entry is not None else None
        )
        rows.append({
            "pdb_id": record.pdb_id,
            "min_symmetry_distance": distance,
            "crystal_contact": None if distance is None else distance < cutoff,
        })

    flags = pd.DataFrame(rows)
    flags["crystal_contact"] = flags["crystal_contact"].astype("boolean")
    return flags


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    descriptors = pd.read_parquet(paths.DESCRIPTORS_PARQUET)
    complexes = descriptors[descriptors["dataset"] == "posebuster"][["pdb_id", "ccd_id"]]

    flags = build_flags(complexes)
    flags.to_parquet(paths.CRYSTAL_CONTACTS_PARQUET, index=False)

    resolved = flags["crystal_contact"].notna().sum()
    log.info("resolved %d/%d complexes", resolved, len(flags))
    log.info("crystal contacts: %d", int(flags["crystal_contact"].sum()))
    log.info("unresolved (no cell, no ligand, or fetch failed): %d",
             int(flags["crystal_contact"].isna().sum()))


if __name__ == "__main__":
    main()
