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


def min_symmetry_distance(path: Path, ccd_id: str) -> tuple[float, float] | None:
    """Closest approach to a symmetry image: (protein-only, any non-ligand).

    Two numbers because two questions. The PoseBusters clash check this exists
    to interrogate compares the ligand against *protein* atoms, so the protein
    -only distance is the one that speaks to it. But a ligand packed against a
    symmetry mate's ordered water or cryoprotectant is still lattice-influenced,
    so the broader distance is kept alongside rather than thrown away.

    Returns None only when the question cannot be asked at all - no cell, no
    spacegroup, or no ligand of that name in the entry.
    """
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_hydrogens()

    # A structure with no spacegroup or no real cell cannot be expanded, so the
    # question is unanswerable rather than answered "no contact". P 1 is NOT
    # disqualifying - a real P1 crystal still packs against its lattice
    # translations, and all 22 P 1 entries here carry genuine physical cells.
    if not structure.spacegroup_hm or structure.cell.volume <= 1.0:
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
    radius = 5.0
    search = gemmi.NeighborSearch(structure[0], structure.cell, radius).populate()

    # "Nothing found" is reported as inf, never as the search radius itself -
    # a censored floor value would be indistinguishable from a real
    # measurement in any downstream histogram or correlation. inf < cutoff is
    # still False, so the boolean flag logic is unaffected.
    closest_protein = float("inf")
    closest_any = float("inf")
    for atom in ligand_atoms:
        for mark in search.find_atoms(atom.pos, "\0", radius=radius):
            # image_idx 0 is the original copy; anything else is a symmetry mate
            if mark.image_idx == 0:
                continue
            cra = mark.to_cra(structure[0])
            if cra.residue.name == ccd_id:
                continue
            position = structure.cell.find_nearest_pbc_position(
                atom.pos, cra.atom.pos, mark.image_idx
            )
            distance = atom.pos.dist(position)
            closest_any = min(closest_any, distance)

            info = gemmi.find_tabulated_residue(cra.residue.name)
            is_protein = info is not None and info.is_amino_acid()
            if is_protein:
                closest_protein = min(closest_protein, distance)

    return closest_protein, closest_any


def build_flags(complexes: pd.DataFrame, cutoff: float = 4.0) -> pd.DataFrame:
    """One row per complex with its closest symmetry contact.

    `cutoff` of 4.0 A is roughly a van der Waals contact between heavy atoms;
    below it the ligand is genuinely touching a neighbouring copy.

    `crystal_contact` is the primary flag - protein-only distance below
    cutoff - and is what `build.py` and later tasks consume. `crystal_contact_any`
    additionally counts contacts against solvent/cryoprotectant symmetry
    mates, which are lattice packing but not what the protein-clash check
    compares against.
    """
    paths.ensure_dirs()
    rows = []
    for record in complexes.itertuples():
        entry = fetch_entry(record.pdb_id)
        result = (
            min_symmetry_distance(entry, record.ccd_id) if entry is not None else None
        )
        protein, any_distance = result if result is not None else (None, None)
        rows.append({
            "pdb_id": record.pdb_id,
            "min_symmetry_distance_protein": protein,
            "min_symmetry_distance_any": any_distance,
            "crystal_contact": None if protein is None else protein < cutoff,
            "crystal_contact_any": None if any_distance is None else any_distance < cutoff,
        })

    flags = pd.DataFrame(rows)
    flags["crystal_contact"] = flags["crystal_contact"].astype("boolean")
    flags["crystal_contact_any"] = flags["crystal_contact_any"].astype("boolean")
    return flags


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    descriptors = pd.read_parquet(paths.DESCRIPTORS_PARQUET)
    complexes = descriptors[descriptors["dataset"] == "posebuster"][["pdb_id", "ccd_id"]]

    flags = build_flags(complexes)
    flags.to_parquet(paths.CRYSTAL_CONTACTS_PARQUET, index=False)

    resolved = flags["crystal_contact"].notna().sum()
    log.info("resolved %d/%d complexes", resolved, len(flags))
    log.info("crystal contacts (protein-only, primary): %d",
             int(flags["crystal_contact"].sum()))
    log.info("crystal contacts (any non-ligand, incl. solvent/cryoprotectant): %d",
             int(flags["crystal_contact_any"].sum()))
    log.info("unresolved (no cell, no spacegroup, no matching ligand, or fetch failed): %d",
             int(flags["crystal_contact"].isna().sum()))


if __name__ == "__main__":
    main()
