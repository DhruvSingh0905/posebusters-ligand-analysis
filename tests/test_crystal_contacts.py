import pandas as pd

from pb import crystal_contacts


def test_unresolvable_complex_stays_na_not_false(monkeypatch):
    """A structure we could not evaluate must never read as 'no contact'."""
    monkeypatch.setattr(crystal_contacts, "fetch_entry", lambda pdb_id: None)
    flags = crystal_contacts.build_flags(
        pd.DataFrame({"pdb_id": ["1ABC"], "ccd_id": ["LIG"]})
    )
    assert flags["crystal_contact"].dtype == "boolean"
    assert flags["crystal_contact"].isna().all()
    assert not (flags["crystal_contact"] == False).any()  # noqa: E712


def test_flag_splits_on_the_cutoff(monkeypatch, tmp_path):
    """Protein-only distance drives crystal_contact; the broad one drives _any."""
    monkeypatch.setattr(crystal_contacts, "fetch_entry", lambda pdb_id: tmp_path)
    distances = {"CLOSE": (3.5, 3.5), "FAR": (9.0, 9.0), "WATERONLY": (float("inf"), 3.2)}
    monkeypatch.setattr(
        crystal_contacts, "min_symmetry_distance",
        lambda path, ccd_id: distances[ccd_id],
    )
    flags = crystal_contacts.build_flags(
        pd.DataFrame({"pdb_id": ["A", "B", "C"], "ccd_id": ["CLOSE", "FAR", "WATERONLY"]})
    ).set_index("pdb_id")

    assert bool(flags.loc["A", "crystal_contact"]) is True
    assert bool(flags.loc["B", "crystal_contact"]) is False
    # a water-only contact is lattice packing but not a protein contact
    assert bool(flags.loc["C", "crystal_contact"]) is False
    assert bool(flags.loc["C", "crystal_contact_any"]) is True
