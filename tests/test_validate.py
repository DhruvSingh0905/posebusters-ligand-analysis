import pandas as pd
import pytest

import pb.validate as validate


def _published_frame(rows):
    """A slice of load_results() shaped like the crystal_structures rows."""
    frame = pd.DataFrame(rows)
    frame["method"] = "crystal_structures"
    return frame


def test_agreement_excludes_na_rows_and_reflects_only_real_disagreement(monkeypatch):
    """n_compared must drop rows NA on either side; agreement must not count them."""
    monkeypatch.setattr(
        validate, "COLUMN_MAP", {"sanitization": "molecule_passes_rdkit_sanity_check"}
    )

    ours = pd.DataFrame({
        "dataset": ["posebuster"] * 5,
        "pdb_id": ["P1", "P2", "P3", "P4", "P5"],
        # P2 is a genuine disagreement; P4 is NA on our side; P5 is fine here
        # but will be NA on the published side below.
        "sanitization": pd.array([True, False, True, None, True], dtype="boolean"),
    })
    published = _published_frame({
        "dataset": ["posebuster"] * 5,
        "pdb_id": ["P1", "P2", "P3", "P4", "P5"],
        "molecule_passes_rdkit_sanity_check": pd.array(
            [True, True, True, True, None], dtype="boolean"
        ),
    })
    monkeypatch.setattr(validate, "load_results", lambda: published)

    out = validate.compare_to_published(ours)

    assert len(out) == 1
    row = out.iloc[0]
    # P4 (NA on ours) and P5 (NA on theirs) are excluded; P1, P2, P3 remain.
    assert row["n_compared"] == 3
    # Only P2 disagrees (False vs True); P1 and P3 agree.
    assert row["n_agree"] == 2
    assert row["agreement"] == pytest.approx(2 / 3)


def test_missing_mapped_column_raises_naming_it(monkeypatch):
    """A COLUMN_MAP key that no longer matches any column must fail loudly."""
    monkeypatch.setattr(
        validate,
        "COLUMN_MAP",
        {
            "sanitization": "molecule_passes_rdkit_sanity_check",
            # Simulates a posebusters upgrade that renamed this raw column -
            # it is absent from both sides, so it can never be compared.
            "ghost_raw_name": "ghost_check",
        },
    )

    ours = pd.DataFrame({
        "dataset": ["posebuster"] * 2,
        "pdb_id": ["P1", "P2"],
        "sanitization": pd.array([True, True], dtype="boolean"),
    })
    published = _published_frame({
        "dataset": ["posebuster"] * 2,
        "pdb_id": ["P1", "P2"],
        "molecule_passes_rdkit_sanity_check": pd.array([True, True], dtype="boolean"),
    })
    monkeypatch.setattr(validate, "load_results", lambda: published)

    with pytest.raises(AssertionError, match="ghost_check"):
        validate.compare_to_published(ours)
