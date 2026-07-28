import pandas as pd

from pb.eligibility import ELIGIBILITY, eligible, eligible_mask


def _frame():
    return pd.DataFrame({
        "sp3_stereochemistry_preserved": pd.array([True, False, True], dtype="boolean"),
        "aromatic_ring_flatness_passes": pd.array([True, True, None], dtype="boolean"),
        "no_clashes_with_protein": pd.array([True, False, True], dtype="boolean"),
        "n_stereocentres": [0, 3, 2],
        "n_aromatic_rings": [0, 1, 2],
    })


def test_gated_check_excludes_structural_zeros():
    mask = eligible_mask(_frame(), "sp3_stereochemistry_preserved")
    assert list(mask) == [False, True, True]


def test_ungated_check_keeps_every_row_that_ran():
    mask = eligible_mask(_frame(), "no_clashes_with_protein")
    assert list(mask) == [True, True, True]


def test_rows_where_check_did_not_run_are_excluded():
    mask = eligible_mask(_frame(), "aromatic_ring_flatness_passes")
    # row 0 has zero aromatic rings, row 2 has a blank check
    assert list(mask) == [False, True, False]


def test_eligible_returns_a_subframe():
    out = eligible(_frame(), "sp3_stereochemistry_preserved")
    assert len(out) == 2
    assert (out["n_stereocentres"] > 0).all()


def test_every_check_has_an_eligibility_entry():
    from pb.build import CHECKS
    assert set(ELIGIBILITY) == set(CHECKS)
