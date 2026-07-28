import numpy as np
import pandas as pd
import pytest

from pb.strata import COFACTOR_CHECKS, association_grid, stratum_frame


@pytest.fixture
def frame():
    return pd.DataFrame({
        "method": ["diffdock"] * 4 + ["vina"] * 2,
        "no_clashes_with_organic_cofactors": pd.array(
            [True, False, True, False, True, True], dtype="boolean"),
        "no_clashes_with_protein": pd.array(
            [True, False, True, False, True, True], dtype="boolean"),
        "has_cofactors": pd.array(
            [True, True, False, False, True, False], dtype="boolean"),
        "n_stereocentres": [1, 1, 1, 1, 1, 1],
        "pdb_id": list("ABCDEF"),
    })


def test_cofactor_check_drops_cofactor_free_complexes(frame):
    out = stratum_frame(frame, "no_clashes_with_organic_cofactors", "diffdock")
    assert len(out) == 2
    assert out["has_cofactors"].all()


def test_non_cofactor_check_keeps_all_complexes(frame):
    out = stratum_frame(frame, "no_clashes_with_protein", "diffdock")
    assert len(out) == 4


def test_stratum_is_restricted_to_one_method(frame):
    out = stratum_frame(frame, "no_clashes_with_protein", "vina")
    assert set(out["method"]) == {"vina"}


def test_cofactor_checks_are_all_registered():
    assert "no_clashes_with_organic_cofactors" in COFACTOR_CHECKS
    assert "no_volume_clash_with_inorganic_cofactors" in COFACTOR_CHECKS
    assert "no_clashes_with_protein" not in COFACTOR_CHECKS


def test_stratum_frame_applies_eligibility_gating():
    """Deleting the eligible() call from stratum_frame must break this test.

    A ligand with no stereocentres cannot fail the chirality check, so it must
    not reach the estimator - its trivial pass would dilute the contrast.
    """
    frame = pd.DataFrame({
        "method": ["diffdock"] * 3,
        "sp3_stereochemistry_preserved": pd.array([True, False, True], dtype="boolean"),
        "n_stereocentres": [0, 2, 3],
        "has_cofactors": pd.array([False] * 3, dtype="boolean"),
        "pdb_id": list("ABC"),
    })
    out = stratum_frame(frame, "sp3_stereochemistry_preserved", "diffdock")
    assert list(out["pdb_id"]) == ["B", "C"]
    assert (out["n_stereocentres"] > 0).all()


def test_stratum_frame_drops_rows_where_check_never_ran():
    frame = pd.DataFrame({
        "method": ["diffdock"] * 3,
        "no_clashes_with_protein": pd.array([True, None, False], dtype="boolean"),
        "n_stereocentres": [1, 1, 1],
        "has_cofactors": pd.array([False] * 3, dtype="boolean"),
        "pdb_id": list("ABC"),
    })
    out = stratum_frame(frame, "no_clashes_with_protein", "diffdock")
    assert list(out["pdb_id"]) == ["A", "C"]


def test_association_grid_skips_thin_strata_and_flags_fdr():
    rng = np.random.default_rng(0)
    n = 120
    failed = rng.random(n) < 0.5
    frame = pd.DataFrame({
        "method": ["diffdock"] * n,
        "no_clashes_with_protein": pd.array(~failed, dtype="boolean"),
        "n_rotatable_bonds": rng.normal(5, 2, n) + failed * 2.0,
        "n_stereocentres": np.ones(n),
        "has_cofactors": pd.array([False] * n, dtype="boolean"),
        "pdb_id": [f"P{i}" for i in range(n)],
    })
    grid = association_grid(
        frame, ["diffdock"], ["n_rotatable_bonds"],
        checks=["no_clashes_with_protein"], n_boot=200, seed=0,
    )
    assert len(grid) == 1
    assert {"d", "lo", "hi", "p_value", "fdr_reject", "abs_d"} <= set(grid.columns)
    assert grid["lo"].iloc[0] < grid["d"].iloc[0] < grid["hi"].iloc[0]

    # A stratum with too few failures is skipped entirely.
    thin = frame.head(20).copy()
    thin["no_clashes_with_protein"] = pd.array([True] * 20, dtype="boolean")
    assert association_grid(
        thin, ["diffdock"], ["n_rotatable_bonds"],
        checks=["no_clashes_with_protein"], n_boot=50, seed=0,
    ).empty
