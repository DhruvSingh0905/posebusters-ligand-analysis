import pandas as pd
import pytest

from pb.strata import COFACTOR_CHECKS, stratum_frame


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
