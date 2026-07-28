import pandas as pd
import pytest

from pb.build import CHECKS


def _row(method, post, pdb_id, overrides=None, blank_all=False):
    """One results-CSV row as the published file writes it: strings, not booleans."""
    row = {
        "dataset": "posebuster",
        "method": method,
        "post-processing": post,
        "pdb_id": pdb_id,
        "ccd_id": "LIG",
        "has_cofactors": "False",
        "sequence_identity": "0.3",
        "rmsd_within_threshold": "True",
        "rmsd": "1.0",
    }
    for check in CHECKS:
        row[check] = "" if blank_all else "True"
    for key, value in (overrides or {}).items():
        row[key] = value
    return row


@pytest.fixture
def synthetic_results():
    """Four rows covering every blank-handling case that matters."""
    return pd.DataFrame([
        # all checks pass, pose produced
        _row("vina", "none", "1AAA"),
        # one real failure
        _row("diffdock", "none", "2BBB", {"no_clashes_with_protein": "False"}),
        # minimised row: flatness checks blank (not run), everything else passes
        _row("vina", "energy minimization", "1AAA", {
            "aromatic_ring_flatness_passes": "",
            "double_bond_flatness_passes": "",
        }),
        # method produced no pose at all
        _row("equibind", "none", "3CCC", {"rmsd_within_threshold": "", "rmsd": ""},
             blank_all=True),
    ])
