import numpy as np
import pandas as pd

from pb.models import DESCRIPTOR_BASIS, fit_check_model, variance_inflation


def test_basis_excludes_the_redundant_size_descriptor():
    assert "mw" in DESCRIPTOR_BASIS
    assert "heavy_atoms" not in DESCRIPTOR_BASIS  # rho = 0.99 with mw


def test_variance_inflation_flags_a_duplicate_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    frame = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.01, 200), "c": rng.normal(size=200)})
    vif = variance_inflation(frame, ["a", "b", "c"])
    assert vif["a"] > 10 and vif["b"] > 10
    assert vif["c"] < 5


def test_model_recovers_the_driving_descriptor():
    """Failure depends on `n_rotatable_bonds` only; `mw` is pure noise."""
    rng = np.random.default_rng(1)
    n = 600
    rot = rng.integers(0, 12, n).astype(float)
    logit = -3.0 + 0.5 * rot
    failed = rng.random(n) < 1 / (1 + np.exp(-logit))

    frame = pd.DataFrame({
        "check_x": pd.array(~failed, dtype="boolean"),
        "n_rotatable_bonds": rot,
        "mw": rng.normal(400, 80, n),
        "pdb_id": [f"P{i}" for i in range(n)],
        "method": "diffdock",
        "has_cofactors": pd.array([False] * n, dtype="boolean"),
        "n_stereocentres": np.ones(n),
    })

    out = fit_check_model(frame, "check_x", "diffdock",
                          descriptors=["n_rotatable_bonds", "mw"])
    driver = out.set_index("descriptor").loc["n_rotatable_bonds"]
    noise = out.set_index("descriptor").loc["mw"]

    assert driver["coef"] > 0
    assert driver["p_value"] < 0.01
    assert noise["p_value"] > 0.05


def test_clustered_errors_widen_with_repeated_ligands():
    """Each ligand appears once per method, so SEs must not shrink with copies."""
    rng = np.random.default_rng(2)
    n = 300
    rot = rng.integers(0, 12, n).astype(float)
    failed = rng.random(n) < 1 / (1 + np.exp(-(-3.0 + 0.5 * rot)))
    base = pd.DataFrame({
        "check_x": pd.array(~failed, dtype="boolean"),
        "n_rotatable_bonds": rot,
        "mw": rng.normal(400, 80, n),
        "pdb_id": [f"P{i}" for i in range(n)],
    })
    duplicated = pd.concat([base] * 4, ignore_index=True)

    one = fit_check_model(base, "check_x", "diffdock",
                          descriptors=["n_rotatable_bonds", "mw"])
    many = fit_check_model(duplicated, "check_x", "diffdock",
                           descriptors=["n_rotatable_bonds", "mw"])

    se_one = one.set_index("descriptor").loc["n_rotatable_bonds", "std_err"]
    se_many = many.set_index("descriptor").loc["n_rotatable_bonds", "std_err"]
    # Naive (unclustered) SEs would fall by about sqrt(4); clustering must not.
    assert se_many > 0.7 * se_one
