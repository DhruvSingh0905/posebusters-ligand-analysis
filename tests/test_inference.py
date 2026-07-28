import numpy as np
import pandas as pd

from pb.inference import Estimate, benjamini_hochberg, cluster_bootstrap_d, cohens_d


def test_cohens_d_sign_and_scale():
    high = pd.Series([10.0, 11.0, 12.0, 11.0])
    low = pd.Series([1.0, 2.0, 3.0, 2.0])
    assert cohens_d(high, low) > 2
    assert cohens_d(low, high) < -2


def test_benjamini_hochberg_rejects_only_small_p():
    p = np.array([0.001, 0.008, 0.04, 0.6, 0.9])
    rejected = benjamini_hochberg(p, alpha=0.05)
    assert rejected[0] and rejected[1]
    assert not rejected[3] and not rejected[4]


def test_benjamini_hochberg_is_monotone():
    """If a p-value is rejected, every smaller p-value is too."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    rejected = benjamini_hochberg(p, alpha=0.1)
    largest_rejected = p[rejected].max() if rejected.any() else -1
    assert all(rejected[p <= largest_rejected])


def test_cluster_bootstrap_recovers_a_planted_effect():
    """Failing rows carry a descriptor shifted by ~1 pooled SD."""
    rng = np.random.default_rng(1)
    n = 300
    failed = rng.random(n) < 0.5
    frame = pd.DataFrame({
        "check_x": pd.array(~failed, dtype="boolean"),
        "descriptor": rng.normal(0, 1, n) + failed * 1.0,
        "pdb_id": [f"P{i}" for i in range(n)],
    })
    est = cluster_bootstrap_d(frame, "check_x", "descriptor", n_boot=400, seed=0)
    assert isinstance(est, Estimate)
    assert 0.6 < est.point < 1.4
    assert est.lo < est.point < est.hi
    assert est.p_value < 0.01
    assert est.n_clusters == n


def test_cluster_bootstrap_widens_with_repeated_ligands():
    """Duplicating every ligand 5x must not shrink the interval."""
    rng = np.random.default_rng(2)
    n = 120
    failed = rng.random(n) < 0.5
    base = pd.DataFrame({
        "check_x": pd.array(~failed, dtype="boolean"),
        "descriptor": rng.normal(0, 1, n) + failed * 0.8,
        "pdb_id": [f"P{i}" for i in range(n)],
    })
    inflated = pd.concat([base] * 5, ignore_index=True)

    narrow = cluster_bootstrap_d(base, "check_x", "descriptor", n_boot=400, seed=3)
    clustered = cluster_bootstrap_d(inflated, "check_x", "descriptor", n_boot=400, seed=3)

    assert clustered.n_clusters == n           # not 5 * n
    width = lambda e: e.hi - e.lo
    assert width(clustered) > 0.5 * width(narrow)


def test_nan_interval_is_not_significant():
    """An estimate whose interval could not be computed is not a finding."""
    nan = float("nan")
    assert not Estimate(0.8, nan, nan, nan, 5, 100, 100).significant()
    assert Estimate(0.8, 0.2, 1.4, 0.01, 50, 100, 100).significant()
    assert not Estimate(0.1, -0.3, 0.5, 0.6, 50, 100, 100).significant()


def test_na_check_rows_join_neither_group():
    """A check that did not run must not count as a pass or as a failure.

    `_d_from_frame` splits on `frame[check] == False`. If NA ever leaked into
    the passing group via `~failed`, poses the checker never examined would
    dilute the contrast and shrink every effect estimate.
    """
    frame = pd.DataFrame({
        "check_x": pd.array([False, False, True, True, None, None], dtype="boolean"),
        "descriptor": [10.0, 12.0, 1.0, 3.0, 99.0, -99.0],
        "pdb_id": list("ABCDEF"),
    })
    with_na = cluster_bootstrap_d(frame, "check_x", "descriptor", n_boot=200, seed=0)

    without_na = cluster_bootstrap_d(
        frame[frame["check_x"].notna()], "check_x", "descriptor", n_boot=200, seed=0
    )

    # The two extreme NA rows would wreck the estimate if they were counted.
    assert with_na.point == without_na.point
    assert with_na.n_fail == 2
