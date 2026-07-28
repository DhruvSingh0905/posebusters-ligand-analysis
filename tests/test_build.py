import pandas as pd

from pb.build import CHECKS, add_validity, _to_nullable_bool


def _prepare(frame):
    frame = frame.copy()
    for column in [*CHECKS, "rmsd_within_threshold"]:
        frame[column] = _to_nullable_bool(pd.Series(frame[column]))
    frame["method_class"] = "other"
    return add_validity(frame)


def test_blank_check_is_not_a_failure(synthetic_results):
    """A minimised row with blank flatness columns is still valid."""
    out = _prepare(synthetic_results)
    minimised = out[out["post-processing"] == "energy minimization"].iloc[0]
    assert minimised["n_checks_run"] == len(CHECKS) - 2
    assert minimised["n_checks_failed"] == 0
    assert bool(minimised["pb_valid"]) is True


def test_all_blank_row_is_not_valid(synthetic_results):
    """A method that produced no pose must not score as valid."""
    out = _prepare(synthetic_results)
    absent = out[out["method"] == "equibind"].iloc[0]
    assert absent["n_checks_run"] == 0
    assert bool(absent["pose_produced"]) is False
    assert bool(absent["pb_valid"]) is False


def test_real_failure_is_counted(synthetic_results):
    out = _prepare(synthetic_results)
    failed = out[out["method"] == "diffdock"].iloc[0]
    assert failed["n_checks_failed"] == 1
    assert bool(failed["pb_valid"]) is False


def test_accurate_but_invalid_requires_both(synthetic_results):
    out = _prepare(synthetic_results)
    failed = out[out["method"] == "diffdock"].iloc[0]
    assert bool(failed["accurate"]) is True
    assert bool(failed["accurate_but_invalid"]) is True

    absent = out[out["method"] == "equibind"].iloc[0]
    assert bool(absent["accurate"]) is False
    assert bool(absent["accurate_but_invalid"]) is False
