import argparse

import pytest

from pb import cli


def _args(**kwargs):
    defaults = {"only": None, "start": None, "fast": False}
    return argparse.Namespace(**{**defaults, **kwargs})


def test_default_runs_every_stage_in_dependency_order():
    names = [s.name for s in cli.select(_args())]
    assert names == [s.name for s in cli.STAGES]
    # crystal_contacts before build, or the contact columns join as all-NA
    assert names.index("crystal_contacts") < names.index("build")
    assert names.index("build") < names.index("analyze")


def test_fast_drops_only_the_independent_verification():
    names = [s.name for s in cli.select(_args(fast=True))]
    assert "validate" not in names
    assert names == [s.name for s in cli.STAGES if s.name != "validate"]


def test_from_resumes_at_a_stage():
    names = [s.name for s in cli.select(_args(start="build"))]
    assert names[0] == "build"
    assert "descriptors" not in names


def test_only_respects_the_order_given():
    names = [s.name for s in cli.select(_args(only=["paper", "figures"]))]
    assert names == ["paper", "figures"]


@pytest.mark.parametrize("args", [_args(only=["nonesuch"]), _args(start="nonesuch")])
def test_unknown_stage_is_rejected(args):
    with pytest.raises(SystemExit):
        cli.select(args)


def test_stage_that_writes_nothing_is_a_failure(monkeypatch):
    """A stage returning cleanly without producing its output must not pass.

    Silent no-ops are the failure mode that matters here: the run reports
    success and the reader trusts stale files from an earlier run.
    """
    stage = cli.Stage("fake", "pb.paths", "does nothing", 0.0,
                      produces=("reports/definitely-not-written.csv",))
    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda name: type("M", (), {"main": staticmethod(lambda: None)}))
    with pytest.raises(RuntimeError, match="did not write"):
        cli.run_stage(stage)


def test_every_stage_declares_what_it_produces():
    for stage in cli.STAGES:
        assert stage.produces, f"{stage.name} declares no outputs, so it cannot be checked"
