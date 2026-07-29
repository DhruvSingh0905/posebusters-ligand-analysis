"""One-shot regeneration of every finding in this repository.

The pipeline is a linear chain with two expensive links. `analyze` bootstraps
~370 effect estimates over 2,000 resamples each (~35 min) and `validate` runs
the PoseBusters checker over 513 structures (~40 min). `--fast` drops the
second, which is an independent verification rather than an input to anything
downstream, so the findings still regenerate in full without it.

    python run.py                 every stage
    python run.py --fast          skip the independent bust verification
    python run.py --from build    resume at a stage
    python run.py --only figures  run specific stages
    python run.py --list          show the chain and what each stage costs
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from dataclasses import dataclass, field

log = logging.getLogger("pb")


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    summary: str
    minutes: float
    produces: tuple[str, ...] = field(default_factory=tuple)
    slow: bool = False
    network: bool = False


# Order is dependency order. crystal_contacts must precede build, or the
# contact columns join as all-NA and every downstream sensitivity is empty.
STAGES: tuple[Stage, ...] = (
    Stage("acquire", "pb.acquire",
          "fetch and unpack the benchmark from Zenodo", 1.0,
          ("data/extracted",), network=True),
    Stage("descriptors", "pb.descriptors",
          "27 RDKit descriptors for 513 crystal ligands", 0.5,
          ("data/processed/ligand_descriptors.parquet",)),
    Stage("crystal_contacts", "pb.crystal_contacts",
          "symmetry-mate contact flags (caches mmCIF from RCSB)", 3.0,
          ("data/processed/crystal_contacts.parquet",), network=True),
    Stage("build", "pb.build",
          "join descriptors and contacts onto 7,695 poses", 0.3,
          ("data/processed/poses_joined.parquet",)),
    Stage("analyze", "pb.analyze",
          "gated, ligand-clustered, FDR-controlled effect tables", 35.0,
          ("reports/tables/association_grid.csv",
           "reports/tables/check_models.csv"), slow=True),
    Stage("replication", "pb.replication",
          "held-out Astex re-estimation and RMSD bands", 1.0,
          ("reports/tables/astex_replication.csv",)),
    Stage("figures", "pb.figures",
          "three report figures", 0.5,
          ("reports/figures",)),
    Stage("report", "pb.report_html",
          "long-form HTML report", 0.2,
          ("reports/benchmark-report.html",)),
    Stage("paper", "pb.paper_html",
          "concise paper (HTML, PDF) and the Pages landing page", 0.2,
          ("reports/paper.html", "index.html")),
    Stage("validate", "pb.validate",
          "independent check: re-run bust on 513 crystal structures", 40.0,
          ("reports/tables/bust_reproduction.csv",), slow=True),
)

BY_NAME = {s.name: s for s in STAGES}


def _render_pdfs() -> None:
    """Both documents print to PDF; missing Chrome is a warning, not a failure."""
    from . import paths
    from .render import to_pdf

    for name in ("benchmark-report.html", "paper.html"):
        html = paths.REPORTS / name
        if html.exists():
            to_pdf(html)


def run_stage(stage: Stage) -> float:
    log.info("── %s: %s", stage.name, stage.summary)
    start = time.perf_counter()
    module = importlib.import_module(stage.module)
    module.main()
    elapsed = time.perf_counter() - start

    from . import paths
    missing = [p for p in stage.produces if not (paths.ROOT / p).exists()]
    if missing:
        raise RuntimeError(f"{stage.name} finished but did not write: {', '.join(missing)}")

    log.info("   done in %s", _fmt(elapsed))
    return elapsed


def _fmt(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.1f} min"


def select(args: argparse.Namespace) -> list[Stage]:
    if args.only:
        unknown = set(args.only) - set(BY_NAME)
        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")
        return [BY_NAME[n] for n in args.only]

    chosen = list(STAGES)
    if args.start:
        if args.start not in BY_NAME:
            raise SystemExit(f"unknown stage: {args.start}")
        names = [s.name for s in STAGES]
        chosen = chosen[names.index(args.start):]
    if args.fast:
        chosen = [s for s in chosen if s.name != "validate"]
    return chosen


def show_chain() -> None:
    print(f"\n{'stage':<18}{'~time':>8}  what it produces")
    print("─" * 74)
    for s in STAGES:
        marks = "".join(["*" if s.slow else " ", "net" if s.network else "   "])
        print(f"{s.name:<18}{_fmt(s.minutes * 60):>8}  {s.summary}  {marks}")
    total = sum(s.minutes for s in STAGES)
    fast = sum(s.minutes for s in STAGES if s.name != "validate")
    print("─" * 74)
    print(f"{'full run':<18}{total:>6.0f} min      (* = slow, net = needs network)")
    print(f"{'--fast':<18}{fast:>6.0f} min      drops the independent bust verification\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Regenerate every table, figure and document in this repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("    python run.py")[0].strip(),
    )
    parser.add_argument("--fast", action="store_true",
                        help="skip `validate` (~40 min); it verifies rather than feeds the analysis")
    parser.add_argument("--from", dest="start", metavar="STAGE",
                        help="resume at this stage and run the rest")
    parser.add_argument("--only", nargs="+", metavar="STAGE",
                        help="run only these stages, in the order given")
    parser.add_argument("--list", action="store_true", help="show the chain and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.list:
        show_chain()
        return 0

    stages = select(args)
    budget = sum(s.minutes for s in stages)
    log.info("running %d stage(s), roughly %.0f min\n", len(stages), budget)

    timings: list[tuple[str, float]] = []
    started = time.perf_counter()
    for stage in stages:
        try:
            timings.append((stage.name, run_stage(stage)))
        except Exception as exc:  # noqa: BLE001 - report which stage, then stop
            log.error("\n%s failed: %s", stage.name, exc)
            log.error("fix the cause and resume with:  python run.py --from %s", stage.name)
            return 1

    if any(s.name in ("report", "paper") for s in stages):
        _render_pdfs()

    total = time.perf_counter() - started
    log.info("\n%s", "─" * 46)
    for name, elapsed in timings:
        log.info("  %-20s %s", name, _fmt(elapsed))
    log.info("  %-20s %s", "total", _fmt(total))

    from . import paths
    tables = len(list((paths.REPORTS / "tables").glob("*.csv")))
    figures = len(list(paths.FIGURES.glob("*.png")))
    log.info("\n%d tables, %d figures, documents in %s",
             tables, figures, paths.REPORTS.relative_to(paths.ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
