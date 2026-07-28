"""Independent replication and the unconditioned view.

Two habits guard against the report's remaining soft spots.

Replication: every effect in this project was discovered and estimated on the
same 428 complexes. The 85 Astex complexes were never touched, so they serve as
a held-out set - an effect that reverses sign there was probably noise.

Unconditioned outcomes: "of accurate poses, the share that are invalid"
conditions on accuracy, which shares an upstream cause with validity (how well
the method did on this ligand). Conditioning on such a variable can induce
association between things that are otherwise unrelated. Reporting validity
against *continuous* RMSD, over all poses, needs no such conditioning.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .build import analysis_population
from .inference import cluster_bootstrap_d
from .strata import stratum_frame

log = logging.getLogger(__name__)


def replicate_on_astex(
    joined: pd.DataFrame,
    methods: list[str],
    pairs: list[tuple[str, str]],
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Re-estimate a shortlist of (check, descriptor) effects on the held-out set."""
    benchmark = analysis_population(joined, dataset="posebuster")
    astex = analysis_population(joined, dataset="astex")

    rows = []
    for method in methods:
        for check, descriptor in pairs:
            main = stratum_frame(benchmark, check, method)
            held = stratum_frame(astex, check, method)
            if held["pdb_id"].nunique() < 20:
                continue

            a = cluster_bootstrap_d(main, check, descriptor, n_boot=n_boot, seed=seed)
            b = cluster_bootstrap_d(held, check, descriptor, n_boot=n_boot, seed=seed)
            if not (np.isfinite(a.point) and np.isfinite(b.point)):
                continue

            rows.append({
                "method": method,
                "check": check,
                "descriptor": descriptor,
                "d_posebuster": a.point,
                "d_astex": b.point,
                "n_astex_clusters": b.n_clusters,
                "same_sign": np.sign(a.point) == np.sign(b.point),
                "both_significant": a.significant() and b.significant(),
            })
    return pd.DataFrame(rows)


def validity_vs_rmsd(
    df: pd.DataFrame,
    method: str,
    edges: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf),
) -> pd.DataFrame:
    """Validity rate across continuous RMSD bands, with no conditioning on accuracy.

    The 2 A threshold throws away the difference between a 0.4 A pose and a 1.9 A
    one. Banding the raw distance keeps it, and covers inaccurate poses too, so
    nothing here conditions on the accuracy outcome.
    """
    frame = df[(df["method"] == method) & df["rmsd"].notna()].copy()
    frame["rmsd_band"] = pd.cut(frame["rmsd"], bins=list(edges))

    out = (
        frame.groupby("rmsd_band", observed=True)
        .agg(n=("pb_valid", "size"),
             valid=("pb_valid", "mean"),
             median_rmsd=("rmsd", "median"))
        .reset_index()
    )
    out.insert(0, "method", method)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from . import paths

    joined = pd.read_parquet(paths.JOINED_PARQUET)
    tables = paths.REPORTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    headline_pairs = [
        ("energy_ratio_within_threshold", "n_rotatable_bonds"),
        ("no_internal_clashes", "n_rotatable_bonds"),
        ("sp3_stereochemistry_preserved", "n_stereocentres"),
        ("no_clashes_with_protein", "n_rotatable_bonds"),
    ]
    methods = ["diffdock", "unimol", "deepdock", "tankbind", "vina", "gold"]

    replication = replicate_on_astex(joined, methods, headline_pairs)
    replication.to_csv(tables / "astex_replication.csv", index=False)
    print(replication.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    requested = len(methods) * len(headline_pairs)
    if len(replication):
        log.info("effects reproducing in sign on held-out Astex: %d/%d",
                 int(replication["same_sign"].sum()), len(replication))
    log.info("pairs producing a comparison: %d/%d requested (%d skipped)",
             len(replication), requested, requested - len(replication))

    population = analysis_population(joined)
    bands = pd.concat([validity_vs_rmsd(population, m) for m in methods])
    bands.to_csv(tables / "validity_vs_rmsd.csv", index=False)
    print(bands.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))


if __name__ == "__main__":
    main()
