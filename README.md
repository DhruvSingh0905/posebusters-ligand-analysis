# PoseBusters ligand-conditioned analysis

The PoseBusters benchmark results tell you which docking *method* fails. They
say nothing about which *molecules* it fails on — the published table carries no
chemistry at all. This joins RDKit descriptors computed from the crystal ligands
onto every pose row and partitions the failures by ligand class.

Findings: [`reports/findings.md`](reports/findings.md)

## Layout

    src/pb/descriptors.py      27 RDKit descriptors per crystal ligand
    src/pb/build.py            join + validity derivation (blank != False)
    src/pb/eligibility.py      which ligands can even fail which check
    src/pb/strata.py           per-method, cofactor-conditioned strata
    src/pb/inference.py        cluster-bootstrap intervals + BH-FDR
    src/pb/crystal_contacts.py symmetry-mate contact flags (needs a PDB cache)
    src/pb/models.py           clustered logistic check models
    src/pb/validate.py         reproduce the paper's rows by running `bust`
    src/pb/analyze.py          gated, clustered, FDR-controlled effect tables
    src/pb/replication.py      held-out Astex replication + unconditioned view
    src/pb/figures.py          report figures
    reports/tables/            association_grid.csv, check_models.csv, and more
    reports/figures/           3 PNGs

## Run

    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python \
        rdkit pandas pyarrow matplotlib statsmodels gemmi posebusters pytest
    export PYTHONPATH=src
    .venv/bin/python -m pb.acquire            # download + unpack from Zenodo
    .venv/bin/python -m pb.descriptors
    .venv/bin/python -m pb.crystal_contacts   # symmetry-contact flags (cached)
    .venv/bin/python -m pb.build              # must run after crystal_contacts, or
                                               # the flag columns are all-<NA>
    .venv/bin/python -m pb.validate           # slow: runs `bust` over 513 structures
    .venv/bin/python -m pb.analyze
    .venv/bin/python -m pb.replication
    .venv/bin/python -m pb.figures

`pb.acquire` fetches both source files from
[Zenodo 8278563](https://zenodo.org/records/8278563) and unpacks the archive.
The 51 MB zip and everything derived from it are untracked; the 1.2 MB results
CSV is committed so the analysis inputs are pinned. `pb.crystal_contacts` caches
downloaded mmCIF entries under `data/pdb_cache/`, so re-runs are fast.
`pb.validate` takes tens of minutes; it is independent verification, not a
dependency of `pb.analyze` or `pb.figures`, and can be run any time after
`pb.build`.
