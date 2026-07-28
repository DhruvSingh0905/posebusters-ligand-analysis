# PoseBusters ligand-conditioned analysis

The PoseBusters benchmark results tell you which docking *method* fails. They
say nothing about which *molecules* it fails on — the published table carries no
chemistry at all. This joins RDKit descriptors computed from the crystal ligands
onto every pose row and partitions the failures by ligand class.

Findings: [`reports/findings.md`](reports/findings.md)

## Layout

    src/pb/descriptors.py   28 RDKit descriptors per crystal ligand
    src/pb/build.py         join + validity derivation (blank != False)
    src/pb/analyze.py       trend tables, effect sizes, stratified controls
    src/pb/figures.py       report figures
    reports/tables/         14 CSVs
    reports/figures/        3 PNGs

## Run

    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python rdkit pandas pyarrow matplotlib
    export PYTHONPATH=src
    .venv/bin/python -m pb.acquire    # download + unpack from Zenodo
    .venv/bin/python -m pb.descriptors
    .venv/bin/python -m pb.build
    .venv/bin/python -m pb.analyze
    .venv/bin/python -m pb.figures

`pb.acquire` fetches both source files from
[Zenodo 8278563](https://zenodo.org/records/8278563) and unpacks the archive.
The 51 MB zip and everything derived from it are untracked; the 1.2 MB results
CSV is committed so the analysis inputs are pinned.
