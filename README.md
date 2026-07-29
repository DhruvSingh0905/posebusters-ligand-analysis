# What breaks a docking pose

Which ligand properties predict which physical-validity failure, across the
PoseBusters benchmark.

**The findings:** [`reports/paper.pdf`](reports/paper.pdf) ·
[`reports/paper.html`](reports/paper.html) — 8 pages, 3 tables, 3 figures.
A longer version with the full method tables is in
[`reports/benchmark-report.html`](reports/benchmark-report.html).

---

## The question

PoseBusters scores docking with 18 physical-validity checks and reports failure
rates per *method*. Its results table carries a PDB code, a chemical-component
code, a cofactor flag and a sequence identity, and nothing about the molecule
being docked, so it cannot say which ligands are hard.

This computes 27 RDKit descriptors for the 428 benchmark ligands, joins them to
all 7,695 published poses, and models each check against ligand chemistry.

## The answer

| | |
|---|---|
| Internal-geometry failure scales with **torsional degrees of freedom** | OR 1.6–9.5 / SD, 4 of 5 methods |
| sp³ chirality failure scales with **stereocentre count**, and nothing else | OR 4.3–10.4 / SD |
| Cofactor clash scales **inversely with molecular weight** | OR 0.15–0.35 / SD, 4 of 5 methods |
| Protein clash, the largest failure term, has **no ligand-level predictor** | 5 of 42 coefficients |

Two limits bound all of it. Only 41 of 126 method × check strata carry enough
failures to model, and molecular weight and rotatable-bond count are correlated
at ρ ≈ 0.73–0.74 throughout, so where both are significant they cannot be
separated.

## Regenerate everything

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python run.py --fast
```

`run.py` chains every stage in dependency order and refuses to report success
for a stage that produced no output.

```
python run.py                 every stage (~82 min)
python run.py --fast          skip the independent bust verification (~42 min)
python run.py --from build    resume at a stage
python run.py --only figures  run specific stages
python run.py --list          show the chain and what each stage costs
```

Two stages are expensive. `analyze` bootstraps ~370 effect estimates over 2,000
resamples each; `validate` re-runs the PoseBusters checker over 513 structures.
The second verifies the analysis rather than feeding it, so `--fast` still
regenerates every finding.

Source data is fetched automatically: the benchmark from
[Zenodo 8278563](https://zenodo.org/records/8278563), deposited structures from
the RCSB PDB (cached under `data/pdb_cache/`). Only the 1.2 MB results CSV is
tracked; everything else is regenerable.

## Layout

```
run.py                      one-shot regeneration
src/pb/
  cli.py                    stage chain, resume, output checks
  acquire.py                fetch and unpack the benchmark
  descriptors.py            27 RDKit descriptors per crystal ligand
  crystal_contacts.py       symmetry-mate contacts via spacegroup expansion
  build.py                  join; blank check is not a failed check
  eligibility.py            which ligands can even fail which check
  strata.py                 per-method, cofactor-conditioned strata
  inference.py              ligand-clustered bootstrap, Benjamini-Hochberg
  models.py                 clustered logistic model per check
  analyze.py                effect tables
  replication.py            held-out Astex, unconditioned RMSD bands
  validate.py               re-run bust against the published rows
  figures.py                figures
  report_html.py            long-form report; shared table builders
  paper_html.py             concise paper
  render.py                 HTML to PDF via headless Chrome
reports/
  paper.{html,pdf}          the findings
  benchmark-report.{html,pdf}
  tables/                   16 CSVs; the source of every number in both documents
  figures/                  3 PNGs
tests/                      45 tests
```

Both documents read their tables from `reports/tables/` at render time and
inline the figures, so neither can drift from the analysis it reports.

## Method notes

Three decisions do most of the work, and getting any of them wrong changes the
conclusions.

**Blank is not False.** A blank check in the published table means the check was
not run. Both flatness columns are blank for all 2,996 energy-minimised rows,
and 85 rows have every check blank because the method returned no pose. Reading
either as a pass inflates validity.

**Eligibility gating.** A molecule with no stereocentres cannot fail the
chirality check. Counting it as a trivial pass deflates the failure rate and
inflates any effect measured against the gating descriptor: gating moves the
observed chirality-failure rate from 21.4% to 36.7% and the associated effect
size from *d* = 1.09 to 0.35. The four cofactor checks are further restricted to
cofactor-bearing complexes.

**Clustering.** Each ligand contributes one pose per method, so treating 2,140
rows as independent observations of 428 ligands reports intervals roughly √5 too
narrow. The bootstrap resamples ligands.

## Data

Buttenschoen M, Morris GM, Deane CM. *PoseBusters: AI-based docking methods fail
to generate physically valid poses or generalise to novel sequences.*
Chem Sci **15**, 3130–3139 (2024).
[doi:10.1039/D3SC04185A](https://doi.org/10.1039/D3SC04185A)
