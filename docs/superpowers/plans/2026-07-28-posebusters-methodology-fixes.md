# PoseBusters Analysis — Methodology Reconfiguration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current pooled, uncontrolled, inference-free marginal analysis with a per-method, eligibility-gated, cluster-inferential pipeline whose published claims survive audit.

**Architecture:** The existing pipeline answers "does descriptor X differ between poses that pass and fail check C?" by pooling 5 deep-learning methods into 2,140 rows, comparing marginal means, and reporting the largest Cohen's *d*. Three structural flaws make those numbers untrustworthy: ligands that *cannot* fail a check are counted in its denominator, methods with wildly different competence are pooled, and 428 ligands are treated as 2,140 independent observations. The reconfiguration inserts four layers ahead of every statistic — **eligibility** (who can fail this check), **stratification** (per method, conditioned on cofactor presence), **inference** (cluster bootstrap by ligand, BH-FDR across the association grid), and **modelling** (one clustered logistic regression per check on a decorrelated descriptor basis, replacing 324 marginal comparisons). Two independent validations are added: reproducing the paper's crystal-structure rows by actually running `bust`, and computing crystal contacts from symmetry mates to test the benchmark's own biggest confound.

**Tech Stack:** Python 3.11, RDKit 2026.3.4, pandas 3.0, pyarrow, matplotlib, statsmodels (new), gemmi (new), posebusters (new), pytest (new).

## Global Constraints

- Python 3.11 via `uv`; virtualenv at `.venv`; all commands run with `PYTHONPATH=src`.
- Existing module layout under `src/pb/` is kept. New modules go alongside, not inside, existing ones.
- Blank values in the results CSV mean *check not run*, never *check failed*. Any new code touching check columns MUST route through the nullable-boolean representation established in `pb.build`.
- A row with `n_checks_run == 0` means the method produced no pose and MUST NOT count as valid.
- All effect estimates are reported with a cluster-bootstrap 95% CI, clustered on `pdb_id`. No point estimate ships without an interval.
- Deep-learning methods are never pooled for an effect estimate. Pooled figures may be shown only alongside the per-method breakdown.
- Random seeds are fixed and passed explicitly. No implicit global RNG state.
- `reports/findings.md` claims must cite the table that supports them; any claim that loses significance under BH-FDR is removed, not softened.

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/conftest.py` | *Create.* Shared fixtures: a synthetic results frame with known blanks. |
| `tests/test_build.py` | *Create.* Locks the blank-handling and `pose_produced` invariants. |
| `tests/test_eligibility.py` | *Create.* Verifies structural-zero exclusion per check. |
| `tests/test_inference.py` | *Create.* Verifies bootstrap CI coverage and BH-FDR monotonicity. |
| `src/pb/eligibility.py` | *Create.* Maps each check to the descriptor that gates it; produces eligible-row masks. |
| `src/pb/inference.py` | *Create.* Cluster bootstrap, bootstrap p-values, Benjamini–Hochberg. |
| `src/pb/crystal_contacts.py` | *Create.* Downloads PDB entries, computes ligand-to-symmetry-mate distance, flags contacts. |
| `src/pb/models.py` | *Create.* Decorrelated descriptor basis; clustered logistic regression per check. |
| `src/pb/validate.py` | *Create.* Runs `bust` on the 513 crystal structures; diffs against the published rows. |
| `src/pb/build.py` | *Modify.* Join the crystal-contact flag; expose `analysis_population()`. |
| `src/pb/analyze.py` | *Rewrite.* Every statistic routed through eligibility + stratification + inference. |
| `src/pb/figures.py` | *Modify.* Error bars on every estimate; per-method facets replace pooled lines. |
| `reports/findings.md` | *Rewrite.* Corrected claims only. |

---

## Task 1: Lock the blank-handling invariants with tests

The correctness property that was silently broken twice. It is the foundation every later
task stands on, so it gets tests before anything else changes.

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_build.py`
- Modify: `pyproject.toml` (create if absent)

**Interfaces:**
- Consumes: `pb.build.load_results`, `pb.build.add_validity`, `pb.build.CHECKS`
- Produces: fixture `synthetic_results` — a `pd.DataFrame` shaped like the published CSV with
  string-typed check columns, used by all later test modules.

- [ ] **Step 1: Add pytest and create the project config**

```bash
uv pip install --python .venv/bin/python pytest statsmodels gemmi posebusters
```

Create `pyproject.toml`:

```toml
[project]
name = "pb-analysis"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the shared fixture**

Create `tests/conftest.py`:

```python
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
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_build.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass against current code**

Run: `.venv/bin/pytest tests/test_build.py -v`
Expected: 4 passed. These lock in behaviour that already works — if any fails, `pb.build` has
regressed and must be fixed before continuing.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/test_build.py
git commit -m "test: lock blank-handling and pose_produced invariants"
```

---

## Task 2: Eligibility layer — exclude ligands that cannot fail a check

Issue #1. `sp3_stereochemistry_preserved` cannot fail on a ligand with zero stereocentres, yet
893 such rows sit in its denominator. This deflates the failure rate and inflates Cohen's *d*
from 0.35 to 1.09.

**Files:**
- Create: `src/pb/eligibility.py`
- Create: `tests/test_eligibility.py`

**Interfaces:**
- Consumes: `pb.build.CHECKS`
- Produces:
  - `ELIGIBILITY: dict[str, str | None]` — check name → descriptor column that must be `> 0`
  - `eligible_mask(df: pd.DataFrame, check: str) -> pd.Series` (boolean, index-aligned)
  - `eligible(df: pd.DataFrame, check: str) -> pd.DataFrame` — rows where the check ran *and* can fail

- [ ] **Step 1: Write the failing test**

Create `tests/test_eligibility.py`:

```python
import pandas as pd

from pb.eligibility import ELIGIBILITY, eligible, eligible_mask


def _frame():
    return pd.DataFrame({
        "sp3_stereochemistry_preserved": pd.array([True, False, True], dtype="boolean"),
        "aromatic_ring_flatness_passes": pd.array([True, True, None], dtype="boolean"),
        "no_clashes_with_protein": pd.array([True, False, True], dtype="boolean"),
        "n_stereocentres": [0, 3, 2],
        "n_aromatic_rings": [0, 1, 2],
    })


def test_gated_check_excludes_structural_zeros():
    mask = eligible_mask(_frame(), "sp3_stereochemistry_preserved")
    assert list(mask) == [False, True, True]


def test_ungated_check_keeps_every_row_that_ran():
    mask = eligible_mask(_frame(), "no_clashes_with_protein")
    assert list(mask) == [True, True, True]


def test_rows_where_check_did_not_run_are_excluded():
    mask = eligible_mask(_frame(), "aromatic_ring_flatness_passes")
    # row 0 has zero aromatic rings, row 2 has a blank check
    assert list(mask) == [False, True, False]


def test_eligible_returns_a_subframe():
    out = eligible(_frame(), "sp3_stereochemistry_preserved")
    assert len(out) == 2
    assert (out["n_stereocentres"] > 0).all()


def test_every_check_has_an_eligibility_entry():
    from pb.build import CHECKS
    assert set(ELIGIBILITY) == set(CHECKS)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_eligibility.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pb.eligibility'`

- [ ] **Step 3: Write the implementation**

Create `src/pb/eligibility.py`:

```python
"""Which ligands are even capable of failing each check.

A ligand with no stereocentres cannot fail the chirality check; one with no
aromatic rings cannot fail aromatic-ring flatness. Counting those ligands in a
check's denominator makes the check look more reliable than it is, and inflates
any effect size measured against the gating descriptor - the comparison ends up
encoding "can this check fire at all" rather than "does this molecule induce
failure". Every statistic in this project is computed on eligible rows only.
"""

from __future__ import annotations

import pandas as pd

from .build import CHECKS

# check -> descriptor that must be non-zero for the check to be able to fail.
# None means every produced pose is eligible.
ELIGIBILITY: dict[str, str | None] = {
    "docked_ligand_successfully_loaded": None,
    "molecule_passes_rdkit_sanity_check": None,
    "molecular_formula_preserved": None,
    "molecular_bonds_preserved": None,
    "sp3_stereochemistry_preserved": "n_stereocentres",
    "double_bond_stereochemistry_preserved": "n_stereo_double_bonds",
    "bond_lengths_within_bounds": None,
    "bond_angles_within_bounds": None,
    "no_internal_clashes": None,
    "aromatic_ring_flatness_passes": "n_aromatic_rings",
    "double_bond_flatness_passes": "n_stereo_double_bonds",
    "energy_ratio_within_threshold": None,
    # cofactor checks are gated by the complex, not the ligand - handled by
    # stratification in pb.analyze rather than here, because `has_cofactors`
    # describes the receptor.
    "no_clashes_with_protein": None,
    "no_clashes_with_organic_cofactors": None,
    "no_clashes_with_inorganic_cofactors": None,
    "no_volume_clash_with_protein": None,
    "no_volume_clash_with_organic_cofactors": None,
    "no_volume_clash_with_inorganic_cofactors": None,
}

assert set(ELIGIBILITY) == set(CHECKS), "eligibility table out of sync with CHECKS"


def eligible_mask(df: pd.DataFrame, check: str) -> pd.Series:
    """Rows where `check` both ran and could have failed."""
    if check not in ELIGIBILITY:
        raise KeyError(f"unknown check: {check}")

    mask = df[check].notna()
    gate = ELIGIBILITY[check]
    if gate is not None:
        mask = mask & (df[gate].fillna(0) > 0)
    return pd.Series(mask, index=df.index).fillna(False).astype(bool)


def eligible(df: pd.DataFrame, check: str) -> pd.DataFrame:
    """Subframe of rows eligible for `check`."""
    return df[eligible_mask(df, check)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_eligibility.py -v`
Expected: 5 passed

- [ ] **Step 5: Measure the damage to the published numbers**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c "
import pandas as pd
from pb import paths
from pb.analyze import primary, cohens_d
from pb.eligibility import eligible
df = pd.read_parquet(paths.JOINED_PARQUET)
dl = primary(df).query('method_class == \"deep learning\"')
for check, desc in [('sp3_stereochemistry_preserved','n_stereocentres'),
                    ('aromatic_ring_flatness_passes','n_aromatic_rings'),
                    ('double_bond_stereochemistry_preserved','n_stereo_double_bonds')]:
    e = eligible(dl, check)
    f = e[e[check]==False][desc].dropna(); p = e[e[check]==True][desc].dropna()
    print(f'{check:42s} n_eligible={len(e):5d}  fail={(e[check]==False).mean():.1%}  d={cohens_d(f,p):+.2f}')
"
```

Expected: failure rates rise sharply and every *d* falls below 0.6. Record the output in the
commit message — these numbers replace the published table in Task 9.

- [ ] **Step 6: Commit**

```bash
git add src/pb/eligibility.py tests/test_eligibility.py
git commit -m "feat: gate every check by whether the ligand can fail it"
```

---

## Task 3: Inference layer — cluster bootstrap and FDR control

Issues #5 and #6. Each ligand appears once per method, so 2,140 rows carry 428 independent
units — a 5× inflation. And 324 associations were computed with the largest reported and no
correction.

**Files:**
- Create: `src/pb/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `pb.eligibility.eligible`
- Produces:
  - `@dataclass Estimate` with fields `point: float`, `lo: float`, `hi: float`,
    `p_value: float`, `n_fail: int`, `n_eligible: int`, `n_clusters: int`
  - `cohens_d(failing: pd.Series, passing: pd.Series) -> float`
  - `cluster_bootstrap_d(df, check, descriptor, cluster="pdb_id", n_boot=2000, seed=0) -> Estimate`
  - `benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray` (boolean, True = reject)

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pb.inference'`

- [ ] **Step 3: Write the implementation**

Create `src/pb/inference.py`:

```python
"""Interval estimates and multiplicity control.

Every ligand is docked by every method, so rows are not independent: 2,140
deep-learning rows carry only 428 independent ligands. Resampling rows would
report intervals about sqrt(5) times too narrow, so the bootstrap resamples
*clusters* - whole ligands, with all their method rows attached.

Because the association grid is 18 checks x 18 descriptors, reporting the
largest effects without correction guarantees false positives. Bootstrap
p-values feed a Benjamini-Hochberg step-up at the reporting boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a cluster-bootstrap interval."""

    point: float
    lo: float
    hi: float
    p_value: float
    n_fail: int
    n_eligible: int
    n_clusters: int

    def significant(self) -> bool:
        return not (self.lo <= 0.0 <= self.hi)


def cohens_d(failing: pd.Series, passing: pd.Series) -> float:
    """Standardised mean difference, positive when failing poses score higher."""
    n_f, n_p = len(failing), len(passing)
    if n_f < 2 or n_p < 2:
        return float("nan")
    var_f, var_p = failing.var(ddof=1), passing.var(ddof=1)
    pooled = ((n_f - 1) * var_f + (n_p - 1) * var_p) / (n_f + n_p - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return float("nan")
    return float((failing.mean() - passing.mean()) / np.sqrt(pooled))


def _d_from_frame(frame: pd.DataFrame, check: str, descriptor: str) -> float:
    failed = frame[check] == False  # noqa: E712
    failing = pd.Series(frame.loc[failed, descriptor]).dropna()
    passing = pd.Series(frame.loc[~failed, descriptor]).dropna()
    return cohens_d(failing, passing)


def cluster_bootstrap_d(
    df: pd.DataFrame,
    check: str,
    descriptor: str,
    cluster: str = "pdb_id",
    n_boot: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Cohen's d with a percentile interval from resampling whole ligands.

    The p-value is the two-sided proportion of bootstrap replicates on the
    opposite side of zero from the point estimate, floored at 1/n_boot since a
    finite bootstrap cannot resolve smaller.
    """
    frame = df[[check, descriptor, cluster]].copy()
    point = _d_from_frame(frame, check, descriptor)

    groups = {key: sub for key, sub in frame.groupby(cluster, observed=True)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)

    replicates: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[k] for k in picked], ignore_index=True)
        value = _d_from_frame(sample, check, descriptor)
        if np.isfinite(value):
            replicates.append(value)

    failed_mask = frame[check] == False  # noqa: E712
    n_fail = int(failed_mask.sum())

    if len(replicates) < 20 or not np.isfinite(point):
        return Estimate(point, float("nan"), float("nan"), float("nan"),
                        n_fail, len(frame), len(keys))

    draws = np.asarray(replicates)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tail = float(np.mean(draws <= 0) if point > 0 else np.mean(draws >= 0))
    p_value = float(min(1.0, max(2 * tail, 1.0 / len(draws))))

    return Estimate(float(point), float(lo), float(hi), p_value,
                    n_fail, len(frame), len(keys))


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Step-up FDR control. Returns a boolean array; True means reject the null."""
    p = np.asarray(p_values, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)

    finite = np.isfinite(p)
    if not finite.any():
        return rejected

    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    thresholds = alpha * np.arange(1, m + 1) / m

    passing = np.nonzero(ranked <= thresholds)[0]
    if len(passing) == 0:
        return rejected

    cutoff = ranked[passing.max()]
    rejected[finite] = values <= cutoff
    return rejected
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_inference.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/pb/inference.py tests/test_inference.py
git commit -m "feat: cluster-bootstrap intervals and BH-FDR control"
```

---

## Task 4: Stratification — per method, conditioned on cofactor presence

Issues #2 and #3. Pooled deep learning is 39% DiffDock and 3% EquiBind, so "the deep-learning
signature" is weighted toward models that barely function. And the cofactor finding
(*d* = −0.29) collapses to −0.02 once restricted to complexes that actually have cofactors —
it was measuring which complexes have cofactors, not which ligands clash.

**Files:**
- Modify: `src/pb/build.py` (add `analysis_population`)
- Create: `src/pb/strata.py`

**Interfaces:**
- Consumes: `pb.eligibility.eligible`, `pb.inference.cluster_bootstrap_d`, `pb.inference.Estimate`
- Produces:
  - `pb.build.analysis_population(df, post="none", dataset="posebuster") -> pd.DataFrame`
  - `COFACTOR_CHECKS: set[str]` — checks that must be conditioned on `has_cofactors`
  - `stratum_frame(df, check, method) -> pd.DataFrame` — eligible rows for one method,
    additionally restricted to cofactor-bearing complexes when the check needs it
  - `association_grid(df, methods, descriptors, n_boot=2000, seed=0) -> pd.DataFrame`
    with columns `method, check, descriptor, d, lo, hi, p_value, n_fail, n_eligible,
    n_clusters, fdr_reject`

- [ ] **Step 1: Add the population helper to `pb.build`**

Append to `src/pb/build.py`:

```python
def analysis_population(
    df: pd.DataFrame, post: str = "none", dataset: str = "posebuster"
) -> pd.DataFrame:
    """The rows every reported statistic is computed on.

    Excludes the crystal-structure reference rows and any pose the method never
    produced, so downstream code never has to remember to filter them.
    """
    return df[
        (df["dataset"] == dataset)
        & (df["post-processing"] == post)
        & (df["method"].isin(DL_METHODS + CLASSICAL_METHODS))
        & df["pose_produced"]
    ].copy()
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_strata.py`:

```python
import pandas as pd
import pytest

from pb.strata import COFACTOR_CHECKS, stratum_frame


@pytest.fixture
def frame():
    return pd.DataFrame({
        "method": ["diffdock"] * 4 + ["vina"] * 2,
        "no_clashes_with_organic_cofactors": pd.array(
            [True, False, True, False, True, True], dtype="boolean"),
        "no_clashes_with_protein": pd.array(
            [True, False, True, False, True, True], dtype="boolean"),
        "has_cofactors": pd.array(
            [True, True, False, False, True, False], dtype="boolean"),
        "n_stereocentres": [1, 1, 1, 1, 1, 1],
        "pdb_id": list("ABCDEF"),
    })


def test_cofactor_check_drops_cofactor_free_complexes(frame):
    out = stratum_frame(frame, "no_clashes_with_organic_cofactors", "diffdock")
    assert len(out) == 2
    assert out["has_cofactors"].all()


def test_non_cofactor_check_keeps_all_complexes(frame):
    out = stratum_frame(frame, "no_clashes_with_protein", "diffdock")
    assert len(out) == 4


def test_stratum_is_restricted_to_one_method(frame):
    out = stratum_frame(frame, "no_clashes_with_protein", "vina")
    assert set(out["method"]) == {"vina"}


def test_cofactor_checks_are_all_registered():
    assert "no_clashes_with_organic_cofactors" in COFACTOR_CHECKS
    assert "no_volume_clash_with_inorganic_cofactors" in COFACTOR_CHECKS
    assert "no_clashes_with_protein" not in COFACTOR_CHECKS
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_strata.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pb.strata'`

- [ ] **Step 4: Write the implementation**

Create `src/pb/strata.py`:

```python
"""Per-method, cofactor-conditioned strata.

Two pooling mistakes are corrected here.

Pooling methods: DiffDock supplies 39% of the pooled deep-learning poses that
pass RMSD and EquiBind 3%, so a pooled effect is mostly a statement about models
that barely function. Effects are estimated per method; a pooled number is only
ever shown beside the breakdown.

Pooling cofactor presence: the cofactor checks pass almost automatically on
complexes with no cofactors (3.4% failure vs 31.3%). Since cofactor-bearing
complexes also carry systematically different ligands (1.56 vs 2.11 aromatic
rings), a pooled comparison measures the receptor, not the ligand. Those checks
are conditioned on `has_cofactors`.
"""

from __future__ import annotations

import pandas as pd

from .eligibility import eligible
from .inference import benjamini_hochberg, cluster_bootstrap_d

COFACTOR_CHECKS = {
    "no_clashes_with_organic_cofactors",
    "no_clashes_with_inorganic_cofactors",
    "no_volume_clash_with_organic_cofactors",
    "no_volume_clash_with_inorganic_cofactors",
}


def stratum_frame(df: pd.DataFrame, check: str, method: str) -> pd.DataFrame:
    """Eligible rows for one check within one method's stratum."""
    frame = df[df["method"] == method]
    frame = eligible(frame, check)
    if check in COFACTOR_CHECKS:
        frame = frame[frame["has_cofactors"].fillna(False).astype(bool)]
    return frame


def association_grid(
    df: pd.DataFrame,
    methods: list[str],
    descriptors: list[str],
    checks: list[str] | None = None,
    min_failures: int = 15,
    min_clusters: int = 30,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Every (method, check, descriptor) association with an interval and an FDR flag.

    Strata too thin to estimate are skipped rather than reported with a wide
    interval, and the skip is visible in the returned frame's absence - callers
    log the count so nothing is dropped silently.
    """
    from .eligibility import ELIGIBILITY

    checks = checks or list(ELIGIBILITY)
    rows = []

    for method in methods:
        for check in checks:
            frame = stratum_frame(df, check, method)
            n_fail = int((frame[check] == False).sum())  # noqa: E712
            n_clusters = frame["pdb_id"].nunique()
            if n_fail < min_failures or n_clusters < min_clusters:
                continue
            if n_fail == len(frame):  # no passing poses to contrast against
                continue

            for descriptor in descriptors:
                est = cluster_bootstrap_d(
                    frame, check, descriptor, n_boot=n_boot, seed=seed
                )
                rows.append({
                    "method": method,
                    "check": check,
                    "descriptor": descriptor,
                    "d": est.point,
                    "lo": est.lo,
                    "hi": est.hi,
                    "p_value": est.p_value,
                    "n_fail": est.n_fail,
                    "n_eligible": est.n_eligible,
                    "n_clusters": est.n_clusters,
                })

    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid

    grid["fdr_reject"] = benjamini_hochberg(grid["p_value"].to_numpy(), alpha=0.05)
    grid["abs_d"] = grid["d"].abs()
    return grid.sort_values("abs_d", ascending=False).reset_index(drop=True)
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_strata.py -v`
Expected: 4 passed

- [ ] **Step 6: Verify the cofactor finding dies as predicted**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c "
import pandas as pd
from pb import paths
from pb.build import analysis_population
from pb.strata import stratum_frame
from pb.inference import cluster_bootstrap_d
df = analysis_population(pd.read_parquet(paths.JOINED_PARQUET))
f = stratum_frame(df, 'no_clashes_with_organic_cofactors', 'diffdock')
est = cluster_bootstrap_d(f, 'no_clashes_with_organic_cofactors', 'n_aromatic_rings', n_boot=500)
print(f'conditioned on cofactor presence: d={est.point:+.2f} CI[{est.lo:+.2f},{est.hi:+.2f}] p={est.p_value:.3f}')
print('significant:', est.significant())
"
```

Expected: interval straddles zero, `significant: False`. This is the retraction evidence for
Task 9.

- [ ] **Step 7: Commit**

```bash
git add src/pb/strata.py src/pb/build.py tests/test_strata.py
git commit -m "feat: per-method and cofactor-conditioned strata for every effect"
```

---

## Task 5: Crystal-contact flag from symmetry mates

Issue #4. The authors dropped 120 of 428 complexes for the journal version precisely because
the ligand touches a crystallisation artifact — and protein clash is the dominant failure mode
at 84%. The 308-complex list is only in Table S2 of the paywalled SI and is not in the archive
or either GitHub repo, so it is computed here instead: download the deposited entry, generate
symmetry images from the spacegroup and unit cell, and measure how close the ligand comes to a
neighbouring copy of the protein.

**Files:**
- Create: `src/pb/crystal_contacts.py`
- Modify: `src/pb/paths.py` (add `PDB_CACHE`, `CRYSTAL_CONTACTS_PARQUET`)
- Modify: `src/pb/build.py` (join the flag)

**Interfaces:**
- Consumes: `pb.paths`
- Produces:
  - `pb.paths.PDB_CACHE: Path`, `pb.paths.CRYSTAL_CONTACTS_PARQUET: Path`
  - `fetch_entry(pdb_id: str) -> Path | None`
  - `min_symmetry_distance(path: Path, ccd_id: str) -> float | None`
  - `build_flags(complexes: pd.DataFrame, cutoff: float = 4.0) -> pd.DataFrame` with columns
    `pdb_id, min_symmetry_distance, crystal_contact`
  - joined column `crystal_contact: boolean` on `poses_joined.parquet`

- [ ] **Step 1: Add the paths**

Modify `src/pb/paths.py`, after `DESCRIPTORS_PARQUET`:

```python
PDB_CACHE = ROOT / "data" / "pdb_cache"
CRYSTAL_CONTACTS_PARQUET = PROCESSED / "crystal_contacts.parquet"
```

And add `PDB_CACHE` to the loop in `ensure_dirs`:

```python
def ensure_dirs() -> None:
    for d in (PROCESSED, REPORTS, FIGURES, PDB_CACHE):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Write the implementation**

Create `src/pb/crystal_contacts.py`:

```python
"""Flag complexes where the ligand touches a neighbouring copy in the crystal.

The published benchmark ships 428 complexes; the journal version reports on 308
after removing those whose ligand contacts a crystallisation artifact. That
308-ID list lives in supplementary Table S2 and is not distributed with the
data, so the flag is recomputed from first principles: read the deposited entry,
expand the unit cell by the spacegroup, and find the closest approach between
the ligand of interest and any symmetry image of the protein.

This matters because protein clash is by far the most-failed check (84% of
deep-learning poses). If clash failures concentrate in contact-bearing
complexes, the headline number is partly an artifact of the benchmark.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request

import gemmi
import pandas as pd

from . import paths

log = logging.getLogger(__name__)

RCSB = "https://files.rcsb.org/download"


def fetch_entry(pdb_id: str) -> "paths.Path | None":
    """Download one mmCIF entry to the cache, or return the cached copy."""
    target = paths.PDB_CACHE / f"{pdb_id.upper()}.cif.gz"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{RCSB}/{pdb_id.upper()}.cif.gz"
    try:
        urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https RCSB URL
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        log.warning("could not fetch %s: %s", pdb_id, exc)
        target.unlink(missing_ok=True)
        return None
    return target


def min_symmetry_distance(path: "paths.Path", ccd_id: str) -> float | None:
    """Closest approach between the named ligand and any symmetry image.

    Returns None when the entry has no ligand of that name, or no cell and
    spacegroup to expand. A large value means the ligand sits well inside its own
    asymmetric unit; a small one means it is packed against a neighbour.
    """
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_hydrogens()

    if structure.spacegroup_hm in ("", "P 1") and structure.cell.volume <= 0:
        return None

    ligand_atoms = [
        atom
        for chain in structure[0]
        for residue in chain
        if residue.name == ccd_id
        for atom in residue
    ]
    if not ligand_atoms:
        return None

    search = gemmi.NeighborSearch(structure[0], structure, 5.0).populate()

    closest = float("inf")
    for atom in ligand_atoms:
        for mark in search.find_atoms(atom.pos, "\0", radius=5.0):
            # image_idx 0 is the original copy; anything else is a symmetry mate
            if mark.image_idx == 0:
                continue
            cra = mark.to_cra(structure[0])
            if cra.residue.name == ccd_id:
                continue
            position = structure.cell.find_nearest_pbc_position(
                atom.pos, cra.atom.pos, mark.image_idx
            )
            closest = min(closest, atom.pos.dist(position))

    return None if closest == float("inf") else float(closest)


def build_flags(complexes: pd.DataFrame, cutoff: float = 4.0) -> pd.DataFrame:
    """One row per complex with its closest symmetry contact.

    `cutoff` of 4.0 A is roughly a van der Waals contact between heavy atoms;
    below it the ligand is genuinely touching a neighbouring copy.
    """
    paths.ensure_dirs()
    rows = []
    for record in complexes.itertuples():
        entry = fetch_entry(record.pdb_id)
        distance = (
            min_symmetry_distance(entry, record.ccd_id) if entry is not None else None
        )
        rows.append({
            "pdb_id": record.pdb_id,
            "min_symmetry_distance": distance,
            "crystal_contact": None if distance is None else distance < cutoff,
        })

    flags = pd.DataFrame(rows)
    flags["crystal_contact"] = flags["crystal_contact"].astype("boolean")
    return flags


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    descriptors = pd.read_parquet(paths.DESCRIPTORS_PARQUET)
    complexes = descriptors[descriptors["dataset"] == "posebuster"][["pdb_id", "ccd_id"]]

    flags = build_flags(complexes)
    flags.to_parquet(paths.CRYSTAL_CONTACTS_PARQUET, index=False)

    resolved = flags["crystal_contact"].notna().sum()
    log.info("resolved %d/%d complexes", resolved, len(flags))
    log.info("crystal contacts: %d", int(flags["crystal_contact"].sum()))
    log.info("unresolved (no cell, no ligand, or fetch failed): %d",
             int(flags["crystal_contact"].isna().sum()))
```

- [ ] **Step 3: Run it**

Run: `PYTHONPATH=src .venv/bin/python -m pb.crystal_contacts`
Expected: 428 entries fetched (a few minutes, ~400 MB cached), a contact count logged.
The count should land in the same neighbourhood as the 120 the authors removed. If it is wildly
different (say under 40 or over 250), stop and report the discrepancy rather than proceeding —
the geometry is wrong and the sensitivity analysis would be meaningless.

- [ ] **Step 4: Join the flag in `pb.build`**

In `src/pb/build.py`, inside `main()`, after `df = add_bins(df)`:

```python
    if paths.CRYSTAL_CONTACTS_PARQUET.exists():
        flags = pd.read_parquet(paths.CRYSTAL_CONTACTS_PARQUET)
        df = df.merge(flags, on="pdb_id", how="left")
        log.info("crystal-contact flag joined: %d complexes flagged",
                 int(flags["crystal_contact"].sum()))
    else:
        df["crystal_contact"] = pd.array([None] * len(df), dtype="boolean")
        df["min_symmetry_distance"] = float("nan")
        log.warning("no crystal-contact flags - run `python -m pb.crystal_contacts`")
```

- [ ] **Step 5: Rebuild and measure the confound**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pb.build
PYTHONPATH=src .venv/bin/python -c "
import pandas as pd
from pb import paths
from pb.build import analysis_population
df = analysis_population(pd.read_parquet(paths.JOINED_PARQUET))
for m in ['diffdock','vina']:
    s = df[df.method==m]
    for flag in [True, False]:
        g = s[s.crystal_contact == flag]
        g = g[g.no_clashes_with_protein.notna()]
        if len(g):
            print(f'{m:9s} crystal_contact={str(flag):5s} n={len(g):4d} protein-clash failure {(g.no_clashes_with_protein==False).mean():.1%}')
"
```

Expected: a materially higher clash-failure rate among contact-bearing complexes. Record the
gap — Task 9 reports the headline both ways.

- [ ] **Step 6: Commit**

```bash
echo "data/pdb_cache/" >> .gitignore
git add src/pb/crystal_contacts.py src/pb/paths.py src/pb/build.py .gitignore
git commit -m "feat: compute crystal-contact flags from symmetry mates"
```

---

## Task 6: Multivariate model replacing 324 marginal comparisons

Issue #10. Molecular weight and heavy-atom count correlate at 0.99; TPSA, HBD and stereocentres
cluster together. "Top 3 descriptors" is frequently three measurements of one latent property.
One clustered logistic regression per (check, method) answers which descriptors matter
*holding the others fixed*, which is what the report has been claiming all along.

**Files:**
- Create: `src/pb/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `pb.strata.stratum_frame`
- Produces:
  - `DESCRIPTOR_BASIS: list[str]` — the decorrelated subset used by every model
  - `variance_inflation(df, descriptors) -> pd.Series`
  - `fit_check_model(df, check, method, descriptors=DESCRIPTOR_BASIS) -> pd.DataFrame`
    with columns `descriptor, coef, std_err, z, p_value, odds_ratio, ci_lo, ci_hi`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pb.models'`

- [ ] **Step 3: Write the implementation**

Create `src/pb/models.py`:

```python
"""One clustered logistic regression per check, replacing marginal comparisons.

Eighteen marginal Cohen's d values per check invite a reading the data cannot
support: molecular weight and heavy-atom count correlate at 0.99, and TPSA, HBD
and stereocentre count move together, so the "top three descriptors" are often
one latent property measured three ways. A single model per check estimates each
descriptor's contribution holding the rest fixed.

Standard errors are clustered on `pdb_id` because each ligand contributes one
row per method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .strata import stratum_frame

# Descriptors kept after dropping near-duplicates. `heavy_atoms` (rho 0.99 with
# `mw`) and `rot_per_heavy_atom` (a ratio of two terms already present) are out.
DESCRIPTOR_BASIS = [
    "mw",
    "n_rotatable_bonds",
    "n_rings",
    "n_aromatic_rings",
    "n_stereocentres",
    "tpsa",
    "clogp",
    "formal_charge",
    "n_halogens",
]


def variance_inflation(df: pd.DataFrame, descriptors: list[str]) -> pd.Series:
    """VIF per descriptor. Above ~10 means the column is nearly a combination of others."""
    frame = df[descriptors].dropna()
    values = {}
    for name in descriptors:
        others = [d for d in descriptors if d != name]
        design = sm.add_constant(frame[others].to_numpy(), has_constant="add")
        fitted = sm.OLS(frame[name].to_numpy(), design).fit()
        values[name] = float("inf") if fitted.rsquared >= 1 else 1 / (1 - fitted.rsquared)
    return pd.Series(values).sort_values(ascending=False)


def fit_check_model(
    df: pd.DataFrame,
    check: str,
    method: str,
    descriptors: list[str] | None = None,
) -> pd.DataFrame:
    """Logistic regression of check failure on standardised descriptors.

    Descriptors are z-scored so coefficients are comparable across columns: each
    is the log-odds change in failure per standard deviation of that descriptor.
    """
    descriptors = descriptors or DESCRIPTOR_BASIS

    frame = df if "method" not in df.columns else stratum_frame(df, check, method)
    frame = frame.dropna(subset=[*descriptors, check, "pdb_id"])
    if frame.empty:
        return pd.DataFrame(columns=["descriptor", "coef", "std_err", "z",
                                     "p_value", "odds_ratio", "ci_lo", "ci_hi"])

    outcome = (frame[check] == False).astype(int).to_numpy()  # noqa: E712
    if outcome.sum() in (0, len(outcome)):
        return pd.DataFrame(columns=["descriptor", "coef", "std_err", "z",
                                     "p_value", "odds_ratio", "ci_lo", "ci_hi"])

    design = frame[descriptors].astype(float)
    spread = design.std(ddof=0).replace(0, np.nan)
    design = ((design - design.mean()) / spread).fillna(0.0)
    design = sm.add_constant(design, has_constant="add")

    fitted = sm.Logit(outcome, design).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": frame["pdb_id"].to_numpy()},
    )

    confidence = fitted.conf_int()
    out = pd.DataFrame({
        "descriptor": fitted.params.index,
        "coef": fitted.params.to_numpy(),
        "std_err": fitted.bse.to_numpy(),
        "z": fitted.tvalues.to_numpy(),
        "p_value": fitted.pvalues.to_numpy(),
        "odds_ratio": np.exp(fitted.params.to_numpy()),
        "ci_lo": np.exp(confidence.iloc[:, 0].to_numpy()),
        "ci_hi": np.exp(confidence.iloc[:, 1].to_numpy()),
    })
    return out[out["descriptor"] != "const"].reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Check collinearity on the real data and fit the headline model**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c "
import pandas as pd
from pb import paths
from pb.build import analysis_population
from pb.models import DESCRIPTOR_BASIS, fit_check_model, variance_inflation
df = analysis_population(pd.read_parquet(paths.JOINED_PARQUET))
print('VIF:'); print(variance_inflation(df.drop_duplicates('pdb_id'), DESCRIPTOR_BASIS).round(2).to_string())
print()
for m in ['diffdock','unimol','vina']:
    out = fit_check_model(df, 'energy_ratio_within_threshold', m)
    if out.empty: print(f'{m}: no fit'); continue
    top = out.reindex(out.p_value.sort_values().index).head(3)
    print(f'{m}: ' + '  '.join(f\"{r.descriptor}={r.coef:+.2f}(p={r.p_value:.3f})\" for r in top.itertuples()))
"
```

Expected: every VIF below 10 (if not, drop the offender from `DESCRIPTOR_BASIS` and re-run).
The per-method output shows whether `n_rotatable_bonds` survives adjustment for `mw` — this is
the direct test of the report's central claim.

- [ ] **Step 6: Commit**

```bash
git add src/pb/models.py tests/test_models.py
git commit -m "feat: clustered logistic models replace marginal comparisons"
```

---

## Task 7: Validate the pipeline by actually running `bust`

Issue #8. Nothing so far confirms the checks are understood correctly. Running PoseBusters on
the 513 crystal structures reproduces the `crystal_structures` rows of the published CSV, which
validates the whole chain end to end.

**Files:**
- Create: `src/pb/validate.py`

**Interfaces:**
- Consumes: `pb.paths.SET_DIRS`, `pb.build.load_results`
- Produces:
  - `run_bust(limit: int | None = None) -> pd.DataFrame` — our own check results per complex
  - `compare_to_published(ours: pd.DataFrame) -> pd.DataFrame` — per-check agreement counts

- [ ] **Step 1: Write the implementation**

Create `src/pb/validate.py`:

```python
"""Reproduce the paper's crystal-structure rows by running PoseBusters ourselves.

Nothing else in this project executes a single check - it consumes the authors'
published results. Re-running `bust` on the 513 crystal ligands against their own
receptors reproduces the `crystal_structures` rows of that CSV, which is the only
end-to-end evidence that the checks are being interpreted correctly. Any
systematic disagreement invalidates the interpretation in the report.
"""

from __future__ import annotations

import logging

import pandas as pd
from posebusters import PoseBusters

from . import paths
from .build import load_results

log = logging.getLogger(__name__)

# bust's own column names -> the published CSV's names
COLUMN_MAP = {
    "sanitization": "molecule_passes_rdkit_sanity_check",
    "molecular_formula": "molecular_formula_preserved",
    "molecular_bonds": "molecular_bonds_preserved",
    "tetrahedral_chirality": "sp3_stereochemistry_preserved",
    "double_bond_stereochemistry": "double_bond_stereochemistry_preserved",
    "bond_lengths": "bond_lengths_within_bounds",
    "bond_angles": "bond_angles_within_bounds",
    "internal_steric_clash": "no_internal_clashes",
    "aromatic_ring_flatness": "aromatic_ring_flatness_passes",
    "double_bond_flatness": "double_bond_flatness_passes",
    "internal_energy": "energy_ratio_within_threshold",
    "minimum_distance_to_protein": "no_clashes_with_protein",
    "minimum_distance_to_organic_cofactors": "no_clashes_with_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors": "no_clashes_with_inorganic_cofactors",
    "volume_overlap_with_protein": "no_volume_clash_with_protein",
    "volume_overlap_with_organic_cofactors": "no_volume_clash_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors": "no_volume_clash_with_inorganic_cofactors",
}


def run_bust(limit: int | None = None) -> pd.DataFrame:
    """Run the redock suite on every crystal ligand against its own receptor."""
    jobs = []
    for dataset, directory in paths.SET_DIRS.items():
        for complex_dir in sorted(directory.iterdir()):
            if not complex_dir.is_dir():
                continue
            name = complex_dir.name
            jobs.append({
                "dataset": dataset,
                "pdb_id": name.partition("_")[0],
                "mol_pred": complex_dir / f"{name}_ligand.sdf",
                "mol_true": complex_dir / f"{name}_ligand.sdf",
                "mol_cond": complex_dir / f"{name}_protein.pdb",
            })
    if limit is not None:
        jobs = jobs[:limit]

    table = pd.DataFrame(jobs)
    log.info("running bust on %d crystal structures", len(table))

    buster = PoseBusters(config="redock")
    results = buster.bust_table(
        table[["mol_pred", "mol_true", "mol_cond"]], full_report=False
    ).reset_index(drop=True)

    return pd.concat([table[["dataset", "pdb_id"]], results], axis=1)


def compare_to_published(ours: pd.DataFrame) -> pd.DataFrame:
    """Per-check agreement between our run and the paper's crystal rows."""
    published = load_results()
    published = published[published["method"] == "crystal_structures"]

    renamed = ours.rename(columns=COLUMN_MAP)
    merged = renamed.merge(
        published, on=["dataset", "pdb_id"], suffixes=("_ours", "_theirs")
    )

    rows = []
    for column in COLUMN_MAP.values():
        ours_col, theirs_col = f"{column}_ours", f"{column}_theirs"
        if ours_col not in merged or theirs_col not in merged:
            continue
        both = merged[[ours_col, theirs_col]].dropna()
        agree = (both[ours_col].astype(bool) == both[theirs_col].astype(bool)).sum()
        rows.append({
            "check": column,
            "n_compared": len(both),
            "n_agree": int(agree),
            "agreement": agree / len(both) if len(both) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("agreement")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths.ensure_dirs()

    ours = run_bust()
    ours.to_parquet(paths.PROCESSED / "our_bust_crystal.parquet", index=False)

    agreement = compare_to_published(ours)
    agreement.to_csv(paths.REPORTS / "tables" / "bust_reproduction.csv", index=False)
    print(agreement.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))

    worst = agreement["agreement"].min()
    log.info("worst per-check agreement: %.1f%%", worst * 100)
    if worst < 0.95:
        log.error("a check disagrees on >5%% of crystal structures - "
                  "the interpretation in reports/findings.md is not safe to publish")
```

- [ ] **Step 2: Smoke-test on 20 complexes before committing to the full run**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c "
import logging; logging.basicConfig(level=logging.INFO)
from pb.validate import run_bust, compare_to_published
ours = run_bust(limit=20)
print(compare_to_published(ours).to_string(index=False))
"
```

Expected: agreement at or near 1.000 for every check. If `PoseBusters(config=...)` raises, check
the installed API with `.venv/bin/python -c "import posebusters; help(posebusters.PoseBusters)"`
and adjust — the package's constructor signature has changed across releases.

- [ ] **Step 3: Run the full validation**

Run: `PYTHONPATH=src .venv/bin/python -m pb.validate`
Expected: 513 complexes, several minutes, worst-check agreement ≥ 0.95. Anything lower is a
blocking finding — report it rather than continuing.

- [ ] **Step 4: Commit**

```bash
git add src/pb/validate.py reports/tables/bust_reproduction.csv
git commit -m "test: reproduce the paper's crystal-structure rows with bust"
```

---

## Task 8: Replication set, continuous RMSD, and the collider caveat

Issues #7 and #11. The 85 Astex complexes are an untouched replication set; the continuous RMSD
column is discarded at the 2 Å threshold; and "accurate but invalid" conditions on an outcome
that shares an upstream cause with validity, which can manufacture association.

**Files:**
- Create: `src/pb/replication.py`

**Interfaces:**
- Consumes: `pb.build.analysis_population`, `pb.strata.association_grid`, `pb.models.fit_check_model`
- Produces:
  - `replicate_on_astex(df, methods, descriptors) -> pd.DataFrame` with columns
    `method, check, descriptor, d_posebuster, d_astex, same_sign, both_significant`
  - `validity_vs_rmsd(df, method, edges) -> pd.DataFrame` — validity rate by RMSD band,
    including poses that fail the 2 Å threshold

- [ ] **Step 1: Write the implementation**

Create `src/pb/replication.py`:

```python
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
    if len(replication):
        log.info("effects reproducing in sign on held-out Astex: %d/%d",
                 int(replication["same_sign"].sum()), len(replication))

    population = analysis_population(joined)
    bands = pd.concat([validity_vs_rmsd(population, m) for m in methods])
    bands.to_csv(tables / "validity_vs_rmsd.csv", index=False)
    print(bands.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src .venv/bin/python -m pb.replication`
Expected: a replication table and an RMSD-band table. Effects that flip sign on Astex are
demoted to "not replicated" in Task 9 regardless of their p-value on the benchmark set.

- [ ] **Step 3: Commit**

```bash
git add src/pb/replication.py reports/tables/astex_replication.csv reports/tables/validity_vs_rmsd.csv
git commit -m "feat: held-out Astex replication and unconditioned RMSD view"
```

---

## Task 9: Rebuild the analysis, figures and report on the new foundation

Everything above produces machinery. This task rewrites the outputs so the published claims are
the ones that survived.

**Files:**
- Rewrite: `src/pb/analyze.py`
- Modify: `src/pb/figures.py`
- Rewrite: `reports/findings.md`

**Interfaces:**
- Consumes: every module built in Tasks 2–8
- Produces: `reports/tables/association_grid.csv`, `reports/tables/check_models.csv`,
  `reports/tables/crystal_contact_sensitivity.csv`, regenerated figures, rewritten report

- [ ] **Step 1: Rewrite `pb.analyze` around the new layers**

Replace the body of `src/pb/analyze.py`'s `main()` with:

```python
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    TABLES.mkdir(parents=True, exist_ok=True)

    from .build import analysis_population, CLASSICAL_METHODS, DL_METHODS
    from .models import DESCRIPTOR_BASIS, fit_check_model
    from .strata import association_grid

    joined = pd.read_parquet(paths.JOINED_PARQUET)
    df = analysis_population(joined)
    methods = DL_METHODS + CLASSICAL_METHODS

    grid = association_grid(df, methods, DESCRIPTOR_BASIS, n_boot=2000, seed=0)
    grid.to_csv(TABLES / "association_grid.csv", index=False)
    log.info("association grid: %d estimates, %d survive BH-FDR",
             len(grid), int(grid["fdr_reject"].sum()))

    models = []
    for method in methods:
        for check in grid["check"].unique():
            fitted = fit_check_model(df, check, method)
            if not fitted.empty:
                fitted.insert(0, "check", check)
                fitted.insert(0, "method", method)
                models.append(fitted)
    model_table = pd.concat(models, ignore_index=True) if models else pd.DataFrame()
    model_table.to_csv(TABLES / "check_models.csv", index=False)

    if df["crystal_contact"].notna().any():
        rows = []
        for method in methods:
            for flag in (True, False):
                sub = df[(df["method"] == method) & (df["crystal_contact"] == flag)]
                sub = sub[sub["no_clashes_with_protein"].notna()]
                if len(sub):
                    rows.append({
                        "method": method,
                        "crystal_contact": flag,
                        "n": len(sub),
                        "protein_clash_failure": float(
                            (sub["no_clashes_with_protein"] == False).mean()  # noqa: E712
                        ),
                    })
        pd.DataFrame(rows).to_csv(
            TABLES / "crystal_contact_sensitivity.csv", index=False
        )

    print(grid[grid["fdr_reject"]]
          .head(25)[["method", "check", "descriptor", "d", "lo", "hi",
                     "n_fail", "n_eligible", "n_clusters"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src .venv/bin/python -m pb.analyze`
Expected: the surviving-effects table. Note how many of the 324 original associations survive
FDR — this number goes in the report.

- [ ] **Step 3: Put error bars on every figure**

In `src/pb/figures.py`, replace `fig_association_heatmap` with a per-method forest plot. Add:

```python
def fig_effect_forest(grid: pd.DataFrame, top_n: int = 18) -> None:
    """Surviving effects as points with cluster-bootstrap intervals, faceted by method."""
    surviving = grid[grid["fdr_reject"]].nlargest(top_n, "abs_d")
    if surviving.empty:
        log.warning("no effects survived FDR - skipping forest plot")
        return

    surviving = surviving.iloc[::-1].reset_index(drop=True)
    labels = [
        f"{r.check.replace('_', ' ')} × {r.descriptor.replace('_', ' ')}  [{r.method}]"
        for r in surviving.itertuples()
    ]
    y = np.arange(len(surviving))

    fig, ax = plt.subplots(figsize=(9, 0.34 * len(surviving) + 1.6))
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    ax.hlines(y, surviving["lo"], surviving["hi"], color=ACCENT, linewidth=2, zorder=2)
    ax.plot(surviving["d"], y, "o", markersize=6, color=ACCENT,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Cohen's d  (cluster-bootstrap 95% CI, BH-FDR controlled)")
    ax.yaxis.grid(False)
    _strip(ax)
    ax.set_title("Effects that survive eligibility gating, clustering and FDR",
                 fontsize=10.5, weight="semibold", loc="left", pad=10)
    fig.savefig(paths.FIGURES / "03_effect_forest.png")
    plt.close(fig)
```

Update `figures.main()` to read `reports/tables/association_grid.csv` and call
`fig_effect_forest(grid)` instead of `fig_association_heatmap(dl)`. Delete the old
`fig_association_heatmap` function and the now-unused `03_association_heatmap.png`.

- [ ] **Step 4: Regenerate and inspect**

Run:

```bash
rm -f reports/figures/*.png
PYTHONPATH=src .venv/bin/python -m pb.figures
```

Open each PNG and check for label collisions and clipped text before accepting them.

- [ ] **Step 5: Rewrite `reports/findings.md`**

Rules for the rewrite, applied strictly:

1. Delete the cofactor finding entirely. It does not survive conditioning (*d* −0.29 → −0.02).
   Add one line to a new "Retracted" section saying so and why.
2. Replace every Cohen's *d* in the effect table with its eligibility-gated value and its
   cluster-bootstrap CI. Any effect whose CI crosses zero, or that fails BH-FDR, is removed.
3. Report the flexibility claim **per method**, not pooled. State explicitly that it is not
   established for DiffDock alone if the Task 6 model says so.
4. Report the headline accuracy/validity table both with and without crystal-contact complexes,
   using `crystal_contact_sensitivity.csv`.
5. Add the Astex replication column to the effect table; mark anything that flips sign
   "not replicated".
6. Replace the "accurate but invalid" framing as the *primary* metric with validity against
   continuous RMSD bands; keep the conditioned metric as secondary with an explicit note that
   conditioning on accuracy can induce association.
7. Add a "Validation" section reporting the `bust` reproduction agreement from Task 7.
8. Update the Limits section: remove limits now addressed, keep the small-denominator caveat.

- [ ] **Step 6: Verify the report cites only surviving numbers**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c "
import pandas as pd, re, pathlib
grid = pd.read_csv('reports/tables/association_grid.csv')
survivors = set(zip(grid[grid.fdr_reject].method, grid[grid.fdr_reject].check, grid[grid.fdr_reject].descriptor))
text = pathlib.Path('reports/findings.md').read_text()
print(f'{len(survivors)} surviving effects available to cite')
print('report mentions cofactor:', 'cofactor' in text.lower())
print('report mentions Retracted section:', 'Retract' in text)
print('report mentions bust reproduction:', 'reproduc' in text.lower())
"
```

Expected: the retraction and validation sections are present.

- [ ] **Step 7: Full pipeline run from clean**

Run:

```bash
rm -rf data/processed reports/tables reports/figures
export PYTHONPATH=src
for stage in acquire descriptors crystal_contacts build analyze replication figures; do
  printf "%-18s " "$stage"
  .venv/bin/python -m pb.$stage >/dev/null 2>&1 && echo OK || echo FAILED
done
.venv/bin/pytest -q
```

Expected: every stage OK, all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/pb/analyze.py src/pb/figures.py reports/
git commit -m "refactor: rebuild analysis and report on gated, clustered, FDR-controlled estimates"
```

---

## Self-Review

**Spec coverage.** Each of the 11 audit issues maps to a task: #1 → Task 2; #2 → Task 4;
#3 → Task 4 + Task 6; #4 → Task 5; #5 → Task 3; #6 → Task 3; #7 → Task 8; #8 → Task 7;
#9 → Task 1; #10 → Task 6; #11 → Task 8. Task 9 propagates all of them into the outputs.

**Placeholders.** None. Every code step contains runnable code; every verification step names
the command and the expected result.

**Type consistency.** `Estimate` is defined in Task 3 and consumed unchanged in Tasks 4 and 8.
`cohens_d` moves from `pb.analyze` to `pb.inference` in Task 3; Task 9's rewrite of
`pb.analyze` no longer defines it, and Task 2's Step 5 diagnostic imports it from `pb.analyze`
while that still holds — run Task 2 before Task 3, as ordered. `eligible`/`eligible_mask`
(Task 2) are consumed by `stratum_frame` (Task 4). `stratum_frame` (Task 4) is consumed by
`fit_check_model` (Task 6) and `replicate_on_astex` (Task 8). `analysis_population` is added to
`pb.build` in Task 4 and used from Task 5 onward.

**Known risks.** Two tasks depend on external services and may need adjustment at execution
time: Task 5 fetches 428 entries from RCSB (~400 MB, rate-limited), and Task 7 depends on the
installed `posebusters` API, whose constructor has changed across releases — its Step 2 is a
20-complex smoke test precisely so this surfaces cheaply.
