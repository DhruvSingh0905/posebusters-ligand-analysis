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
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .eligibility import ELIGIBILITY
from .strata import stratum_frame

# Descriptors kept after dropping near-duplicates. `heavy_atoms` (rho 0.99 with
# `mw`) and `rot_per_heavy_atom` (a ratio of two terms already present) are out
# by construction.
#
# `tpsa` and `n_rings` were dropped after `variance_inflation` on the real
# analysis population (Task 6, Step 5) flagged VIF >= 10 for `mw` (27.80),
# `tpsa` (17.01) and `n_rings` (12.91) - not from any single pairwise
# correlation (max pairwise rho among the original nine was 0.83, n_rings vs
# n_aromatic_rings), but from `mw` being well reconstructed by a combination of
# the others (R^2 = 0.96). `mw` is the size term this project's central claim
# is contrasted against, so it stays deliberately despite having had the
# highest initial VIF: it is one of the two competing hypotheses (size vs.
# flexibility), not a nuisance covariate to be dropped for statistical
# convenience. `tpsa` (rho 0.78 with n_stereocentres, 0.65 with mw - the
# TPSA/stereocentre cluster the module docstring already calls out) and
# `n_rings` (rho 0.83 with n_aromatic_rings, a near-strict superset) are the
# redundant members of their respective clusters and were dropped instead.
# Re-running `variance_inflation` on the remaining seven descriptors gives a
# maximum VIF of 8.84 (`mw`).
DESCRIPTOR_BASIS = [
    "mw",
    "n_rotatable_bonds",
    "n_aromatic_rings",
    "n_stereocentres",
    "clogp",
    "formal_charge",
    "n_halogens",
]

_EMPTY_COLUMNS = ["descriptor", "coef", "std_err", "z", "p_value",
                  "odds_ratio", "ci_lo", "ci_hi"]


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

    # The unit test passes a hand-built frame with a synthetic check name that
    # is not in the real registry; real callers always use a registered check.
    # Test membership rather than catching KeyError, so a genuine fault in the
    # eligibility chain (e.g. a missing `has_cofactors` column) still raises
    # instead of silently falling through to an ungated, method-pooled frame.
    if "method" in df.columns and check in ELIGIBILITY:
        frame = stratum_frame(df, check, method)
    else:
        frame = df
    frame = frame.dropna(subset=[*descriptors, check, "pdb_id"])
    if frame.empty:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    outcome = (frame[check] == False).astype(int).to_numpy()  # noqa: E712
    if outcome.sum() in (0, len(outcome)):
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    design = frame[descriptors].astype(float)
    spread = design.std(ddof=0).replace(0, np.nan)
    design = ((design - design.mean()) / spread).fillna(0.0)
    design = sm.add_constant(design, has_constant="add")

    # Thin or near-degenerate strata (e.g. a handful of failures, or a
    # descriptor that separates the outcome perfectly within the stratum) can
    # make the MLE diverge. Treat those the same as the other "can't estimate
    # here" cases above rather than letting the loop crash on real data.
    try:
        fitted = sm.Logit(outcome, design).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": frame["pdb_id"].to_numpy()},
        )
    except (PerfectSeparationError, np.linalg.LinAlgError):
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    if not fitted.mle_retvals.get("converged", True):
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

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
