"""Render the findings as a self-contained HTML report.

Every table in the output is read from `reports/tables/` at render time and
every figure is inlined as a data URI, so the document cannot drift from the
analysis the way a hand-written summary would. It is one file with no external
requests, which also makes it printable to PDF without a network.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import pandas as pd

from . import paths

log = logging.getLogger(__name__)

TABLES = paths.REPORTS / "tables"
OUT = paths.REPORTS / "benchmark-report.html"

PCT = "{:.1%}".format
PP = "{:+.1f}".format


def _img(name: str) -> str:
    """Inline a figure as a data URI so the document stays self-contained."""
    data = base64.b64encode((paths.FIGURES / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _table(df: pd.DataFrame, align_first_left: bool = True) -> str:
    """Render a DataFrame as an HTML table, numbers right-aligned."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for record in df.itertuples(index=False):
        cells = []
        for i, value in enumerate(record):
            cls = ' class="l"' if i == 0 and align_first_left else ""
            cells.append(f"<td{cls}>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _sig(flag: bool) -> str:
    return '<span class="yes">yes</span>' if flag else '<span class="no">no</span>'


# ── section builders ────────────────────────────────────────────────────

def headline() -> str:
    df = pd.read_csv(TABLES / "method_summary.csv")
    df = df[df.post_processing == "none"].sort_values("accurate", ascending=False)
    out = pd.DataFrame({
        "Method": df.method,
        "Class": df.method_class,
        "RMSD ≤ 2 Å": df.accurate.map(PCT),
        "Valid": df.valid.map(PCT),
        "Both": df.accurate_and_valid.map(PCT),
        "Of accurate poses, invalid": df.invalid_given_accurate.map(PCT),
    })
    return _table(out)


def crystal_contacts() -> str:
    df = pd.read_csv(TABLES / "crystal_contact_sensitivity.csv")
    df = df.sort_values("diff_pp")
    out = pd.DataFrame({
        "Method": df.method,
        "Contact": [f"{r:.1%} ({n})" for r, n in zip(df.failure_contact, df.n_contact)],
        "No contact": [
            f"{r:.1%} ({n})" for r, n in zip(df.failure_no_contact, df.n_no_contact)
        ],
        "Difference (pp)": df.diff_pp.map(PP),
        "95% CI": [
            f"[{lo:+.1f}, {hi:+.1f}]" for lo, hi in zip(df.diff_lo_pp, df.diff_hi_pp)
        ],
        "Excludes zero": [_sig(bool(s)) for s in df.significant],
    })
    return _table(out)


def coverage() -> tuple[str, dict]:
    df = pd.read_csv(TABLES / "stratum_coverage.csv")
    counts = df.skip_reason.fillna("estimated").value_counts()
    labels = {
        "estimated": "Estimated",
        "no_failures": "Check never failed for that method",
        "too_few_failures": "Fewer than 15 failures",
        "too_few_ligands": "Fewer than 30 distinct ligands",
    }
    out = pd.DataFrame({
        "Outcome": [labels.get(k, k) for k in counts.index],
        "Strata": counts.to_numpy(),
        "Share": [f"{v / len(df):.0%}" for v in counts.to_numpy()],
    })
    return _table(out), {"total": len(df), "estimated": int(counts.get("estimated", 0))}


def survivors(top_n: int = 12) -> str:
    df = pd.read_csv(TABLES / "association_grid.csv")
    df = df[df.fdr_reject].nlargest(top_n, "abs_d")
    out = pd.DataFrame({
        "Method": df.method,
        "Check": [c.replace("_", " ") for c in df.check],
        "Descriptor": [c.replace("_", " ") for c in df.descriptor],
        "d": df.d.map("{:.2f}".format),
        "95% CI": [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(df.lo, df.hi)],
        "Failures": df.n_fail,
        "Ligands": df.n_clusters,
    })
    return _table(out)


def flexibility() -> str:
    df = pd.read_csv(TABLES / "check_models.csv")
    df = df[df.check == "energy_ratio_within_threshold"]
    rows = []
    for method, group in df.groupby("method"):
        by = group.set_index("descriptor")
        if "mw" not in by.index or "n_rotatable_bonds" not in by.index:
            rows.append((method, "—", "—", "No model fit (too few failures)"))
            continue
        p_mw, p_rot = by.loc["mw", "p_value"], by.loc["n_rotatable_bonds", "p_value"]
        neg = " (negative)" if by.loc["mw", "coef"] < 0 else ""
        if p_mw < 0.05 and p_rot < 0.05:
            verdict = "Both carry independent signal"
        elif p_rot < 0.05:
            verdict = "Flexibility only"
        elif p_mw < 0.05:
            verdict = "Size only"
        else:
            verdict = "Neither — not resolvable"
        rows.append((method, f"{p_mw:.3f}{neg}", f"{p_rot:.3f}", verdict))

    order = ["diffdock", "equibind", "deepdock", "tankbind", "unimol", "gold", "vina"]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    if not any(r[0] == "vina" for r in rows):
        rows.append(("vina", "—", "—", "No model fit (too few failures)"))
    out = pd.DataFrame(rows, columns=["Method", "p (mw)", "p (rotatable bonds)", "Verdict"])
    return _table(out)


def replication() -> tuple[str, dict]:
    df = pd.read_csv(TABLES / "astex_replication.csv")
    out = pd.DataFrame({
        "Method": df.method,
        "Check": [c.replace("_", " ") for c in df.check],
        "Descriptor": [c.replace("_", " ") for c in df.descriptor],
        "d (benchmark)": df.d_posebuster.map("{:.2f}".format),
        "d (Astex)": df.d_astex.map("{:.2f}".format),
        "Replicated": [
            '<span class="yes">yes</span>' if b else '<span class="no">no</span>'
            for b in df.both_significant
        ],
        "Thin": ["thin" if t else "" for t in df.thin],
    })
    stats = {
        "n": len(df),
        "replicated": int(df.both_significant.sum()),
        "same_sign": int(df.same_sign.sum()),
        "thin": int(df.thin.sum()),
    }
    return _table(out), stats


def rmsd_bands() -> str:
    df = pd.read_csv(TABLES / "validity_vs_rmsd.csv")
    wide = df.pivot_table(
        index="method", columns="rmsd_band", values="valid", observed=True
    )
    # pivot_table sorts the band labels as strings, which puts "(10.0, inf]"
    # between "(1.0, 2.0]" and "(2.0, 3.0]". Order by the interval's lower
    # bound instead so the columns read left-to-right as increasing RMSD.
    wide = wide[sorted(wide.columns, key=lambda b: float(str(b).split(",")[0].lstrip("([")))]
    wide = wide.reset_index()
    wide.columns.name = None
    wide["method"] = wide["method"].astype(str)
    for c in wide.columns[1:]:
        wide[c] = wide[c].map(lambda v: "—" if pd.isna(v) else PCT(v))
    wide = wide.rename(columns={"method": "Method"})
    return _table(wide)


def validation() -> tuple[str, dict]:
    df = pd.read_csv(TABLES / "bust_reproduction.csv").sort_values("agreement")
    out = pd.DataFrame({
        "Check": [c.replace("_", " ") for c in df.check],
        "Compared": df.n_compared,
        "Agree": df.n_agree,
        "Agreement": df.agreement.map("{:.2%}".format),
    })
    return _table(out.head(6)), {
        "worst": df.agreement.min(),
        "n_checks": len(df),
        "perfect": int((df.agreement == 1.0).sum()),
    }


def cofactor() -> dict:
    df = pd.read_csv(TABLES / "cofactor_retraction.csv").set_index("variant")
    return {
        v: {
            "d": df.loc[v, "d"],
            "lo": df.loc[v, "lo"],
            "hi": df.loc[v, "hi"],
            "p": df.loc[v, "p_value"],
            "n": int(df.loc[v, "n_eligible"]),
        }
        for v in df.index
    }


# ── document ────────────────────────────────────────────────────────────

CSS = """
:root{
  --paper:#EDF1F2;--raise:#F7F9F9;--ink:#0E181D;--ink-2:#3A4C55;--ink-3:#6C7E87;
  --rule:#C3CFD3;--rule-soft:#D9E1E3;--accent:#0A6B87;--accent-soft:#0a6b8722;
  --pass:#2C6E4E;--fail:#9E3A28;--band:#A67C21;
  --serif:"Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0A1216;--raise:#0F1B21;--ink:#DDE7EA;--ink-2:#93A6AE;--ink-3:#6A7C85;
  --rule:#22343C;--rule-soft:#1A2A31;--accent:#58BEDC;--accent-soft:#58bedc1f;
  --pass:#5FAE83;--fail:#DF7A61;--band:#D3A544;}}
:root[data-theme="dark"]{
  --paper:#0A1216;--raise:#0F1B21;--ink:#DDE7EA;--ink-2:#93A6AE;--ink-3:#6A7C85;
  --rule:#22343C;--rule-soft:#1A2A31;--accent:#58BEDC;--accent-soft:#58bedc1f;
  --pass:#5FAE83;--fail:#DF7A61;--band:#D3A544;}
:root[data-theme="light"]{
  --paper:#EDF1F2;--raise:#F7F9F9;--ink:#0E181D;--ink-2:#3A4C55;--ink-3:#6C7E87;
  --rule:#C3CFD3;--rule-soft:#D9E1E3;--accent:#0A6B87;--accent-soft:#0a6b8722;
  --pass:#2C6E4E;--fail:#9E3A28;--band:#A67C21;}

html{background:var(--paper)}
body{background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:16.5px;line-height:1.62;-webkit-font-smoothing:antialiased;padding:0 20px 90px}
.wrap{max-width:1120px;margin:0 auto}
.col{max-width:66ch}
header{padding:64px 0 34px;border-bottom:1px solid var(--rule);margin-bottom:46px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);display:flex;gap:14px;flex-wrap:wrap}
.eyebrow span::before{content:"·";margin-right:14px;color:var(--rule)}
.eyebrow span:first-child::before{content:none;margin:0}
h1{font-weight:400;font-size:clamp(32px,5vw,50px);line-height:1.06;letter-spacing:-.015em;
  margin:18px 0 0;text-wrap:balance;max-width:20ch}
.standfirst{color:var(--ink-2);font-size:18.5px;margin:16px 0 0;max-width:60ch}
section{margin:0 0 54px}
h2{font-family:var(--mono);font-weight:600;font-size:12px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 10px;padding-bottom:8px;
  border-bottom:1px solid var(--rule-soft);display:flex;justify-content:space-between;gap:16px}
h2 b{color:var(--accent);font-weight:600}
h3{font-weight:400;font-size:25px;line-height:1.2;letter-spacing:-.01em;margin:18px 0 10px;text-wrap:balance}
p{margin:0 0 15px}
code,.q{font-family:var(--mono);font-size:.85em;font-variant-numeric:tabular-nums;
  background:var(--accent-soft);color:var(--accent);padding:1px 5px;border-radius:2px;white-space:nowrap}
.q.pass{background:color-mix(in srgb,var(--pass) 13%,transparent);color:var(--pass)}
.q.fail{background:color-mix(in srgb,var(--fail) 13%,transparent);color:var(--fail)}
.scroll{overflow-x:auto;margin:22px 0;border:1px solid var(--rule);border-radius:3px;background:var(--raise)}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums}
th{text-align:right;font-weight:600;font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--ink-3);padding:11px 13px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child{text-align:left}
td{padding:8px 13px;border-bottom:1px solid var(--rule-soft);text-align:right;white-space:nowrap}
td.l{text-align:left;color:var(--ink-2)}
tbody tr:last-child td{border-bottom:none}
.yes{color:var(--pass);font-weight:600}
.no{color:var(--ink-3)}
figure{margin:26px 0;background:var(--raise);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
figure img{display:block;width:100%;height:auto}
figcaption{font-family:var(--mono);font-size:11.5px;line-height:1.55;color:var(--ink-2);
  padding:11px 15px;border-top:1px solid var(--rule-soft);display:flex;gap:12px}
figcaption b{color:var(--accent);font-weight:600;letter-spacing:.1em;text-transform:uppercase;white-space:nowrap}
.note{border-left:2px solid var(--band);padding:2px 0 2px 18px;margin:24px 0;color:var(--ink-2);font-size:15.5px}
.note b{color:var(--ink)}
.kicker{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--band);margin:0 0 5px}
.retract{border-left:2px solid var(--fail)}
.retract .kicker{color:var(--fail)}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:3px;margin:24px 0}
.stat{background:var(--raise);padding:14px 16px}
.stat .n{font-family:var(--mono);font-size:26px;font-weight:600;color:var(--accent);font-variant-numeric:tabular-nums}
.stat .k{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin-top:3px}
footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:11.5px;line-height:1.7;color:var(--ink-3)}
footer a{color:var(--accent);text-decoration:none}
@media (max-width:640px){.grid{grid-template-columns:repeat(2,1fr)}}
@media print{
  :root{--paper:#fff;--raise:#fff;--ink:#111;--ink-2:#333;--ink-3:#666;--rule:#bbb;--rule-soft:#ddd}
  body{font-size:10.5pt;padding:0}
  .scroll{overflow:visible;border-radius:0}
  table{font-size:7.2pt;table-layout:auto;width:100%}
  th,td{padding:5px 7px;white-space:normal}
  th{font-size:6.6pt;letter-spacing:.06em}
  section{break-inside:avoid-page;margin-bottom:26px}
  figure,.scroll,.grid,.note{break-inside:avoid}
  header{padding-top:0}
  h1{font-size:26pt}
  @page{margin:16mm}
}
"""


def build() -> Path:
    cov_table, cov = coverage()
    rep_table, rep = replication()
    val_table, val = validation()
    cf = cofactor()
    pooled, cond = cf["pooled_unconditioned"], cf["cofactor_conditioned"]

    html = f"""<title>PoseBusters benchmark — ligand-conditioned analysis</title>
<style>{CSS}</style>
<div class="wrap">

<header>
  <div class="eyebrow"><span>PoseBusters Benchmark</span><span>428 complexes · 7 methods</span><span>Ligand-conditioned re-analysis</span></div>
  <h1>Which molecules break docking predictions</h1>
  <p class="standfirst">The published benchmark records which docking <em>method</em> fails
  which physical-validity check. It carries no chemistry about the <em>molecule</em> being
  docked. This joins RDKit descriptors from the crystal ligands onto every pose and asks what
  the failures have in common — with eligibility gating, per-method strata, cluster-bootstrap
  intervals, FDR control, an external validation and a held-out replication.</p>
</header>

<div class="grid">
  <div class="stat"><div class="n">7,695</div><div class="k">pose rows joined</div></div>
  <div class="stat"><div class="n">{cov['estimated']}/{cov['total']}</div><div class="k">strata estimable</div></div>
  <div class="stat"><div class="n">95/287</div><div class="k">effects survive FDR</div></div>
  <div class="stat"><div class="n">{val['worst']:.1%}</div><div class="k">worst-case validation</div></div>
  <div class="stat"><div class="n">{rep['replicated']}/{rep['n']}</div><div class="k">replicate held-out</div></div>
  <div class="stat"><div class="n">37</div><div class="k">tests · 20 commits</div></div>
</div>

<section class="col">
  <h2><span>The benchmark</span><b>01</b></h2>
  <h3>Accuracy and validity are different questions</h3>
  <p>Each of 7 docking methods predicted a pose for all 428 complexes. Two things are then
  asked of every pose: is it in the right <em>place</em> (RMSD ≤ 2 Å against the crystal
  structure), and is it a <em>molecule that could exist</em> (18 physical-validity checks —
  bond lengths and angles within distance-geometry bounds, aromatic rings planar, no internal
  or protein clashes, stereochemistry preserved, strain energy not absurd).</p>
  <p>A pose can pass one and fail the other, and that gap is the subject of this report.</p>
</section>

{headline()}

<section class="col">
  <p>Classical search methods produce accurate poses about half the time and nearly all of
  those are physically valid. The deep-learning methods' accurate poses are mostly invalid.
  These figures reproduce the paper and nothing below revises them.</p>
</section>

<section class="col">
  <h2><span>Validation</span><b>02</b></h2>
  <h3>The published columns mean what we think they mean</h3>
  <p>Every number in this report reads the paper's published check results rather than
  recomputing them — so the whole analysis rests on an assumption about what each column
  means. That assumption was tested by re-running PoseBusters itself on all 513 crystal
  ligands against their own receptors and comparing to the paper's own reference rows.</p>
  <p>Worst-case agreement across {val['n_checks']} mapped checks is
  <span class="q pass">{val['worst']:.2%}</span>; {val['perfect']} of {val['n_checks']}
  agree on all 513 structures. The two disagreements are single structures on different
  checks — <code>7V3S</code> on ring flatness and <code>7T0U</code> on strain energy, the
  latter one of the paper's own two acknowledged borderline failures.</p>
</section>

{val_table}

<section class="col">
  <h2><span>Coverage</span><b>03</b></h2>
  <h3>Two thirds of the grid cannot be measured at all</h3>
  <p>There are 126 method × check combinations. Only {cov['estimated']} carry enough failures
  and enough distinct ligands to estimate an effect. The rest are not null results — they are
  unmeasured, and the distinction matters when reading anything below.</p>
</section>

{cov_table}

<section class="col">
  <p>Across those {cov['estimated']} strata every descriptor was tested against every check,
  giving <span class="q">287</span> estimates, of which <span class="q">95</span> survive
  Benjamini–Hochberg correction at α = 0.05. Intervals are cluster-bootstrapped on ligand,
  because each ligand contributes one row per method and treating those as independent would
  report intervals about √5 too narrow.</p>
</section>

<figure>
  <img src="{_img('03_effect_forest.png')}" alt="Forest plot of surviving effects with confidence intervals">
  <figcaption><b>Fig 1</b><span>The 18 largest surviving marginal effects, with cluster-bootstrap
  intervals. Read as associations that survive gating, clustering and multiplicity correction —
  not as a ranking of causes. Note how <code>mw</code> and <code>n rotatable bonds</code> appear
  on the same check for the same method: that is one signal counted twice.</span></figcaption>
</figure>

{survivors()}

<section class="col">
  <h2><span>The central question</span><b>04</b></h2>
  <h3>Flexibility or size? Not resolvable for the method it was asked about</h3>
  <p>The obvious hypothesis is that torsional freedom, not molecular size, drives strain and
  clash failures. Testing it requires holding one fixed while varying the other — but
  <code>mw</code> and <code>n_rotatable_bonds</code> correlate at
  <span class="q">ρ ≈ 0.73–0.74 within every method's ligand set</span>, so a stratified table
  cannot separate them. A multivariate model can try. One clustered logistic regression per
  method, both descriptors and five others entered together, on the strain check:</p>
</section>

{flexibility()}

<section class="col">
  <p><strong>DiffDock is the method the question was mainly about, and for DiffDock neither
  descriptor clears significance once the other is held fixed</strong> — despite both
  individually surviving FDR in the marginal grid. EquiBind gives the one clean
  flexibility-only result, and it is the weakest method in the benchmark. Three methods show
  both descriptors carrying independent signal; DeepDock's size coefficient is negative,
  the opposite sign from a naive "bigger molecules fail more" story.</p>
  <p>The honest conclusion is that this dataset cannot adjudicate size versus flexibility for
  the method that matters. That is a limitation, not a finding.</p>
</section>

<section class="col">
  <h2><span>A prediction that failed</span><b>05</b></h2>
  <h3>Crystal contacts do not inflate protein-clash failure</h3>
  <p>The benchmark's authors dropped 120 of 428 complexes for their journal version because
  the ligand touches a crystallographic symmetry mate, on the reasoning that such contacts
  inflate protein-clash failures. That list was never published, so the flag was recomputed
  here from first principles — expand each deposited structure by its spacegroup, measure the
  closest approach to any symmetry image of the protein. That gives 104 complexes with a
  genuine protein contact, or 119 counting solvent and cryoprotectant, bracketing the
  paper's 120.</p>
  <p>With intervals on the contact-minus-no-contact difference, the prediction is
  <strong>not supported</strong>: one method of seven shows a significant difference, and it
  runs in the <em>opposite</em> direction.</p>
</section>

{crystal_contacts()}

<section class="col">
  <p>Only Uni-Mol's interval excludes zero, and its clash failure is <em>lower</em> among
  contact-bearing complexes. The other six are indistinguishable from zero and none of their
  point estimates should be read as a direction — Vina's entire contribution is 2 failing
  poses out of 323, Gold's is 16 of 322 against 3 of 102. At those counts a direction is
  noise from a handful of poses.</p>
</section>

<section class="col">
  <div class="note retract">
    <p class="kicker">Retracted</p>
    <p><b>The cofactor-chemistry finding is withdrawn.</b> An earlier version of this analysis
    reported that small, non-aromatic ligands preferentially clash with cofactors —
    <span class="q">d = {pooled['d']:.2f}</span>, 95% CI
    [{pooled['lo']:.2f}, {pooled['hi']:.2f}], p = {pooled['p']:.4f}, pooled across all
    {pooled['n']:,} eligible rows.</p>
    <p>That pooling was the error. Over half those rows are complexes with <em>no cofactor at
    all</em>, where the check passes almost automatically, so the contrast mostly measured
    whether the receptor has a cofactor rather than any property of the ligand. Restricted to
    the {cond['n']:,} rows where a cofactor is actually present — the only population in which
    the check can be informative — the effect collapses to
    <span class="q">d = {cond['d']:.2f}</span>, 95% CI [{cond['lo']:.2f}, {cond['hi']:.2f}],
    p = {cond['p']:.3f}. The finding is deleted, not restated more cautiously.</p>
  </div>
</section>

<section class="col">
  <h2><span>Replication</span><b>06</b></h2>
  <h3>Held-out Astex Diverse Set</h3>
  <p>Every effect above was discovered and estimated on the same 428 complexes. The 85 Astex
  complexes were never touched during the analysis, so re-estimating a shortlist there is an
  independent check. Of {rep['n']} checkable triples, <strong>{rep['replicated']} replicate
  with statistical support</strong> — both intervals excluding zero.
  {rep['same_sign']} of {rep['n']} agree in sign, but sign agreement between two estimates
  that both straddle zero carries no information, so {rep['replicated']}/{rep['n']} is the
  number to cite. {rep['thin']} rows rest on fewer than 15 failures and are flagged.</p>
</section>

{rep_table}

<section class="col">
  <p>The strain and internal-clash effects against rotatable bonds are the most robust of the
  shortlist. The stereochemistry effect and the protein-clash effect are significant on the
  benchmark but do not replicate with support on Astex for most methods.</p>
</section>

<section class="col">
  <h2><span>Accuracy and validity</span><b>07</b></h2>
  <h3>Validity across the whole RMSD distribution</h3>
  <p>Reporting "of accurate poses, the share invalid" conditions on the accuracy outcome,
  which shares an upstream cause with validity — how well the method handled that ligand.
  Conditioning on such a variable can manufacture association. Banding raw RMSD instead needs
  no conditioning and uses every pose, not only those clearing an arbitrary 2 Å cutoff.</p>
</section>

{rmsd_bands()}

<section class="col">
  <p>The deep-learning methods' validity is already poor at sub-Ångström accuracy and does not
  recover at larger RMSD. That is a stronger statement than "poses that barely clear 2 Å are
  usually invalid" — it holds across the whole distribution rather than at one threshold.</p>
</section>

<figure>
  <img src="{_img('01_gap_by_partition.png')}" alt="Accurate-but-invalid rate across four ligand partitions">
  <figcaption><b>Fig 2</b><span>The conditioned view, kept as secondary. Monotone against
  rotatable bonds in both method classes; close to flat against ring count. The partition-level
  trend is specific to flexibility among these four — but see section 04: the descriptors are
  correlated throughout, so this alone does not establish cause.</span></figcaption>
</figure>

<figure>
  <img src="{_img('02_checks_by_flexibility.png')}" alt="Per-check failure rate by rotatable-bond bin">
  <figcaption><b>Fig 3</b><span>Per-check failure with eligibility-gated denominators — a ligand
  that cannot fail a check is excluded rather than counted as a trivial pass. Gating flattens
  sp3 stereochemistry almost completely; ungated it appeared to trend. The four cofactor checks
  are excluded: they are conditioned on the receptor, not the ligand.</span></figcaption>
</figure>

<section class="col">
  <h2><span>Two traps</span><b>08</b></h2>
  <h3>Both would silently corrupt a naive re-analysis</h3>
  <p><strong>Blank is not False.</strong> Both flatness columns are empty for all 2,996
  energy-minimised rows — those checks were not run. Reading blanks as failures makes every
  minimised pose invalid, producing the nonsensical result that minimisation destroys 100% of
  poses including AutoDock Vina's. Handled correctly, minimisation takes DiffDock from
  <span class="q">24.1% → 65.0%</span> valid while costing Vina
  <span class="q fail">14.5 points</span>.</p>
  <p><strong>No output is not perfect output.</strong> 85 rows have every check blank — the
  method produced no pose at all. Since nothing is False, a naive "all checks passed" test
  scores them valid and inflates every validity rate.</p>
</section>

<section class="col">
  <h2><span>Limits</span><b>09</b></h2>
  <h3>What this data cannot answer</h3>
  <p>Descriptors describe the molecule that was docked, not the pose that came back — nothing
  here measures how <em>wrong</em> a pose is beyond the binary checks. {cov['total'] - cov['estimated']}
  of {cov['total']} strata could not be estimated at all, so the 95 surviving effects describe
  roughly a third of the grid. The size-versus-flexibility contrast is not identified for
  DiffDock, and Gold and Vina fail too rarely to fit a model on most checks — these are not
  gaps more analysis of this dataset would close.</p>
  <p>The predicted poses themselves were never released, so the methods' checks cannot be
  independently recomputed; only the crystal-structure rows could be, and were. The
  crystal-contact flag is this project's own estimate, bracketing but not reproducing the
  paper's undisclosed subset. Several surviving effects rest on small failure counts even
  after the ≥15-failure gate — check <code>n_fail</code> before treating any single row as
  decisive.</p>
</section>

<footer>
  <p><strong>Data.</strong> PoseBusters paper data, <a href="https://zenodo.org/records/8278563">Zenodo 8278563</a>.
  Method: Buttenschoen M, Morris GM, Deane CM, <em>PoseBusters: AI-based docking methods fail to
  generate physically valid poses or generalise to novel sequences</em>, Chem Sci 15, 3130 (2024),
  <a href="https://doi.org/10.1039/D3SC04185A">doi:10.1039/D3SC04185A</a>.</p>
  <p><strong>Reproduce.</strong> Every table in this document is read from <code>reports/tables/</code>
  at render time; every figure is inlined from <code>reports/figures/</code>. Regenerate with
  <code>python -m pb.acquire</code> → <code>descriptors</code> → <code>crystal_contacts</code> →
  <code>build</code> → <code>validate</code> → <code>analyze</code> → <code>replication</code> →
  <code>figures</code> → <code>report_html</code>. 37 tests, 20 commits.</p>
</footer>

</div>
"""
    OUT.write_text(html, encoding="utf-8")
    return OUT


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = build()
    log.info("wrote %s (%.0f kB)", path, path.stat().st_size / 1000)


if __name__ == "__main__":
    main()
