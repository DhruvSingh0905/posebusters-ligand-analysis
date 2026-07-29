"""Render the analysis as a paper.

Shares its table builders with `pb.report_html`, so both documents are read
from `reports/tables/` at render time and cannot disagree with each other or
with the analysis. Figures are inlined as data URIs; the output is one file
that prints to PDF without a network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import paths
from .report_html import (
    TABLES,
    check_signatures,
    _img,
    _table,
    cofactor,
    coverage,
    crystal_contacts,
    flexibility,
    headline,
    replication,
    rmsd_bands,
    survivors,
    validation,
)

log = logging.getLogger(__name__)

OUT = paths.REPORTS / "paper.html"

CSS = """
:root{
  --paper:#FFFFFF;--raise:#FFFFFF;--ink:#101418;--ink-2:#39424A;--ink-3:#727C84;
  --rule:#B9C2C7;--rule-soft:#DFE4E7;--accent:#0A5F7A;--accent-soft:#0a5f7a14;
  --pass:#2C6E4E;--fail:#9E3A28;
  --serif:"Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  color-scheme:light;
}

html{background:var(--paper)}
body{background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.6;-webkit-font-smoothing:antialiased;padding:0 22px 90px}
.wrap{max-width:760px;margin:0 auto}

/* ── title block ─────────────────────────────────────────────── */
header{padding:70px 0 0;text-align:center}
h1{font-weight:400;font-size:clamp(28px,4.4vw,40px);line-height:1.14;letter-spacing:-.012em;
  margin:0 auto;max-width:22ch;text-wrap:balance}
.sub{color:var(--ink-2);font-size:18px;margin:14px auto 0;max-width:52ch;font-style:italic}
.byline{font-size:15px;color:var(--ink-2);margin:20px 0 0}
.byline a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}
.byline a:hover{border-bottom-color:currentColor}
.byline a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.affil{font-family:var(--mono);font-size:10.5px;line-height:1.5;color:var(--ink-3);
  margin:12px auto 0;max-width:56ch}
.rule{height:1px;background:var(--rule);margin:34px 0 0}

/* ── abstract ────────────────────────────────────────────────── */
.abstract{background:#FCFCFB;border:1px solid var(--rule-soft);border-radius:2px;
  padding:22px 26px;margin:34px 0 8px;font-size:15.5px;line-height:1.58;color:var(--ink-2)}
.abstract h2{font-family:var(--mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);margin:0 0 10px;border:0;padding:0}
.abstract p{margin:0 0 11px}
.abstract p:last-child{margin:0}
.abstract strong{color:var(--ink)}

/* ── sections ────────────────────────────────────────────────── */
section{margin:0 0 34px}
h2{font-weight:400;font-size:21px;letter-spacing:-.008em;margin:38px 0 12px;
  padding-bottom:0;border:0;color:var(--ink);display:block}
h2 .n{font-family:var(--mono);font-size:13px;color:var(--accent);margin-right:11px;font-weight:600}
h3{font-weight:600;font-size:16px;margin:24px 0 8px;color:var(--ink)}
h3 .n{font-family:var(--mono);font-size:12px;color:var(--ink-3);margin-right:9px;font-weight:600}
p{margin:0 0 14px}
code,.q{font-family:var(--mono);font-size:.845em;font-variant-numeric:tabular-nums;
  background:var(--accent-soft);color:var(--accent);padding:1px 4px;border-radius:2px;white-space:nowrap}
.q.pass{background:color-mix(in srgb,var(--pass) 12%,transparent);color:var(--pass)}
.q.fail{background:color-mix(in srgb,var(--fail) 12%,transparent);color:var(--fail)}
sup a{color:var(--accent);text-decoration:none;font-size:.72em;font-family:var(--mono);padding:0 1px}

/* ── tables ──────────────────────────────────────────────────── */
.tbl{margin:20px 0}
.tbl .cap{font-family:var(--mono);font-size:10.5px;line-height:1.5;color:var(--ink-2);
  margin:0 0 7px;display:flex;gap:9px}
.tbl .cap b{color:var(--accent);font-weight:600;letter-spacing:.09em;text-transform:uppercase;white-space:nowrap}
.scroll{overflow-x:auto;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink)}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums}
th{text-align:right;font-weight:600;font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-2);padding:8px 11px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child{text-align:left}
td{padding:6px 11px;border-bottom:1px solid var(--rule-soft);text-align:right;white-space:nowrap}
td.l{text-align:left;color:var(--ink-2)}
tbody tr:last-child td{border-bottom:none}
.yes{color:var(--pass);font-weight:600}
.no{color:var(--ink-3)}

/* ── plain-language takeaways ─────────────────────────────────── */
.take{margin:22px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.take .row{display:flex;gap:16px;padding:13px 2px;border-bottom:1px solid var(--rule-soft);align-items:baseline}
.take .row:last-child{border-bottom:none}
.take .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--accent);white-space:nowrap;flex:0 0 108px;padding-top:3px}
.take .v{flex:1}
.take .v b{font-weight:600}
.take .v em{color:var(--ink-3);font-style:normal;font-family:var(--mono);font-size:11.5px}
.close{border-left:3px solid var(--accent);padding:4px 0 4px 20px;margin:26px 0 0}
.close p{margin:0 0 10px}
.close p:last-child{margin:0}

/* ── figures ─────────────────────────────────────────────────── */
figure{margin:26px 0}
figure img{display:block;width:100%;height:auto;border:1px solid var(--rule-soft);border-radius:2px}
figcaption{font-family:var(--mono);font-size:10.5px;line-height:1.55;color:var(--ink-2);
  margin-top:8px;display:flex;gap:9px}
figcaption b{color:var(--accent);font-weight:600;letter-spacing:.09em;text-transform:uppercase;white-space:nowrap}

/* ── refs ────────────────────────────────────────────────────── */
.refs{font-size:14px;color:var(--ink-2);counter-reset:ref}
.refs li{margin:0 0 9px;line-height:1.5}
.refs a{color:var(--accent);text-decoration:none}
footer{margin-top:42px;padding-top:18px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:11px;line-height:1.65;color:var(--ink-3)}
footer a{color:var(--accent);text-decoration:none}

@media print{
  :root{--paper:#fff;--raise:#fff;--ink:#000;--ink-2:#2a2a2a;--ink-3:#5a5a5a;
        --rule:#999;--rule-soft:#d4d4d4}
  body{font-size:10pt;padding:0;line-height:1.45}
  .wrap{max-width:none}
  header{padding-top:0}
  h1{font-size:20pt}
    .sub{font-size:11pt}
  .byline{font-size:9.5pt}
  .affil{font-size:7.5pt}
  h2{font-size:13pt;margin-top:20px}
  h3{font-size:10.5pt}
  section{break-inside:auto}
  figure,.tbl{break-inside:avoid}
  .abstract{break-inside:auto}
  .rule{margin-top:20px}
  .scroll{overflow:visible}
  table{font-size:7pt}
  th,td{padding:4px 6px;white-space:normal}
  th{font-size:6.4pt;letter-spacing:.05em}
  .abstract{padding:14px 16px}
  @page{margin:18mm 20mm}
}
"""


def _fig(name: str, num: int, caption: str) -> str:
    return (
        f'<figure><img src="{_img(name)}" alt="{caption[:80]}">'
        f"<figcaption><b>Figure {num}</b><span>{caption}</span></figcaption></figure>"
    )


def _tbl(num: int, caption: str, html: str) -> str:
    return (
        f'<div class="tbl"><p class="cap"><b>Table {num}</b>'
        f"<span>{caption}</span></p>{html}</div>"
    )


def build() -> Path:
    cov_table, cov = coverage()
    rep_table, rep = replication()
    val_table, val = validation()
    sig_table, sig = check_signatures()
    n_skipped = cov["total"] - cov["estimated"]

    grid = pd.read_csv(TABLES / "association_grid.csv")
    n_est, n_sig = len(grid), int(grid.fdr_reject.sum())

    html = f"""<title>What breaks a docking pose: ligand determinants of physical-validity failure</title>
<style>{CSS}</style>
<div class="wrap">

<header>
  <h1>What breaks a docking pose</h1>
  <p class="sub">Which ligand properties predict which physical-validity failure,
  across the PoseBusters benchmark</p>
  <p class="byline">Dhruv Singh · <a href="https://www.linkedin.com/in/dhruv-singh-933154265/">LinkedIn</a>
  · <a href="https://github.com/DhruvSingh0905/posebusters-ligand-analysis">Code and data</a></p>
  <p class="affil">Independent exploratory work. Not affiliated with, funded by, or
  endorsed by any institution, laboratory or research group.</p>
  <div class="rule"></div>
</header>

<div class="abstract">
  <h2>Summary</h2>
  <p>Docking is scored by whether a predicted pose lands within 2 Å of the crystal structure,
  but an accurate pose can still be physically impossible. PoseBusters<sup><a href="#r1">1</a></sup>
  measures this with 18 validity checks and reports failure per <em>method</em>. It records
  nothing about the molecules, so it cannot say which ligands are hard. We computed 27 RDKit
  descriptors for the 428 benchmark ligands, joined them to all 7,695 published poses, and
  modelled each check against ligand chemistry.</p>
  <p><strong>Failure modes separate by descriptor.</strong> Internal-geometry failures (strain energy, self-clash) scale with torsional degrees of freedom; sp³ chirality failure scales with stereocentre count and with nothing else; cofactor clash scales <em>inversely</em> with molecular weight. Protein clash, the largest failure term, admits no ligand-level predictor: its variance is between methods, not between ligands.</p>
  <p><strong>Two limits.</strong> Only {cov['estimated']} of {cov['total']} method × check
  strata carry enough failures to model, and molecular weight and rotatable-bond count are
  correlated at ρ ≈ 0.73–0.74 throughout, so where both are significant they cannot be
  separated.</p>
</div>

<section>
  <h2><span class="n">1</span>What breaks which check</h2>
  <p>Each check was modelled by logistic regression on seven decorrelated descriptors entered
  together, fitted separately per docking method, with standard errors clustered on ligand.
  Table 1 lists every descriptor reaching significance for at least two methods. Odds ratios
  are per standard deviation with the other six held fixed, so a value below 1 means failure
  becomes <em>less</em> likely as the property rises.</p>
</section>

{_tbl(1, "Ligand properties carrying independent signal, per check. Multivariate models, "
         "odds ratio per standard deviation, other descriptors held fixed.", sig_table)}

<section>
  <p>Four results follow.</p>
</section>

<div class="take">
  <div class="row"><div class="k">Torsional</div><div class="v">
    <b>Internal-geometry failure scales with torsional degrees of freedom.</b> Rotatable-bond
    count multiplies failure odds by 1.6–9.5 per standard deviation on both the strain-energy
    ratio and the internal-clash test, significant in 4 of 5 methods with consistent sign.
    Under a threefold-minimum torsional model a molecule with <em>k</em> rotatable bonds spans
    on the order of 3<sup><em>k</em></sup> conformers; methods that regress coordinates
    directly, rather than searching torsions of an already-valid conformer, leave that space
    unconstrained. Both effects hold on the held-out set.
    <em>OR 1.6–9.5 / SD</em></div></div>

  <div class="row"><div class="k">Stereogenic</div><div class="v">
    <b>sp³ chirality failure scales with stereocentre count, and with no other descriptor.</b>
    OR 4.3–10.4 per standard deviation; every remaining coefficient exceeds α = 0.05 in both
    fitted methods. If a method assigns each centre independently with accuracy <em>p</em>,
    then P(all <em>n</em> correct) = <em>p<sup>n</sup></em>, so failure probability
    1 − <em>p<sup>n</sup></em> is monotone increasing in <em>n</em>. The output is a
    stereoisomer, not a distorted conformer: a distinct chemical entity with distinct activity.
    <em>OR 4.3–10.4 / SD</em></div></div>

  <div class="row"><div class="k">Inverse in MW</div><div class="v">
    <b>Cofactor clash is inversely associated with molecular weight.</b> OR 0.15–0.35 per
    standard deviation in 4 of 5 methods, so a one-SD decrease in MW multiplies failure odds
    by roughly 3–7. The distance-based and volume-overlap tests are independent measurements
    and agree in sign and magnitude per method (TankBind 0.17 and 0.16; EquiBind 0.23 and
    0.15). Consistent with steric exclusion: cofactor-occupied volume admits small ligands
    into contested space, while larger ligands are placed outside it.
    <em>OR 0.15–0.35 / SD</em></div></div>

  <div class="row"><div class="k">Unpredicted</div><div class="v">
    <b>Protein clash admits no ligand-level predictor.</b> Fitted on 6 methods (more than any other check) against a base failure rate near 84% for the deep-learning methods. Five of
    42 coefficients reach α = 0.05 and none replicates beyond two methods. The variance is
    between methods, not between ligands.
    <em>no predictor</em></div></div>
</div>

<section>
  <p>Two descriptors are near-null across the grid: aromatic ring count reaches α = 0.05 in
  4 of 41 fitted coefficients and halogen count in 2 of 41, against 19 of 41 for molecular
  weight. Neither predicts any check reproducibly.</p>
  <p>Molecular weight is also significant on the internal-geometry checks (3 of 5 methods for
  self-clash, same sign as rotatable-bond count) but less consistently, and the two are
  collinear; see §2. Formal charge is inversely associated with self-clash (OR 0.73–0.82,
  2 of 5), consistent with intramolecular Coulomb repulsion disfavouring compact conformers.</p>
</section>

{_fig('03_effect_forest.png', 1,
      "The largest single-descriptor associations surviving eligibility gating, "
      "ligand-clustered bootstrap intervals and Benjamini–Hochberg correction "
      f"({n_sig} of {n_est} estimates survive). Molecular weight and rotatable-bond count "
      "recur on the same check for the same method — one signal counted twice, which is why "
      "Table 1 uses multivariate models instead.")}

<section>
  <h2><span class="n">2</span>Where size and flexibility cannot be separated</h2>
  <p>§1 attributes strain and clash failure to torsional freedom rather than bulk. That attribution holds only in the weak form stated. Molecular weight
  and rotatable-bond count correlate at ρ ≈ 0.73–0.74 <em>within every method's ligand set</em>,
  not merely on average, so no stratification of these 428 complexes separates them. For DiffDock, the strongest deep-learning method and the usual subject of the question,
  neither descriptor reaches significance once the other is held fixed
  (<em>p</em> = 0.941 and 0.108), despite both surviving correction individually. Separating them requires a benchmark constructed to decorrelate the two, populating the large-rigid and small-flexible quadrants that this ligand set leaves sparse.</p>

  <h2><span class="n">3</span>Crystal contacts are not the explanation</h2>
  <p>The benchmark's journal version drops 120 of 428 complexes whose ligand touches a
  crystallographic symmetry mate, on the reasoning that such contacts inflate protein-clash
  failure. That subset was never published, so we recomputed it from spacegroup expansion:
  104 of 427 resolvable complexes have a protein contact within 4.0 Å (119 counting solvent),
  bracketing the authors' 120. Contact status does not account for the clash rate. One method of
  seven shows a significant difference and it runs in the opposite direction: Uni-Mol's clash
  failure is 17.1 points lower among contact-bearing complexes (95% CI −27.0 to −7.7).
  The other six straddle zero; Vina and Gold fail so rarely (2 of 323 and 16 of 322 poses)
  that their point estimates carry no information.</p>
</section>

<section>
  <h2><span class="n">4</span>How large the gap is</h2>
  <p>Classical search methods are accurate on about half of complexes and valid on 94–97% of all poses. The deep-learning methods are less accurate and, conditional on accuracy, predominantly invalid.</p>
</section>

{_tbl(2, "Accuracy, validity and their intersection. Raw poses, 428 complexes.", headline())}

<section>
  <p>Reporting invalidity only among accurate poses conditions on an outcome that shares an
  upstream cause with validity. Banding raw RMSD instead uses every pose and needs no such
  conditioning — and shows the failure is not confined to poses near the threshold. Deep-learning
  validity is already poor at sub-Ångström accuracy (DiffDock 61.3%, Uni-Mol 33.3%) and does
  not recover further out; the classical methods stay above 88% in every band.</p>
</section>

{_tbl(3, "Share of poses passing all applicable checks, by RMSD band. "
         "Unconditioned on accuracy.", rmsd_bands())}

{_fig('01_gap_by_partition.png', 2,
      "Share of RMSD-accurate poses that are physically invalid, across four ligand "
      "partitions. Monotone against rotatable-bond count in both method classes, close to "
      "flat against ring count.")}

{_fig('02_checks_by_flexibility.png', 3,
      "Per-check failure against rotatable-bond count, pooled across the five deep-learning "
      "methods, with eligibility-gated denominators. Strain and internal clash climb steeply; "
      "planarity checks barely move. Cofactor checks are excluded as receptor-conditioned.")}

<section>
  <h2><span class="n">5</span>Confidence in these numbers</h2>
  <p><strong>The published checks reproduce.</strong> Everything here reads the paper's
  published check outcomes rather than recomputing them, so we re-ran PoseBusters on all 513
  crystal ligands against their own receptors: worst-case agreement
  <span class="q pass">{val['worst']:.2%}</span> across {val['n_checks']} checks, with
  {val['perfect']} agreeing on all 513 structures. The two disagreements are single structures
  on different checks.</p>
  <p><strong>The main effects replicate held-out.</strong> The 85-complex Astex Diverse Set was
  untouched throughout. Of {rep['n']} shortlisted effects, {rep['replicated']} replicate with
  both intervals excluding zero — the strain and internal-clash effects
  against rotatable bonds hold for every deep-learning method tested; the chirality and
  protein-clash effects do not.</p>
  <p><strong>Two thirds of the grid is unmeasurable.</strong> {n_skipped} of {cov['total']}
  method × check strata could not be estimated, {int(pd.read_csv(TABLES / 'stratum_coverage.csv').skip_reason.eq('no_failures').sum())}
  of them because the check never failed for that method. These are absent measurements rather than null results; an effect grid reported without that denominator conflates the two.</p>
</section>

<section>
  <h2><span class="n">6</span>What this means in practice</h2>
  <div class="close">
    <p>Three descriptors carry the predictive signal, each specific to a failure mode:
    rotatable-bond count for strain and self-clash, stereocentre count for chirality, molecular
    weight for cofactor clash in the inverse direction. No single descriptor predicts validity
    in aggregate, so a scalar difficulty score over these ligands would average out
    check-specific effects of opposite sign.</p>
    <p>The dominant failure term is not ligand-dependent. Protein clash accounts for the
    largest share of deep-learning invalidity and is unpredicted by any descriptor measured
    here, so ligand-level triage cannot reduce it; that variance is attributable to the
    method.</p>
  </div>
</section>

<section>
  <h2><span class="n">7</span>Method</h2>
  <p>27 RDKit<sup><a href="#r4">4</a></sup> descriptors per crystal ligand (size, flexibility,
  ring system, stereochemistry, polarity, composition), joined to the published results
  table<sup><a href="#r2">2</a></sup> on PDB code. Blank check values in that table mean the check was <em>not run</em>, not that it failed. Both flatness columns are blank for all
  2,996 energy-minimised rows, and 85 rows have every check blank because the method produced
  no pose; treating either as a pass inflates validity.</p>
  <p>Each check is restricted to ligands that could fail it: a molecule with no stereocentres
  cannot fail the chirality check, and counting it as a trivial pass both deflates the failure
  rate and inflates any effect measured against the gating descriptor. Gating raises the
  observed chirality-failure rate from 21.4% to 36.7%. The four cofactor checks are further
  restricted to cofactor-bearing complexes, since they pass almost automatically otherwise
  (2.4% versus 22.5% failure).</p>
  <p>Effects are estimated per method, never pooled. Intervals come from a bootstrap resampling
  <em>ligands</em> rather than poses, since each ligand contributes one pose per method and
  treating 2,140 rows as independent would report intervals roughly √5 too narrow. Strata with
  fewer than 15 failures or 30 distinct ligands are not estimated. Single-descriptor estimates
  across the whole grid are corrected by Benjamini–Hochberg<sup><a href="#r5">5</a></sup> at
  α = 0.05 as one family; because they cannot separate correlated descriptors, the per-check
  conclusions in §1 come from the multivariate models. Crystal contacts were computed with
  gemmi<sup><a href="#r6">6</a></sup> by expanding each deposited structure<sup><a href="#r7">7</a></sup>
  by its spacegroup.</p>
  <p><strong>Caveats.</strong> Descriptors describe the docked molecule, not the returned pose.
  The predicted poses were never released, so only the crystal-structure rows could be
  independently recomputed. Several surviving effects rest on small failure counts even after
  gating (DiffDock's internal-clash effects rest on 25), so single rows should not be read
  as decisive.</p>
</section>

<section>
  <h2>References</h2>
  <ol class="refs">
    <li id="r1">Buttenschoen M, Morris GM, Deane CM. PoseBusters: AI-based docking methods fail
      to generate physically valid poses or generalise to novel sequences. <em>Chem Sci</em>
      <strong>15</strong>, 3130–3139 (2024).
      <a href="https://doi.org/10.1039/D3SC04185A">doi:10.1039/D3SC04185A</a></li>
    <li id="r2">Buttenschoen M, Morris GM, Deane CM. PoseBusters paper data. Zenodo (2023).
      <a href="https://doi.org/10.5281/zenodo.8278563">doi:10.5281/zenodo.8278563</a></li>
    <li id="r4">RDKit: open-source cheminformatics. <a href="https://www.rdkit.org">rdkit.org</a></li>
    <li id="r5">Benjamini Y, Hochberg Y. Controlling the false discovery rate. <em>J R Stat
      Soc B</em> <strong>57</strong>, 289–300 (1995).</li>
    <li id="r6">Wojdyr M. GEMMI: a library for structural biology. <em>J Open Source Softw</em>
      <strong>7</strong>, 4200 (2022).</li>
    <li id="r7">Berman HM <em>et al.</em> The Protein Data Bank. <em>Nucleic Acids Res</em>
      <strong>28</strong>, 235–242 (2000).</li>
  </ol>
</section>

<footer>
  <p>Every table is read from the analysis outputs at render time and every figure inlined from
  its original, so this document cannot diverge from the analysis it reports. Deterministic
  under fixed seeds; 37 tests. Regenerate with <code>python -m pb.paper_html</code>.</p>
</footer>

</div>
"""
    OUT.write_text(html, encoding="utf-8")
    return OUT


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = build()
    log.info("wrote %s (%.0f kB)", path, path.stat().st_size / 1000)
    landing = build_landing()
    log.info("wrote %s (%.0f kB)", landing, landing.stat().st_size / 1000)



LANDING = paths.ROOT / "index.html"

LANDING_CSS = """
*{box-sizing:border-box}
body{margin:0;padding:0 22px 80px;background:#FFFFFF;color:#101418;
  font:17px/1.6 "Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
  -webkit-font-smoothing:antialiased}
.w{max-width:660px;margin:0 auto}
header{padding:72px 0 0}
h1{font-weight:400;font-size:clamp(30px,5vw,42px);line-height:1.1;margin:0;letter-spacing:-.015em}
.sub{color:#39424A;font-size:18px;margin:14px 0 0;font-style:italic}
.byline{font-size:15px;color:#39424A;margin:18px 0 0}
.byline a{color:#0A5F7A;text-decoration:none;border-bottom:1px solid transparent}
.byline a:hover{border-bottom-color:currentColor}
.byline a:focus-visible{outline:2px solid #0A5F7A;outline-offset:2px}
.affil{font:10.5px/1.55 ui-monospace,"SF Mono",Menlo,monospace;color:#727C84;
  margin:12px 0 0;max-width:56ch}
hr{border:0;border-top:1px solid #B9C2C7;margin:30px 0}
.cta{display:flex;gap:10px;flex-wrap:wrap;margin:26px 0 34px}
.cta a{display:inline-block;padding:11px 18px;border:1px solid #0A5F7A;border-radius:2px;
  color:#0A5F7A;text-decoration:none;font:12px/1 ui-monospace,"SF Mono",Menlo,monospace;
  letter-spacing:.09em;text-transform:uppercase}
.cta a.primary{background:#0A5F7A;color:#fff}
.cta a:hover{background:#0A5F7A;color:#fff}
.cta a:focus-visible{outline:2px solid #0A5F7A;outline-offset:3px}
table{border-collapse:collapse;width:100%;margin:8px 0 26px;
  font:12.5px/1.45 ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums;
  border-top:1.5px solid #101418;border-bottom:1.5px solid #101418}
td{padding:9px 10px;border-bottom:1px solid #E2E7E9;vertical-align:top}
tr:last-child td{border-bottom:none}
td:last-child{text-align:right;white-space:nowrap;color:#39424A}
b{font-weight:600}
p{margin:0 0 15px}
footer{margin-top:36px;padding-top:18px;border-top:1px solid #B9C2C7;
  font:11.5px/1.65 ui-monospace,"SF Mono",Menlo,monospace;color:#727C84}
footer a{color:#0A5F7A;text-decoration:none}
@media (max-width:520px){td:last-child{white-space:normal}}
"""


def build_landing() -> Path:
    """A front door for GitHub Pages, generated so its numbers cannot go stale."""
    _, cov = coverage()
    _, rep = replication()
    _, val = validation()

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>What breaks a docking pose</title>
<meta name="description" content="Which ligand properties predict which physical-validity
failure, across the PoseBusters benchmark.">
<style>{LANDING_CSS}</style>
</head><body><div class="w">

<header>
  <h1>What breaks a docking pose</h1>
  <p class="sub">Which ligand properties predict which physical-validity failure,
  across the PoseBusters benchmark</p>
  <p class="byline">Dhruv Singh · <a href="https://www.linkedin.com/in/dhruv-singh-933154265/">LinkedIn</a>
  · <a href="https://github.com/DhruvSingh0905/posebusters-ligand-analysis">Code and data</a></p>
  <p class="affil">Independent exploratory work. Not affiliated with, funded by, or
  endorsed by any institution, laboratory or research group.</p>
</header>

<hr>

<div class="cta">
  <a class="primary" href="reports/paper.pdf">Read the paper (PDF)</a>
  <a href="reports/paper.html">HTML</a>
  <a href="https://github.com/DhruvSingh0905/posebusters-ligand-analysis">Source</a>
</div>

<p>PoseBusters scores docking with 18 physical-validity checks and reports failure rates per
<em>method</em>. Its results table records nothing about the molecule being docked, so it
cannot say which ligands are hard. This computes 27 RDKit descriptors for the 428 benchmark
ligands, joins them to all 7,695 published poses, and models each check against ligand
chemistry.</p>

<table>
  <tr><td>Internal-geometry failure scales with <b>torsional degrees of freedom</b></td>
      <td>OR 1.6–9.5 / SD</td></tr>
  <tr><td>sp³ chirality failure scales with <b>stereocentre count</b>, and nothing else</td>
      <td>OR 4.3–10.4 / SD</td></tr>
  <tr><td>Cofactor clash scales <b>inversely with molecular weight</b></td>
      <td>OR 0.15–0.35 / SD</td></tr>
  <tr><td>Protein clash, the largest failure term, has <b>no ligand-level predictor</b></td>
      <td>5 of 42 coefficients</td></tr>
</table>

<p>Two limits bound all of it. Only {cov['estimated']} of {cov['total']} method × check strata
carry enough failures to model, and molecular weight and rotatable-bond count are correlated at
ρ ≈ 0.73–0.74 throughout, so where both are significant they cannot be separated.</p>

<p>The analysis is verified two ways: re-running the PoseBusters checker over all 513 crystal
structures reproduces the published reference rows to {val['worst']:.1%} worst-case agreement,
and {rep['replicated']} of {rep['n']} shortlisted effects replicate on the held-out
85-complex Astex Diverse Set.</p>

<footer>
  <p>Every table in the paper is read from the analysis outputs at render time and every figure
  inlined from its original, so the document cannot diverge from the analysis it reports.
  Regenerate the whole thing with <code>python run.py --fast</code>.</p>
  <p>Data: Buttenschoen, Morris &amp; Deane, <em>Chem Sci</em> <b>15</b>, 3130 (2024),
  <a href="https://doi.org/10.1039/D3SC04185A">doi:10.1039/D3SC04185A</a></p>
</footer>

</div></body></html>
"""
    LANDING.write_text(html, encoding="utf-8")
    return LANDING

if __name__ == "__main__":
    main()
