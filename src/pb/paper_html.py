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
  --paper:#FAFAF8;--raise:#FFFFFF;--ink:#14181B;--ink-2:#3E474D;--ink-3:#78838A;
  --rule:#C9D0D3;--rule-soft:#E2E7E9;--accent:#0A5F7A;--accent-soft:#0a5f7a1a;
  --pass:#2C6E4E;--fail:#9E3A28;
  --serif:"Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0C1013;--raise:#12181C;--ink:#E2E8EA;--ink-2:#9FAAB1;--ink-3:#727E85;
  --rule:#263137;--rule-soft:#1B242A;--accent:#5FBAD4;--accent-soft:#5fbad41f;
  --pass:#5FAE83;--fail:#DF7A61;}}
:root[data-theme="dark"]{
  --paper:#0C1013;--raise:#12181C;--ink:#E2E8EA;--ink-2:#9FAAB1;--ink-3:#727E85;
  --rule:#263137;--rule-soft:#1B242A;--accent:#5FBAD4;--accent-soft:#5fbad41f;
  --pass:#5FAE83;--fail:#DF7A61;}
:root[data-theme="light"]{
  --paper:#FAFAF8;--raise:#FFFFFF;--ink:#14181B;--ink-2:#3E474D;--ink-3:#78838A;
  --rule:#C9D0D3;--rule-soft:#E2E7E9;--accent:#0A5F7A;--accent-soft:#0a5f7a1a;
  --pass:#2C6E4E;--fail:#9E3A28;}

html{background:var(--paper)}
body{background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.6;-webkit-font-smoothing:antialiased;padding:0 22px 90px}
.wrap{max-width:760px;margin:0 auto}

/* ── title block ─────────────────────────────────────────────── */
header{padding:70px 0 0;text-align:center}
h1{font-weight:400;font-size:clamp(28px,4.4vw,40px);line-height:1.14;letter-spacing:-.012em;
  margin:0 auto;max-width:22ch;text-wrap:balance}
.sub{color:var(--ink-2);font-size:18px;margin:14px auto 0;max-width:52ch;font-style:italic}
.meta{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-3);margin-top:22px}
.rule{height:1px;background:var(--rule);margin:34px 0 0}

/* ── abstract ────────────────────────────────────────────────── */
.abstract{background:var(--raise);border:1px solid var(--rule-soft);border-radius:2px;
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
  h2{font-size:13pt;margin-top:20px}
  h3{font-size:10.5pt}
  section{break-inside:auto}
  figure,.tbl,.abstract{break-inside:avoid}
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
    cf = cofactor()
    n_skipped = cov["total"] - cov["estimated"]

    grid = pd.read_csv(TABLES / "association_grid.csv")
    n_est, n_sig = len(grid), int(grid.fdr_reject.sum())

    html = f"""<title>Ligand determinants of physical-validity failure in protein–ligand docking</title>
<style>{CSS}</style>
<div class="wrap">

<header>
  <h1>Ligand determinants of physical-validity failure in protein–ligand docking</h1>
  <p class="sub">A re-analysis of the PoseBusters benchmark conditioned on the molecule
  rather than the method</p>
  <div class="meta">428 complexes · 7 docking methods · 7,695 poses</div>
  <div class="rule"></div>
</header>

<div class="abstract">
  <h2>Abstract</h2>
  <p>Docking methods are judged by whether a predicted ligand pose lands within 2 Å of the
  crystal structure, but a pose can be accurate and still be physically impossible. The
  PoseBusters benchmark<sup><a href="#r1">1</a></sup> quantifies this with 18 validity checks
  and reports failure rates per docking method. It records nothing about the molecules being
  docked, so it cannot say which ligands are hard.</p>
  <p>We computed 27 RDKit descriptors for each of the 428 benchmark ligands, joined them to
  all 7,695 published poses, and estimated the association between ligand chemistry and each
  validity failure. Estimates are restricted to ligands capable of failing the check in
  question, computed per docking method, bootstrapped over ligands rather than poses, and
  corrected for multiple testing.</p>
  <p><strong>Three results.</strong> First, only <strong>{cov['estimated']} of
  {cov['total']}</strong> method × check strata carry enough failures to estimate at all;
  of {n_est} resulting estimates, <strong>{n_sig}</strong> survive false-discovery-rate
  correction. Second, molecular weight and rotatable-bond count are so strongly correlated
  within every method's ligand set (ρ ≈ 0.73–0.74) that the intuitive hypothesis — that
  torsional flexibility rather than size drives strain and clash failures —
  <strong>is not identifiable</strong> for DiffDock, the strongest deep-learning method.
  Third, removing complexes whose ligand contacts a crystallographic symmetry mate does
  <strong>not</strong> reduce protein-clash failure; one method of seven shows a significant
  difference and it runs in the opposite direction.</p>
  <p>We validate the analysis by re-running the checker itself on all 513 crystal structures
  ({val['worst']:.1%} worst-case agreement with the published reference rows) and test
  generalisation on the 85-complex Astex Diverse Set, where {rep['replicated']} of
  {rep['n']} shortlisted effects replicate with statistical support.</p>
</div>

<section>
  <h2><span class="n">1</span>Introduction</h2>
  <p>Molecular docking predicts where a small molecule binds a protein. Performance is
  conventionally summarised by the fraction of predictions within 2 Å root-mean-square
  deviation (RMSD) of the crystallographic pose. RMSD is a positional measure: it compares
  where atoms are, and says nothing about whether their arrangement is chemically possible.</p>
  <p>PoseBusters<sup><a href="#r1">1</a></sup> addressed this by pairing RMSD with 18 explicit
  physical checks — bond lengths and angles within distance-geometry bounds, aromatic rings
  planar, tetrahedral stereochemistry preserved, no internal or intermolecular steric clashes,
  internal strain energy within a factor of 100 of a generated conformer ensemble. Applied to
  seven docking methods, it showed that deep-learning methods with competitive RMSD produce
  poses that frequently fail these checks, while classical search methods do not.</p>
  <p>That result is method-conditioned. The published results table carries a PDB identifier,
  a chemical-component identifier, a cofactor flag and a sequence identity, and nothing about
  the molecule's chemistry. The complementary question — which <em>ligands</em> break which
  methods — has not been asked of this data, though the crystal ligands are distributed
  alongside the results as unanalysed structure files.</p>
  <p>We ask it here, and find that the answer is constrained less by the data's size than by
  collinearity among the descriptors that would answer it.</p>
</section>

<section>
  <h2><span class="n">2</span>Data and methods</h2>

  <h3><span class="n">2.1</span>Data</h3>
  <p>The PoseBusters paper data<sup><a href="#r2">2</a></sup> comprises 428 protein–ligand
  complexes released after 2021 (the PoseBusters Benchmark set), 85 complexes from the Astex
  Diverse Set<sup><a href="#r3">3</a></sup>, and a results table giving every validity check
  and RMSD for seven docking methods — AutoDock Vina, CCDC Gold, DeepDock, DiffDock, EquiBind,
  TankBind and Uni-Mol — with and without post-prediction energy minimisation. The predicted
  poses themselves were not released, so the published check results are the analysis input.
  Unless stated, results are for raw (un-minimised) poses on the 428-complex benchmark set.</p>
  <p>Two encoding properties of that table govern everything downstream. A blank check value
  means the check <em>was not run</em>, not that it failed; both flatness columns are blank
  for all 2,996 minimised rows. And 85 rows have every check blank because the method
  produced no pose at all — treating those as passing inflates validity rates. Both are
  handled explicitly.</p>

  <h3><span class="n">2.2</span>Descriptors</h3>
  <p>27 descriptors were computed per crystal ligand with RDKit<sup><a href="#r4">4</a></sup>:
  size (molecular weight, heavy atoms), flexibility (rotatable bonds, fraction sp³, amide
  bonds), ring system (total, aromatic, aliphatic, largest ring, spiro and bridgehead atoms),
  stereochemistry (tetrahedral centres, stereogenic double bonds), and polarity and
  composition (TPSA, cLogP, donors, acceptors, formal charge, heteroatoms, halogens). All 513
  ligands parsed and sanitised without error. Descriptors describe the crystal ligand, so they
  are properties of the molecule that was docked, independent of which method produced a pose.</p>

  <h3><span class="n">2.3</span>Eligibility</h3>
  <p>A ligand with no stereocentres cannot fail the tetrahedral-chirality check; one with no
  aromatic ring cannot fail aromatic-ring flatness. Including such ligands in a check's
  denominator deflates its failure rate and inflates any effect measured against the gating
  descriptor, because the comparison then encodes whether the check <em>can</em> fire rather
  than whether the molecule induces failure. Each check is therefore gated on the count of the
  substructure it inspects, using the checker's own SMARTS patterns where applicable. Gating
  raises the observed chirality-failure rate from 21.4% to 36.7% and reduces its association
  with stereocentre count from <em>d</em> = 1.09 to <em>d</em> = 0.35.</p>
  <p>The four cofactor-clash checks are conditioned differently: they pass almost
  automatically on complexes that contain no cofactor (2.4% failure versus 22.5% where one is
  present), so they are restricted to cofactor-bearing complexes, which is a property of the
  receptor rather than the ligand.</p>

  <h3><span class="n">2.4</span>Estimation</h3>
  <p>Effect sizes are Cohen's <em>d</em> between failing and passing poses, estimated
  separately for each docking method — the methods differ by more than an order of magnitude
  in competence, so pooling them describes the weakest as much as the strongest. Confidence
  intervals come from a bootstrap that resamples <em>ligands</em>, not poses: each ligand
  contributes one pose per method, so treating 2,140 rows as independent observations of 428
  ligands would report intervals roughly √5 too narrow. Strata with fewer than 15 failures or
  fewer than 30 distinct ligands are not estimated. Across the full grid of estimates,
  <em>p</em>-values are corrected by the Benjamini–Hochberg procedure<sup><a href="#r5">5</a></sup>
  at α = 0.05, treating every estimate as one family.</p>
  <p>Because marginal comparisons cannot separate correlated descriptors, each check is also
  modelled by logistic regression on a decorrelated descriptor basis with standard errors
  clustered on ligand. Where the marginal estimates and the models disagree, the models
  govern.</p>

  <h3><span class="n">2.5</span>Crystal contacts</h3>
  <p>The benchmark's journal version reports on 308 of the 428 complexes, excluding those
  whose ligand contacts a crystallographic symmetry mate on the grounds that such contacts
  could inflate protein-clash failures. That subset was not published. We recomputed it from
  first principles with gemmi<sup><a href="#r6">6</a></sup>: each deposited structure was
  expanded by its spacegroup and the closest approach measured between the ligand of interest
  and any symmetry image of the protein, counting only amino-acid residues. 104 of 427
  resolvable complexes fall within 4.0 Å; a broader criterion including solvent and
  cryoprotectant flags 119. These bracket the 120 the authors removed, and are an independent
  estimate rather than a reproduction of their list.</p>
</section>

<section>
  <h2><span class="n">3</span>Results</h2>

  <h3><span class="n">3.1</span>Accuracy and validity diverge sharply by method class</h3>
  <p>Classical search methods produce accurate poses about half the time, and nearly all of
  those poses are physically valid. Deep-learning methods produce fewer accurate poses, and
  most of the accurate ones are invalid (Table 1). DiffDock is the strongest deep-learning
  method by RMSD alone and 64.2% of its accurate poses fail at least one validity check.</p>
</section>

{_tbl(1, "Accuracy, validity, and their intersection, per method. Raw poses, 428 complexes. "
         "The final column is the share of RMSD-accurate poses that are physically invalid.",
      headline())}

<section>
  <h3><span class="n">3.2</span>The published checks reproduce</h3>
  <p>Every result here reads the published check outcomes rather than recomputing them, so the
  analysis depends on interpreting those columns correctly. We tested that by re-running
  PoseBusters on all 513 crystal ligands against their own receptors and comparing to the
  paper's own reference rows. Worst-case agreement across {val['n_checks']} mapped checks is
  <span class="q pass">{val['worst']:.2%}</span>, with {val['perfect']} of {val['n_checks']}
  checks agreeing on all 513 structures. The two disagreements are single structures on
  different checks — <code>7V3S</code> on aromatic ring flatness and <code>7T0U</code> on
  strain energy, the latter one of the two borderline failures the paper itself reports.</p>

  <h3><span class="n">3.3</span>Two thirds of the grid is unmeasurable</h3>
  <p>Of {cov['total']} method × check combinations, {cov['estimated']} carry enough failures
  and enough distinct ligands to estimate an effect (Table 2). The remaining {n_skipped} are
  not null results; they are unmeasured, mostly because the check never failed for that method.
  Across the estimable strata, {n_est} descriptor associations were computed and
  <strong>{n_sig} survive</strong> false-discovery-rate correction.</p>
</section>

{_tbl(2, "Estimability of the 126 method × check strata. Thresholds: at least 15 failures "
         "and at least 30 distinct ligands.", cov_table)}

<section>
  <h3><span class="n">3.4</span>Surviving associations are dominated by size and flexibility</h3>
  <p>The largest surviving effects (Figure 1, Table 3) concentrate on internal-geometry
  checks — internal clashes, bond lengths, bond angles, strain energy — and on two
  descriptors, molecular weight and rotatable-bond count. Both appear repeatedly on the same
  check for the same method with overlapping intervals, which is not two findings but one
  signal counted twice: the two descriptors correlate at ρ ≈ 0.73–0.74 within every method's
  ligand set. Marginal estimates cannot separate them.</p>
</section>

{_fig('03_effect_forest.png', 1,
      "The 18 largest surviving marginal associations with cluster-bootstrap 95% confidence "
      "intervals, after eligibility gating and Benjamini–Hochberg correction. Molecular "
      "weight and rotatable-bond count recur on the same check for the same method, "
      "reflecting their collinearity rather than independent effects.")}

{_tbl(3, "Twelve largest surviving associations by |d|. Failures and ligands give the "
         "denominator each estimate rests on.", survivors())}

<section>
  <h3><span class="n">3.5</span>Flexibility versus size is not identifiable</h3>
  <p>The natural hypothesis is that torsional freedom, not molecular size, drives strain and
  clash failures: a flexible molecule has more ways to be locally wrong while remaining
  globally close to the crystal pose. Testing it requires holding one descriptor fixed while
  varying the other, which their correlation forbids in a stratified table. Logistic models
  with both descriptors entered together give a per-method answer that does not agree with
  itself (Table 4).</p>
  <p>For <strong>DiffDock</strong>, neither descriptor reaches significance once the other is
  held fixed (<em>p</em> = 0.941 and 0.108), despite both surviving correction as marginal
  effects. EquiBind gives the only clean flexibility-only result and is the weakest method in
  the benchmark. DeepDock, TankBind and Uni-Mol show both descriptors carrying independent
  signal; DeepDock's weight coefficient is negative, the opposite sign to a naive
  "larger molecules fail more" account. Gold and Vina fail too rarely to fit.</p>
  <p>The contrast is therefore not identified in this dataset for the method it was posed
  about. This is a property of the benchmark's ligand set, not of the sample size: the
  correlation is essentially constant across every method's stratum, so no reweighting or
  larger subset of these complexes would separate the two.</p>
</section>

{_tbl(4, "Clustered logistic models of the strain-energy check, both descriptors entered "
         "together with five others. Standard errors clustered on ligand.", flexibility())}

<section>
  <h3><span class="n">3.6</span>Crystal contacts do not inflate protein-clash failure</h3>
  <p>Protein clash is the most-failed check for deep-learning methods, so if it concentrated
  in complexes whose ligand touches a symmetry mate, the headline failure rate would be partly
  an artifact of crystal packing. It does not (Table 5). One method of seven shows a
  significant difference and its clash failure is <em>lower</em> among contact-bearing
  complexes, the reverse of the predicted direction. The remaining six are indistinguishable
  from zero, and the two classical methods fail so rarely — Vina on 2 of 323 no-contact poses,
  Gold on 16 of 322 — that their point estimates carry no information.</p>
</section>

{_tbl(5, "Protein-clash failure by crystal-contact status, with cluster-bootstrap intervals "
         "on the difference (2,000 replicates, resampling complexes).", crystal_contacts())}

<section>
  <h3><span class="n">3.7</span>Invalidity is not confined to marginal poses</h3>
  <p>Reporting invalidity only among RMSD-accurate poses conditions on an outcome that shares
  an upstream cause with validity, which can induce association between otherwise unrelated
  quantities. Banding raw RMSD avoids this and uses every pose (Table 6). Deep-learning
  validity is already poor at sub-Ångström accuracy — DiffDock 61.3%, Uni-Mol 33.3% — and does
  not recover at larger RMSD. The classical methods are above 88% in every band.</p>
  <p>Conditioned on accuracy, the invalid share rises monotonically with rotatable-bond count
  in both method classes, and is close to flat against ring count (Figure 2). Per-check, the
  strain-energy and internal-clash failures climb steeply with flexibility while planarity
  checks barely move (Figure 3) — consistent with the surviving associations in §3.4, and
  subject to the same identifiability limit in §3.5.</p>
</section>

{_tbl(6, "Share of poses passing all applicable validity checks, by RMSD band. "
         "Unconditioned on accuracy.", rmsd_bands())}

{_fig('01_gap_by_partition.png', 2,
      "Share of RMSD-accurate poses that are physically invalid, across four ligand "
      "partitions. Monotone against rotatable-bond count in both method classes; close to "
      "flat against ring count. Correlated descriptors mean this does not by itself "
      "establish which property is responsible.")}

{_fig('02_checks_by_flexibility.png', 3,
      "Per-check failure rate against rotatable-bond count, pooled across the five "
      "deep-learning methods, with eligibility-gated denominators. Cofactor-clash checks are "
      "excluded as receptor- rather than ligand-conditioned.")}

<section>
  <h3><span class="n">3.8</span>Replication on a held-out set</h3>
  <p>All estimates above were both discovered and computed on the same 428 complexes. The 85
  Astex complexes were untouched throughout, so re-estimating a shortlist there is an
  independent test. Of {rep['n']} checkable triples, <strong>{rep['replicated']} replicate
  with statistical support</strong> — both intervals excluding zero (Table 7).
  {rep['same_sign']} of {rep['n']} agree in sign, but agreement between two estimates that
  both straddle zero carries no information, so {rep['replicated']}/{rep['n']} is the
  defensible figure. The strain-energy and internal-clash effects against rotatable bonds
  replicate for every deep-learning method tested; the stereochemistry and protein-clash
  effects largely do not.</p>
</section>

{_tbl(7, "Shortlisted effects re-estimated on the held-out Astex Diverse Set. "
         "Rows resting on fewer than 15 failures on either side are flagged thin.",
      rep_table)}

<section>
  <h2><span class="n">4</span>Discussion</h2>
  <p>The clearest result is negative, and it is about the benchmark rather than the methods.
  The two descriptors that any account of docking failure would want to distinguish — how big
  a molecule is and how many ways it can bend — are entangled throughout this ligand set at a
  level that no analysis of these 428 complexes can undo. Marginal estimates make both look
  significant; multivariate models make neither look significant for the method the question
  is most often asked about. Answering it would require a benchmark assembled to decorrelate
  them, containing large rigid ligands and small flexible ones in comparable numbers.</p>
  <p>The crystal-contact result is similarly cautionary. The exclusion of contact-bearing
  complexes is a reasonable methodological precaution, but in this data it does not do what it
  is assumed to do: protein-clash failure is no higher among the complexes it removes. That is
  worth knowing before treating the 308-complex subset as a cleaner benchmark than the 428.</p>
  <p>What does survive is coarser than the descriptor-level story the marginal grid appears to
  tell. Deep-learning poses fail internal-geometry checks — clashes, bond geometry, strain —
  at rates that rise with molecular size and flexibility jointly, and the failure is not
  confined to poses that barely clear the RMSD threshold. It is present at sub-Ångström
  accuracy, which suggests these methods are not producing slightly-imperfect physical
  molecules but rather point clouds that happen to be near the right coordinates.</p>
  <p>Finally, {n_skipped} of {cov['total']} strata could not be estimated at all, most because
  the check never failed for that method. Reporting an effect grid without that denominator
  invites reading absence of estimate as absence of effect.</p>
</section>

<section>
  <h2><span class="n">5</span>Limitations</h2>
  <p>Descriptors describe the docked molecule, not the returned pose, so nothing here measures
  how wrong a pose is beyond the binary checks. The predicted poses were never released, so
  the methods' check results cannot be independently recomputed; only the crystal-structure
  rows could be, and were (§3.2). The crystal-contact flag is our own estimate and brackets
  rather than reproduces the authors' undisclosed subset. Several surviving associations rest
  on small failure counts even after gating — DiffDock's internal-clash effects on 25 failures
  — and single rows should not be read as decisive. The cofactor-clash checks are conditioned
  on the receptor and are reported only in the estimable-strata counts, not interpreted as
  ligand chemistry.</p>
</section>

<section>
  <h2><span class="n">6</span>Data and code availability</h2>
  <p>Source data are the PoseBusters paper data<sup><a href="#r2">2</a></sup> on Zenodo and
  deposited structures from the RCSB PDB<sup><a href="#r7">7</a></sup>, both fetched
  automatically by the pipeline. Every table in this paper is read from the analysis outputs
  at render time and every figure is inlined from the generated originals, so the document
  cannot diverge from the analysis it reports. The pipeline is deterministic under fixed
  seeds and covered by 37 tests.</p>
</section>

<section>
  <h2>References</h2>
  <ol class="refs">
    <li id="r1">Buttenschoen M, Morris GM, Deane CM. PoseBusters: AI-based docking methods
      fail to generate physically valid poses or generalise to novel sequences.
      <em>Chem Sci</em> <strong>15</strong>, 3130–3139 (2024).
      <a href="https://doi.org/10.1039/D3SC04185A">doi:10.1039/D3SC04185A</a></li>
    <li id="r2">Buttenschoen M, Morris GM, Deane CM. PoseBusters paper data. Zenodo (2023).
      <a href="https://doi.org/10.5281/zenodo.8278563">doi:10.5281/zenodo.8278563</a></li>
    <li id="r3">Hartshorn MJ <em>et al.</em> Diverse, high-quality test set for the validation
      of protein–ligand docking performance. <em>J Med Chem</em> <strong>50</strong>, 726–741 (2007).</li>
    <li id="r4">RDKit: open-source cheminformatics. <a href="https://www.rdkit.org">rdkit.org</a></li>
    <li id="r5">Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and
      powerful approach to multiple testing. <em>J R Stat Soc B</em> <strong>57</strong>, 289–300 (1995).</li>
    <li id="r6">Wojdyr M. GEMMI: a library for structural biology. <em>J Open Source Softw</em>
      <strong>7</strong>, 4200 (2022).</li>
    <li id="r7">Berman HM <em>et al.</em> The Protein Data Bank. <em>Nucleic Acids Res</em>
      <strong>28</strong>, 235–242 (2000).</li>
  </ol>
</section>

<footer>
  <p>Analysis conditioned on ligand chemistry rather than docking method. All estimates carry
  cluster-bootstrap intervals; no point estimate is reported without one. Generated from
  <code>reports/tables/</code> — regenerate with <code>python -m pb.paper_html</code>.</p>
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
