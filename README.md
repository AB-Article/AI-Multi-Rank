# A differential theory of quality–resource trade-offs — reproducibility archive

Data, analysis code and results for:

> D. Ignatov, *A Differential Theory of Quality–Resource Trade-offs for AI Model
> Selection: The Multiresource Frontier and Its Minimax Price.*

Everything reported in the article, its Online Appendix and its technical report
is produced by the code in this archive from the data in this archive. There are
no hidden steps, no manual edits to any number, and no network access at
analysis time.

**Archive DOI:** `10.5281/zenodo.0000000` — replace with the DOI Zenodo mints
for this release before quoting it.

---

## Quick start

```bash
python3 -m pip install -r requirements.txt
./run_all.sh guards    # about a minute: the six consistency guards
./run_all.sh fast      # 160 scripts, less the eight slowest stages
./run_all.sh           # the full pipeline
./run_all.sh list      # print the execution order and exit
```

Wall clock depends on the machine and is dominated by a handful of stages, so no
single figure is quoted. Every stage prints its own elapsed seconds and the logs
under `results/logs/` keep them, so a run measures itself. For scale, on two
cores: most stages take seconds, `07`, `09` and `14` take three to six minutes
each, and the eight stages `fast` skips are the longest of all — `82`, `116`,
`130` and `167` are tens of minutes apiece. **Budget hours, not minutes, for a
full run, and start with `fast`.**

Outputs land in `results/` (numbers, tables and run logs) and `figures/` (PDFs).
Nothing outside those two directories is written, so a run can always be checked
by diffing them against the archived copies:

```bash
./run_all.sh && git diff --stat results figures
```

The pipeline needs `numpy`, `scipy`, `pandas`, `matplotlib` and `mpmath`; five
scripts also need `sympy`. Every random seed is fixed, so results are
reproducible on a given platform — the two Monte-Carlo additions of this
revision, `167` and `168`, were verified **bit-identical** across two
independent runs.

**No network access is needed or attempted.** Exactly three of the 162 numbered
scripts can reach the network and none is needed for any published number; all
three are excluded from every mode of `run_all.sh`.

---

## What reproduces, and how exactly

Verified on Python 3.13 / numpy 2.5.3 / scipy 1.18.1 / pandas 3.0.5 /
matplotlib 3.11.2 / mpmath 1.4.1 / sympy 1.14.0.

| Claim in the article | Value | Reproduced by |
|---|---|---|
| The curve determines the frontier: reconstruction by quadrature | 159 of 159 checks at 40 digits | `134_ode_first.py` |
| Every analytic identity in the three documents | 683 identity checks, worst residual 1.9 × 10⁻⁴⁹ | `162_verification_index.py` |
| Archive median fitted floor share | 0.9556 over 117 frontiers; 0.9910 over the 92 second-order-readable | `158_floor_share_null.py` |
| …reproduced by a frontier with **no floor at all** | at shape index β = 0.101 (archive median design) | `158_floor_share_null.py` |
| Admissible-ceiling violations, as once reported | 89 of 117 | `129_plane_corpus.py` |
| …expected under a design-matched null, which **withdraws that count** | 100.3 of 117 | `167_ceiling_null.py` |
| …calibrated rejections at the 5 % level | 0 of 111 (median triple), 5 of 88 (widest triple) | `167_ceiling_null.py` |
| …power against a floor at twice its admissible value | 0.010 and 0.096 | `167_ceiling_null.py` |
| Held-out prediction on 2025 withheld models | R² = 0.9897 on log-error, 2.60 % median relative error | `68_prediction_assessment.py`, pooled by `99_cure_chart.py` |
| Cure chart ties the standard form out of sample | 2.57 % on 2013 points against 2.60 % on 2025 | `99_cure_chart.py` |
| Arc–depth identity: √N<sub>eff</sub> beats N<sub>eff</sub> and S | 76 of 90 frontiers, median R² = 0.9992 | `104_tower_and_arc.py` |
| Harmonic rule against the resource-aware choice | 90.94 % on 20 206 Pareto pairs, against 77.41 % | `45_accuracy_only.py` |
| Derived price against the rules in use | median regret 0.0010 nats, 96 % in band on 26 frontiers | `11_selection_rules.py` |
| …ordering tested, not eyeballed | Friedman χ² = 113.97, p = 5.8 × 10⁻²¹, Nemenyi CD = 2.356 | `166_rules_significance.py` |
| Rules in use return a median of five distinct models | 19× apart in resource for 1.72 bits, on 96 frontiers | `168_rule_spread.py` |
| …and a unit change moves the one non-scale-free rule | 94 of 96 frontiers | `168_rule_spread.py` |
| Pareto points optimal at **no** price | median 62.5 %, 5 of 11 reachable | `155_price_envelope.py` |

Each row names the script that writes the number. The complete script-by-script
map is under "Full provenance" below.

---

## Layout

```
.
├── run_all.sh              driver: reproduces every number and figure
├── requirements.txt        pinned dependency floor
├── LICENSE                 Apache-2.0 (code)
├── LICENSE-DATA            CC BY 4.0 (data compilations)
├── CITATION.cff            machine-readable citation metadata
├── .zenodo.json            Zenodo release metadata
├── README.md               this file
├── PRACTICAL_PIPELINE.md   the deployment protocol, its levels and its gates
├── data/                   inputs (see "Data provenance" below)
├── scripts/                162 numbered scripts + 6 consistency guards
├── results/                generated numbers, tables and run logs
├── figures/                generated figures
└── examples/
    ├── operating_point.py  the rule below as runnable code, with a self-test
    └── model_selection/    the decision rule end to end, stdlib only
```

**Conventions.** `data/` holds inputs only: **none of the files the maps name
sit under `data/`**, and none more are in both places. Every derived file is
written under `results/`, and `figures/` holds what the figure scripts draw.
Script numbers run to `168` with gaps at `78`, `85`, `94`, `101`, `106`, `112`
and `118`: `94` was merged into `93` and `78` into `77`, and the rest are slots
vacated when the statement they verified was withdrawn or absorbed. No row of
the provenance map points at a missing script.

`scripts/` is what a reviewer needs; "Full provenance" below is what a reader
needs who wants to know which script produced a particular sentence.

---

## Data provenance

Every source is public and none is credentialed.

### Upstream sources and their licences

All inputs are public and none is credentialed. The archive redistributes
flattened extracts of published records --- accuracy and cost numbers, never
images or model weights --- and the upstream licence and citation apply to the
records each row covers.

**The licence column was verified against each upstream repository's own
licence file** on 2026-11-14, by reading `LICENSE` at the default branch rather
than by consulting a third-party index. Where a repository carries no licence
file, this table says so instead of guessing: that is not a formality, because
redistribution of those records is not covered by any grant.

| Source | Field | Upstream | Citation | Licence (verified) |
|---|---|---|---|---|
| `timm` results tables | vision | `huggingface/pytorch-image-models`, `results/` | Wightman (2019) | Apache-2.0 |
| ImageNet-ReaL, -V2, -Sketch, -R, -A | vision | accuracy numbers via the `timm` tables above; no images are redistributed | Beyer et al. (2020); Recht et al. (2019); Wang et al. (2019); Hendrycks et al. (2021a,b) | the numbers come under `timm`'s Apache-2.0; each dataset keeps its own terms, which this archive does not exercise |
| NAS-Bench-201 | NAS topology | `D-X-Y/NAS-Bench-201` | Dong and Yang (2020) | MIT |
| HW-NAS-Bench | NAS latency | `GATECH-EIC/HW-NAS-Bench` | Li et al. (2021) | MIT |
| NATS-Bench SSS | NAS size | `D-X-Y/NATS-Bench` | Dong et al. (2022) | MIT |
| TabArena / `tabrepo` | tabular AutoML | `autogluon/tabrepo` | Erickson et al. (2025) | Apache-2.0 |
| MedMNIST v2, MedMNIST+ | medical imaging | `MedMNIST/MedMNIST` (published results) | Yang et al. (2023); Doerrich et al. (2025) | Apache-2.0 for the code; the datasets are CC BY 4.0 |
| Matbench Discovery | materials | `janosh/matbench-discovery` | Riebesell et al. (2025) | MIT |
| Gadre et al. grids | language models | `mlfoundations/scaling` | Gadre et al. (2024) | MIT |
| Porian et al. grids | language models | `formll/resolving-scaling-law-discrepancies` | Porian et al. (2024) | MIT text, but the copyright line is an unedited template; treat the holder as the repository's authors |
| **YAHPO Gym** | HPO | `slds-lmu/yahpo_gym` | Pfisterer et al. (2022) | **no licence file upstream** --- see below |
| **Chinchilla reanalysis** | language models | `epoch-research/analyzing-chinchilla` | Besiroglu et al. (2024) | **no licence file upstream** --- see below |

**The two unlicensed sources.** `slds-lmu/yahpo_gym` and
`epoch-research/analyzing-chinchilla` carry no licence file, so no
redistribution right is granted by default and the records drawn from them are
redistributed here on the strength of their being published figures, cited to
their authors. Before a Zenodo release, either obtain written permission, or
replace those two extracts with a script that regenerates them from upstream on
the user's machine. The two files concerned are `data/field_hpo_yahpo.csv` and
the Chinchilla rows of `data/grid_factorial.csv`. No conclusion in the article
rests on either: the class gate reports which frontiers are in class, and
dropping both leaves the corpus at 115 frontiers.

Upstream commit identifiers were not recorded for every source when the
extracts were taken, and are not reconstructed here. The three that were
recorded are `9225883` (Chinchilla reanalysis), `0dc60d9` (Porian et al.) and
`a003c49` (Gadre et al.).

---

### Derived inputs

Every derived file lives under `results/`; `data/` holds inputs only.

| File | Contents | Producer |
|---|---|---|
| `results/table_all_frontiers.csv` | the 117-frontier build, one row per frontier | `09_all_fields.py` |
| `results/table_universal_points.csv` | pooled points on the universal curve | `10_universal_curve.py` |
| `results/minimax_verification.csv` | Bayes and minimax grid searches | `19_minimax_verification.py` |
| `results/rules_theorem_verification.csv` | the selection identities, verified | `11_selection_rules.py` |
| `results/frontiers/*.csv` | the 66 per-frontier extracts the later scripts read | `09_all_fields.py` |

The copies of these that shipped under `data/` in the first release were outputs
of an earlier revision of the pipeline, and `data/` was never read for them.
They have been moved to `results/`, which is where every script writes them.
Two carry visibly older numbers --- `table_all_frontiers.csv` differs from a
rerun by up to `1.5e-3` in `eps_inf` and `0.61` in `xi_span`, with every
screening verdict unchanged --- so **run `./run_all.sh` once before a release**
and commit what it regenerates.

`00_fetch_data.py` clones the two upstream benchmark repositories whose extracts
are already archived under `data/`, and `13_factorial_grids.py` rebuilds the
factorial grids from them. Neither is needed to reproduce a published number;
see "Full rebuild from upstream sources" below.

### Known limitations

The archive runs from the archived frontiers with **no refitting of any
published table**: where the article tests a published fit, it tests the
published numbers. Three limitations are inherited and documented rather than
worked around.

1. **Living upstream documentation.** `timm`'s tables are append-only and
   historically stable; some benchmark leaderboards are living text. Exact
   re-extraction needs the contemporaneous revision. No conclusion depends on
   it: the gate in `139_class_gate_corpus.py` reports which frontiers are in
   class, and the article's scope section reports what fails first.
2. **Published item counts.** Only 36 of the frontiers publish an evaluation-set
   size, which is what the precision gate and the certified set need. Where it
   is missing the archive says so rather than imputing it.
3. **A fitted floor is not a measurement.** This is a *finding* of the article,
   not a defect of the archive, but it governs how `data/` should be read: the
   fitted irreducible error on any frontier here is a monotone readout of the
   shape index, and `158_floor_share_null.py` demonstrates it on frontiers whose
   floor is exactly zero.

---

## Pipeline stages

`run_all.sh` executes 160 of the 162 numbered scripts in the order the work was
written, which is also a dependency order. Each is independently runnable from
`scripts/`.

**Frontiers and laws** (`01`–`19`). Build the frontiers, verify the duality and
regret identities, fit the laws, run the extrapolation and selection
experiments, and check the factorial grids and the minimax statements.

**Spectral extension** (`20`–`46`). The channel count, the running floor, the
bend and path-length diagnostics, the width coordinate with its calibrated null,
the floor axis, and the accuracy-only collapse with its 20 206-pair test.

**Exchange rate, duality, arc** (`47`–`62`). The local exchange rate, the
exponent transfer, the AM–GM master identity, axis self-duality, the
one-dimensional minimax, and the arc with its covering numbers.

**Regret law and microfoundation** (`63`–`79`). The co-failure panel that
refutes conditional independence, the five regret identities, held-out
prediction on 117 frontiers, the count axis, and the floor as a readout of the
shape index.

**The gauge-invariant calculus** (`80`–`120`). The chart test with
design-matched nulls, de Finetti on a count axis, the chart verdict, the
divisibility ladders, the horizon lattice and the attempt chain.

**The channel–depth plane** (`121`–`138`). The curve as the primitive, both
rungs as Lipschitz bounds, the corpus read on the plane, the design-matched
divisibility nulls, and the thirteen-axis audit of the reformulation.

**The practical pipeline** (`139`–`157`). The class gate, the precision gate,
the certified set and its measured coverage, the design rule, the sequential
minimax design and the input-size axis.

**This revision** (`158`–`168`). The zero-floor null for the fitted floor share,
the exponent decomposition, the arc deficit read at its crossover, the plug-in
rung's node-spacing bias, the published fixed-exponent rule, the plug-in regret,
the three-point-floor re-score of the rule table, Friedman and Nemenyi, **the
admissible-ceiling null that retires the 89-of-117 count** (`167`), and **the
loss-free measurement of how far apart the rules in use are** (`168`).

**Guards** (five, unnumbered). `run_all.sh guards` runs them; see below.

The eight stages `fast` mode skips are `30`, `63`, `82`, `116`, `124`, `127`,
`130` and `167` — the ones the provenance map documents as slow. They dominate
the wall clock of a full run.

---

## Full provenance: every script and what it writes

The table above names one script per headline claim. The four tables below are
the complete map: every item of the article and its appendices, the script that
produces it, and the file it writes. The *script* and *output* columns are
maintained and authoritative. The *paper item* column keeps the wording and the
section numbers of the manuscript revision that wrote each row, so read it as a
description rather than as a current cross-reference.

### Numerical verification of the analytic identities

| Paper item | Script | Output |
|---|---|---|
| Duality, canonical reference, rank, normal form, Fisher metric (Table S5) | `02_verify_theory.py` | `results/theorem_verification.csv` |
| Running exponents, several objectives, soft-max family (Table S5) | `05_verify_generalisations.py` | `results/generalisation_verification.csv` |
| Proposition 33, fixed-ratio selection is single-channel (Table S5) | `16_verify_fixed_ratio.py` | `results/fixed_ratio_verification.csv` |
| Minimax characterization, Lemma 15, Remark 16: 86.6% of random frontiers (Table S6) | `19_minimax_verification.py` | `results/minimax_verification.csv` |
| Single-channel identities: elasticity, curvature, regret band, three-point rule, concavity (Table S7) | `11_selection_rules.py` | `results/rules_theorem_verification.csv` |
| Nested duality, nested canonical reference (Technical report, Thm. `nested`) | `20_nested_verification.py` | `results/nested_verification.csv` |
| Path length, arc identity, total length pi, K-atom bound, Gamma width, speed limit (Table 11) | `34_path_length.py` | `results/table_path_verification.csv` |
| Width coordinate: zeta = ln g = logit p0, slope = alpha, eps = 2 eps_inf at zeta = 0, cumulant closure, Gamma branch | `38_width_coordinate.py` | `results/table_width_verification.csv` |

### Experiments

| Paper item | Script | Output |
|---|---|---|
| Frontier construction for the vision corpus | `01_build_frontiers.py` | `results/frontiers/*.csv`, `results/frontier_index.csv` |
| Corpora and screening (Section 8.1, Table 3; details in S2.1) | `09_all_fields.py` | `results/table_all_frontiers.csv` |
| Canonical selection on measured frontiers (Section 8.2) | `06_selection_example.py`, `07_new_corpora.py` | `results/table_selection_example.csv`, `results/table_nb201_selection.csv` |
| Regret from estimation error, against Prop. `misquote` (Online Appendix) | `164_plugin_regret.py` | `results/table_plugin_regret.csv` |
| Which frontiers count: the funnel and Table `namedrules` re-read on the bandwidth-free three-point floor (paper, "Rules in use") | `165_floor_filter_sensitivity.py` | `results/table_floor_filter_funnel.csv`, `table_floor_filter_per_frontier.csv`, `table_rules_summary_3pt.csv` |
| Is the ordering real: Friedman, Nemenyi and bootstrap intervals for Table `namedrules`, the setting of Demsar (2006) | `166_rules_significance.py` | `results/table_rules_significance.csv`, `table_rules_nemenyi.csv` |
| **The admissible-ceiling test, read against a design-matched null (paper, Table `ceilnull`).** The null is the incumbent form itself: `eps = eps_inf + beta t^-alpha` is the two-atom chart, which **saturates** the axiom wall (`N_eff = rho - 1` and `eps_inf = eps/(1+N_eff)` identically), so a frontier of that form can never violate its own ceiling and the ceiling test is a goodness-of-fit test of the form whose null is the form. Per frontier: fit the form, take its residual scale, draw replicates on the frontier's own nodes, re-run the whole pipeline on each. Three statistics, differing only in *where* the pointwise bound is read -- the incumbent minimum over up to `16719` triples, the median over consecutive triples, and the single widest triple fixed in advance. **This script retires the `89 of 117` count** reported in earlier drafts: under the null the incumbent procedure reports violations on a median of *every* replicate and an expected count *above* the 89 observed, because `N_eff = s^2/kappa` diverges as `kappa -> 0+` and a minimum over thousands of heavy-tailed draws is a selection and not a bound (symptom with no simulation needed: on `16` of `117` frontiers the companion wall returns exactly `0`, which no floored frontier can have). Power is measured against the deformed wall `N_eff = theta(e^dS - 1)`, integrated in closed form to `s = alpha (1 - e^-dS)^(1/theta)`, whose floor exceeds its ceiling by `ln theta` at depth; the two integration constants place the truth on each frontier's own node and error range. A sigma-sweep gives the per-model precision a measurement would have to deliver | `167_ceiling_null.py` | `results/table_ceiling_null_per_frontier.csv`, `table_ceiling_null_summary.csv` |
| **How large the selection decision is, measured with no objective at all (paper, "How large the decision is").** Table `namedrules` scores rules against this paper's own loss, so its first row is close to an identity; this script instead reports only what the measured frontier says. The eight fit-free rules on the `96`-frontier build -- no floor needed, the widest population in the work -- return a median of **five distinct models**; the extreme picks lie a median `19x` apart in resource (IQR `4x` to `236x`) for a median `1.72` bits; the spread exceeds `10x` on `52` of `96` and buys under one bit on `29`; the median exchange rate across the disagreement is `12x` per bit; and the one rule in current use that is not scale-free moves its pick on `94` of `96` under a `1e3` change of unit | `168_rule_spread.py` | `results/table_rule_spread_per_frontier.csv`, `table_rule_spread_summary.csv` |
| Published fixed-exponent rule, MnasNet (paper, "Rules in use") | `163_published_rule.py` | `results/table_published_rule.csv` |
| Additivity, dimension, reparameterization (Section 8.3; details in S2.4) | `03_fit_laws.py`, `07_new_corpora.py`, `09_all_fields.py` | `results/table_additive_vs_multiplicative.csv`, `table_exponents.csv`, `table_exponent_factorisation.csv`, `table_unit_covariance.csv`, `table_frontier_spectrum.csv`, `table_single_resource_alpha.csv`, `table_nb201_pairs.csv`, `table_tabarena_pairs.csv`, `table_all_lawtest.csv` |
| Latency and energy on one device (S2.3, Table S2) | `07_new_corpora.py` | `results/table_nb201_within_device.csv` |
| Held-out extrapolation (Section 8.4, Table 4) | `04_extrapolation.py`, `07_new_corpora.py` | `results/table_extrapolation.csv`, `table_extrapolation_summary.csv`, `table_extrapolation_wins.csv`, `table_new_corpora_extrap.csv` |
| Effective channel count and single-channel collapse (Section 8.5, Table 5) | `09_all_fields.py`, `10_universal_curve.py` | `results/table_channel_count.csv`, `table_universal_curve.csv`, `table_universal_points.csv` |
| Multi-channel cross-field collapse (S2.5, Table S3) | `08_cross_field.py` | `results/table_cross_field_frontiers.csv`, `table_cross_field_points.csv` |
| Factorial grids (Section 8.6) | `13_factorial_grids.py` | `data/grid_factorial.csv` |
| Factorial test of N = n (Table 6, Figure 1) | `14_test_N_equals_n.py`, `17_factorial_summary.py`, `18_figures_factorial.py` | `results/table_factorial_N_test*.csv`, `table_factorial_summary.csv`, `figures/fig_factorial.pdf` |
| Robustness of the factorial results (S2.6) | `15_factorial_robustness.py` | `results/table_factorial_robustness.csv` |
| Nested reference on the language-model grids (Technical report, Thm. `nested`) | `21_nested_reference_lm.py` | `results/table_nested_reference_lm.csv` |
| Fitting-free rules (Section 8.7, Table 8; Algorithm S1, Figures S1–S2, S2.7) | `11_selection_rules.py`, `12_figures_rules.py` | `results/table_rules_summary.csv`, `table_rules_per_frontier.csv`, `table_rules_by_field.csv`, `table_rules_admissibility.csv`, `table_half_slope_sensitivity.csv`, `table_nonconcavity_by_field.csv`, `figures/fig_rules_theory.pdf`, `figures/fig_rules_empirical.pdf` |
| Floor bound and label noise (Appendix C.2; S2.8, Table S4) | `11_selection_rules.py` | `results/table_floor_label_noise.csv` |

`table_rules_summary.csv` also carries the three rules named in the literature
--- accuracy per unit cost, the ratio of accuracy to `log10` cost, and a hard
budget --- which `11_selection_rules.py` scores beside the paper's own. The
ratio rule's `ratio_log10_unitflip` column in `table_rules_per_frontier.csv`
records whether its pick survives a thousand-fold change of the resource unit.

### The spectral extension

| Paper item | Script | Output |
|---|---|---|
| Held-out floors, floor drift, null simulations (Section 8.8, Table S9) | `22_holdout_floor.py` | `results/table_holdout_floor.csv`, `table_holdout_floor_per_frontier.csv`, `table_floor_drift.csv`, `table_holdout_nulls.csv` |
| Resolution number and channel count (Section 8.9, Table 9; calibration in S5.2, Table S10) | `23_resolution_channels.py` | `results/table_resolution_frontiers.csv`, `table_resolution_summary.csv`, `table_resolution_sim.csv` |
| Floor-free exponent: identities and estimation (Section 8.10; S5.3, Table S11) | `24_floor_free_exponent.py` | `results/table_floor_free_verification.csv`, `table_floor_free_per_frontier.csv`, `table_floor_free_summary.csv` |
| Budget law: verification and iso-FLOP comparison (Section 8.11; S5.4, Table S12) | `25_budget_law.py` | `results/table_budget_verification.csv`, `table_budget_law.csv` |
| Shared exponents out of sample; floors vs ReaL labels; shrinkage (Sections 8.12, 8.14; S5.5) | `26_shared_shrinkage.py` | `results/table_shared_exponents.csv`, `table_floor_label_noise_fano.csv`, `table_shrinkage.csv` |
| Reciprocity of interaction and exponent (Section 8.13, Table 10) | `27_resolution_model.py` | `results/table_resolution_model.csv` |
| Figure 3 (resolution number, calibration, reciprocity) | `28_figure_spectral.py` | `figures/fig_spectral.pdf` |
| Figure 1 (one measured frontier and the collapse) | `29_figure_overview.py` | `figures/fig_overview.pdf` |
| Surprisal identity, running floor, measured spectral width (Section 8.9, Table 9; verification in S3; per-window detail in S5.5) | `30_surprisal_running_floor.py` | `results/table_surprisal_verification.csv`, `table_running_floor.csv`, `table_running_floor_summary.csv`, `table_spectral_width.csv`, `table_p0_selection.csv` |
| Figure 4 (identity, running floor, measurable resolution) | `31_figure_surprisal.py` | `figures/fig_surprisal.pdf` |
| Selection law and bend divergence (Section 10.12, Figure 5) | `32_bend_divergence.py` | `results/table_bend_verification.csv`, `table_bend_per_frontier.csv`, `table_bend_summary.csv` |
| Figure 5 (master equation, geometry of the bend, bend vs. width) | `33_figure_bend.py` | `figures/fig_bend.pdf` |
| Path length, Gamma dichotomy, local floor, truncation experiment, worked example (Section 10.13, Tables 11-13) | `34_path_length.py` | `results/table_path_verification.csv`, `table_path_per_frontier.csv`, `table_path_summary.csv`, `table_path_truncation.csv`, `table_path_truncation_summary.csv`, `table_path_example.csv` |
| Figure 6 (windows on the universal arc, length identity, Gamma dichotomy) | `35_figure_path.py` | `figures/fig_path.pdf` |
| Width coordinate: affinity of zeta, slope vs. fitted exponent, half-error floor, membership statistic R and its synthetic power (Section 10.14, Table 13, Online Appendix S4.12) | `38_width_coordinate.py` | `results/table_width_verification.csv`, `table_width_per_frontier.csv`, `table_width_summary.csv`, `table_width_power.csv` |
| Figure 10 (affine width coordinate, slope vs. exponent, the R statistic) | `39_figure_width.py` | `figures/fig_width.pdf` |

### Estimator calibration

These five scripts supply simulated nulls for statistics whose analytic nulls
are identities in the exact derivatives of `S` but whose plug-in estimators
carry a bandwidth. They read only tables already written by `11`, `27`, `38`
and `22`, plus `data/grid_factorial.csv` and
`data/table_floor_label_noise.csv`, so they can be run last and in any order.

| Paper item | Script | Output |
|---|---|---|
| Membership statistic against a per-frontier calibrated null; two- vs. three-atom (Online Appendix `tab:Rcal`, Table S13) | `40_width_null_calibration.py` | `results/table_width_null_calibration.csv`, `table_width_null_summary.csv`, `table_width_null_three.csv` |
| Why that null is needed: R against window offset and bandwidth on exact two-atom data (Section 5 of the note, S5.18) | `41_width_window_profile.py` | `results/table_width_window_profile.csv`, `table_width_bandwidth_centred.csv`, `table_width_exact_moments.csv` |
| Power of the resolution-model reciprocity against an independent-parameter null (Section 8 of the note, S4.3) | `42_resolution_power.py` | `results/table_resolution_power.csv` |
| The fitted floor is not axis-invariant, so it does not estimate the Bayes error (S5.12) | `43_floor_axis.py` | `results/table_floor_axis.csv` |
| Binomial sampling noise as a lower bound on sigma in the resolution number (S5.13) | `44_noise_binomial.py` | `results/table_noise_binomial.csv` |
| The accuracy-only collapse: rules under both objectives, the two-candidate rule on every Pareto pair, floor sensitivity, and the three-point floor (Section 3 of the paper and Online Appendix `tab:acconly`, S5.21) | `45_accuracy_only.py` | `results/table_acconly_cells.csv`, `table_acconly_rules.csv`, `table_acconly_pairs.csv`, `table_acconly_pairs_strata.csv`, `table_acconly_floor.csv`, `table_acconly_threepoint.csv` |
| Figure 2 of the paper: the canonical value, the rule against the truth, and the blind spot | `46_figure_acconly.py` | `figures/fig_acconly.pdf` |

`40` and `42` are the slow ones: `40` simulates 600 two-atom draws and 400
three-atom draws per frontier at four bandwidths (about fifteen minutes for
the 28-frontier panel), and `42` profiles the power-mean fit over seven values
of theta for six tasks in each of twelve repetitions under two generators
(about ten minutes). Both take their seeds from the module header, so the
tables they write reproduce the numbers quoted in the paper exactly.
`45` and `46` are deterministic and take under a minute each; `45` must run
before `46` is read for its numbers, though `46` recomputes what it draws. `41`,
`43` and `44` are fast. `40` requires `38` to have been run (it reads
`results/table_width_per_frontier.csv` for the panel definition and the fitted
floors), `42` requires `14` and `27`, `43` requires `11`, and `44` requires
`22`. All five import the shared estimator through `scripts/_calib.py`, which
reproduces the local-quadratic estimator of `38` verbatim so that the
simulated null is the null of the estimator actually used.

Scripts `22`, `24`, `26`, and `30` import the frontier builders of `09` and
`11` through `scripts/_loaders.py`, which executes their module-level code
without running their `__main__` blocks; `23` and `25` do the same for `14`.
Run `09`, `11`, and `14` first, since `23`, `25`, and `27` read
`results/table_channel_count.csv` and `results/table_factorial_N_test_lm.csv`.
Script `38` uses the same loaders and takes roughly eight minutes (bootstrap
over Pareto points plus the synthetic power study); `39` recomputes the width
profiles directly and is fast, but reads `38`'s tables, so run `38` first.
`23` reuses `results/table_resolution_frontiers.csv` if it already exists;
delete that file to recompute from the frontiers. Run `24` before `30`: the
latter reads the BIC channel counts from
`results/table_floor_free_per_frontier.csv` rather than refitting them, so the
two scripts report the same `N_hat`. `30` takes a few minutes, almost all of
it in the two null ensembles of the running-floor panel; reduce `nboot` in
`floor_panel` for a quick pass. `31` reads only the tables written by `30`.

`32` and `34` import the frontier builders through `scripts/_loaders.py` in
the same way as `22`, `24`, `26`, and `30`, so `09` and `11` must have been
run first. `34` reads `results/table_floor_free_per_frontier.csv` only for the
cross-check of its local floor rule against the fitted floor; it recomputes
its own three-parameter fits, so it is self-contained otherwise. `34` takes
about two minutes, almost all of it in the 200-draw bootstrap of the path
length and the width drift. `35` reads only
`results/table_path_per_frontier.csv`.

Note on the estimator in `32` and `34`: slope and curvature come from the same
kernel-weighted local quadratic with bandwidth `span/4` used in `24`, and the
outer tenth of each window is trimmed. This smooths curvature and so biases
the path length `L` downward. That bias is now moot for the `L < pi` reading,
which is **withdrawn** outright (script `132`, and the note at `92` above): a
floored frontier's admissible arc is `2 arccos sqrt(eps_inf) < pi` whatever its
channel count, so a window arc below `pi` carries no information about the
channel count and no improvement in bandwidth could have rescued it. What the
paper reports is the agreement between `L` and its floor-free prediction, and
`N >= L/pi` read on the complete sweep.

Note on the verification in `30`: the normalized resource is the **logit of
the floor share**, `zeta = ln(p_0 / (1 - p_0))`, not its reciprocal. The
orientation is easy to invert, and the high-precision check in `verify()` is
what catches it.

---

---

---

## Six consistency guards

The archive audits itself against the article's sources. Each guard takes no
arguments, reads only this archive and the sources one level above, and prints
`CLEAN` or the offenders.

| guard | checks |
|---|---|
| `_guard_nums.py` | every high-precision number (3–6 decimals) in the three `.tex` files must match some value in some CSV under `results/` or `data/` |
| `_guard_titles.py` | every quoted statement and section title in the prose documents and in all script docstrings, against the real titles in the `.tex` files |
| `_guard_readme.py` | every structural count this archive states about itself — figures, tables, numbered statements, page counts, script gaps |
| `_guard_prov.py` | every "the convention of script N" claim, against which scripts actually declare the jet bandwidth and the noise estimator |
| `_guard_refwords.py` | every `Theorem/Table/Figure~\ref{...}` in the `.tex` files, against the environment that declares that label |
| `_guard_snippet.py` | the ten-line rule printed below must be character-identical to `examples/operating_point.py`, whose self-test then runs it on every archived frontier |

Run all six after any edit to the article or the scripts. LaTeX compiles
cleanly whatever the prose says about a statement's name, so nothing else
catches these.

---

## The selection rule, in ten lines

The whole method, for a practitioner who wants nothing else from this archive.
No floor, no fit, no bandwidth, no declared price — three accuracies and the
resource they were measured at:

```python
import math

def operating_point(points, q0=0.0, grain=1.0):
    """points: iterable of (name, quality Q, resource t > 0).
    q0: chance level.  grain: items per delivered unit.

    Returns the name whose effective channel count N_eff = s^2 / kappa is
    closest to 1/grain -- the right angle of the article -- with N_eff read
    from three accuracies, or None if no interior triple is second-order
    readable.  No floor, no fit, no bandwidth, no declared price.
    """
    # one point per resource value, keeping the most accurate; then sort by cost
    best_at = {}
    for name, Q, t in points:
        if not (t > 0):
            continue
        eps = (1.0 - Q) / (1.0 - q0)                  # chance-corrected error
        if not (eps > 0):
            continue
        x = math.log(t)
        if x not in best_at or eps < best_at[x][1]:
            best_at[x] = (name, eps)
    grid = sorted((x, n, e) for x, (n, e) in best_at.items())
    if len(grid) < 3:
        return None

    pick, best_gap = None, float("inf")
    for i in range(1, len(grid) - 1):                 # every interior triple
        (x0, _, e0), (x1, n1, e1), (x2, _, e2) = grid[i - 1], grid[i], grid[i + 1]
        l0, l1, l2 = math.log(e0), math.log(e1), math.log(e2)
        s = -(l2 - l0) / (x2 - x0)                    # s = -d ln eps / d ln t
        kappa = -2.0 * (l0 / ((x0 - x1) * (x0 - x2))  # kappa = -d2 ln eps / d ln t2
                        + l1 / ((x1 - x0) * (x1 - x2))
                        + l2 / ((x2 - x0) * (x2 - x1)))
        if s <= 0 or kappa <= 0:
            continue                                  # not readable at this triple
        gap = abs(math.log(s * s / kappa * grain))     # log distance from N_eff = 1/grain
        if gap < best_gap:
            pick, best_gap = n1, gap
    return pick
```

The spacing cancels identically in `s*s/kap`, so the triple need not be evenly
spaced and no bandwidth is chosen. A frontier on which no interior triple has
`s > 0` and `kap > 0` is not second-order readable, and the rule correctly
declines to name a point rather than naming a wrong one.

Two cautions the article establishes and this snippet cannot:
a fitted irreducible error is not a measurement of the irreducible error, so do
not feed one in; and a rule that a change of resource unit can overturn is a
price and not a measurement, which is why no unit appears above.

A maintained implementation, independent of this archive and installable with
`pip install nn-rank`, is at <https://github.com/ABrain-One/nn-rank>. This
archive alone reproduces the numbers reported in the article.

---

## Full rebuild from upstream sources

| Step | Command | Produces | Network |
|---|---|---|---|
| timm and Matbench records | `python scripts/00_fetch_data.py` | `data/timm_raw/`, `data/matbench_models.csv` | yes |
| NAS-Bench-201 and TabArena | `python scripts/extract_local.py nb201 --nb201-file <path>/NAS-Bench-201-v1_1-096897.pth`; `python scripts/extract_local.py tabrepo` | `data/nb201_accuracy.csv`, TabArena metrics | yes (large files) |
| Factorial grids | `python scripts/13_factorial_grids.py` | `data/grid_factorial.csv` | yes (pinned commits below) |
| YAHPO, MedMNIST, NATS-SSS | not scripted --- see the note below | `data/field_hpo_yahpo.csv`, `data/field_medicine_medmnist_published.csv`, `data/field_vision_size_natssss.csv` | — |

**The three unscripted extracts.** These were flattened by hand from upstream
into this archive's common schema, and the flattening step was not kept as a
script. What the files are is nonetheless fully determined by their contents,
which is what a reader needs to check them against upstream:

| File | Rows | Schema | Upstream records |
|---|---|---|---|
| `field_hpo_yahpo.csv` | 14 400 | `field, frontier, item, quality, chance, cost_timetrain, cost_timepredict` | YAHPO Gym surrogate evaluations, one row per configuration, `frontier` naming the scenario and OpenML task (e.g. `yahpo|rbv2_svm|40981`) |
| `field_medicine_medmnist_published.csv` | 840 | `field, frontier, item, quality, chance, cost_pixels, cost_gmacs, cost_params_M` | published MedMNIST+ results, `frontier` naming dataset and protocol (e.g. `medmnist|blood|endToEnd`), `item` the model at its input resolution |
| `field_vision_size_natssss.csv` | 98 304 | `field, frontier, item, quality, chance, cost_flops_M, cost_params_M, cost_latency_s` | NATS-Bench SSS, `item` the channel-width string (e.g. `8:8:8:8:8`), one row per architecture per dataset |

`chance` is the chance-level accuracy the article divides by, and is a property
of the evaluation set rather than of the extract. Every other column is copied
from upstream without transformation, so any row can be checked against the
upstream record it names. Reconstructing the three extraction scripts is the
one open item in this archive; it would not change a number, because the
scripts below read these files and never the upstream APIs.

---

---

---

## Six things called `phi`

The archive uses the letter for six unrelated quantities. They are never the
same number and nothing but context separates them, so they are tabulated here:

| symbol | meaning | where |
|---|---|---|
| `varphi = 2 arctan sqrt(N_eff)` | the **channel angle** of the paper's `eq:angle`, in `(0, pi)`, `0` at the floor and `pi` at the pure power law | `135`, the paper's Prop. "In gauge-free coordinates both rungs are Lipschitz bounds" |
| `L = int sqrt(kappa) dx` | the **Fisher-Rao arc length**, unbounded above; earlier revisions wrote it `phi`, and the arc-depth identity `L = 2 sqrt(N_eff)/|sigma| + const` is about this one | `104`, `34`, `58`, `143` |
| `phi` | the **failure correlation** between two candidates on the same items, in `[0,1]`, printed by `select_model.py` and tabulated as `median_phi` | `145`, `examples/model_selection` |
| `phi = eps_inf_quoted / eps_inf` | the **floor-misquote factor** of the accuracy-only sections, in `(0, inf)`, at `1` when the floor is quoted correctly | `supplement.tex`, `theory_supplement.tex` |
| `phi(u) = u^beta` | the **Bernstein inner function** of the subordination argument, composed with the outer clock | `theory_supplement.tex`, proof of Prop. "Spectral breadth" |
| `phi = alpha_quoted / alpha` | the **exponent-misquote factor** of the borrowed-exponent bound, in `(0, inf)`, at `1` when the exponent is right | `theory_supplement.tex`, proof of "the bound on a misquoted exponent" |

The paper and both appendices write `varphi` for the first and `L` for the
second; both appendices write `phi` for the fourth, the technical report also
for the fifth and the sixth, and the failure correlation never appears in them.

---

---

---

## Releasing to GitHub and Zenodo

The archive is laid out as a stand-alone repository: its root is the repository
root, so `data/`, `scripts/`, `results/`, `figures/` and `examples/` sit
directly under it, as in the predecessor archive
[`AB-Article/AI-selection`](https://github.com/AB-Article/AI-selection).

**Before releasing**

1. Rebuild the three documents and refresh the frozen cross-references:
   `sh ../tools/freeze-labels.sh` from the source directory one level up.
2. `./run_all.sh` from the archive root, then `git diff --stat results figures`.
   The diff is the release's own regression test: anything that moves must be
   explained, and nothing should move that the changelog does not mention.
3. `./run_all.sh guards`. All six must read `ok`. Four of them need the
   article's `.tex` sources beside the archive and will report `skipped` in a
   clone that does not have them, which is the expected state for a reader.
4. Check that no file exceeds 50 MB (GitHub warns) and none exceeds 100 MB
   (GitHub refuses): `find . -type f -size +50M`. The largest file in this
   release is `data/nb201_hwnas.csv` at 13 MB, so no Git LFS is needed.

**GitHub**

```bash
git init -b main
git add -A
git commit -m "Reproducibility archive for <article title>"
git remote add origin git@github.com:AB-Article/<repo>.git
git push -u origin main
git tag -a v2.0.0 -m "Submission"   # the tag Zenodo archives
git push origin v2.0.0
```

**Zenodo**

Zenodo mints a DOI from a GitHub *release*, not from a push, and it reads
`.zenodo.json` from the tagged tree for its metadata. So:

1. Enable the repository in Zenodo (`zenodo.org/account/settings/github/`)
   **before** creating the release; Zenodo only sees releases made after the
   switch is on.
2. Create a GitHub release from the tag. Zenodo archives the tarball and mints
   both a version DOI and a concept DOI that always resolves to the newest
   version.
3. `.zenodo.json` already declares the title, description, creator, keywords,
   licence and the related identifiers - `isNewVersionOf` the predecessor
   archive, `isSupplementTo` the predecessor preprint, and `isSupplementedBy`
   the maintained `nn-rank` implementation. Check them on the Zenodo record
   rather than assuming.
4. Put the **version** DOI in four places and keep them consistent:

   | file | placeholder to replace |
   |---|---|
   | `README.md`, "Archive DOI" | `10.5281/zenodo.0000000` |
   | `CITATION.cff`, `identifiers` | `10.5281/zenodo.0000000` |
   | `../refs.bib`, `@string{ignatovZenodoID}` | `XXXXXXX` (one edit feeds the DOI and the URL) |
   | the article's data-availability statement | `[TO BE COMPLETED BEFORE SUBMISSION]` in `../cover_letter.txt` |

   `grep -rn "zenodo.0000000\|XXXXXXX\|TO BE COMPLETED"` from the source
   directory finds all four.
5. Re-run `./run_all.sh guards` after editing those files, since the guards
   read them.

**What is deliberately not in the repository**

The article's LaTeX sources, its PDFs and the cover letter. The archive is the
data, the code and the results; the four guards that audit the archive against
the article are author tools and skip cleanly without them.

---

---

## Licence

Code is Apache-2.0 (`LICENSE`). Data compilations are CC BY 4.0 (`LICENSE-DATA`). The
underlying upstream benchmarks remain under their own licences; this archive
redistributes measurements drawn from public leaderboards, published tables and
documentation, with sources given above.

## Contact

Dmitry Ignatov — Computer Vision Lab, CAIDAS & IFI, University of Würzburg
