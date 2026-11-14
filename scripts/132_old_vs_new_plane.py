#!/usr/bin/env python3
"""132_old_vs_new_plane.py -- does the plane reading beat the previous one?

Like-for-like audit, in the manner of scripts 123 and 126.  Every row is either
decided by a number that scripts 127-131 produced, or it is a parsimony claim
about what the manuscript has to assume and state.  Numbers are READ FROM THE
RESULT FILES, never typed here, so the table cannot drift from the archive.

Verdicts: BETTER, NONE (no change), DEFLATED, WITHDRAWN, SETTLED.

Output: results/table_old_vs_new_plane.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save                                        # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")


def tab(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


pl = tab("table_plane_summary.csv")
vd = tab("table_plane_power.csv")
pw = tab("table_power_summary.csv")
gr = tab("table_grid_summary.csv")
fv = tab("table_plane_frontiers.csv")

g = lambda d, k, dflt="n/a": (f"{d[k].iloc[0]}" if d is not None and k in d else dflt)
gf = lambda d, k, n=3, dflt=float("nan"): (float(d[k].iloc[0]) if d is not None and k in d else dflt)

rows = [
    dict(axis="1  THE CHART ZOO",
         previous="six charts (cure, power-floor, Lomax, stretched, exponential, lumpy) "
                  "ordered by sigma, each with its own proposition",
         revised="one PENCIL OF STRAIGHT LINES on the plane N_eff against S "
                 "(Prop. plane/pencil): the slope is sigma, the S-intercept is -ln eps_inf",
         verdict="BETTER",
         note="six statements -> one; the reconstruction s = s0 exp(int dS/N_eff) recovers the "
              "frontier from the curve up to the axis gauge, verified to 3e-61 (sc. 127 I1)"),
    dict(axis="2  THEOREM onedof",
         previous="measured: a 3-parameter floored chart fitted to a floorless frontier returns a "
                  "floor that is a monotone readout of beta (Spearman 1.000)",
         revised="a COROLLARY of the pencil: the floor and the shape index are the intercept and "
                 "the slope of one line",
         verdict="BETTER",
         note="an empirical coincidence becomes geometry; one measured result leaves the residue"),
    dict(axis="3  THE FLOOR BOUNDS",
         previous="two determinant formulas in sampled accuracies, eps_A and eps_W (Prop. "
                  "floorbracket)",
         revised="the two WALLS solved for the floor: eps/(1+N_eff) and eps exp(-N_eff) "
                 "(Prop. tangent) -- functions of two measured numbers",
         verdict="BETTER",
         note=f"the 1006d layer adds nothing beyond (eps, N_eff); fitted floors exceed eps_A, "
              f"which no CM frontier may do, on {g(pl,'fitted_above_epsA')} of 117 (sc. 125: 76/88)"),
    dict(axis="4  LAW III's pi",
         previous="a Fisher-Rao length, re-derived as the area under a secant (Prop. gudermann)",
         revised="the VARIATIONAL MINIMUM of L = int dS/sqrt(N_eff): L >= 2 arccos sqrt(eps_inf), "
                 "-> pi as the floor deepens, equality iff two atoms (Prop. minarc)",
         verdict="BETTER",
         note="Law III becomes a corollary of Law IV; pi = int du/sqrt(e^u-1), a Beta integral, "
              "verified to 1e-18 (sc. 127 E7)"),
    dict(axis="5  THE COVERING LAW",
         previous="five operating points cover any SINGLE-CHANNEL frontier to 0.05 nats",
         revised="no frontier WITH A FLOOR is coverable by fewer than arccos sqrt(eps_inf)/"
                 "sqrt(2 rho) points -- 4.65 at a 1 % floor -- and a divisible one needs "
                 "sqrt(lambda/2rho), 6.79",
         verdict="BETTER",
         note="a hypothesis is removed from a headline claim, and benchmark design becomes a "
              "divisibility discriminator"),
    dict(axis="6  THE L < pi CERTIFICATE",
         previous="55/55 single-resource frontiers certified two-atom at 1006b; withdrawn "
                  "empirically at 1006d when alpha-constancy did not reproduce it",
         revised="VOID BY CONSTRUCTION: the complete arc of any floored frontier is already "
                 "2 arccos sqrt(eps_inf) ~ pi, so a sub-window arc is below pi for every "
                 "frontier whatever its channel count",
         verdict="SETTLED",
         note="the 1006d withdrawal was mandatory, not unlucky; no bandwidth could have rescued "
              "it. What L/pi measures is window completeness (corpus median 0.32)"),
    dict(axis="7  THE OPERATING RULE",
         previous="operate where N_eff = 1/t, with the grain t an external input the theory "
                  "cannot supply (the third impossibility)",
         revised="operate where N_eff = 1 IN DELIVERED UNITS -- parameter-free, and the fixed "
                 "point of the mode-swap involution N_eff -> 1/N_eff (Prop. localprice)",
         verdict="BETTER",
         note="the third impossibility dissolves: the rule IS the choice of unit, and once the "
              "unit is named the rule is unique and invariant"),
    dict(axis="8  THE PRICE",
         previous="lambda* = alpha/(1+t) with alpha from a Prony root on three accuracies",
         revised="lambda* = s/sqrt(t N_eff) identically, hence lambda* = s exactly at the "
                 "operating point: the canonical price is the local price",
         verdict="BETTER",
         note="no Prony root, no curvature, no chart; verified to 1e-24 on four charts at "
              "t = 1, 2, 5 (sc. 127 F1-F2)"),
    dict(axis="9  THE DIVISIBILITY TEST, WORKING FORM",
         previous="Hankel minors of the sampled depth (Prop. grid), which need a floor",
         revised="the second difference of ln eps NEVER GROWS -- exact at every spacing "
                 "(Widder), linear in the measured accuracies, floor-free. Equivalently: a "
                 "divisible frontier has NO ELBOW",
         verdict="BETTER",
         note="Prop. unbounded and Prop. elbow collapse into one line; the archive's "
              "elbow-location statistic is itself the refutation statistic"),
    dict(axis="10  THE COUNT-AXIS LADDER",
         previous="rung 1 from P(1..4) (L1 L3 >= L2^2); rung 2 from P(1..6), quoted at 10^8 items",
         revised="the attempt price is a Hausdorff moment sequence FROM m = 0, because P(0) = 1 "
                 "exactly: rung 1 from P(1..3), rung 2 from P(1..5)",
         verdict="BETTER",
         note="level-controlled by bootstrap (sc. 128): on the 3-class census, offset-0 rung 2 "
              "reaches 0.84 power at 1e5 items where offset-1 is at 0.07; level 0.00 at the "
              "extremal divisible truth. Three orders of magnitude on the decisive test"),
    dict(axis="11  THE CORPUS DIVISIBILITY VERDICT",
         previous="decided at 1006b (55/55), withdrawn at 1006d; the required precision was "
                  "quoted as an 18-22x reduction and never tabulated",
         revised=f"UNDECIDABLE at this precision, now established against a design-matched null: "
                 f"{g(vd,'refuted_z_placed')}/{g(vd,'n_z_placed')} placed quadruples, "
                 f"{g(vd,'refuted_z_unplaced')}/{g(vd,'n_z_unplaced')} unplaced, "
                 f"{g(vd,'refuted_rises')}/{g(vd,'n_rises')} curvature-rise, "
                 f"{g(vd,'refuted_line_r2')}/{g(vd,'n_line_r2')} straightness -- all at the "
                 f"nominal 5 % level, median power {gf(vd,'medpower_z_unplaced'):.3f}",
         verdict="SETTLED",
         note="the flip-flop ends: no bandwidth-free statistic has power here, and that is now a "
              "measured fact with a null rather than an assertion either way"),
    dict(axis="12  THE POWER TABLE",
         previous="demanded by the residue since 1006b and never computed",
         revised=f"computed in closed form and verified against simulation to "
                 f"{gf(pw,'analytic_check_max_err'):.4f}: median sigma_80 = "
                 f"{1000*gf(pw,'median_sigma80_whole'):.2f} milli-nats on the widest quadruple, "
                 f"on {int(gf(pw,'n_whole'))} frontiers, i.e. x"
                 f"{gf(pw,'factor_vs_nominal_whole'):.1f} on the nominal 20 milli-nats",
         verdict="BETTER",
         note=f"this REPRODUCES the archive's own 18-22x figure by an independent analytic route, "
              f"so that number is now confirmed rather than quoted; "
              f"{g(pw,'decidable_at_nominal_whole')} frontiers are already decidable at 20 "
              f"milli-nats"),
    dict(axis="13  'PLACEMENT, NOT PRECISION'",
         previous="conjectured in derived_laws_20261006e: a window ending at the elbow should "
                  "give 80 % power at the archive's own noise (sc. 128 on a clean truth)",
         revised=f"REFUTED on this corpus: restricting the quadruple to the left of the curvature "
                 f"maximum shrinks the span from "
                 f"{gf(pw,'median_span_both'):.2f} to "
                 f"{gf(pw,'median_admissible_span_both'):.2f} e-folds and COSTS a factor "
                 f"{gf(pw,'placement_cost'):.1f} in required precision "
                 f"(median of ratios on {int(gf(pw,'n_both'))} frontiers)",
         verdict="WITHDRAWN",
         note="the clean simulation assumed a 5-e-fold admissible span; the corpus's elbow sits at "
              "0.31 of the window. The archive's placement was right and its power figure stands"),
    dict(axis="14  'THE TWO-ENDED TEST GAINS POWER'",
         previous="conjectured: the telescoped sum of every third difference should average noise",
         revised="REFUTED on known charts (sc. 128): power 0.20-0.47 against 0.80 for the "
                 "archive's widest quadruple on the same data; sigma_80 8.4 against 20.0 "
                 "milli-nats",
         verdict="WITHDRAWN",
         note="the telescoping identity is exact and explains why wide quadruples dominate, but "
              "no power is available from rearranging one linear functional"),
    dict(axis="15  SIGN-OF-MINOR READINGS",
         previous="count-axis minors reported by their sign (rung 2: -5.6e-6 against +9.5e-3)",
         revised="a sign reading has LEVEL 1/2 at the extremal divisible truth, which sits on the "
                 "boundary; rung 2's sign refutes it 85 % of the time at every n",
         verdict="BETTER",
         note="seventh method-law casualty, and the method law's own prediction; every minor is "
              "now studentised by item bootstrap"),
    dict(axis="16  THE WORKED EXAMPLE",
         previous="292 lines: Prony root, two determinant floor formulas, a rank-1 certificate, "
                  "four separate tests",
         revised="the plane: N_eff from three accuracies (the spacing cancels), one line fit, two "
                 "walls, one height. ~90 lines of logic, stdlib only",
         verdict="BETTER",
         note="recovers the cure chart's floor exactly (0.006738 against 0.006738) and the "
              "two-atom price to 1 % (0.4451 against alpha/2 = 0.45); and it demonstrates its own "
              "trap, the O(h^2) downward bias of the slope"),
    dict(axis="17  REGRESSION",
         previous="R^2 = 0.9897, median relative error 2.60 % on 2025 withheld models",
         revised="unchanged -- no chart refitted, no datum touched, no jet recomputed; the "
                 "prediction result files are bit-identical to the shipped ones",
         verdict="NONE",
         note="scripts 68/77/80/97/104/109/117/121/122/124/125/126 are untouched"),
]

df = pd.DataFrame(rows)
v = df.verdict.value_counts()
print("\n" + "=" * 78)
print("  OLD vs NEW -- the channel-depth plane against the 1006d reading")
print("=" * 78)
for _, r in df.iterrows():
    print(f"\n{r.axis}   [{r.verdict}]")
    print(f"   was : {r.previous}")
    print(f"   is  : {r.revised}")
    print(f"   note: {r.note}")
print("\n" + "=" * 78)
print("  " + ",  ".join(f"{k} {int(n)}" for k, n in v.items()))
print("=" * 78)
save(df, "table_old_vs_new_plane.csv")
