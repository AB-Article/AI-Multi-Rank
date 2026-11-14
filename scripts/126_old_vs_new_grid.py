#!/usr/bin/env python3
"""126_old_vs_new_grid.py -- does the bandwidth-free reading beat the jet reading?

A like-for-like audit, in the manner of script 123.  Same 117 frontiers, same
data, nothing refitted.  The incumbent column is the archive's own jet
(h = 0.40*span, 9-point grid) and its fitted three-parameter chart; the revised
column is Prop. 'grid' / 'floorbracket' / 'grainprice' / 'grainminimax', which
consume measured accuracies only.

The verdicts are not uniform, and that is the point of running it: two of the
previous revision's claims do not survive a bandwidth-free reading and are
recorded here as withdrawn.

Inputs : results/table_grid_summary.csv   (script 125)
         results/table_grid_frontiers.csv (script 125)
         results/table_grid_laws.csv      (script 124)
         results/table_elbow_summary.csv  (script 122)
Output : results/table_old_vs_new_grid.csv
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
from _loaders import save                                       # noqa: E402


def main():
    g = pd.read_csv(os.path.join(RES, "table_grid_summary.csv")).iloc[0]
    f = pd.read_csv(os.path.join(RES, "table_grid_frontiers.csv"))
    laws = pd.read_csv(os.path.join(RES, "table_grid_laws.csv"))
    elb = pd.read_csv(os.path.join(RES, "table_elbow_summary.csv")).iloc[0]
    nlaw, nfail = len(laws), int((laws.passed == 0).sum())

    rows = [
        dict(axis="1  UNFORCED CHOICES in the test path",
             previous="a kernel bandwidth h = 0.40*span, a 9-point interior grid, "
                      "and a fitted three-parameter chart for the floor",
             revised="none: every test is a determinant of measured accuracies "
                     "(Prop. grid)",
             verdict="BETTER",
             note=f"{nlaw} analytic checks, {nfail} failures (script 124)"),
        dict(axis="2  ORDER and ACCURACIES for a divisibility verdict",
             previous="third order, 4 accuracies, no floor (Prop. divdiff); or second "
                      "order through an estimated jet (Prop. wallrefutes)",
             revised="second order, 3 accuracies and a floor, exact and bandwidth-free "
                     "(eq. dmoments)",
             verdict="BETTER",
             note="(d0 d2 - d1^2)/h^2 -> kappa*DeltaS - s^2: the wall in finite differences"),
        dict(axis="3  AN EXTRA RUNG, on a compute axis",
             previous="none; the second rung existed only on a count axis "
                      "(6 all-fail rates, 10^8 items)",
             revised="the 5-accuracy minor of the sampled depth",
             verdict="BETTER",
             note="refutes 292/300 random indivisible frontiers against 232/300 for the "
                  "3-accuracy minor (script 124)"),
        dict(axis="4  THE FLOOR",
             previous="a fitted three-parameter chart, or Aitken's Delta^2 used as a "
                      "point estimate",
             revised="two sharp upper bounds from three accuracies: eps_A (axiom) and "
                     "eps_W (composition law), each exact on its own extremal chart",
             verdict="BETTER",
             note=f"and it refutes the incumbent floors: the fitted floor exceeds eps_A, "
                  f"which no completely monotone frontier may do, on "
                  f"{int(g.fitted_above_eps_A)}/{int(g.n_floor_room)} frontiers "
                  f"(against 53/92 for the weaker monofloor bound)"),
        dict(axis="5  THE CORPUS DIVISIBILITY VERDICT",
             previous="settled: L < pi certifies two atoms on 55/55 single-resource "
                      "frontiers, and a bounded spectrum is never divisible",
             revised=f"NOT reproduced without a bandwidth: alpha recovered from wide "
                     f"triples has median IQR/median {g.median_alpha_spread:.2f} along "
                     f"the frontier, and {int(g.rank1_at_10pct)}/{int(g.n_frontiers)} "
                     f"frontiers are two-atom at a 10% tolerance",
             verdict="WITHDRAWN",
             note="Prop. unbounded stands as a theorem; its application to this corpus "
                  "does not. The arc certificate is a plug-in reading, and the question "
                  "the Scope section posed -- whether it survives a bandwidth-free "
                  "reading -- is now answered: it does not, at published precision"),
        dict(axis="6  POWER of the floor test on real windows",
             previous=f"the jet's refutation band exceeds the count-axis bracket on "
                      f"{int(elb.band_gt_020)}/{int(elb.n_frontiers)} and "
                      f"{int(elb.band_gt_0094)}/{int(elb.n_frontiers)} frontiers",
             revised=f"the bandwidth-free room ln(eps_A/eps_W) exceeds it on "
                     f"{int(g.decided_by_bracket_020)}/{int(g.n_floor_room)} and "
                     f"{int(g.decided_by_bracket_0094)}/{int(g.n_floor_room)}; median "
                     f"room {g.median_floor_room_nats:.3f} nats, IQR "
                     f"[{g.floor_room_q25:.3f}, {g.floor_room_q75:.3f}]",
             verdict="DEFLATED",
             note="the published windows sit deep, i.e. right of the elbow, which is "
                  "exactly where Prop. elbow says the test has no power; the earlier "
                  "figure was bandwidth-dependent and is superseded"),
        dict(axis="7  DESIGN REQUIREMENT",
             previous="five to six operating points (Thm. covering)",
             revised="five to six points on a GEOMETRIC grid, and exact tests read on "
                     "WIDE triples",
             verdict="BETTER",
             note=f"narrow consecutive triples pass ln-eps convexity on "
                  f"{100*g.triples_axiom_pass/g.triples_tested:.0f}% of "
                  f"{int(g.triples_tested)} cases against 83% for triples spanning two "
                  f"thirds of the window: the defect scales as h^2 and the noise does "
                  f"not, so narrow triples measure noise"),
        dict(axis="8  THE PRICE",
             previous="lambda = sqrt(kappa/t) with kappa from a kernel, or alpha/2 with "
                      "alpha fitted",
             revised="lambda = alpha/(1+t) with alpha exact from three accuracies "
                     "(Prop. grainprice), and a minimax form for an unknown grain",
             verdict="UNRESOLVED on this corpus",
             note=f"median |ln(lambda_grid/lambda_jet)| = "
                  f"{g.median_abs_log_price_ratio:.2f}; the two prices select the same "
                  f"point on {int(g.same_pick)}/{int(g.n_price)} frontiers and agree "
                  f"within the 0.05-nat band on {int(g.disagreements_inside_tie_band)}"
                  f"/{int(g.n_price)}. The closed form is exact only on a certified "
                  f"two-atom window, and by axis 5 this corpus has none, so the "
                  f"difference cannot be adjudicated here"),
        dict(axis="9  THE GRAIN",
             previous="an external input, silently defaulted to t = 1",
             revised="an interval, with minimax point 1/sqrt(t1 t2) and worst-case cost "
                     "ln cosh((1/4) ln(t2/t1)) (Prop. grainminimax)",
             verdict="BETTER",
             note="the archive's own implied-grain IQR [1.0, 7.3] costs 0.119 nats, more "
                  "than twice the 0.05-nat band inside which the recommendation is "
                  "constant: the grain has to be declared"),
        dict(axis="10  REGRESSION",
             previous="R^2 = 0.9897, median relative error 2.60% on 2025 withheld models",
             revised="unchanged -- no chart refitted, no datum touched, no jet recomputed; "
                     "scripts 124-126 add no fit",
             verdict="NONE",
             note="scripts 68/77/80/97/104/109/117/121/122 are untouched and their "
                  "outputs are bit-identical"),
        dict(axis="11  PARSIMONY of the test apparatus",
             previous="a jet, a kernel, a fitted chart, and four separate tests",
             revised="two sampled sequences: the error (axiom) and the depth "
                     "(composition law)",
             verdict="BETTER",
             note="every rung of both hierarchies is a Hankel minor of measured numbers"),
    ]
    df = pd.DataFrame(rows)
    save(df, "table_old_vs_new_grid.csv")

    print("=" * 78)
    print("DOES THE BANDWIDTH-FREE READING BEAT THE JET READING?")
    print("=" * 78)
    for r in rows:
        print(f"\n{r['axis']}")
        print(f"   previous : {r['previous']}")
        print(f"   revised  : {r['revised']}")
        print(f"   VERDICT  : {r['verdict']}")
        print(f"   note     : {r['note']}")
    v = df.verdict.value_counts()
    print("\n" + "-" * 78)
    print("  ".join(f"{k}: {int(n)}" for k, n in v.items()))
    print("Net: the apparatus is strictly cheaper and strictly more exact; two claims of")
    print("the previous revision -- the corpus divisibility verdict and the power of the")
    print("floor test on published windows -- do not survive it and are withdrawn here.")
    print("Nothing that was fitted has moved.")


if __name__ == "__main__":
    main()
