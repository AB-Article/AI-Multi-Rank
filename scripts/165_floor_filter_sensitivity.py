#!/usr/bin/env python3
"""WHICH FRONTIERS COUNT, AND WHO DECIDES -- THE HEADLINE TABLE READ TWICE.
   (Paper, "Rules in use", Table 'namedrules'.)

Table 'namedrules' is measured on the frontiers where the canonical point is
ATTAINABLE: those carrying a usable floor with 2*eps_inf between the smallest
and the largest measured error.  Until now that test used eps_inf from the
three-parameter fit -- the same fitted floor that the paper's first
contribution argues is "not a measurement of the irreducible error".  Selecting
the evaluation sample with an estimator the paper discredits is a fair thing
for a referee to object to, so this script does the obvious control: it repeats
the funnel and the whole table with the BANDWIDTH-FREE three-point floor of
script 11 in its place, and reports what moves.

Two outputs.

  results/table_floor_filter_funnel.csv
      the funnel, estimator by estimator: single-resource frontiers built, those
      carrying a usable floor, those where the canonical point is attainable,
      and the overlap between the two attainable sets.

  results/table_rules_summary_3pt.csv
      Table 'namedrules' recomputed end to end with the three-point floor: the
      floor sets the canonical pick AND scores every rule's regret, so this is a
      genuine re-reading and not a re-filtering.

Nothing here is a new measurement; both floors are already in
table_rules_per_frontier.csv.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import selection_rules, save          # noqa: E402

sr = selection_rules()
FLOOR_LO, FLOOR_HI = 1e-3, 0.5


def usable(f):
    return bool(np.isfinite(f) and FLOOR_LO < f < FLOOR_HI)


def main():
    frs = sr.single_resource_frontiers()
    rows = []
    for fr in frs:
        field, name, costname, x, e = fr
        rec = sr.analyse(fr)
        f_fit = rec.get("eps_inf", np.nan)
        f_3pt = rec.get("eps_inf_3pt", np.nan)
        att_fit = bool(rec.get("attainable"))
        att_3pt = bool(usable(f_3pt) and e.min() <= 2 * f_3pt <= e.max())
        rows.append(dict(field=field, frontier=name, cost=costname, n_pareto=len(e),
                         eps_inf_fitted=f_fit, eps_inf_threepoint=f_3pt,
                         usable_fitted=bool(rec.get("floor")),
                         usable_threepoint=usable(f_3pt),
                         attainable_fitted=att_fit,
                         attainable_threepoint=att_3pt))
    D = pd.DataFrame(rows)

    funnel = pd.DataFrame([
        dict(stage="single-resource frontiers built", fitted_floor=len(D),
             three_point_floor=len(D)),
        dict(stage="carries a usable floor",
             fitted_floor=int(D.usable_fitted.sum()),
             three_point_floor=int(D.usable_threepoint.sum())),
        dict(stage="canonical point attainable",
             fitted_floor=int(D.attainable_fitted.sum()),
             three_point_floor=int(D.attainable_threepoint.sum())),
        dict(stage="attainable under BOTH estimators",
             fitted_floor=int((D.attainable_fitted & D.attainable_threepoint).sum()),
             three_point_floor=int((D.attainable_fitted & D.attainable_threepoint).sum())),
        dict(stage="attainable under EITHER estimator",
             fitted_floor=int((D.attainable_fitted | D.attainable_threepoint).sum()),
             three_point_floor=int((D.attainable_fitted | D.attainable_threepoint).sum())),
    ])
    save(funnel, "table_floor_filter_funnel.csv")
    save(D, "table_floor_filter_per_frontier.csv")

    # ---- the whole table again, scored by the three-point floor -------------
    recs = []
    for fr in frs:
        field, name, costname, x, e = fr
        f_3pt = sr.analyse(fr).get("eps_inf_3pt", np.nan)
        if not usable(f_3pt) or not (e.min() <= 2 * f_3pt <= e.max()):
            continue
        recs.append(sr.analyse(fr, floor_override=f_3pt))
    S3 = sr.summarise(pd.DataFrame(recs))
    S3.insert(1, "subset", "attainable, three-point floor")
    save(S3, "table_rules_summary_3pt.csv")

    fit = pd.read_csv(os.path.join(os.path.dirname(HERE), "results",
                                   "table_rules_summary.csv"))
    fit = fit[fit.subset == "all attainable"].set_index("rule")
    print(f"     funnel: {len(D)} built -> "
          f"{int(D.usable_fitted.sum())}/{int(D.usable_threepoint.sum())} with a floor "
          f"-> {int(D.attainable_fitted.sum())}/{int(D.attainable_threepoint.sum())} "
          f"attainable (fitted/three-point)")
    print(f"     the two attainable sets agree on "
          f"{int((D.attainable_fitted & D.attainable_threepoint).sum())} frontiers")
    print("     rule                   fitted -> three-point   (in band)")
    for r in sr.RULES:
        if r in fit.index and (S3.rule == r).any():
            a = fit.loc[r]; b = S3[S3.rule == r].iloc[0]
            print(f"       {r:20s} {a.frac_inband:5.0%} -> {b.frac_inband:5.0%}"
                  f"      median regret {a.median_regret:.4f} -> {b.median_regret:.4f}")


if __name__ == "__main__":
    main()
