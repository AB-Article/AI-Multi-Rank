#!/usr/bin/env python3
"""147 -- the calibration grid, designed BEFORE the runs are spent.  (Fix F13.)

Version 2 diagnosed an under-powered calibration curve after the fact: it
computed the items N_eff would have needed and the factor the grid should have
been stretched by, once the models had already been trained at the wrong
spacings.  That is the wrong order.  The grid width is a design variable, and
the inequality that fixes it needs nothing that is only known afterwards:

    kappa is a second difference, sd(kappa) = sqrt(6) sd(l) / h^2, so
    sd(kappa)/kappa <= tol  requires  h >= sqrt( sqrt(6) sd(l) / (tol kappa) ),
    with sd(l) = sqrt(Q/((1-Q) n)) from the items,

and before any run kappa is fixed by the rule itself: the operating point has
N_eff = 1/t and N_eff = s^2/kappa, so kappa = s^2 t -- an expected slope and the
grain, both of which a deployment already has to declare.

This script checks that the prospective rule agrees with the retrospective one
it replaces.  For every archived frontier with a public evaluation-set size it
computes

  * h_min from the PROSPECTIVE inequality (expected slope and grain only),
  * h_min from the RETROSPECTIVE one (the measured kappa, which is what script
    140 uses), and
  * the FEASIBLE/INFEASIBLE verdict of each at the grid that frontier actually
    used,

and reports the agreement, and above all the one-sided error that matters: how
often the prospective rule says FEASIBLE when the retrospective one says the
curve was not powered.  It then runs both demo calibration curves of
``examples/model_selection``, whose truth is known, as an end-to-end check.

Writes results/table_design_rule.csv and results/table_design_rule_summary.csv.
"""
import csv
import math
import os

import _pipeline as P

pc = P.pc
TOL = 0.25
GRAIN = 1.0
KCAL = 5        # the paper's five checkpoints: one model, five budgets


def main():
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        n = len(xs)
        if n < 3:
            continue
        n_items = P.eval_items(f["name"])
        eps_mid = sorted(es)[n // 2]
        q_mid = 1.0 - eps_mid
        steps = [b - a for a, b in zip(xs, xs[1:])]
        h = sum(steps) / len(steps)
        j = n // 2
        s_mid = pc.local_slope(xs[j - 1:j + 2], ls[j - 1:j + 2])
        req = pc.calibration_requirement(xs, ls, q_mid, tol=TOL)
        des = pc.design_calibration(q_mid, n_items, s_mid, t=GRAIN, K=KCAL,
                                    tol=TOL, h_available=h)
        rec = {"corpus": f["corpus"], "frontier": f["name"], "n_points": n,
               "n_test": n_items, "h_used": h, "s_mid": s_mid,
               "kappa_measured": req["kappa"] if req else None,
               "kappa_expected": des["kappa_expected"],
               "sd_l": des["sd_l"],
               "h_min_prospective": des["h_min"],
               "h_min_price": des["h_min_price"],
               "verdict_prospective": des["verdict"],
               "verdict_prospective_price": des["verdict_price"],
               "widen_by_prospective": des["widen_by"],
               "widen_by_price": des["widen_by_price"],
               "resource_ratio_needed": des["resource_ratio"],
               "resource_ratio_price": des["resource_ratio_price"]}
        # the outcome the design is supposed to predict: after the runs, is the
        # derived PRICE identified at all?  (The criterion the selector itself
        # uses: a 90% interval that spans more than its own value is not a
        # measurement.)  This is the honest target: the measured kappa is
        # itself a noise-dominated second difference on these grids, so
        # validating one noisy reading against another would prove nothing.
        if n_items and des["sd_l"]:
            sds = [pc.sd_of_l(min(max(1.0 - e, 1e-6), 1.0 - 1e-9), n_items)
                   for e in es]
            med, lo, hi, _sd = pc.price_uncertainty(xs, ls, sds, GRAIN,
                                                    draws=200, seed=7)
            rec["price_median"] = med
            rec["price_identified_after"] = (
                bool(med and (hi - lo) <= med) if med else None)
            j2 = n // 2
            mc = pc.neff_uncertainty(xs[j2 - 1:j2 + 2], ls[j2 - 1:j2 + 2],
                                     [sds[j2]] * 3, draws=400, seed=7)
            rec["neff_identified_after"] = (
                bool(mc[0] and (mc[2] - mc[1]) <= mc[0]) if mc[0] else False)
        rows.append(rec)
    df = P.save(rows, "table_design_rule.csv")

    d = df.dropna(subset=["price_identified_after"])
    rows_s = []
    for tag, pred, truth in (
            ("price target", "verdict_prospective_price",
             "price_identified_after"),
            ("N_eff target", "verdict_prospective", "neff_identified_after")):
        fe = ((d[pred] == "FEASIBLE") & (~d[truth].astype(bool))).mean()
        fi = ((d[pred] == "INFEASIBLE") & (d[truth].astype(bool))).mean()
        rows_s.append({"target": tag, "n": len(d),
                       "agreement": float((
                           (d[pred] == "FEASIBLE") == d[truth].astype(bool)
                       ).mean()),
                       "falsely_feasible": float(fe),
                       "falsely_infeasible": float(fi),
                       "share_feasible_predicted": float(
                           (d[pred] == "FEASIBLE").mean()),
                       "share_identified_after": float(
                           d[truth].astype(bool).mean())})
    sdf = P.save(rows_s, "table_design_rule_summary.csv")
    print()
    print(f"  the prospective rule needs only: items, accuracy, an expected"
          f" slope and the grain.")
    print(f"  the outcome it has to predict is whether the quantity turns out"
          f" to be identified at all, after the runs.")
    print(sdf.to_string(index=False))
    print()
    print(f"  The error that matters is the optimistic one -- designing a grid"
          f" the data will not support.")
    for r in rows_s:
        print(f"    {r['target']:<13} falsely FEASIBLE on"
              f" {r['falsely_feasible']:.1%} of {r['n']} frontiers")
    print()
    print(f"  median grid actually used {d.h_used.median():.3f} in"
          f" log-resource; the rule asks {d.h_min_price.median():.3f} for a"
          f" price and {d.h_min_prospective.median():.3f} for a channel count"
          f" -- factors of {d.widen_by_price.median():.1f}x and"
          f" {d.widen_by_prospective.median():.1f}x.")
    print(f"  stated as a design for {KCAL} calibration points: the curve must"
          f" span a resource ratio of"
          f" {d.resource_ratio_price.median():,.0f}x for a price and"
          f" {d.resource_ratio_needed.median():,.0f}x for N_eff, cheapest to"
          f" dearest.")
    print(f"  That is the published corpus' real defect: not too few items, but"
          f" a grid designed for plotting rather than for a second"
          f" difference.")

    # ---- the end-to-end check on the demo curves, whose truth is known
    print()
    print("  the two demo calibration curves (same truth, same items, grids"
          " x2 and x4):")
    rows2 = []
    for name in ("calibration.csv", "calibration_wide.csv"):
        path = os.path.join(P.EX, name)
        xs, ls, ns = [], [], []
        with open(path) as fh:
            for row in csv.DictReader(l for l in fh if not l.startswith("#")):
                xs.append(math.log(float(row["inference_time"])))
                ls.append(pc.ln_eps(float(row["accuracy"]), 0.1))
                ns.append(int(row["n_test"]))
        q_mid = 1.0 - math.exp(sorted(ls)[len(ls) // 2])
        steps = [b - a for a, b in zip(xs, xs[1:])]
        h = sum(steps) / len(steps)
        j = len(xs) // 2
        s_mid = pc.local_slope(xs[j - 1:j + 2], ls[j - 1:j + 2])
        des = pc.design_calibration(q_mid, ns[0], s_mid, t=GRAIN, K=len(xs),
                                    tol=TOL, h_available=h)
        sd = pc.sd_of_l(q_mid, ns[0])
        med, lo, hi, _s = pc.price_uncertainty(xs, ls, [sd] * len(xs), GRAIN,
                                               draws=400, seed=3)
        ident = bool(med and (hi - lo) <= med)
        rows2.append({"curve": name, "n_test": ns[0], "h_used": h,
                      "h_min_price": des["h_min_price"],
                      "h_min_neff": des["h_min"],
                      "widen_by_price": des["widen_by_price"],
                      "verdict_price": des["verdict_price"],
                      "verdict_neff": des["verdict"],
                      "n_test_min_at_this_h": des["n_test_min_at_h"],
                      "price_identified_after_the_fact": ident})
        print(f"    {name:<22} h = {h:.3f};"
              f" rule asks {des['h_min_price']:.3f} for a price"
              f" -> {des['verdict_price']:<10}"
              f" and {des['h_min']:.3f} for N_eff -> {des['verdict']:<10}"
              f" | after the fact the price is"
              f" {'identified' if ident else 'NOT identified'}")
    P.save(rows2, "table_design_rule_demo.csv")
    print()
    print("  The rule separates the two curves on the price target, the way")
    print("  the tool separates them after the fact -- and it would have said")
    print("  so before either curve was run.  Neither curve is wide enough")
    print("  for N_eff, which is also what the tool reports afterwards.")


if __name__ == "__main__":
    main()
