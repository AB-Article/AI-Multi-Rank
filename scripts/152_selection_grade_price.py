#!/usr/bin/env python3
"""152 -- how sharp must the AUTOMATIC price be?  (Fix F20.)

The theory's price is not a free parameter: at the operating point
lambda* = s/sqrt(t N_eff) = sqrt(kappa/t), so once the grain is declared the
price is determined.  Nothing has to be set by hand.  What is uncertain is not
the price but its ESTIMATE, and F13 judged that estimate against the wrong
target: the precision kappa needs to be measured to a relative tolerance.

A selector needs something much weaker.  It needs the price's own interval to
be narrower than the interval of prices over which the WINNER DOES NOT CHANGE
-- a candidate's "reign".  Reigns are a property of the candidates in hand and
cost nothing to measure, so this is the design rule a deployment should check.

Measured here on every archived frontier with a public evaluation-set size:

  * the width of the derived price's own 90% interval, in doublings;
  * the median width of a candidate's reign, in doublings;
  * the ratio -- the factor by which the price must sharpen before it names a
    single candidate -- and what that costs by each of the two routes.

Because lambda* = sqrt(kappa/t), sd(lambda)/lambda = (1/2) sd(kappa)/kappa and
sd(kappa) = sqrt(6) sd(l)/h^2, so a factor F of sharpening is sqrt(F) in grid
width or F^2 in items.  The sweep is the cheap route by a wide margin.

Writes results/table_selection_grade.csv and
results/table_selection_grade_summary.csv.
"""
import math

import numpy as np

import _pipeline as P

pc = P.pc
LN2 = math.log(2.0)


def main():
    rows = []
    for f in P.frontiers():
        xs, es = list(f["x"]), list(f["e"])
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in o]; es = [es[i] for i in o]
        ls = [math.log(e) for e in es]
        S = [-l for l in ls]
        n = len(xs)
        if n < 5:
            continue
        ni = P.eval_items(f["name"])
        if not ni:
            continue
        sds = [pc.sd_of_l(min(max(1 - e, 1e-6), 1 - 1e-9), ni) for e in es]
        plane = pc.read_plane(xs, ls, 1.0)
        lam = plane.get("price")
        if not lam or lam <= 0:
            continue
        med, lo, hi, psd = pc.price_uncertainty(xs, ls, sds, 1.0,
                                                draws=300, seed=3)
        if not med or not (hi > lo > 0):
            continue
        steps = [b - a for a, b in zip(xs, xs[1:])]
        h = sum(steps) / len(steps)
        models = [{"name": i, "eps": es[i], "x": {"r": xs[i]}}
                  for i in range(n)]
        d = pc.design_for_selection(models, {"r": lam}, "r", lo, hi,
                                    h_available=h)
        # for reference, the kappa-grade rule of F13 on the same frontier
        j = n // 2
        s_mid = pc.local_slope(xs[j - 1:j + 2], ls[j - 1:j + 2])
        k13 = pc.design_calibration(min(max(1 - es[j], 1e-6), 1 - 1e-9), ni,
                                    s_mid, t=1.0, K=5, h_available=h)
        rows.append({
            "corpus": f["corpus"], "frontier": f["name"], "n_candidates": n,
            "n_test": ni, "h_used": h, "lambda_hat": lam,
            "price_interval_doublings": d["price_interval_doublings"],
            "median_reign_doublings": d["median_reign_doublings"],
            "n_reigns_in_interval": d["n_reigns_in_interval"],
            "verdict_selection": d["verdict"],
            "sharpen_by": d.get("sharpen_by"),
            "widen_sweep_by": d.get("widen_grid_by"),
            "more_items_by": d.get("more_items_by"),
            "h_needed_selection": d.get("h_needed"),
            "span_needed_selection": d.get("span_needed"),
            "h_needed_price_F13": k13.get("h_min_price"),
            "h_needed_neff_F13": k13.get("h_min"),
            "span_needed_neff_F13": k13.get("resource_ratio"),
        })
    df = P.save(rows, "table_selection_grade.csv")
    ok = df[df.verdict_selection == "FEASIBLE"]
    bad = df.dropna(subset=["sharpen_by"])
    summ = [{
        "n_frontiers": len(df),
        "already_sharp_enough": len(ok),
        "median_price_interval_doublings": float(
            df.price_interval_doublings.median()),
        "median_reign_doublings": float(df.median_reign_doublings.median()),
        "median_sharpen_by": float(bad.sharpen_by.median()),
        "median_widen_sweep_by": float(bad.widen_sweep_by.median()),
        "median_more_items_by": float(bad.more_items_by.median()),
        "median_h_used": float(df.h_used.median()),
        "median_h_needed_selection": float(bad.h_needed_selection.median()),
        "median_span_needed_selection": float(bad.span_needed_selection.median()),
        "median_h_needed_neff_F13": float(df.h_needed_neff_F13.median()),
        "median_span_needed_neff_F13": float(df.span_needed_neff_F13.median()),
    }]
    sdf = P.save(summ, "table_selection_grade_summary.csv")
    r = summ[0]
    print()
    print(f"  on {len(df)} frontiers with a public item count:")
    print(f"    derived price's own 90% interval : median"
          f" {r['median_price_interval_doublings']:.2f} doublings")
    print(f"    a candidate's reign over prices  : median"
          f" {r['median_reign_doublings']:.2f} doublings")
    print(f"    => sharpen the price by a median factor"
          f" {r['median_sharpen_by']:.1f} before it names one candidate")
    print(f"       ({r['already_sharp_enough']} of {len(df)} frontiers are"
          f" already sharp enough)")
    print()
    print(f"    the two routes, from lambda* = sqrt(kappa/t):")
    print(f"      widen the sweep by x{r['median_widen_sweep_by']:.1f}"
          f"  (h from {r['median_h_used']:.3f} to"
          f" {r['median_h_needed_selection']:.3f}, a span of"
          f" {r['median_span_needed_selection']:,.0f}x over five points)")
    print(f"      or buy x{r['median_more_items_by']:,.0f} more items")
    print()
    print(f"    for contrast, F13's kappa-grade rule asks h >="
          f" {r['median_h_needed_neff_F13']:.2f}, a span of"
          f" {r['median_span_needed_neff_F13']:,.0f}x -- a far stricter target,")
    print(f"    because measuring the channel count is a far stricter demand"
          f" than naming a winner.")
    print()
    print("  So the automatic price is not blocked by anything in the theory.")
    print("  It is blocked by a calibration sweep that nobody designs, and the")
    print("  sweep it needs is a modest one: one model, five budgets, spanning")
    print("  the order of ten in resource rather than the order of a thousand.")


if __name__ == "__main__":
    main()
