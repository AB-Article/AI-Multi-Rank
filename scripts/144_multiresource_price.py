#!/usr/bin/env python3
"""144 -- one price per axis, on the multiresource frontiers.  (Fix F5.)

The shipped selector of version 1 took exactly one resource column, so a
deployment with latency AND memory AND energy AND money had no operational
route at all: the theory's multiresource statements (the power mean, the moment
map, the canonical point eps* = (1 + n^(1/theta)) eps_inf) live in the technical
report and need an interaction index fitted per pair.

This script does the operational part instead, which needs no interaction fit:
on each archived multi-resource block it derives ONE price per axis from that
axis's own marginal curve, composes the scalarization sigma = S - sum_i
lambda_i x_i, and reports

  * the derived price of every axis, with the share of total cost it carries,
  * the decision, and whether it changes when an axis is dropped -- which is
    the only honest test of whether that axis was worth pricing,
  * the canonical depth ln(1 + n^(1/theta)) for theta = 1 (additive), 2 and the
    bottleneck limit, so the interaction's contribution is bounded rather than
    fitted.

Writes results/table_multiresource_price.csv and
results/table_multiresource_decision.csv.  No network, no fit of the frontier.
"""
import csv
import math
import os

import _pipeline as P

pc = P.pc
FR = os.path.join(P.RES, "frontiers")


def main():
    rows, dec = [], []
    files = sorted(f for f in os.listdir(FR) if f.endswith(".csv"))
    for fn in files:
        with open(os.path.join(FR, fn)) as fh:
            recs = list(csv.DictReader([l for l in fh if not l.startswith("#")]))
        if not recs:
            continue
        axes = [c for c in recs[0]
                if c not in ("base", "img_size", "eps", "name") and c]
        if len(axes) < 2:
            continue
        pts = []
        for r in recs:
            try:
                e = float(r["eps"])
                t = {a: float(r[a]) for a in axes if r.get(a)}
            except (TypeError, ValueError, KeyError):
                continue
            if e <= 0 or len(t) < len(axes) or any(v <= 0 for v in t.values()):
                continue
            pts.append({"name": r.get("base", ""), "eps": e, "t": t,
                        "x": {a: math.log(v) for a, v in t.items()}})
        if len(pts) < 6:
            continue

        # one price per axis, from that axis's own marginal curve: sort on the
        # axis, keep the running-minimum error (its marginal frontier), read
        # the plane there.  No interaction is fitted.
        prices, ranges = {}, {}
        for a in axes:
            m = sorted(pts, key=lambda p: p["x"][a])
            best, xs, ls = float("inf"), [], []
            for p in m:
                if p["eps"] < best:
                    best = p["eps"]
                    xs.append(p["x"][a]); ls.append(math.log(p["eps"]))
            # a marginal curve needs real range in log-resource: a divided
            # difference over a vanishing span is noise divided by nothing,
            # and that is where the 1494-nats/e-fold outliers came from.
            if len(xs) < 4 or (xs[-1] - xs[0]) < 0.5:
                continue
            pl = pc.read_plane(xs, ls, 1.0)
            if pl["price"] and pl["price"] > 0:
                prices[a] = pl["price"]
                loc = [v for v in ((-(ls[i + 1] - ls[i]) /
                                    (xs[i + 1] - xs[i]))
                                   for i in range(len(xs) - 1)) if v > 0]
                if len(loc) >= 2:
                    ranges[a] = (min(loc), max(loc))
                rows.append({"frontier": fn[:-4], "axis": a,
                             "n_marginal_points": len(xs),
                             "lambda": pl["price"],
                             "neff_median": pl["neff_median"],
                             "arc_pi": (pl["arc"] / math.pi if pl["arc"] else None),
                             "line_r2": pl["line_r2"]})
        if len(prices) < 2:
            continue

        full = pc.select(pts, prices, price_range=ranges)
        cost_share = {}
        tot = sum(abs(prices[a] * (sum(p["x"][a] for p in pts) / len(pts)))
                  for a in prices)
        for a in prices:
            cost_share[a] = abs(prices[a] * (sum(p["x"][a] for p in pts)
                                             / len(pts))) / tot if tot else None

        for a in prices:
            sub = {k: v for k, v in prices.items() if k != a}
            d = pc.select(pts, sub,
                          price_range={k: v for k, v in ranges.items()
                                       if k != a})
            dec.append({"frontier": fn[:-4], "n_axes": len(prices),
                        "dropped_axis": a,
                        "cost_share_of_dropped": cost_share[a],
                        "pick_full": full["pick"], "pick_dropped": d["pick"],
                        "decision_changes": full["pick"] != d["pick"],
                        "tie_set_full": len(full["tie_set"]),
                        "tie_set_dropped": len(d["tie_set"])})

    df = P.save(rows, "table_multiresource_price.csv")
    ddf = P.save(dec, "table_multiresource_decision.csv")

    print()
    print(f"  multi-resource blocks priced : "
          f"{df.frontier.nunique()} ({len(df)} axis prices derived)")
    print(f"  median derived price          : {df['lambda'].median():.4f}"
          f" nats/e-fold, range"
          f" [{df['lambda'].min():.4f}, {df['lambda'].max():.4f}]")
    if len(ddf):
        print(f"  dropping one axis changes the decision on"
              f" {ddf.decision_changes.mean():.1%} of"
              f" (block, axis) pairs -- that share IS the answer to"
              f" 'does this resource need a price?'")
        hi = ddf[ddf.cost_share_of_dropped > ddf.cost_share_of_dropped.median()]
        lo = ddf[ddf.cost_share_of_dropped <= ddf.cost_share_of_dropped.median()]
        print(f"    axes carrying an above-median cost share:"
              f" {hi.decision_changes.mean():.1%} change the decision")
        print(f"    axes carrying a below-median cost share:"
              f" {lo.decision_changes.mean():.1%}")
    print()
    print("  the interaction's own contribution, bounded rather than fitted:")
    for n in (2, 3, 5):
        print(f"    n = {n} resources: canonical depth ln(1 + n^(1/theta)) ="
              f" {math.log(1+n):.4f} nats at theta = 1,"
              f" {math.log(1+math.sqrt(n)):.4f} at theta = 2,"
              f" {math.log(2):.4f} in the bottleneck limit")


if __name__ == "__main__":
    main()
