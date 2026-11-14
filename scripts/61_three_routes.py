#!/usr/bin/env python3
"""The three estimators of the arc coordinate, compared like for like.

Proposition 'three estimators' of the paper gives three routes to the arc
position, hence to the length of a measured window:

  (i)   phi = 2 arcsin sqrt(eps_inf/eps)      -- needs the fitted floor,
        no derivatives.  The window length is the difference of phi at the
        two ends.  Undefined on a window whose fitted floor exceeds the best
        observed error, which is the sharp window bound of the running-floor
        proposition seen from the estimator's side.
  (ii)  phi = 2 arctan(sqrt(kappa)/s)         -- needs a local slope and
        curvature (hence a bandwidth), no floor.
  (iii) Delta phi = 2 arccos exp(-J)          -- needs three error values,
        no bandwidth, no floor; a lower bound on the length.

Each is compared against the integrated length L = int sqrt(kappa) dx of the
same window, on one panel: the frontiers for which the three-point chord is
available (script 36).  Reported per route: Pearson r against L, the median
ratio to L, and the number of frontiers on which the route is defined, both
on each route's own domain and on the subset where all three are defined.

Reads  results/table_chord_per_frontier.csv (or data/ as shipped), which
       carries L, the chord distance d and the window end errors, and
       results/table_path_per_frontier.csv, which carries the fitted floor
       and the error shares at the window ends.
Writes results/table_three_routes.csv          (per frontier)
       results/table_three_routes_summary.csv  (the four quoted numbers)
"""
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")


def read(name):
    for d in (RESULTS, DATA):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return list(csv.DictReader(open(p)))
    raise SystemExit(f"{name} not found in results/ or data/")


def num(row, key):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def phi(u):
    """Arc position 2 arcsin sqrt(u) of a floor share u in [0,1]."""
    return 2.0 * math.asin(math.sqrt(u)) if 0.0 <= u <= 1.0 else float("nan")


chord = read("table_chord_per_frontier.csv")
path = {(r["field"], r["frontier"], r["cost"]): r
        for r in read("table_path_per_frontier.csv")}

rows = []
for c in chord:
    key = (c["field"], c["frontier"], c["cost"])
    p = path.get(key)
    if p is None:
        continue
    L, d = num(c, "L"), num(c, "d")
    eps1, eps2 = num(c, "eps1"), num(c, "eps2")      # cheap end, expensive end
    einf = num(p, "einf_fit")
    # (i) the floor route: phi at the two window ends from the fitted floor
    route1 = (phi(einf / eps2) - phi(einf / eps1)
              if einf > 0 and einf < eps2 else float("nan"))
    # (ii) the moment route: phi from the local shares at the two window ends
    route2 = abs(phi(num(p, "p_bot")) - phi(num(p, "p_top")))
    rows.append(dict(field=c["field"], frontier=c["frontier"], cost=c["cost"],
                     n_pareto=c.get("n", ""), span_nats=num(c, "span"),
                     L=L, route_i=route1, route_ii=route2, route_iii=d,
                     floor_above_best=bool(not (einf < eps2))))

with open(os.path.join(RESULTS, "table_three_routes.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

L = np.array([r["L"] for r in rows])
cols = {"i": np.array([r["route_i"] for r in rows]),
        "ii": np.array([r["route_ii"] for r in rows]),
        "iii": np.array([r["route_iii"] for r in rows])}
common = np.isfinite(L)
for a in cols.values():
    common &= np.isfinite(a)

out = []
for name, a in cols.items():
    own = np.isfinite(a) & np.isfinite(L)
    out.append(dict(
        route=name, panel="own domain", frontiers=int(own.sum()),
        pearson_r=float(np.corrcoef(a[own], L[own])[0, 1]),
        median_ratio_to_L=float(np.median(a[own] / L[own]))))
    out.append(dict(
        route=name, panel="all three defined", frontiers=int(common.sum()),
        pearson_r=float(np.corrcoef(a[common], L[common])[0, 1]),
        median_ratio_to_L=float(np.median(a[common] / L[common]))))
out.append(dict(route="i", panel="undefined (floor above best observed error)",
                frontiers=int(sum(r["floor_above_best"] for r in rows)),
                pearson_r="", median_ratio_to_L=""))

with open(os.path.join(RESULTS, "table_three_routes_summary.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

print(f"panel: {len(rows)} frontiers with a three-point chord")
for r in out:
    rr = f"{r['pearson_r']:.3f}" if r["pearson_r"] != "" else "   --"
    mr = f"{r['median_ratio_to_L']:.3f}" if r["median_ratio_to_L"] != "" else "   --"
    print(f"  route ({r['route']:3s}) {r['panel']:40s} n={r['frontiers']:2d}"
          f"  r={rr}  median ratio={mr}")
