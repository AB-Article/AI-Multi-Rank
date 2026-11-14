#!/usr/bin/env python3
"""operating_point.py -- the decision rule of the article in ten lines, and a
self-test that runs it on every frontier in the archive.

This file is the EXACT code printed in README.md, "The selection rule, in ten
lines", followed by a self-test.  The two are kept identical by
scripts/_guard_snippet.py, so the README cannot drift from runnable code.

Stdlib only: no numpy, no pandas, nothing to install.

    python3 examples/operating_point.py          # self-test on the archive
    python3 examples/operating_point.py --demo   # one worked frontier
"""
from __future__ import annotations

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


# --------------------------------------------------------------------- self-test
def _frontiers(root):
    import csv, glob, os
    for path in sorted(glob.glob(os.path.join(root, "results", "frontiers", "*.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows or "eps" not in rows[0]:
            continue
        cols = list(rows[0].keys())
        res = [c for c in cols
               if c not in ("base", "img_size", "eps", "name", "model") and "eps" not in c]
        if not res:
            continue
        pts = []
        for row in rows:
            try:
                e, t = float(row["eps"]), float(row[res[0]])
            except (TypeError, ValueError):
                continue
            if e > 0 and t > 0:
                pts.append((row.get("base") or row.get("name") or "?", 1.0 - e, t))
        if len(pts) >= 5:
            yield os.path.basename(path)[:-4], res[0], pts


def _selftest(root):
    named = declined = 0
    for name, resource, pts in _frontiers(root):
        pick = operating_point(pts)
        if pick is None:
            declined += 1
        else:
            named += 1
    total = named + declined
    print(f"operating_point: ran on {total} archived frontiers with no exception")
    print(f"  named an operating point : {named}")
    print(f"  declined (not readable)  : {declined}")
    assert total > 0, "no frontiers found; run this from the archive root"
    return 0


def _demo(root):
    for name, resource, pts in _frontiers(root):
        pick = operating_point(pts)
        print(f"frontier   {name}")
        print(f"resource   {resource}")
        print(f"candidates {len(pts)}")
        print(f"pick       {pick}")
        print()
        print("The spacing cancels identically in s*s/kappa, so the triple need not")
        print("be evenly spaced and no bandwidth is chosen.  A frontier on which no")
        print("interior triple has s > 0 and kappa > 0 is not second-order readable,")
        print("and the rule declines to name a point rather than naming a wrong one.")
        return 0
    print("no frontier found")
    return 1


if __name__ == "__main__":
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.exit(_demo(root) if "--demo" in sys.argv else _selftest(root))
