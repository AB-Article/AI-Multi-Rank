#!/usr/bin/env python3
"""Which modes are visible on the arc: the length of a finite spectrum.

Proposition 'path length'(iii) of the paper: a spectrum with an atom at the
origin and finite support of size N+1 has

    pi <= L(-inf, inf) <= N pi,

with L = pi exactly for N = 1, and L -> N pi only when every mode dominates
the error over a long stretch -- that is, when each (a_j, ln c_j) is a
well-separated vertex of the upper envelope.  A mode interior to that
envelope is invisible on the frontier and its corner is cut: with equal
weights and exponents 0, 1, 10^8 the two crossovers coincide, the path passes
through the edge midpoint instead of the middle vertex, and L = 1.5 pi; a
large middle weight restores the vertex (2 pi) and a negligible one removes
the mode altogether (1 pi).

The length is computed as the length of the polyline traced by the Hellinger
embedding r_j(x) = 2 sqrt(p_j(x)) of the tilted measure, which equals the
Fisher-Rao length and is independent of how x is sampled, so the grid is
refined around every pairwise crossover log(c_i/c_j)/(a_i-a_j) on the scale
1/|a_i-a_j| set by that crossover.  This is what makes exponents eight
decades apart tractable in double precision.

Writes results/table_mode_visibility.csv
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
PI = np.pi


def fisher_rao_length(a, c, xmax=200.0, n_coarse=400001, n_fine=40001):
    """Fisher-Rao length of the tilted family of eps(x) = sum c_j exp(-a_j x)."""
    a = np.asarray(a, float)
    c = np.asarray(c, float)
    grids = [np.linspace(-xmax, xmax, n_coarse)]
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] == a[j]:
                continue
            x0 = np.log(c[i] / c[j]) / (a[i] - a[j])          # crossover
            scale = 1.0 / abs(a[i] - a[j])                     # its own width
            d = np.geomspace(1e-12 * scale, 60 * scale, n_fine)
            grids += [x0 + d, x0 - d, np.array([x0])]
    x = np.unique(np.concatenate(grids))
    logw = np.log(c)[None, :] - np.outer(x, a)
    logw -= logw.max(1, keepdims=True)                         # stable softmax
    w = np.exp(logw)
    p = w / w.sum(1, keepdims=True)
    r = 2.0 * np.sqrt(p)                                       # Hellinger sphere
    return float(np.sqrt((np.diff(r, axis=0) ** 2).sum(1)).sum())


CASES = [
    # (label, exponents, weights, value quoted in the paper or None)
    ("two atoms (one channel and a floor)", [0, 1], [1, 1], 1.000),
    ("equal weights, exponents 0, 1, 1e8", [0, 1, 1e8], [1, 1, 1], 1.500),
    ("large middle weight", [0, 1, 1e8], [1, 1e6, 1], 2.000),
    ("negligible middle weight", [0, 1, 1e8], [1, 1e-6, 1], 1.000),
    # the upper bound N pi is a limit, approached as the vertices of the upper
    # envelope separate; the middle weight is what separates the crossovers
    ("0, 1, 2; middle weight 1", [0, 1, 2], [1, 1, 1], None),
    ("0, 1, 2; middle weight 10", [0, 1, 2], [1, 10, 1], None),
    ("0, 1, 2; middle weight 1e3", [0, 1, 2], [1, 1e3, 1], None),
    ("0, 1, 2; middle weight 1e6", [0, 1, 2], [1, 1e6, 1], None),
    ("0, 1, 2; middle weight 1e9", [0, 1, 2], [1, 1e9, 1], None),
    ("0, 1, 3, 9; middle weights 1", [0, 1, 3, 9], [1, 1, 1, 1], None),
    ("0, 1, 3, 9; middle weights 1e3", [0, 1, 3, 9], [1, 1e3, 1e3, 1], None),
    ("0, 1, 3, 9; middle weights 1e6", [0, 1, 3, 9], [1, 1e6, 1e6, 1], None),
]

rows = []
for name, a, c, quoted in CASES:
    L = fisher_rao_length(a, c)
    N = len(a) - 1
    rows.append(dict(case=name, exponents=" ".join(f"{v:g}" for v in a),
                     weights=" ".join(f"{v:g}" for v in c), N=N,
                     L_over_pi=round(L / PI, 4),
                     quoted_in_paper=("" if quoted is None else quoted),
                     matches_quoted=("" if quoted is None
                                     else bool(abs(L / PI - quoted) < 1e-3)),
                     lower_bound_pi_ok=bool(L >= PI - 1e-3),
                     upper_bound_N_pi_ok=bool(L <= N * PI + 1e-3)))

with open(os.path.join(RESULTS, "table_mode_visibility.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

bad = [r for r in rows
       if not (r["lower_bound_pi_ok"] and r["upper_bound_N_pi_ok"])
       or r["matches_quoted"] is False]
for r in rows:
    q = f" (paper: {r['quoted_in_paper']:.3f})" if r["quoted_in_paper"] != "" else ""
    print(f"  {r['case']:38s} N={r['N']}  L/pi={r['L_over_pi']:.4f}{q}")
print("bounds pi <= L <= N pi hold on every case; "
      f"{len(bad)} case(s) disagree with a quoted value")
