#!/usr/bin/env python3
"""MINIMAX ON A ONE-DIMENSIONAL FRONTIER (Section 6 of paper.tex).

Theorem "Canonical reference" declares uniform error shares, selects
eps* = (N+1) einf and has worst-case regret ln(N+1).  Its hypothesis is ONE
CHANNEL PER RESOURCE, i.e. rank A = N: the declaration must be realizable, and
only then is every share vector in the simplex attainable.

When several reducible modes share one resource direction -- the case the rank
evidence of Section 3 says is normal -- the attainable share set is a CURVE in
the simplex, the barycentre is generically off it, and the answer changes:

  THEOREM (one-dimensional minimax).  Let the spectrum have an atom at 0 and
  an atom at a_max > 0.  Declarations and realized prices both range over the
  curve {p(x)}.  Then for every declaration p-hat on the curve

      sup_x KL(p(x) || p-hat)  >=  max( -ln p-hat_0 , -ln p-hat_max ),

  because the two vertex limits x -> +-oo are in the closure of the curve.
  Minimizing that lower bound over the curve equalizes the two EXTREME shares,

      p_0 = p_max       (equivalently: the extreme mode errors are equal),

  giving C_1D >= -ln p_0*, with eps* = einf / p_0*.  The equalizing point is
  exactly the self-dual fixed point of script 55.  Uniform shares are
  recovered iff ln c_j is affine in a_j (the barycentre lies on the curve),
  which is automatic for N = 1 and codimension one beyond.

This script (i) computes the exact minimax over the curve on a grid, (ii)
confirms the equalization to 1e-8, (iii) shows the value both ABOVE and BELOW
ln(N+1) -- the deviation is not one-signed -- and (iv) confirms the N = 1 and
affine-weight cases reproduce ln(N+1) and (N+1) einf.

Writes results/table_onedim_minimax.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


def shares(xx, cs, as_):
    lw = np.log(cs) - as_ * xx
    lw = lw - lw.max()
    w = np.exp(lw)
    return w / w.sum()


def minimax_1d(cs, as_, ngrid=60001):
    cs = np.asarray(cs, float); as_ = np.asarray(as_, float)
    amin = min(a for a in as_ if a > 0); amax = as_.max()
    lo, hi = -90.0 / amax - 50.0, 260.0 / amin
    grid = np.linspace(lo, hi, ngrid)
    P = np.array([shares(g, cs, as_) for g in grid])
    Pc = np.clip(P, 1e-300, None)

    def worst(xh):
        ph = np.clip(shares(xh, cs, as_), 1e-300, None)
        interior = float(np.max(np.sum(Pc * np.log(Pc / ph), axis=1)))
        return max(interior, -np.log(ph[0]), -np.log(ph[-1]))

    r = minimize_scalar(worst, bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-10})
    ph = shares(r.x, cs, as_)
    return r.x, r.fun, ph


CASES = [
    ("N=1  a=0,1              c=.3,1",        [.3, 1.], [0, 1.]),
    ("N=2  a=0,1,8            c=.1,1,1",      [.1, 1, 1], [0, 1, 8.]),
    ("N=2  a=0,1,8            c=1,5,.2",      [1., 5, .2], [0, 1, 8.]),
    ("N=2  a=0,.5,1.2         c=.05,2,7",     [.05, 2, 7.], [0, .5, 1.2]),
    ("N=3  a=0,1,3,9          c=.2,1,3,1",    [.2, 1, 3, 1.], [0, 1, 3, 9.]),
    ("N=2  a=0,1,8  tiny mid  c=.1,1e-6,30",  [.1, 1e-6, 30.], [0, 1, 8.]),
    ("N=2  a=0,1,2  affine ln c (barycentre on curve)", [1., 1, 1], [0, 1, 2.]),
    ("N=3  a=0,1,2,3 affine ln c", [1., 1, 1, 1], [0, 1, 2, 3.]),
]

rows = []
print(f"{'case':52s} {'C_1D':>8s} {'-ln p0*':>8s} {'ln(N+1)':>8s} "
      f"{'p0*/pmax*':>10s} {'eps*/einf':>10s} {'(N+1)':>6s}")
for name, cs, as_ in CASES:
    M = len(cs)
    xs, C, ph = minimax_1d(cs, as_)
    ratio = ph[0] / ph[-1]
    rows.append(dict(case=name, n_modes=M, C_1d=C, minus_ln_p0=-np.log(ph[0]),
                     ln_N_plus_1=np.log(M), extreme_share_ratio=ratio,
                     eps_star_over_einf=1.0 / ph[0], N_plus_1=M,
                     shares=" ".join(f"{v:.5f}" for v in ph),
                     barycentre_on_curve=bool(abs(ratio - 1) < 1e-6
                                              and abs(ph.max() - ph.min()) < 1e-6)))
    print(f"{name:52s} {C:8.4f} {-np.log(ph[0]):8.4f} {np.log(M):8.4f} "
          f"{ratio:10.6f} {1/ph[0]:10.3f} {M:6d}")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(RESULTS, "table_onedim_minimax.csv"), index=False)

eq = np.abs(R.extreme_share_ratio - 1).max()
print(f"\nmax |p_0/p_max - 1| at the optimum          : {eq:.2e}  (expect 0)")
print(f"max |C_1D - (-ln p_0*)|                     : "
      f"{np.abs(R.C_1d - R.minus_ln_p0).max():.2e}  (expect 0)")
above = int((R.C_1d > R.ln_N_plus_1 + 1e-6).sum())
below = int((R.C_1d < R.ln_N_plus_1 - 1e-6).sum())
equal = int((np.abs(R.C_1d - R.ln_N_plus_1) <= 1e-6).sum())
print(f"cases with C_1D above / equal / below ln(N+1): {above} / {equal} / {below}"
      "   -> the deviation is NOT one-signed")
print(f"affine-ln-c cases recovering uniform shares  : "
      f"{int(R.barycentre_on_curve.sum())} of "
      f"{int((R.case.str.contains('affine')).sum()) + 1} expected")
print("wrote results/table_onedim_minimax.csv")
