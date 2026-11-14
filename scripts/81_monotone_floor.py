#!/usr/bin/env python3
"""THE MONOTONE FLOOR BOUND, MEASURED.
   (Section "The floor is not a parameter", Thm. 'monofloor', part (ii).)

Theorem 'monofloor' (script 80) says that for every completely monotone
frontier the LOCAL floor

    F(x) = kappa eps / (s^2 + kappa)

is non-increasing in the resource and bounds eps_inf from above, with equality
(a constant F) iff the frontier is floor + exactly one reducible channel.  The
paper's previous bound was eps_inf <= eps(x_0 + X), the best observed error.
F is never worse and is usually much better, because it discounts the part of
the observed error that the curve is still visibly reducing.

This script measures, on every single-resource Pareto frontier of the archive:

  (i)   F at the right edge of the window, against eps(x_0+X);
  (ii)  the tightening factor eps(x_0+X)/F;
  (iii) whether F is monotone non-increasing across the window (it must be, up
        to noise and bandwidth, for any frontier obeying the axiom) -- an
        excursion beyond the design-matched null of script 82 is evidence
        against complete monotonicity on that window, which is the first
        ONE-SIDED test of the axiom the theory has had;
  (iv)  the fitted floor of the three-parameter law, for reference.

Outputs: results/table_monotone_floor_bound.csv
         results/table_monotone_floor_summary.csv
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save, all_frontiers

ROOT = os.path.dirname(HERE)
FRONT = os.path.join(ROOT, "results", "frontiers")
HFRAC, NGRID = 0.40, 9


def frontiers():
    """All 117 single-resource Pareto frontiers of the archive (script 68's builder)."""
    for fr in all_frontiers():
        yield fr["name"], fr["corpus"], fr["x"], fr["e"]


def local_jets(x, S, hfrac=HFRAC, ng=NGRID):
    """local-quadratic (S, s, kappa) on an interior grid; h = hfrac * window."""
    Z = x[-1] - x[0]; h = Z * hfrac
    grid = np.linspace(x[0] + 0.15 * Z, x[-1] - 0.15 * Z, ng)
    out = []
    for g in grid:
        d = x - g; w = np.exp(-0.5 * (d / h) ** 2)
        M = np.column_stack([np.ones_like(d), d, 0.5 * d ** 2])
        Wr = np.sqrt(w)[:, None]
        b, *_ = np.linalg.lstsq(M * Wr, S * Wr[:, 0], rcond=None)
        out.append((g, b[0], b[1], -b[2]))
    return np.array(out)                      # x, S, s, kappa


def fit_floor(x, eps):
    y = np.log(eps)

    def r(th):
        ei, lc, la = th
        return np.log(np.exp(ei) + np.exp(lc) * np.exp(-np.exp(la) * x)) - y
    best = None
    for e0 in (-6., -3., -1.5):
        for a0 in (-1., 0., 1.):
            try:
                s = least_squares(r, [e0, np.log(eps[0]), a0], method="lm", max_nfev=8000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    return float(np.exp(best.x[0])) if best is not None else np.nan


def run():
    rows = []
    for key, axis, x, eps in frontiers():
        o = np.argsort(x); x, eps = x[o], eps[o]
        S = -np.log(eps)
        J = local_jets(x, S)
        s, k = J[:, 2], J[:, 3]
        ok = (s > 0) & (k > 0)
        if ok.sum() < 5:
            rows.append(dict(frontier=key, corpus=axis, n=len(x), usable=0))
            continue
        Sg, sg, kg = J[ok, 1], s[ok], k[ok]
        epsg = np.exp(-Sg)
        F = kg * epsg / (sg ** 2 + kg)
        eps_edge = float(eps.min())
        rows.append(dict(
            frontier=key, corpus=axis, n=len(x), usable=1,
            eps_best=eps_edge,
            F_left=float(F[0]), F_right=float(F[-1]),
            bound_new=float(F[-1]), bound_old=eps_edge,
            tighten=float(eps_edge / max(F[-1], 1e-12)),
            monotone=int(np.all(np.diff(F) <= 1e-12)),
            dlogF_dS=float(np.polyfit(Sg, np.log(F), 1)[0]),
            w_median=float(np.median(kg / sg ** 2)),
            floor_fitted=fit_floor(x, eps)))
    P = pd.DataFrame(rows)
    U = P[P.usable == 1]
    summ = pd.DataFrame([
        dict(quantity="single-resource frontiers", value=float(len(P))),
        dict(quantity="usable (s>0, kappa>0 on >=5 grid points)", value=float(len(U))),
        dict(quantity="median tightening eps_best / F_right", value=float(U.tighten.median())),
        dict(quantity="mean tightening", value=float(U.tighten.mean())),
        dict(quantity="tightening at the 75th percentile", value=float(U.tighten.quantile(.75))),
        dict(quantity="max tightening", value=float(U.tighten.max())),
        dict(quantity="frontiers where the new bound is tighter", value=float((U.tighten > 1).sum())),
        dict(quantity="median F_right (new bound)", value=float(U.F_right.median())),
        dict(quantity="median eps_best (old bound)", value=float(U.eps_best.median())),
        dict(quantity="median fitted floor (3-parameter law)", value=float(U.floor_fitted.median())),
        dict(quantity="fitted floor above the new bound (count)",
             value=float((U.floor_fitted > U.F_right * 1.000001).sum())),
        dict(quantity="F monotone non-increasing across the window (count)",
             value=float(U.monotone.sum())),
        dict(quantity="median d lnF / dS  (0 = floor chart, see script 82)",
             value=float(U.dlogF_dS.median()))])
    return P, summ


if __name__ == "__main__":
    print(__doc__.split("Outputs")[0])
    P, S = run()
    U = P[P.usable == 1]
    print(U.groupby("corpus")[["tighten", "w_median", "dlogF_dS"]].median().to_string())
    print()
    print(S.to_string(index=False))
    save(P, "table_monotone_floor_bound.csv")
    save(S, "table_monotone_floor_summary.csv")
