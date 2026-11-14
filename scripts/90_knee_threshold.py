#!/usr/bin/env python3
"""WHEN DOES A KNEE EXIST AT ALL?  THE SEPARATION THRESHOLD 3+2 sqrt 2.
   (Section "Law II", Prop. 'knee'; Section "What the evidence says".)

The canonical point is w=1.  On a frontier with a floor (an atom at a=0) w runs
from 0 to infinity and w=1 is always attained.  Without a floor it is NOT: for a
two-atom spectrum {a1,a2} with ratio r = a2/a1 > 1 the invariant w is unimodal
in the resource with maximum

    W := sup_x w = (r-1)^2 / (4r) ,     attained at mixing share theta = 1/(1+r),

so the canonical point exists if and only if W >= 1, i.e. if and only if

    r  >=  3 + 2 sqrt 2  =  (1+sqrt 2)^2  =  5.828427 ... .

Two channels closer than that factor produce NO knee: the frontier has no point
of unit coefficient of variation, "operate where kappa=s^2" is empty, and the
correct instruction is the general rule of script 89, operate where |ln N_eff|
is smallest -- which is the maximum of w.

The threshold also explains, as a theorem rather than as estimator noise, why
the three routes to the canonical point disagree.  Near r = 3+2 sqrt 2 the
equation w=1 has a DOUBLE root, so dw/dx -> 0 at the knee and the located
position inherits a square-root, not linear, sensitivity to noise:

    |x_hat - x*|  ~  sqrt( 2 sigma_w / |w''| )  ~  (W-1)^{-1/2} .

Measured here by simulation, and tested on the archive in script 91.

Outputs: results/table_knee_threshold.csv
         results/table_knee_tangency.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save

RNG = np.random.default_rng(20261005)
RSTAR = 3 + 2 * np.sqrt(2)
HFRAC, NGRID = 0.40, 9


def w_exact(a1, a2, x):
    a = np.array([a1, a2]); p = np.array([.5, .5])
    l = np.log(p) - a * x; l -= l.max(); wt = np.exp(l); wt /= wt.sum()
    s = (wt * a).sum(); kap = (wt * (a - s) ** 2).sum()
    return kap / s ** 2


def run():
    rows = []
    for r in (2.0, 3.0, 4.0, 5.0, 5.5, RSTAR - 1e-3, RSTAR + 1e-3, 6.5, 8.0, 12.0, 20.0, 100.0):
        a1 = 1.0; a2 = r
        xs = np.linspace(-6 / a1, 30 / a1, 400001)
        W = max(w_exact(a1, a2, x) for x in np.linspace(-8, 40, 60001))
        rows.append(dict(exponent_ratio=r, W_measured=W, W_formula=(r - 1) ** 2 / (4 * r),
                         knee_exists=bool(W >= 1)))
    T = pd.DataFrame(rows)
    save(T, "table_knee_threshold.csv")
    print(T.to_string(index=False))
    print(f"\n  threshold: W=1  <=>  r = 3 + 2 sqrt 2 = {RSTAR:.6f}")

    # ------------------------------------------------- tangency sensitivity -
    # window centred on the knee of each frontier, spanning the natural
    # transition width 1/(a2-a1); scatter reported in NATS of log-error, which
    # is the unit in which the paper quotes the route disagreement.
    out = []
    for r in (6.0, 6.5, 7.5, 10.0, 16.0, 30.0, 60.0):
        W = (r - 1) ** 2 / (4 * r)
        a1, a2 = 1.0, r
        xstar = np.log(r) / (r - 1)
        half = 4.0 / (r - 1)
        xs = np.linspace(max(xstar - half, 1e-6), xstar + half, 61)
        eps = .5 * np.exp(-a1 * xs) + .5 * np.exp(-a2 * xs)
        S = -np.log(eps)
        locS = []
        for _ in range(400):
            Sn = S + RNG.normal(0, 0.02, len(S))
            Z = xs[-1] - xs[0]; h = HFRAC * Z
            grid = np.linspace(xs[0] + .15 * Z, xs[-1] - .15 * Z, 41)
            ws, Sg = [], []
            for g in grid:
                d = xs - g; wq = np.exp(-.5 * (d / h) ** 2)
                M = np.column_stack([np.ones_like(d), d, .5 * d ** 2])
                Wr = np.sqrt(wq)[:, None]
                b, *_ = np.linalg.lstsq(M * Wr, Sn * Wr[:, 0], rcond=None)
                if b[1] > 0 and -b[2] > 0:
                    ws.append(-b[2] / b[1] ** 2); Sg.append(b[0])
            if ws:
                ws = np.array(ws); Sg = np.array(Sg)
                locS.append(Sg[np.argmin(np.abs(np.log(ws)))])
        locS = np.array(locS)
        out.append(dict(exponent_ratio=r, W=W, W_minus_1=W - 1,
                        knee_depth_sd_nats=float(locS.std()),
                        knee_depth_iqr_nats=float(np.subtract(*np.percentile(locS, [75, 25]))),
                        inv_sqrt_W_minus_1=float(1 / np.sqrt(max(W - 1, 1e-9)))))
    O = pd.DataFrame(out)
    save(O, "table_knee_tangency.csv")
    print()
    print("  tangency: scatter of the located knee DEPTH (nats) under 0.02-nat noise")
    print(O.to_string(index=False))
    from scipy.stats import spearmanr
    rho = spearmanr(O.knee_depth_sd_nats, O.inv_sqrt_W_minus_1).statistic
    print(f"\n  Spearman(knee-depth scatter, (W-1)^-1/2) = {rho:.3f}  (n={len(O)})")
    print(f"  scatter at W-1 = {O.W_minus_1.iloc[0]:.3f}: {O.knee_depth_sd_nats.iloc[0]:.2f} nats"
          f"   vs at W-1 = {O.W_minus_1.iloc[-1]:.2f}: {O.knee_depth_sd_nats.iloc[-1]:.2f} nats")


if __name__ == "__main__":
    run()
