#!/usr/bin/env python3
"""THE KNEE CONDITION, AND THE ROUTE DISAGREEMENT, MEASURED ON THE ARCHIVE.
   (Section "Law II", Prop. 'knee'; Section "What the evidence says", table
   'kneearchive'.)

Proposition 'knee' (script 90) says the canonical point w=1 exists only where
the invariant reaches 1, and that near the threshold its location inherits a
square-root sensitivity to noise.  Both are tested here on all 117
single-resource Pareto frontiers of the archive, with the local-quadratic jets
of script 81 (h = 0.40 of the window, 9 interior points).

Per frontier we report
    W      = max w over the window          (knee exists in-window iff W >= 1)
    S_knee = depth in nats at argmin |ln w| (the general rule of script 89)
    S_star = median of S w                  (nats per effective channel; the
                                             floorless chart's conserved
                                             quantity, = (1-beta)/beta)
    N_eff  = 1/w at the right edge
and the three routes to the canonical point, all as DEPTHS in nats so that they
are comparable without a chart:
    route kappa=s^2 :  S at argmin |ln w|                     (chart-free)
    route eps=2eps0 :  -ln(2 eps_inf) from the 3-parameter fit (needs a floor)
    route eps_star  :  (1-beta)/beta from the stable fit       (needs beta)

Two predictions are then checked:
  K1  the two chart-dependent routes disagree MORE on frontiers with small W,
      because that is where the knee is a near-tangency (Spearman and a split
      at W=1);
  K2  the old rule eps = 2 eps_inf is undefined or outside the window far more
      often than the general rule, which is defined everywhere.

Outputs: results/table_knee_archive.csv
         results/table_knee_archive_summary.csv
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save, all_frontiers

HFRAC, NGRID = 0.40, 9


def jets(x, S, hfrac=HFRAC, ng=NGRID):
    Z = x[-1] - x[0]; h = Z * hfrac
    grid = np.linspace(x[0] + .15 * Z, x[-1] - .15 * Z, ng)
    out = []
    for g in grid:
        d = x - g; wq = np.exp(-.5 * (d / h) ** 2)
        M = np.column_stack([np.ones_like(d), d, .5 * d ** 2])
        Wr = np.sqrt(wq)[:, None]
        b, *_ = np.linalg.lstsq(M * Wr, S * Wr[:, 0], rcond=None)
        out.append((g, b[0], b[1], -b[2]))
    return np.array(out)                           # x, S, s, kappa


def fit_floor(x, y):
    def r(th):
        return np.log(np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * x)) - y
    best = None
    for e0 in (-6., -3., -1.5):
        for a0 in (-1., 0., 1.):
            try:
                s = least_squares(r, [e0, y[0], a0], method="lm", max_nfev=8000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    return (np.exp(best.x[0]) if best else np.nan)


def fit_beta(x, y):
    best, bb = None, np.nan
    for b0 in (0.2, 0.4, 0.7):
        for off in (0.3, 1.5, 5.0):
            def r(th):
                u = np.clip(x - th[3], 1e-9, None)
                return th[0] - (np.exp(th[1]) * u) ** np.exp(th[2]) - y
            try:
                s = least_squares(r, [y[0], 0.0, np.log(b0), x[0] - off],
                                  method="lm", max_nfev=8000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best, bb = s, float(np.exp(s.x[2]))
    return bb


def run():
    rows = []
    for fr in all_frontiers():
        x, e = fr["x"], fr["e"]
        o = np.argsort(x); x, e = x[o], e[o]
        if len(x) < 7 or e.min() <= 0:
            continue
        y = np.log(e); S = -y
        J = jets(x, S)
        ok = (J[:, 2] > 0) & (J[:, 3] > 0)
        if ok.sum() < 5:
            continue
        g, Sg, s, kap = J[ok, 0], J[ok, 1], J[ok, 2], J[ok, 3]
        w = kap / s ** 2
        W = float(w.max())
        i = int(np.argmin(np.abs(np.log(w))))
        S_knee = float(Sg[i])
        S_star = float(np.median(Sg * w))
        eps_inf = fit_floor(x, y)
        beta = fit_beta(x, y)
        S_route_floor = float(-np.log(2 * eps_inf)) if (np.isfinite(eps_inf)
                                                       and 0 < 2 * eps_inf < 1) else np.nan
        S_route_beta = float((1 - beta) / beta) if (np.isfinite(beta) and 0 < beta < 1) else np.nan
        rows.append(dict(corpus=fr["corpus"], frontier=fr["name"], n=len(x),
                         span_nats=float(x[-1] - x[0]), S_lo=float(Sg.min()),
                         S_hi=float(Sg.max()), W=W, knee_in_window=bool(W >= 1),
                         S_knee=S_knee, S_star_invariant=S_star,
                         N_eff_right=float(1 / w[-1]), beta_fit=beta, eps_inf_fit=eps_inf,
                         S_route_floor=S_route_floor, S_route_beta=S_route_beta,
                         route_gap=abs(S_route_floor - S_route_beta)
                         if np.isfinite(S_route_floor) and np.isfinite(S_route_beta) else np.nan,
                         floor_route_in_window=bool(np.isfinite(S_route_floor)
                                                    and Sg.min() <= S_route_floor <= Sg.max()),
                         beta_route_in_window=bool(np.isfinite(S_route_beta)
                                                   and Sg.min() <= S_route_beta <= Sg.max())))
    T = pd.DataFrame(rows)
    save(T, "table_knee_archive.csv")

    G = T.dropna(subset=["route_gap"])
    lo, hi = G[G.W < 1], G[G.W >= 1]
    rho = spearmanr(G.W, G.route_gap).statistic if len(G) > 5 else np.nan
    pv = spearmanr(G.W, G.route_gap).pvalue if len(G) > 5 else np.nan
    srows = [
        dict(quantity="frontiers with a usable second-order reading", value=len(T)),
        dict(quantity="median W = max w over the window", value=float(T.W.median())),
        dict(quantity="frontiers with W >= 1 (canonical point exists in-window)",
             value=int(T.knee_in_window.sum())),
        dict(quantity="fraction with W >= 1", value=float(T.knee_in_window.mean())),
        dict(quantity="implied exponent ratio at the median W "
                      "(r solving (r-1)^2/4r = W)",
             value=float(1 + 2 * T.W.median() + 2 * np.sqrt(T.W.median() * (T.W.median() + 1)))),
        dict(quantity="median S* (nats per effective channel, invariant route)",
             value=float(T.S_star_invariant.median())),
        dict(quantity="median S* implied by the fitted beta, (1-b)/b",
             value=float(T.S_route_beta.median())),
        dict(quantity="K1  Spearman(W, |route gap|)", value=float(rho)),
        dict(quantity="K1  p-value", value=float(pv)),
        dict(quantity="K1  median route gap, W < 1 (nats)", value=float(lo.route_gap.median())),
        dict(quantity="K1  median route gap, W >= 1 (nats)", value=float(hi.route_gap.median())),
        dict(quantity="K1  n with W < 1 / W >= 1", value=float(len(lo)) ),
        dict(quantity="K2  eps=2eps_inf route defined", value=int(T.S_route_floor.notna().sum())),
        dict(quantity="K2  eps=2eps_inf route inside the window",
             value=int(T.floor_route_in_window.sum())),
        dict(quantity="K2  eps_star route inside the window",
             value=int(T.beta_route_in_window.sum())),
        dict(quantity="K2  |ln N_eff| rule defined (always, by construction)", value=len(T)),
    ]
    Sm = pd.DataFrame(srows)
    save(Sm, "table_knee_archive_summary.csv")
    print(Sm.to_string(index=False))


if __name__ == "__main__":
    run()
