#!/usr/bin/env python3
"""THE ATTEMPT CHART: THE CHAIN WRITTEN DOWN AS A CHART, AND HELD OUT.
   (Section "The price is a frontier", Prop. 'chain', 'attemptfloor';
    Online Appendix sec. "The attempt chain".)

The chain  x --(m)--> attempts --(P)--> error  with P(k) = int g^k d nu gives a
chart with nothing added by hand.  Take the smallest non-degenerate task, two
difficulty classes, and the smallest saturating clock, one obstacle type:

        m(x) = Lambda (1 - (1-M0) e^{-c (x - x_min)}) ,   M(x) = m(x)/Lambda
        eps(x) = w e^{-B1 M(x)} + (1-w) e^{-B2 M(x)} ,    B_i = Lambda * (-ln g_i)

Five parameters (w, B1, B2, c, M0) and then, with nothing further fitted,

        eps_inf  =  w e^{-B1} + (1-w) e^{-B2}   =   P(Lambda)

-- the floor is the all-fail rate at the attempt budget, NOT a free parameter.
Two special cases are already in the paper: w = 1 is the cure chart (one
difficulty class), and M0 -> 0 puts the left edge of the window at chance.

This is the one chart the attempt chain adds, and it is the only place where the
revision can buy ACCURACY rather than structure.  Protocol: script 68's / 110's,
same frontiers, same three splits, same leave-one-out, so the columns are
comparable line by line with the paper's chart table.

  INTERIOR  leave one interior point out, predict it
  FORWARD   fit the first 2/3 (in log-resource), predict the rest
  BACKWARD  fit the last 2/3, predict the first third

Also reported: the derived floor against the incumbent's fitted floor, and the
implied attempt budget split  Lambda = B1/(-ln g1)  (identifiable only up to the
product, which is the point of Prop. 'chain': the clock and the task are
separable only on a count axis).

Outputs: results/table_attempt_chart_frontiers.csv
         results/table_attempt_chart_summary.csv
         results/table_attempt_chart_params.csv
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers, _module              # noqa: E402

PF = _module("110_powerfloor_chart.py", "pf")                  # its __main__ is not run


# ---------------------------------------------------------- the attempt chart --
def _unpack(th, xm):
    lw, lB1, ldB, lc, lM = th
    w = 1.0 / (1.0 + np.exp(-lw))
    B1 = np.exp(np.clip(lB1, -12.0, 12.0))
    B2 = B1 + np.exp(np.clip(ldB, -12.0, 12.0))                # ordering B1 < B2
    c = np.exp(np.clip(lc, -8.0, 8.0))
    M0 = 1.0 / (1.0 + np.exp(-lM))
    return w, B1, B2, c, M0


def fit_attempt(x, e):
    """eps = w e^{-B1 M} + (1-w) e^{-B2 M},  M = 1 - (1-M0) e^{-c (x-x_min)}."""
    y = np.log(e)
    xm = float(x[0])

    def mdl(th, xx):
        w, B1, B2, c, M0 = _unpack(th, xm)
        M = 1.0 - (1.0 - M0) * np.exp(-c * (np.atleast_1d(xx) - xm))
        return np.log(w * np.exp(-B1 * M) + (1.0 - w) * np.exp(-B2 * M) + 1e-300)

    span = max(float(x[-1] - x[0]), 1e-6)
    S0, Smax = float(-y[0]), float(-y.min())
    best = None
    starts = [(0.0, max(Smax, 1e-3), max(0.5 * Smax, 1e-3), 1.0 / span, 0.3),
              (2.5, max(1.6 * S0, 1e-3), max(6.0 * Smax, 1e-2), 1.0 / span, 0.4),
              (-1.0, max(Smax, 1e-3), max(2.0 * Smax, 1e-2), 0.5 / span, 0.5)]
    for lw, B1g, dBg, cg, M0g in starts:
        th0 = [lw, np.log(B1g), np.log(dBg), np.log(cg), np.log(M0g / (1 - M0g))]
        try:
            s = least_squares(lambda th: mdl(th, x) - y, th0,
                              x_scale="jac", max_nfev=250)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    if best is None:
        return None, None
    return (lambda xx: np.exp(mdl(best.x, xx))), best


def attempt_params(x, e, sol):
    w, B1, B2, c, M0 = _unpack(sol.x, float(x[0]))
    eps_inf = float(w * np.exp(-B1) + (1 - w) * np.exp(-B2))
    return dict(w=float(w), B1=float(B1), B2=float(B2), c=float(c), M0=float(M0),
                eps_inf_derived=eps_inf,
                floor_share_derived=float(eps_inf / e.min()),
                budget_spent_at_left_edge=float(M0),
                budget_spent_at_right_edge=float(
                    1.0 - (1.0 - M0) * np.exp(-c * (x[-1] - x[0]))))


# ------------------------------------------------------------ the comparison --
METHODS = ["power", "cure", "law", "stable", "powerfloor", "law_beta", "law2",
           "attempt", "interp"]
NPAR = dict(PF.NPAR, attempt=5)
REGIMES = PF.REGIMES


def predict(xtr, etr, xte, m):
    if m == "attempt":
        f, _ = fit_attempt(xtr, etr)
        if f is None:
            return np.full_like(np.atleast_1d(xte), np.nan, dtype=float)
        return f(xte)
    return PF.predict(xtr, etr, xte, m)


def assess(fr):
    x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
    o = np.argsort(x)
    x, e = x[o], e[o]
    k = (e > 0) & np.isfinite(e)
    x, e = x[k], e[k]
    n = len(x)
    if n < 7:
        return None
    rec = {m: {r: [] for r in REGIMES} for m in METHODS}
    for i in range(1, n - 1):
        msk = np.ones(n, bool)
        msk[i] = False
        for m in METHODS:
            try:
                p = float(np.atleast_1d(predict(x[msk], e[msk], x[i], m))[0])
            except Exception:
                continue
            if np.isfinite(p) and p > 0:
                rec[m]["INTERIOR"].append(abs(np.log(p) - np.log(e[i])))
    cut = int(np.ceil(n * 2 / 3))
    for lbl, tr, te in [("FORWARD", slice(0, cut), slice(cut, n)),
                        ("BACKWARD", slice(n - cut, n), slice(0, n - cut))]:
        for m in METHODS:
            try:
                p = np.atleast_1d(predict(x[tr], e[tr], x[te], m))
            except Exception:
                continue
            g = np.isfinite(p) & (p > 0)
            if g.any():
                rec[m][lbl] += list(np.abs(np.log(p[g]) - np.log(e[te][g])))
    row = dict(corpus=fr["corpus"], name=fr["name"], n=n,
               span=float(x.max() - x.min()))
    for m in METHODS:
        for lbl in REGIMES:
            row[f"{m}_{lbl}"] = float(np.median(rec[m][lbl])) if rec[m][lbl] else np.nan
    return row


def _one(fr):
    r = assess(fr)
    x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
    o = np.argsort(x)
    x, e = x[o], e[o]
    k = (e > 0) & np.isfinite(e)
    x, e = x[k], e[k]
    d = None
    if len(x) >= 7:
        f, sol = fit_attempt(x, e)
        if sol is not None:
            d = dict(corpus=fr["corpus"], name=fr["name"])
            d.update(attempt_params(x, e, sol))
            d["resid_nats"] = float(np.median(np.abs(np.log(f(x)) - np.log(e))))
    return r, d


def run():
    from multiprocessing import Pool
    frs = list(all_frontiers())
    with Pool(processes=max(1, min(4, os.cpu_count() or 2))) as pool:
        got = pool.map(_one, frs, chunksize=1)
    rows = [r for r, _ in got if r]
    pars = [d for _, d in got if d]
    T = pd.DataFrame(rows)
    save(T, "table_attempt_chart_frontiers.csv")
    P = pd.DataFrame(pars)
    save(P, "table_attempt_chart_params.csv")

    out = []
    for m in METHODS:
        d = dict(chart=m, parameters=NPAR[m])
        for lbl in REGIMES:
            col = T[f"{m}_{lbl}"]
            d[f"median_{lbl}"] = round(float(col.median()), 4)
            d[f"rel_{lbl}_pct"] = round(float((np.exp(col.median()) - 1) * 100), 2)
            d[f"n_{lbl}"] = int(col.notna().sum())
        out.append(d)
    S = pd.DataFrame(out)
    save(S, "table_attempt_chart_summary.csv")
    print("\n   held-out relative error, per cent (lower is better)")
    print(S[["chart", "parameters"] + [f"rel_{l}_pct" for l in REGIMES]
            + [f"n_{l}" for l in REGIMES]].to_string(index=False))

    print("\n   head-to-head, paired over frontiers (wins / n, Wilcoxon p)")
    for ref in ("law", "powerfloor", "law2"):
        print(f"   against {ref}:")
        for m in ("attempt",):
            line = [f"     {m:10}"]
            for lbl in REGIMES:
                a, b = T[f"{m}_{lbl}"].values, T[f"{ref}_{lbl}"].values
                g = np.isfinite(a) & np.isfinite(b)
                wins = int((a[g] < b[g]).sum())
                try:
                    p = wilcoxon(a[g], b[g]).pvalue
                except Exception:
                    p = np.nan
                line.append(f"{lbl[:3]} {wins:3d}/{int(g.sum()):3d} p={p:.3f}")
            print("  ".join(line))

    if len(P):
        print("\n   the derived floor and the budget already spent in the window")
        print(f"     median derived floor share eps_inf/eps_min : "
              f"{P.floor_share_derived.median():.4f}")
        print(f"     median attempt budget spent at left edge   : "
              f"{P.budget_spent_at_left_edge.median():.4f}")
        print(f"     median attempt budget spent at right edge  : "
              f"{P.budget_spent_at_right_edge.median():.4f}")
        print(f"     median in-window residual (nats)           : "
              f"{P.resid_nats.median():.4f}")
    return T, S, P


if __name__ == "__main__":
    print(__doc__.splitlines()[0])
    run()
