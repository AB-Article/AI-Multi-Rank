#!/usr/bin/env python3
"""THE POWER-FLOOR CHART: THE EMPTY CELL OF THE CHART LINE, HELD OUT.
   (Section "The floor is not a parameter", Prop. 'powerfloor'.)

The similarity solutions of the horizon lattice are h_k = (x-x_0)/(k-beta), with
sigma = beta/(1-beta), so the compositional charts form ONE line in sigma:

    sigma = -1          cure          floor, approached EXPONENTIALLY
    sigma in (-1, 0)    ** empty **   floor, approached as a POWER LAW
    sigma = 0           Lomax/Gamma   no floor, power-law frontier
    sigma > 0           stretched     no floor
    sigma < -1          lumpy         (the incumbent two-atom chart)

The empty cell is a legitimate chart, and it is the one in which a real frontier
with a real floor most plausibly sits:

    S = Lam - C (x - x_0)^beta ,  beta < 0,   eps_inf = e^{-Lam} > 0

with S' completely monotone (verified to order 7, script 108), and then, with
nothing further fitted,

    N_eff = |sigma| DeltaS ,   |sigma| = -beta/(1-beta) in (0,1)

so it interpolates exactly between the cure chart (|sigma| = 1, N_eff = DeltaS)
and the floorless charts (|sigma| -> 0).  It has never been fitted.

Protocol: script 68's and 99's, on the same frontiers and the same splits, so the
numbers are directly comparable to Table 'prediction' and Table 'cure'.
    INTERIOR  leave one interior point out, predict it
    FORWARD   train on the cheapest 2/3, predict the dearest 1/3
    BACKWARD  train on the dearest 2/3, predict the cheapest 1/3

Charts (parameters in brackets):
    power      [2]  c e^{-alpha x}
    cure       [3]  exp{-lam(1-e^{-c(x-x0)})}             divisible, floored
    law        [3]  eps_inf + c e^{-alpha x}              indivisible, floored
    stable     [3]  exp{-(alpha(x-x0))^beta}              divisible, floorless
    powerfloor [4]  exp{-Lam + C (x-x0)^beta}, beta<0     divisible, floored  NEW
    law_beta   [4]  eps_inf + c exp{-(alpha(x-x0))^beta}  the archive's 4-par
    law2       [5]  eps_inf + c1 e^{-a1 x} + c2 e^{-a2 x}
    interp     [-]  log-linear through the neighbours

Outputs: results/table_powerfloor_frontiers.csv
         results/table_powerfloor_summary.csv
         results/table_powerfloor_identity.csv
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
from _loaders import save, all_frontiers                      # noqa: E402


# --------------------------------------------------------------- the charts ---
def fit_cure(x, e):
    y = np.log(e)
    x0 = x[0]

    def mdl(th, xx):
        lam, c = np.exp(th[0]), np.exp(th[1])
        u0 = 1.0 / (1.0 + np.exp(-th[2]))
        return -lam * (1.0 - u0 * np.exp(-c * (np.atleast_1d(xx) - x0)))
    best, S0 = None, -y[0]
    for lam0 in (S0 * 1.5, -np.log(e.min()) + 2.0):
        if not np.isfinite(lam0) or lam0 <= 0:
            continue
        for c0 in (0.3, 1.2):
            u00 = np.clip(1.0 - S0 / lam0, 1e-6, 1 - 1e-6)
            try:
                s = least_squares(lambda th: mdl(th, x) - y,
                                  [np.log(lam0), np.log(c0), np.log(u00 / (1 - u00))],
                                  max_nfev=20000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    return (None if best is None else (lambda xx: np.exp(mdl(best.x, xx)))), best


def fit_law(x, e, floor=True, k=1):
    def mdl(th, xx):
        v = np.full_like(np.atleast_1d(xx), np.exp(th[0]) if floor else 0.0, dtype=float)
        for j in range(k):
            v = v + np.exp(th[1 + 2 * j]) * np.exp(-np.exp(th[2 + 2 * j])
                                                   * np.atleast_1d(xx))
        return v
    th0 = [np.log(max(e.min() * .5, 1e-9))]
    for j in range(k):
        th0 += [np.log(e.max() / (j + 1)), np.log(.4 * (j + 1) ** 2)]
    r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), np.array(th0),
                      max_nfev=40000)
    return (lambda xx: mdl(r.x, xx)), r


def fit_stable(x, e):
    y = np.log(e)

    def mdl(th, xx):
        la, lb, x0 = th
        u = np.clip(np.atleast_1d(xx) - x0, 1e-9, None)
        return -(np.exp(la) * u) ** np.exp(lb)
    best = None
    for b0 in (-1.2, -0.3):
        for off in (1.0, 4.0):
            try:
                s = least_squares(lambda th: mdl(th, x) - y, [0.0, b0, x[0] - off],
                                  max_nfev=20000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    return (None if best is None else (lambda xx: np.exp(mdl(best.x, xx)))), best


def fit_powerfloor(x, e):
    """S = Lam - C (x-x0)^beta with beta = -b < 0:  divisible, floored, power-law
    approach to the floor.

    The naive parameterisation has a flat ridge: as b -> 0 with C b fixed the
    chart degenerates to the Lomax/log chart, and an optimiser wanders along it.
    We parameterise ALONG the ridge instead, which removes it:

        S(x) = L0 + (G/b) * ( 1 - (u/u0)^{-b} ),      u = x - x_min + u0

    so L0 = S(x_min) exactly, G is the local log-slope scale, b the shape and u0
    the distance back to the origin; as b -> 0 the expression tends smoothly to
    L0 + G ln(u/u0).  The floor is Lam = L0 + G/b, i.e. eps_inf = e^{-Lam}, and
    beta = -b, sigma = beta/(1-beta) = -b/(1+b).
    """
    y = np.log(e)
    xm = x[0]
    S0 = float(-y[0])
    span = max(x[-1] - x[0], 1e-6)
    G0 = max((-y.min() - -y.max()) / span, 1e-3) if len(x) > 1 else 1.0
    G0 = max(float((-y.min() + y.max()) / span), 1e-3)

    def mdl(th, xx):
        L0, lG, lb, lu = th
        G = np.exp(np.clip(lG, -20.0, 20.0))
        b = np.exp(np.clip(lb, np.log(1e-4), np.log(20.0)))
        u0 = np.exp(np.clip(lu, -20.0, 20.0))
        u = (np.atleast_1d(xx) - xm + u0) / u0
        return -(L0 + (G / b) * (1.0 - u ** (-b)))        # = ln eps

    best = None
    for lb in (np.log(0.15), np.log(0.6), np.log(2.5)):
        for lu in (np.log(0.3 * span + 1e-6), np.log(2.0 * span + 1e-6)):
            try:
                s = least_squares(lambda th: mdl(th, x) - y,
                                  [S0, np.log(G0), lb, lu],
                                  x_scale="jac", max_nfev=600)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    if best is None:
        return None, None
    return (lambda xx: np.exp(mdl(best.x, xx))), best


def powerfloor_params(x, e, sol):
    """(Lam, beta, sigma, x0) from a fit_powerfloor solution."""
    L0, lG, lb, lu = sol.x
    G = float(np.exp(np.clip(lG, -20.0, 20.0)))
    b = float(np.exp(np.clip(lb, np.log(1e-4), np.log(20.0))))
    u0 = float(np.exp(np.clip(lu, -20.0, 20.0)))
    return dict(Lam=float(L0 + G / b), beta=float(-b),
                sigma=float(-b / (1.0 + b)), x0=float(x[0] - u0),
                G=float(G), b=float(b), u0=float(u0), L0=float(L0))


def fit_law_beta(x, e):
    """the archive's four-parameter chart eps_inf + c exp{-(alpha(x-x0))^beta}."""
    y = np.log(e)

    def mdl(th, xx):
        f, c, la, lb = np.exp(th[0]), np.exp(th[1]), th[2], th[3]
        u = np.clip(np.atleast_1d(xx) - (x[0] - 1.0), 1e-9, None)
        return np.log(f + c * np.exp(-(np.exp(la) * u) ** np.exp(lb)))
    best = None
    for b0 in (-1.0, -0.2):
        try:
            s = least_squares(lambda th: mdl(th, x) - y,
                              [np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), 0.0, b0],
                              max_nfev=30000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    return (None if best is None else (lambda xx: np.exp(mdl(best.x, xx)))), best


def predict(xtr, etr, xte, m):
    if m == "cure":
        f, _ = fit_cure(xtr, etr)
    elif m == "stable":
        f, _ = fit_stable(xtr, etr)
    elif m == "powerfloor":
        f, _ = fit_powerfloor(xtr, etr)
    elif m == "law_beta":
        f, _ = fit_law_beta(xtr, etr)
    elif m == "law":
        f = fit_law(xtr, etr)[0]
    elif m == "law2":
        f = fit_law(xtr, etr, k=2)[0]
    elif m == "power":
        f = fit_law(xtr, etr, floor=False)[0]
    elif m == "interp":
        o = np.argsort(xtr)
        return np.exp(np.interp(xte, xtr[o], np.log(etr[o])))
    else:
        raise ValueError(m)
    if f is None:
        return np.full_like(np.atleast_1d(xte), np.nan, dtype=float)
    return f(xte)


METHODS = ["power", "cure", "law", "stable", "powerfloor", "law_beta", "law2", "interp"]
NPAR = {"power": 2, "cure": 3, "law": 3, "stable": 3, "powerfloor": 4,
        "law_beta": 4, "law2": 5, "interp": 0}
REGIMES = ["INTERIOR", "FORWARD", "BACKWARD"]


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
    row = dict(corpus=fr["corpus"], name=fr["name"], n=n, span=float(x.max() - x.min()))
    for m in METHODS:
        for lbl in REGIMES:
            row[f"{m}_{lbl}"] = float(np.median(rec[m][lbl])) if rec[m][lbl] else np.nan
    return row


def identity_check(fr):
    """fit the power-floor chart to the whole frontier and read off what it then
    asserts with no further freedom:  N_eff = |sigma| DeltaS  at every point."""
    x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
    o = np.argsort(x)
    x, e = x[o], e[o]
    k = (e > 0) & np.isfinite(e)
    x, e = x[k], e[k]
    if len(x) < 7:
        return None
    f, sol = fit_powerfloor(x, e)
    if f is None:
        return None
    P = powerfloor_params(x, e, sol)
    lam, beta, sig, x0 = P["Lam"], P["beta"], P["sigma"], P["x0"]
    G, b, u0, L0 = P["G"], P["b"], P["u0"], P["L0"]
    # S = L0 + (G/b)(1 - (u/u0)^{-b});  DeltaS = (G/b)(u/u0)^{-b};
    # s = G (u/u0)^{-b-1}/u0;  kappa = G(b+1)(u/u0)^{-b-2}/u0^2;
    # so N_eff = G (u/u0)^{-b}/(b+1)  and  N_eff/DeltaS = b/(1+b) = |sigma|.
    with np.errstate(all="ignore"):
        q = np.clip((x - x[0] + u0) / u0, 1e-12, None) ** (-b)
        dS = (G / b) * q
        Neff = G * q / (1.0 + b)
        ratio = float(np.median(Neff / dS)) if np.all(dS > 0) else np.nan
    return dict(corpus=fr["corpus"], name=fr["name"], Lam=lam, beta=beta,
                sigma=sig, x0=x0, eps_inf=float(np.exp(-lam)),
                floor_share=float(np.exp(-lam) / e.min()),
                Neff_over_dS=ratio, abs_sigma=float(abs(sig)),
                identity_err=float(abs(ratio - abs(sig))),
                resid_nats=float(np.median(np.abs(np.log(f(x)) - np.log(e)))))


def run():
    rows, ident = [], []
    for fr in all_frontiers():
        r = assess(fr)
        if r:
            rows.append(r)
        i = identity_check(fr)
        if i:
            ident.append(i)
    T = pd.DataFrame(rows)
    save(T, "table_powerfloor_frontiers.csv")
    out = []
    for m in METHODS:
        d = dict(chart=m, parameters=NPAR[m])
        for lbl in REGIMES:
            col = T[f"{m}_{lbl}"]
            d[f"median_{lbl}"] = round(float(col.median()), 4)
            d[f"n_{lbl}"] = int(col.notna().sum())
        out.append(d)
    S = pd.DataFrame(out)
    save(S, "table_powerfloor_summary.csv")
    print("\n   held-out median |log-error| residual, nats (lower is better)")
    print(S.to_string(index=False))
    print("\n   relative error, per cent (exp(median) - 1)")
    rel = S.copy()
    for lbl in REGIMES:
        rel[f"median_{lbl}"] = ((np.exp(S[f"median_{lbl}"]) - 1) * 100).round(2)
    print(rel[["chart", "parameters"] + [f"median_{l}" for l in REGIMES]].to_string(index=False))

    print("\n   head-to-head against the incumbent floored chart 'law' (3 par.)")
    for m in ["cure", "stable", "powerfloor", "law_beta", "law2"]:
        line = [f"   {m:11}"]
        for lbl in REGIMES:
            a, b = T[f"{m}_{lbl}"].values, T[f"law_{lbl}"].values
            g = np.isfinite(a) & np.isfinite(b)
            wins = int((a[g] < b[g]).sum())
            try:
                p = wilcoxon(a[g], b[g]).pvalue
            except Exception:
                p = np.nan
            line.append(f"{lbl}: {wins}/{g.sum()} wins, p={p:.3g}")
        print("  ".join(line))

    I = pd.DataFrame(ident)
    save(I, "table_powerfloor_identity.csv")
    print(f"\n   the free identity N_eff = |sigma| DeltaS, on {len(I)} fitted frontiers:")
    print(f"      median |N_eff/DeltaS - |sigma|| = {I.identity_err.median():.2e}")
    print(f"      median |sigma| = {I.abs_sigma.median():.3f}  "
          f"(cure chart is 1, floorless is 0)")
    print(f"      median fitted beta = {I.beta.median():.3f}, "
          f"median floor share = {I.floor_share.median():.3f}, "
          f"median in-window residual = {I.resid_nats.median():.4f} nats")
    print(f"      frontiers with |sigma| in (0,1), i.e. genuinely in the new cell: "
          f"{int(((I.abs_sigma > 0.02) & (I.abs_sigma < 0.98)).sum())}/{len(I)}")
    return T, S, I


if __name__ == "__main__":
    run()
