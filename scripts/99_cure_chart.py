#!/usr/bin/env python3
"""THE CURE CHART, HELD OUT AGAINST THE OTHER THREE-PARAMETER CHARTS.
   (Section "The price is a frontier", Prop. 'cure'; Section "What the evidence
    says", Table 'curepred'.)

The composition law admits a chart the archive has never fitted: the divisible
chart WITH a floor,

    eps(x) = exp{ -lam (1 - e^{-c(x-x_0)}) } ,    eps_inf = e^{-lam},

the promotion-time cure model.  It has the same three parameters as the two
charts it competes with, and it is the unique chart for which

    N_eff = ln(eps/eps_inf)      (channels = nats remaining, exactly)
    sigma = -1 ,   kappa/s = c , G = s^2/kappa - ln eps = -ln eps_inf
    eps* = e eps_inf             (the canonical point, one nat above the floor)
    L    = 2 sqrt(-ln eps_inf)

so if it predicts as well as the others, every one of those identities becomes
available at no cost in accuracy, and the operating rule needs no fit at all.

Protocol: exactly script 68's, on the same frontiers and the same splits, so the
numbers are directly comparable to Table 'prediction'.

    INTERIOR   leave one interior point out, predict it
    FORWARD    train on the cheapest 2/3, predict the dearest 1/3
    BACKWARD   train on the dearest 2/3, predict the cheapest 1/3

Charts compared (parameter count in brackets):
    cure   [3]  exp{-lam(1-e^{-c(x-x0)})}            divisible, floored
    law    [3]  eps_inf + c e^{-alpha x}             indivisible, floored
    stable [3]  exp{-(alpha(x-x0))^beta}             divisible, floorless
    law2   [5]  eps_inf + c1 e^{-a1 x} + c2 e^{-a2 x}
    power  [2]  c e^{-alpha x}
    interp [-]  log-linear through the neighbours

Outputs: results/table_cure_prediction_frontiers.csv
         results/table_cure_prediction_summary.csv
         results/table_cure_identities_archive.csv
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save, all_frontiers


# ------------------------------------------------------------------- charts ---
def fit_cure(x, e):
    """eps = exp{-lam (1 - e^{-c (x - x0)})}, fitted in log-error.

    Parameters (log lam, log c, x0).  S = lam (1 - u0 e^{-c(x-x_min)}) with
    u0 = e^{-c(x_min-x0)} in (0,1], so the chart is reparameterised as
    (log lam, log c, logit u0) to keep every trial admissible.
    """
    y = np.log(e); x0 = x[0]

    def mdl(th, xx):
        lam = np.exp(th[0]); c = np.exp(th[1]); u0 = 1.0 / (1.0 + np.exp(-th[2]))
        return -lam * (1.0 - u0 * np.exp(-c * (np.atleast_1d(xx) - x0)))   # = ln eps

    best = None
    S0 = -y[0]
    for lam0 in (S0 * 1.5, -np.log(e.min()) + 2.0):
        if not np.isfinite(lam0) or lam0 <= 0:
            continue
        for c0 in (0.3, 1.2):
            u00 = np.clip(1.0 - S0 / lam0, 1e-6, 1 - 1e-6)
            th0 = [np.log(lam0), np.log(c0), np.log(u00 / (1 - u00))]
            try:
                sol = least_squares(lambda th: mdl(th, x) - y, th0, max_nfev=20000)
            except Exception:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
    if best is None:
        return None, None
    return (lambda xx: np.exp(mdl(best.x, xx))), best


def fit_law(x, e, floor=True, k=1):
    def mdl(th, xx):
        f = np.exp(th[0]) if floor else 0.0
        v = np.full_like(np.atleast_1d(xx), f, dtype=float)
        for j in range(k):
            v = v + np.exp(th[1 + 2 * j]) * np.exp(-np.exp(th[2 + 2 * j]) * np.atleast_1d(xx))
        return v
    th0 = [np.log(max(e.min() * .5, 1e-9))]
    for j in range(k):
        th0 += [np.log(e.max() / (j + 1)), np.log(.4 * (j + 1) ** 2)]
    r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), np.array(th0), max_nfev=40000)
    return (lambda xx: mdl(r.x, xx)), r


def fit_stable(x, e):
    """exp{-(alpha(x-x0))^beta}: the divisible floorless chart, free origin."""
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
    if best is None:
        return None, None
    return (lambda xx: np.exp(mdl(best.x, xx))), best


def predict(xtr, etr, xte, method):
    if method == "cure":
        f, _ = fit_cure(xtr, etr)
        return f(xte) if f is not None else np.full_like(np.atleast_1d(xte), np.nan, dtype=float)
    if method == "law":
        return fit_law(xtr, etr)[0](xte)
    if method == "law2":
        return fit_law(xtr, etr, k=2)[0](xte)
    if method == "stable":
        f, _ = fit_stable(xtr, etr)
        return f(xte) if f is not None else np.full_like(np.atleast_1d(xte), np.nan, dtype=float)
    if method == "power":
        return fit_law(xtr, etr, floor=False)[0](xte)
    if method == "interp":
        o = np.argsort(xtr)
        return np.exp(np.interp(xte, xtr[o], np.log(etr[o])))
    raise ValueError(method)


# law2 (five parameters) is already tabulated by script 68; the comparison here is
# between the three THREE-parameter charts plus two parameter-free baselines.
METHODS = ["cure", "law", "stable", "power", "interp"]
REGIMES = ["INTERIOR", "FORWARD", "BACKWARD"]


def assess(fr):
    x, e = fr["x"], fr["e"]; n = len(x)
    o = np.argsort(x); x, e = x[o], e[o]
    rec = {m: {r: [] for r in REGIMES} for m in METHODS}
    po = {m: [] for m in METHODS}
    for i in range(1, n - 1):
        msk = np.ones(n, bool); msk[i] = False
        for m in METHODS:
            try:
                p = float(np.atleast_1d(predict(x[msk], e[msk], x[i], m))[0])
            except Exception:
                continue
            if not np.isfinite(p) or p <= 0:
                continue
            rec[m]["INTERIOR"].append(abs(np.log(p) - np.log(e[i])))
            po[m].append((np.log(p), np.log(e[i])))
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
    return row, {m: np.array(po[m]) for m in METHODS}


# ----------------------------------- the cure chart's identities, on the data ---
def cure_identities(fr):
    """Fit the cure chart to the whole frontier and read off what it then asserts
    with no further freedom: eps* = e eps_inf, N_eff = ln(eps/eps_inf), L = 2 sqrt(lam)."""
    x, e = fr["x"], fr["e"]
    o = np.argsort(x); x, e = x[o], e[o]
    f, sol = fit_cure(x, e)
    if f is None:
        return None
    lam = float(np.exp(sol.x[0])); c = float(np.exp(sol.x[1]))
    u0 = float(1.0 / (1.0 + np.exp(-sol.x[2])))
    einf = float(np.exp(-lam))
    S = -np.log(e)
    # the canonical point: DeltaS = 1 nat, i.e. eps = e * eps_inf
    estar = np.e * einf
    inside = bool(e.min() <= estar <= e.max())
    resid = float(np.median(np.abs(np.log(f(x)) - np.log(e))))
    return dict(corpus=fr["corpus"], frontier=fr["name"], n=len(x),
                span=float(x[-1] - x[0]), lam=lam, c=c, u0=u0, eps_inf=einf,
                median_abs_resid_nats=resid,
                eps_star=estar, canonical_inside_window=int(inside),
                arc_total=2.0 * np.sqrt(lam),
                nats_remaining_right=float(np.log(e.min() / einf)) if einf > 0 else np.nan)


def run():
    Fr = list(all_frontiers())
    rows, po, ident = [], {m: [] for m in METHODS}, []
    for i, fr in enumerate(Fr, 1):
        try:
            r, p = assess(fr)
        except Exception as ex:
            print("  skip", fr["name"], ex, flush=True); continue
        print(f"  [{i}/{len(Fr)}] {fr['name'][:40]:40s} "
              f"cure={r['cure_INTERIOR']:.4f} law={r['law_INTERIOR']:.4f} "
              f"stable={r['stable_INTERIOR']:.4f}", flush=True)
        rows.append(r)
        for m in METHODS:
            if p[m].size:
                po[m].append(p[m])
        ci = cure_identities(fr)
        if ci is not None:
            ident.append(ci)
    T = pd.DataFrame(rows)
    save(T, "table_cure_prediction_frontiers.csv")

    out = []
    for corpus, g in list(T.groupby("corpus")) + [("ALL", T)]:
        d = dict(corpus=corpus, frontiers=len(g), median_n=float(g.n.median()),
                 median_span=round(float(g.span.median()), 2))
        for m in METHODS:
            for lbl in REGIMES:
                d[f"{m}_{lbl}"] = round(float(g[f"{m}_{lbl}"].median()), 4)
        out.append(d)
    S2 = pd.DataFrame(out)
    save(S2, "table_cure_prediction_summary.csv")
    I = pd.DataFrame(ident)
    save(I, "table_cure_identities_archive.csv")

    # pooled relative error, the paper's own headline metric
    pooled = []
    for m in METHODS:
        if not po[m]:
            continue
        A = np.vstack(po[m]); A = A[np.isfinite(A).all(1)]
        ss_res = np.sum((A[:, 0] - A[:, 1]) ** 2)
        ss_tot = np.sum((A[:, 1] - A[:, 1].mean()) ** 2)
        pooled.append(dict(method=m, n_points=len(A),
                           R2_log=round(float(1 - ss_res / ss_tot), 4),
                           median_abs_nats=round(float(np.median(np.abs(A[:, 0] - A[:, 1]))), 4),
                           median_rel_pct=round(float(100 * np.median(
                               np.abs(np.exp(A[:, 0] - A[:, 1]) - 1))), 2)))
    Pl = pd.DataFrame(pooled)
    save(Pl, "table_cure_prediction_pooled.csv")
    return T, S2, I, Pl


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    T, S2, I, Pl = run()
    print(f"\n{len(T)} frontiers.  MEDIAN |ln eps_pred - ln eps_obs| (nats)\n")
    print(S2.to_string(index=False), "\n")
    print("POOLED held-out (INTERIOR), the paper's headline metric\n")
    print(Pl.to_string(index=False), "\n")
    for lbl in REGIMES:
        a = T[f"cure_{lbl}"]; b = T[f"law_{lbl}"]; c = T[f"stable_{lbl}"]
        m = np.isfinite(a) & np.isfinite(b)
        try:
            p_ab = float(wilcoxon(a[m], b[m]).pvalue)
        except Exception:
            p_ab = np.nan
        m2 = np.isfinite(a) & np.isfinite(c)
        try:
            p_ac = float(wilcoxon(a[m2], c[m2]).pvalue)
        except Exception:
            p_ac = np.nan
        print(f"{lbl:9s} cure {a.median():.4f} | law {b.median():.4f} | stable {c.median():.4f} "
              f"  cure better than law on {int((a[m] < b[m]).sum())}/{int(m.sum())} "
              f"(signed-rank p={p_ab:.3g}); than stable on "
              f"{int((a[m2] < c[m2]).sum())}/{int(m2.sum())} (p={p_ac:.3g})")
    print(f"\nCURE-CHART IDENTITIES ON THE ARCHIVE ({len(I)} frontiers fitted):")
    print(f"  median in-window fit residual      : {I.median_abs_resid_nats.median():.4f} nats")
    print(f"  median fitted floor eps_inf        : {I.eps_inf.median():.5f}")
    print(f"  median total arc 2 sqrt(-ln eps_inf): {I.arc_total.median():.3f}")
    print(f"  canonical point eps* = e eps_inf inside the measured window on "
          f"{int(I.canonical_inside_window.sum())}/{len(I)} frontiers")
    print(f"  median nats remaining at the right edge: "
          f"{I.nats_remaining_right.median():.3f}")
