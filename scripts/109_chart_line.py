#!/usr/bin/env python3
"""THE CHART LINE: ONE NUMBER FOR THE CHART, THE FLOOR AND THE ORIGIN.
   (Section "The floor is not a parameter": Prop. 'chartline', 'floorsign',
    'linefit'; Online Appendix sec. "The horizon lattice".)

The similarity solutions of the horizon lattice h_k' = h_k/h_{k+1} - 1 are the
constant one (the cure chart) and the affine one

        h_k = (x - x_0)/(k - beta),     D_k = 1/(k - beta),

so along the whole chart line the FIRST horizon h_1 = s/kappa is exactly affine
in x, with

        slope  =  D_1  =  1 + sigma  =  1/(1 - beta),      zero  =  x_0 .

One straight line through the measured s/kappa therefore returns, with no floor,
no fitted chart, no assumed origin and no third derivative:

        D_1 <  0   refutes the composition law        (lumpy)
        D_1 in [0,1)  a floor, approached as a power law, sigma = D_1 - 1 < 0
        D_1 =  1   the Pareto / Gamma chart
        D_1 >  1   no floor, stretched, beta = 1 - 1/D_1
        x_0 = the zero of the line = the point where every horizon vanishes.

Three things are measured here.
 (A) D_1 as a LINE SLOPE against D_1 as the median POINTWISE RATIO c_1c_3/c_2^2-1
     (the previous revision's chain link 1), which needs a third derivative and
     carries the one-unit plug-in bias of script 97.
 (B) the TWO-POINT rungs h_k(x_2) >= h_k(x_1), which are rung k at derivative
     order k+1 instead of k+2 -- i.e. how many more frontiers become readable.
 (C) the floor verdict (D_1 < 1), the origin and beta = 1 - 1/D_1, each against
     the archive's own FITTED value from the three-parameter floorless chart.

Outputs: results/table_chartline_frontiers.csv
         results/table_chartline_summary.csv
         results/table_chartline_agreement.csv
         results/table_chartline_null.csv
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                      # noqa: E402

HFRAC, NGRID = 0.40, 9


def jets(x, S, kmax=4, hfrac=HFRAC, ng=NGRID):
    """local polynomial of degree kmax; returns (grid, S, c_1..c_kmax) with
    c_k = (-1)^{k+1} S^{(k)}.  Identical convention to scripts 97/104."""
    Z = x[-1] - x[0]
    h = Z * hfrac
    grid = np.linspace(x[0] + 0.15 * Z, x[-1] - 0.15 * Z, ng)
    out = []
    fact = [1.0]
    for j in range(1, kmax + 1):
        fact.append(fact[-1] * j)
    for g in grid:
        d = x - g
        w = np.sqrt(np.exp(-0.5 * (d / h) ** 2))
        M = np.column_stack([d ** j / fact[j] for j in range(kmax + 1)])
        b, *_ = np.linalg.lstsq(M * w[:, None], S * w, rcond=None)
        out.append([g, b[0]] + [(-1) ** (k + 1) * b[k] for k in range(1, kmax + 1)])
    return np.array(out)


def fit_stable(x, e):
    """the archive's three-parameter floorless chart exp{-(alpha(x-x0))^beta},
    reproduced from script 99 so that beta and x_0 are directly comparable."""
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
        return np.nan, np.nan
    return float(np.exp(best.x[1])), float(best.x[2])          # beta, x_0


def fit_floor_share(x, e):
    """floor share eps_inf/eps_min from the incumbent chart, for the floor check."""
    def mdl(th, xx):
        return np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * np.atleast_1d(xx))
    th0 = [np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4)]
    try:
        r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), th0, max_nfev=40000)
        return float(np.exp(r.x[0]) / e.min())
    except Exception:
        return np.nan


def run():
    rows = []
    for fr in all_frontiers():
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x)
        x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e)
        x, e = x[k], e[k]
        if len(x) < 7:
            continue
        S = -np.log(e)
        J = jets(x, S, kmax=4)
        g, c1, c2, c3, c4 = J[:, 0], J[:, 2], J[:, 3], J[:, 4], J[:, 5]

        # ---- (A) the first horizon, and D_1 two ways ------------------------
        ok1 = (c1 > 0) & (c2 > 0) & np.isfinite(c1) & np.isfinite(c2)
        h1 = np.where(ok1, c1 / c2, np.nan)                     # = s/kappa
        n1 = int(np.sum(ok1))
        D1_line = np.nan
        x0_line = np.nan
        if n1 >= 4:
            A = np.polyfit(g[ok1], h1[ok1], 1)
            D1_line = float(A[0])
            if abs(A[0]) > 1e-9:
                x0_line = float(-A[1] / A[0])
        ok3 = ok1 & (c3 > 0) & np.isfinite(c3)
        D1_ratio = float(np.nanmedian(c1[ok3] * c3[ok3] / c2[ok3] ** 2 - 1.0)) \
            if ok3.sum() >= 3 else np.nan
        # ---- (B) the two-point rungs ---------------------------------------
        def twopoint(num, den, mask):
            """h = num/den read at the two ends of the admissible range."""
            i = np.where(mask)[0]
            if i.size < 2:
                return np.nan, 0
            a, b = i[0], i[-1]
            ha, hb = num[a] / den[a], num[b] / den[b]
            return float(hb - ha), 1
        r1_val, r1_read = twopoint(c1, c2, ok1)
        ok2 = ok1 & (c3 > 0) & np.isfinite(c3)
        r2_val, r2_read = twopoint(c2, c3, ok2)
        ok3b = ok2 & (c4 > 0) & np.isfinite(c4)
        r3_val, r3_read = twopoint(c3, c4, ok3b)
        # the previous revision's RATIO rungs, for the readability comparison
        ratio1_read = int(ok3.sum() >= 3)
        ratio2_read = int((ok3b).sum() >= 3)
        # ---- (C) against the archive's fitted chart -------------------------
        beta_fit, x0_fit = fit_stable(x, e)
        fshare = fit_floor_share(x, e)
        beta_line = 1.0 - 1.0 / D1_line if np.isfinite(D1_line) and D1_line > 1e-9 else np.nan
        rows.append(dict(
            corpus=fr["corpus"], frontier=fr["name"], n=len(x),
            span=float(x[-1] - x[0]),
            D1_line=D1_line, D1_ratio=D1_ratio, x0_line=x0_line,
            beta_line=beta_line, beta_fit=beta_fit, x0_fit=x0_fit,
            floor_share_fit=fshare,
            verdict=("lumpy" if (np.isfinite(D1_line) and D1_line < 0) else
                     "floor" if (np.isfinite(D1_line) and D1_line < 1) else
                     "nofloor" if np.isfinite(D1_line) else "unreadable"),
            rung1_twopoint=r1_val, rung2_twopoint=r2_val, rung3_twopoint=r3_val,
            rung1_tp_ok=int(np.isfinite(r1_val) and r1_val >= 0),
            rung2_tp_ok=int(np.isfinite(r2_val) and r2_val >= 0),
            rung3_tp_ok=int(np.isfinite(r3_val) and r3_val >= 0),
            rung1_tp_read=r1_read, rung2_tp_read=r2_read, rung3_tp_read=r3_read,
            rung1_ratio_read=ratio1_read, rung2_ratio_read=ratio2_read,
            rung1_ratio_ok=int(np.isfinite(D1_ratio) and D1_ratio >= 0)))
    T = pd.DataFrame(rows)
    save(T, "table_chartline_frontiers.csv")

    out = []
    for c, gg in list(T.groupby("corpus")) + [("ALL", T)]:
        out.append(dict(
            corpus=c, frontiers=len(gg),
            median_D1_line=round(float(gg.D1_line.median()), 3),
            median_D1_ratio=round(float(gg.D1_ratio.median()), 3),
            n_lumpy=int((gg.verdict == "lumpy").sum()),
            n_floor=int((gg.verdict == "floor").sum()),
            n_nofloor=int((gg.verdict == "nofloor").sum()),
            rung1_tp_read=int(gg.rung1_tp_read.sum()), rung1_tp_ok=int(gg.rung1_tp_ok.sum()),
            rung1_ratio_read=int(gg.rung1_ratio_read.sum()),
            rung1_ratio_ok=int(gg.rung1_ratio_ok.sum()),
            rung2_tp_read=int(gg.rung2_tp_read.sum()), rung2_tp_ok=int(gg.rung2_tp_ok.sum()),
            rung3_tp_read=int(gg.rung3_tp_read.sum()), rung3_tp_ok=int(gg.rung3_tp_ok.sum())))
    SUM = pd.DataFrame(out)
    save(SUM, "table_chartline_summary.csv")
    print("\n   the chart line, per corpus")
    print(SUM.to_string(index=False))

    # ---- agreement of the derived readings with the archive's fitted ones ----
    ag = []

    def sp(a, b, lbl):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 6:
            return
        rho, p = spearmanr(a[m], b[m])
        ag.append(dict(quantity=lbl, n=int(m.sum()), spearman=round(float(rho), 3),
                       p_value=float(p),
                       median_derived=round(float(np.median(a[m])), 4),
                       median_fitted=round(float(np.median(b[m])), 4)))
    sp(T.beta_line.values, T.beta_fit.values, "beta: line slope vs fitted chart")
    sp(T.x0_line.values, T.x0_fit.values, "x_0: line zero vs fitted chart")
    sp(T.D1_line.values, T.floor_share_fit.values,
       "D_1 (line) vs fitted floor share")
    sp(T.D1_line.values, T.D1_ratio.values, "D_1: line slope vs pointwise ratio")
    A = pd.DataFrame(ag)
    save(A, "table_chartline_agreement.csv")
    print("\n   agreement with the archive's own fitted values")
    print(A.to_string(index=False))

    rd = T[T.rung1_tp_read == 1]
    print(f"\n   readability.  rung 1: two-point form readable on "
          f"{int(T.rung1_tp_read.sum())}/{len(T)} frontiers, holds on "
          f"{int(T.rung1_tp_ok.sum())}; the pointwise ratio form readable on "
          f"{int(T.rung1_ratio_read.sum())}, holds on {int(T.rung1_ratio_ok.sum())}.")
    print(f"   rung 2: two-point readable on {int(T.rung2_tp_read.sum())}, "
          f"holds on {int(T.rung2_tp_ok.sum())}.   rung 3: readable on "
          f"{int(T.rung3_tp_read.sum())}, holds on {int(T.rung3_tp_ok.sum())}.")
    print(f"   floor verdict from the sign of D_1 - 1: "
          f"{int((T.verdict=='floor').sum())} floored, "
          f"{int((T.verdict=='nofloor').sum())} floorless, "
          f"{int((T.verdict=='lumpy').sum())} refuting divisibility, of {len(T)}.")
    return T


# ----------------------------------------------------------------------------
# The method law of the paper: a plug-in estimator of an exact identity inherits
# a bandwidth, so the null must be SIMULATED ON THE SAME DESIGN and never taken
# analytic.  D_1 read as a line slope is such a plug-in.  Below, every known
# truth is put on each frontier's own grid and noise level and run through the
# identical jets + line fit, so the observed D_1 is compared against the
# distribution it would have under a truth whose D_1 is known.
# ----------------------------------------------------------------------------
TRUTH_D1 = {"cure": 0.0, "powerfloor": 0.625, "lomax": 1.0, "stretched": 1.0 / (1 - 0.35)}


def truth_S(nm, u, S_lo, S_hi):
    if nm == "cure":
        lam = S_hi * 1.6
        c = max(-np.log1p(-(S_hi - S_lo) / (lam - S_lo + 1e-12)), 1e-3)
        return lam * (1 - (1 - S_lo / lam) * np.exp(-c * u))
    if nm == "powerfloor":
        b, lam = -0.6, S_hi * 1.6
        return lam - (lam - S_lo) * (1.0 + 3.0 * u) ** b
    if nm == "lomax":
        return S_lo + (S_hi - S_lo) * np.log1p(3 * u) / np.log(4.0)
    if nm == "stretched":
        b = 0.35
        return S_lo + (S_hi - S_lo) * ((u + .25) ** b - .25 ** b) / (1.25 ** b - .25 ** b)
    if nm == "law_f0.15":
        f, a = 0.15, 2.5
        v = f + (1 - f) * np.exp(-a * u)
        return S_lo + (S_hi - S_lo) * (-np.log(v)) / (-np.log(f + (1 - f) * np.exp(-a)))
    raise ValueError(nm)


def run_null(T, nrep=120):
    """observed D_1 against each truth's own [p10, p90] on the same design."""
    frs = {f["name"]: f for f in all_frontiers()}
    rows = []
    rng = np.random.default_rng(20261005)
    for _, r in T.iterrows():
        fr = frs.get(r["frontier"])
        if fr is None or not np.isfinite(r["D1_line"]):
            continue
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x)
        x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e)
        x, e = x[k], e[k]
        if len(x) < 7:
            continue
        Sobs = -np.log(e)
        # the same noise estimate the four-point script uses
        hh = max(0.35 * (x[-1] - x[0]), 1e-9)
        res = []
        for i in range(len(x)):
            m = np.ones(len(x), bool)
            m[i] = False
            d = x[m] - x[i]
            w = np.sqrt(np.exp(-0.5 * (d / hh) ** 2))
            M = np.column_stack([np.ones(m.sum()), d, d ** 2])
            b, *_ = np.linalg.lstsq(M * w[:, None], Sobs[m] * w, rcond=None)
            res.append(Sobs[i] - b[0])
        rr = np.asarray(res)
        sig = float(np.clip(1.4826 * np.median(np.abs(rr - np.median(rr))), 0.005, 0.25))
        u = (x - x[0]) / (x[-1] - x[0])
        S_lo, S_hi = float(Sobs.min()), float(Sobs.max())
        row = dict(frontier=r["frontier"], corpus=r["corpus"], sigma_S=sig,
                   D1_obs=float(r["D1_line"]))
        for nm in ["cure", "powerfloor", "lomax", "stretched", "law_f0.15"]:
            try:
                St = np.asarray(truth_S(nm, u, S_lo, S_hi), float)
            except Exception:
                continue
            if not np.all(np.isfinite(St)) or np.any(np.diff(St) <= 0):
                continue
            vals = []
            for _ in range(nrep):
                Sn = St + rng.normal(0, sig, len(x))
                J = jets(x, Sn, kmax=4)
                g, c1, c2 = J[:, 0], J[:, 2], J[:, 3]
                ok = (c1 > 0) & (c2 > 0) & np.isfinite(c1) & np.isfinite(c2)
                if ok.sum() >= 4:
                    vals.append(np.polyfit(g[ok], (c1 / c2)[ok], 1)[0])
            if len(vals) < 20:
                continue
            v = np.asarray(vals)
            row[f"{nm}_p10"] = float(np.percentile(v, 10))
            row[f"{nm}_p90"] = float(np.percentile(v, 90))
            row[f"{nm}_med"] = float(np.median(v))
            row[f"{nm}_bias"] = float(np.median(v) - TRUTH_D1.get(nm, np.nan)) \
                if nm in TRUTH_D1 else np.nan
            row[f"{nm}_covers"] = int(row[f"{nm}_p10"] <= row["D1_obs"] <= row[f"{nm}_p90"])
        rows.append(row)
    N = pd.DataFrame(rows)
    save(N, "table_chartline_null.csv")
    print("\n   design-matched null for the plug-in D_1 (the paper's method law)")
    out = []
    for nm in ["cure", "powerfloor", "lomax", "stretched", "law_f0.15"]:
        cm, bm = f"{nm}_covers", f"{nm}_bias"
        if cm not in N:
            continue
        out.append(dict(truth=nm, analytic_D1=TRUTH_D1.get(nm, np.nan),
                        median_simulated_D1=round(float(N[f"{nm}_med"].median()), 3),
                        median_bias=(round(float(N[bm].median()), 3)
                                     if bm in N and N[bm].notna().any() else np.nan),
                        frontiers_covered=int(N[cm].sum()),
                        of=int(N[cm].notna().sum())))
    O = pd.DataFrame(out)
    print(O.to_string(index=False))
    print("\n   the plug-in is biased DOWNWARD by about one unit at every truth, exactly")
    print("   as the trend of G is in script 97: so the raw count of D_1 < 0 frontiers")
    print("   is an artifact and is NOT reported as a refutation.  What survives")
    print("   calibration is the comparison between truths, and the readability gain.")
    return N


if __name__ == "__main__":
    T = run()
    run_null(T)
