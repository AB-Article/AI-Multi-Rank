#!/usr/bin/env python3
"""THE SPECTRAL BREADTH PARAMETER  (Section "What the law predicts" of paper.tex).

Script 68 shows that the three-parameter law predicts a held-out model to a
median 2.6% but PLATEAUS: training on sixteen models is no better than training
on five.  A plateau is misspecification, not estimation error, and this script
locates it and removes it.

Where it is NOT.  The plateau is not the two-atom truncation of Law I.  Fitting
the full completely monotone class -- a non-negative measure on a grid of
exponents, by NNLS with a roughness penalty, which is the standard positive
inversion of a Laplace transform -- ties with the three-parameter law.  Eighty
free weights buy nothing, because a grid plus a ridge penalty smooths away
exactly the broad measures the data want.

Where it is.  Give the reducible part one shape parameter,

    eps(x) = eps_inf + c exp( -( alpha (x - x0) )^beta ),                  (SB)

which is the two-atom law at beta = 1.  For 0 < beta <= 1, exp(-u^beta) is the
Laplace transform of a one-sided stable density and is therefore COMPLETELY
MONOTONE: (SB) is a member of Law I with a broad spectral measure, not a
departure from it.  beta < 1 is a wide spectrum, beta = 1 a single channel,
beta > 1 outside the class altogether.

One trap, recorded because it nearly produced the opposite conclusion.  With x0
pinned to the low end of the measured window, the fitted beta is 1.24 to 1.30
and 56-67% of frontiers exceed 1, which would refute complete monotonicity.
Freeing x0 reverses it: the median beta falls to 0.8-1.06 and the corpora sit
inside the class.  A shape parameter and an origin are confounded over a finite
window, and the origin must be free.

Writes (results/):
    table_breadth_curve.csv      learning curve, every method, per frontier
    table_breadth_summary.csv    pooled by training-set size
    table_breadth_beta.csv       fitted beta per frontier, pinned vs free origin
    table_breadth_profile.csv    profile interval for beta, per frontier
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import least_squares, nnls

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)
_src = open(os.path.join(HERE, "68_prediction_assessment.py")).read().split("if __name__")[0]
_ns = {"__name__": "m68", "__file__": os.path.join(HERE, "68_prediction_assessment.py")}
exec(compile(_src, "68_prediction_assessment.py", "exec"), _ns)
build, predict = _ns["build"], _ns["predict"]

MS = [3, 4, 5, 6, 8, 12, 16]


def fit_sb(x, e, free_origin=True):
    """(SB); x0 free unless pinned to the window edge."""
    def mk(t):
        f, c, a, b = np.exp(t[0]), np.exp(t[1]), np.exp(t[2]), np.exp(t[3])
        x0 = t[4] if free_origin else x.min()
        return lambda xx: f + c * np.exp(-(a * np.maximum(np.atleast_1d(xx) - x0, 0.0)) ** b)
    t0 = np.array([np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4), 0.0, x.min() - 1.0])
    if not free_origin:
        t0 = t0[:4]
    r = least_squares(lambda t: np.log(mk(t)(x)) - np.log(e), t0, max_nfev=60000)
    return mk(r.x), float(np.exp(r.x[3]))


def fit_cm(x, e, ngrid=80, lam=1e-4):
    """the whole completely monotone class, by non-negative least squares."""
    span = max(x.max() - x.min(), 1e-9)
    a = np.r_[0.0, np.exp(np.linspace(np.log(.05 / span), np.log(20 / span), ngrid - 1))]
    A = np.exp(-np.outer(x, a)); s = 1 / np.maximum(e, 1e-12)
    w, _ = nnls(np.vstack([A * s[:, None], np.eye(len(a)) * lam]),
                np.r_[np.ones_like(e), np.zeros(len(a))])
    return lambda xx: np.exp(-np.outer(np.atleast_1d(xx), a)) @ w


def spread(x, m):
    """m training models spread evenly over the window, as an experimenter would pick."""
    g = np.linspace(x.min(), x.max(), m); idx = []
    for t in g:
        for j in np.argsort(np.abs(x - t)):
            if j not in idx:
                idx.append(int(j)); break
    return np.array(sorted(idx))


def noise_proxy(x, e):
    """irreducible scatter: leave-one-out local quadratic residual."""
    n = len(x); r = []
    for i in range(2, n - 2):
        c = np.polyfit(np.r_[x[i - 2:i], x[i + 1:i + 3]],
                       np.log(np.r_[e[i - 2:i], e[i + 1:i + 3]]), 2)
        r.append(abs(np.log(e[i]) - np.polyval(c, x[i])))
    return float(np.median(r)) if r else np.nan


def profile_beta(x, e, grid=None):
    """profile the fit over beta with every other parameter refitted; return the
       minimiser and the set within (1+4/(n-4)) of the minimum RSS."""
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.3), np.log(3.0), 19))
    def rss(b):
        def mk(t):
            f, c, a = np.exp(t[0]), np.exp(t[1]), np.exp(t[2]); x0 = t[3]
            return lambda xx: f + c * np.exp(-(a * np.maximum(np.atleast_1d(xx) - x0, 0.0)) ** b)
        t0 = np.array([np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4), x.min() - 1.0])
        def resid(t):
            v = mk(t)(x)
            if not np.all(np.isfinite(v)) or np.any(v <= 0): return np.full_like(e, 1e3)
            r = np.log(v) - np.log(e)
            return np.where(np.isfinite(r), r, 1e3)
        try:
            return float(np.sum(least_squares(resid, t0, max_nfev=20000).fun ** 2))
        except Exception:
            return np.inf
    rs = np.array([rss(b) for b in grid])
    if not np.any(np.isfinite(rs)): return None
    best = np.nanmin(rs); thr = best * (1 + 4.0 / max(len(x) - 4, 1))
    inset = grid[rs <= thr]
    return dict(beta_hat=float(grid[np.argmin(rs)]), lo=float(inset.min()),
                hi=float(inset.max()), width=float(inset.max() / inset.min()),
                covers_one=bool(inset.min() <= 1 <= inset.max()))


if __name__ == "__main__":
    Fr = build()
    rows, brows = [], []
    for fr in Fr:
        x, e = fr["x"], fr["e"]; o = np.argsort(x); x, e = x[o], e[o]; n = len(x)
        nz = noise_proxy(x, e) if n >= 9 else np.nan
        if n >= 15:
            try:
                _, bp = fit_sb(x, e, False); _, bf = fit_sb(x, e, True)
                brows.append(dict(corpus=fr["corpus"], name=fr["name"], n=n,
                                  beta_pinned=bp, beta_free=bf,
                                  outside_class=bool(bf > 1.05)))
            except Exception:
                pass
        for m in MS:
            if n < m + 3:
                continue
            tr = spread(x, m); te = np.setdiff1d(np.arange(n), tr)
            r = dict(corpus=fr["corpus"], name=fr["name"], n=n, m=m, noise=nz)
            for meth in ["law", "law2", "power", "interp"]:
                try:
                    p = np.atleast_1d(predict(x[tr], e[tr], x[te], meth))
                    r[meth] = float(np.median(np.abs(np.log(np.maximum(p, 1e-12)) - np.log(e[te]))))
                except Exception:
                    r[meth] = np.nan
            for lbl, fn in [("cm", lambda: fit_cm(x[tr], e[tr])),
                            ("sb", lambda: fit_sb(x[tr], e[tr], True)[0])]:
                try:
                    p = np.maximum(np.atleast_1d(fn()(x[te])), 1e-12)
                    r[lbl] = float(np.median(np.abs(np.log(p) - np.log(e[te]))))
                except Exception:
                    r[lbl] = np.nan
            rows.append(r)
    T = pd.DataFrame(rows); B = pd.DataFrame(brows)
    T.to_csv(os.path.join(RES, "table_breadth_curve.csv"), index=False)
    B.to_csv(os.path.join(RES, "table_breadth_beta.csv"), index=False)
    M = ["law", "law2", "power", "cm", "sb", "interp"]
    S = []
    for m in MS:
        g = T[T.m == m].dropna(subset=["law"])
        if len(g) < 5:
            continue
        d = dict(m=m, frontiers=len(g), noise=round(float(g.noise.median()), 4))
        for c in M:
            d[c] = round(float(g[c].median()), 4)
        d["sb_vs_law_pct"] = round(100 * (1 - d["sb"] / d["law"]), 1)
        d["gap_to_noise_closed_pct"] = round(
            100 * (d["law"] - d["sb"]) / max(d["law"] - d["noise"], 1e-9), 1)
        S.append(d)
    S = pd.DataFrame(S); S.to_csv(os.path.join(RES, "table_breadth_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("LEARNING CURVE: train on m models spread over the window, predict the rest")
    print("median |ln eps_pred - ln eps_obs|, nats\n")
    print(S.to_string(index=False))
    print(f"\n(SB) beats the three-parameter law on "
          f"{100*(T.sb < T.law).mean():.0f}% of (frontier, m) cases")
    print("\nfitted beta (frontiers with >=15 Pareto points)\n")
    print(B.groupby("corpus")[["beta_pinned", "beta_free"]].median().round(3).to_string())
    print(f"\n  median beta, pinned origin {B.beta_pinned.median():.2f}  "
          f"-> {100*(B.beta_pinned>1.05).mean():.0f}% outside the class (ARTIFACT)")
    print(f"  median beta, free   origin {B.beta_free.median():.2f}  "
          f"-> {100*(B.beta_free>1.05).mean():.0f}% outside the class")
    prows = []
    for fr in Fr:
        x, e = fr["x"], fr["e"]; o = np.argsort(x); x, e = x[o], e[o]
        if len(x) < 15: continue
        pr = profile_beta(x, e)
        if pr: prows.append(dict(corpus=fr["corpus"], name=fr["name"], n=len(x), **pr))
    P = pd.DataFrame(prows); P.to_csv(os.path.join(RES, "table_breadth_profile.csv"), index=False)
    print("\nIS beta IDENTIFIED?  profile over beta, all other parameters refitted\n")
    print(f"  {'corpus':<13} {'#':>3} {'beta_hat':>9} {'1s width':>9} {'covers 1':>9}")
    for c, g in list(P.groupby("corpus")) + [("ALL", P)]:
        print(f"  {c:<13} {len(g):>3} {g.beta_hat.median():9.2f} {g.width.median():8.1f}x "
              f"{100*g.covers_one.mean():8.0f}%")
    print("\nwrote table_breadth_curve.csv, table_breadth_summary.csv, "
          "table_breadth_beta.csv, table_breadth_profile.csv")
