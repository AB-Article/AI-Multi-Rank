#!/usr/bin/env python3
"""WHICH CHART? A SECOND-ORDER TEST WITH DESIGN-MATCHED NULLS.
   (Section "The floor is not a parameter", Prop. 'chartode' and the paragraph
   "Which chart the archive is in".)

Script 80 shows that in the gauge-invariant coordinates the two charts of the
theory are two first-order ODEs for the single invariant w = kappa/s^2:

    floor + one reducible channel :  dw/dS = + w (1 + w)     (w rises)
    floorless stable (Hougaard)   :  dw/dS = - w / S         (w falls)

They differ at SECOND order, so the test needs only (s, kappa) -- no third
derivative, no fit, no floor, no origin, no units.  The statistic is the trend
of the collapse coordinate along the log-error,

    T  =  d zeta / d S ,      zeta = ln(kappa/s^2) = ln w ,

estimated by a local quadratic.  Per the paper's own method rule a plug-in
reading of an exact identity inherits a bandwidth, so T is NEVER compared with
its analytic value: for every frontier we simulate truths on that frontier's
own x-grid, its own S range and its own estimated noise, and compare T with
those nulls.  Truth families:

    floor + 1 channel, p0(end) in {0.25, 0.50, 0.75, 0.90}
    floor + 2 channels (exponent ratio 3)
    floorless stable,  beta in {0.15, 0.25, 0.35, 0.50, 0.70, 0.90}

Because T is monotone in beta within the stable family, inverting the null
medians gives a FIT-FREE, null-calibrated estimator of the shape index, which
is reported per frontier and compared with the fitted beta of script 69/76.

The same machinery calibrates the two one-sided diagnostics of script 81: the
trend of the local floor ln F, which Thm. 'monofloor' forces to be <= 0, and
the event {fitted floor > F_right}.

Outputs: results/table_chart_test_per_frontier.csv
         results/table_chart_test_nulls.csv
         results/table_beta_from_invariant.csv
         results/table_chart_test_summary.csv
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save, all_frontiers

RNG = np.random.default_rng(20261005)
HFRAC, NGRID, NREP = 0.40, 9, 100
P0_SET = (0.25, 0.50, 0.75, 0.90)
BETA_SET = (0.15, 0.25, 0.35, 0.50, 0.70, 0.90)


# ------------------------------------------------------------- estimators ---
def hat_ops(x, hfrac=HFRAC, ng=NGRID):
    """Pre-build the local-quadratic hat operators: b = H_g @ S gives (S,S',S'')."""
    Z = x[-1] - x[0]; h = Z * hfrac
    grid = np.linspace(x[0] + 0.15 * Z, x[-1] - 0.15 * Z, ng)
    H = []
    for g in grid:
        d = x - g; w = np.exp(-0.5 * (d / h) ** 2)
        M = np.column_stack([np.ones_like(d), d, 0.5 * d ** 2])
        A = M.T * w
        H.append(np.linalg.solve(A @ M, A))
    return np.stack(H)                               # (ng, 3, n)


def _slope(Xg, Yg, W):
    """weighted least-squares slope of Y on X, row-wise (W is 0/1 mask)."""
    n = W.sum(1)
    mx = (Xg * W).sum(1) / n; my = (Yg * W).sum(1) / n
    dx = (Xg - mx[:, None]) * W; dy = (Yg - my[:, None]) * W
    den = (dx * dx).sum(1)
    out = np.full(len(n), np.nan)
    good = (n >= 5) & (den > 1e-12)
    out[good] = (dx * dy).sum(1)[good] / den[good]
    return out


def stats_from(H, Smat):
    """Vectorised over replicates. Smat (nrep, n) -> arrays of T and dlnF/dS."""
    B = np.einsum("gkn,rn->rgk", H, Smat)            # (nrep, ng, 3)
    Sg, s, kap = B[..., 0], B[..., 1], -B[..., 2]
    W = ((s > 0) & (kap > 0)).astype(float)
    with np.errstate(all="ignore"):
        z = np.log(np.where(W > 0, kap / np.maximum(s, 1e-300) ** 2, 1.0))
        lF = np.log(np.where(W > 0, kap * np.exp(-Sg) / (s ** 2 + kap), 1.0))
    W = W * np.isfinite(z) * np.isfinite(lF)
    return _slope(Sg, z, W), _slope(Sg, lF, W)


def stats(x, S, H=None):
    if H is None:
        H = hat_ops(x)
    T, G = stats_from(H, S[None, :])
    return float(T[0]), float(G[0])


def noise_level(x, S):
    """leave-one-out local-quadratic residual scale, in nats of log-error."""
    Z = x[-1] - x[0]; h = max(Z * 0.35, 1e-9); r = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        d = x[m] - x[i]; w = np.exp(-0.5 * (d / h) ** 2)
        M = np.column_stack([np.ones_like(d), d, 0.5 * d ** 2])
        Wr = np.sqrt(w)[:, None]
        try:
            b, *_ = np.linalg.lstsq(M * Wr, S[m] * Wr[:, 0], rcond=None)
        except Exception:
            continue
        r.append(S[i] - b[0])
    r = np.asarray(r)
    return float(np.clip(1.4826 * np.median(np.abs(r - np.median(r))), 0.005, 0.25))


# ------------------------------------------------------------------ truths ---
def gen_floor(x, S, p0_end, k_ch=1):
    Smin, Smax = S.min(), S.max(); Z = x[-1] - x[0]
    ei = np.exp(-Smax) * p0_end
    A, B = np.exp(-Smin) - ei, np.exp(-Smax) - ei
    if A <= 0 or B <= 0:
        return None
    al = np.log(A / B) / Z
    if k_ch == 1:
        return ei + A * np.exp(-al * (x - x[0]))
    e = ei + 0.5 * A * np.exp(-al * (x - x[0])) + 0.5 * A * np.exp(-3 * al * (x - x[0]))
    return e if (e > 0).all() else None


def gen_stable(x, S, beta):
    Smin, Smax = S.min(), S.max(); Z = x[-1] - x[0]
    q = (Smax / max(Smin, 1e-9)) ** (1.0 / beta)
    if not np.isfinite(q) or q <= 1.0001 or q > 1e12:
        return None
    d = Z / (q - 1.0)
    return np.exp(-Smin * ((x - (x[0] - d)) / d) ** beta)


TRUTHS = ([(f"floor_p0_{p:.2f}", lambda x, S, p=p: gen_floor(x, S, p)) for p in P0_SET]
          + [("floor_2ch", lambda x, S: gen_floor(x, S, 0.5, k_ch=2))]
          + [(f"stable_beta_{b:.2f}", lambda x, S, b=b: gen_stable(x, S, b)) for b in BETA_SET])


def null_dist(x, S, gen, sig, H, nrep=NREP):
    eps = gen(x, S)
    if eps is None:
        return np.array([]), np.array([])
    Sm = -np.log(eps)[None, :] + sig * RNG.standard_normal((nrep, len(x)))
    T, G = stats_from(H, Sm)
    m = np.isfinite(T)
    return T[m], G[m]


def fit_floor_share(x, eps):
    """p_0 at the right edge implied by the frontier's OWN 3-parameter floor fit."""
    y = np.log(eps)

    def r(th):
        return np.log(np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * x)) - y
    best = None
    for e0 in (-6., -3., -1.5):
        for a0 in (-1., 0., 1.):
            try:
                sol = least_squares(r, [e0, y[0], a0], method="lm", max_nfev=8000)
            except Exception:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
    if best is None:
        return np.nan
    return float(np.clip(np.exp(best.x[0]) / eps.min(), 0.02, 0.98))


def fit_beta(x, eps):
    """the fitted shape index, free origin (script 69's form), for comparison."""
    y = np.log(eps)

    def r(th):
        lc, la, lb, x0 = th
        u = np.clip(x - x0, 1e-9, None)
        return lc - (np.exp(la) * u) ** np.exp(lb) - y
    best = None
    for b0 in (-1.0, -0.3, 0.0):
        for off in (0.5, 2.0, 6.0):
            try:
                s = least_squares(r, [y[0], 0.0, b0, x[0] - off], method="lm", max_nfev=6000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    return float(np.exp(best.x[2])) if best is not None else np.nan


# --------------------------------------------------------------------- run ---
def run():
    per, nulls = [], []
    for fr in all_frontiers():
        x, eps = fr["x"], fr["e"]
        o = np.argsort(x); x, eps = x[o], eps[o]
        S = -np.log(eps)
        H = hat_ops(x)
        T, G = stats(x, S, H)
        if not np.isfinite(T):
            continue
        sig = noise_level(x, S)
        p0f = fit_floor_share(x, eps)
        row = dict(corpus=fr["corpus"], frontier=fr["name"], n=len(x),
                   span=float(x[-1] - x[0]), sigma=sig, T_obs=T, dlogF_obs=G,
                   beta_fit=fit_beta(x, eps), p0_fitted=p0f)
        # THE internal-consistency test: the floor chart's null at the frontier's
        # OWN fitted floor share.  If the archive were in the floor chart, T_obs
        # would sit inside this null; the fitted floor share and the measured
        # invariant are then two readings of one quantity.
        if np.isfinite(p0f):
            Tn, Gn = null_dist(x, S, lambda xx, SS, p=p0f: gen_floor(xx, SS, p), sig, H)
            if Tn.size >= 20:
                row["T_null_ownp0_med"] = float(np.median(Tn))
                row["T_null_ownp0_p90"] = float(np.percentile(Tn, 90))
                row["exceeds_own_null"] = float(T > np.percentile(Tn, 90))
        med = {}
        for name, gen in TRUTHS:
            Tn, Gn = null_dist(x, S, gen, sig, H)
            if Tn.size < 20:
                med[name] = np.nan; continue
            med[name] = float(np.median(Tn))
            nulls.append(dict(corpus=fr["corpus"], frontier=fr["name"], truth=name,
                              n_rep=int(Tn.size), T_med=float(np.median(Tn)),
                              T_p10=float(np.percentile(Tn, 10)),
                              T_p90=float(np.percentile(Tn, 90)),
                              dlogF_med=float(np.median(Gn)),
                              dlogF_p90=float(np.percentile(Gn, 90)),
                              p_T_above_obs=float(np.mean(Tn > T))))
        # which single truth is closest, and the calibrated beta
        cand = {k: v for k, v in med.items() if np.isfinite(v)}
        if cand:
            row["closest_truth"] = min(cand, key=lambda k: abs(cand[k] - T))
            row["best_floor_truth"] = min([k for k in cand if k.startswith("floor")],
                                          key=lambda k: abs(cand[k] - T), default="")
        # invert the stable family's monotone map T(beta) -> beta
        bs = [(b, med.get(f"stable_beta_{b:.2f}", np.nan)) for b in BETA_SET]
        bs = [(b, t) for b, t in bs if np.isfinite(t)]
        row["beta_invariant"] = np.nan
        if len(bs) >= 2:
            bb = np.array([b for b, _ in bs]); tt = np.array([t for _, t in bs])
            k = np.argsort(tt)
            if tt[k][0] <= T <= tt[k][-1]:
                row["beta_invariant"] = float(np.interp(T, tt[k], bb[k]))
            elif T > tt[k][-1]:
                row["beta_invariant"] = float(bb[k][-1])
            else:
                row["beta_invariant"] = float(bb[k][0])
        per.append(row)
    P = pd.DataFrame(per); N = pd.DataFrame(nulls)
    return P, N


def summarise(P, N):
    rows = [dict(quantity="frontiers with a usable second-order reading", value=float(len(P))),
            dict(quantity="observed T = dzeta/dS, median", value=float(P.T_obs.median())),
            dict(quantity="observed T, p10", value=float(P.T_obs.quantile(.10))),
            dict(quantity="observed T, p90", value=float(P.T_obs.quantile(.90)))]
    for name, _ in TRUTHS:
        g = N[N.truth == name]
        if len(g) == 0:
            continue
        # pooled: fraction of frontiers whose observed T lies inside that truth's [p10,p90]
        m = P.merge(g[["frontier", "T_med", "T_p10", "T_p90"]], on="frontier", how="inner")
        inside = float(np.mean((m.T_obs >= m.T_p10) & (m.T_obs <= m.T_p90)))
        rows.append(dict(quantity=f"null {name}: median T (pooled) / frac obs inside p10-p90",
                         value=float(g.T_med.median())))
        rows.append(dict(quantity=f"null {name}: fraction of frontiers inside p10-p90",
                         value=inside))
    # primary comparison: the arc route of script 77, which the paper recommends
    arc_path = os.path.join(os.path.dirname(HERE), "results", "table_shape_from_arc.csv")
    if os.path.exists(arc_path):
        A = pd.read_csv(arc_path)
        m = P.merge(A, left_on="frontier", right_on="vkey", how="inner") \
             .dropna(subset=["beta_invariant", "beta_chord"])
        if len(m) > 4:
            rows += [dict(quantity="beta_invariant vs beta_chord (arc route): Spearman",
                          value=float(spearmanr(m.beta_invariant, m.beta_chord).statistic)),
                     dict(quantity="beta_invariant vs beta_chord: n", value=float(len(m))),
                     dict(quantity="beta_invariant median (same frontiers)",
                          value=float(m.beta_invariant.median())),
                     dict(quantity="beta_chord median (same frontiers)",
                          value=float(m.beta_chord.median()))]
    ok = P.dropna(subset=["beta_invariant", "beta_fit"])
    ok = ok[(ok.beta_fit > 0.05) & (ok.beta_fit < 3)]
    if len(ok) > 4:
        rows += [dict(quantity="beta_invariant vs a per-frontier free-origin fit: Spearman",
                      value=float(spearmanr(ok.beta_invariant, ok.beta_fit).statistic)),
                 dict(quantity="beta_invariant vs beta_fit: n", value=float(len(ok))),
                 dict(quantity="beta_invariant median", value=float(ok.beta_invariant.median())),
                 dict(quantity="beta_fit median", value=float(ok.beta_fit.median()))]
    # pooled chart comparison: distance of the observed median T to each null median
    for name, _ in TRUTHS:
        g = N[N.truth == name]
        if len(g) == 0:
            continue
        m = P.merge(g[["frontier", "T_med"]], on="frontier", how="inner")
        rows.append(dict(quantity=f"null {name}: median |T_obs - T_null| per frontier",
                         value=float(np.median(np.abs(m.T_obs - m.T_med)))))
    # THE internal-consistency test
    Q = P.dropna(subset=["T_null_ownp0_p90"]) if "T_null_ownp0_p90" in P else P.iloc[:0]
    if len(Q):
        rows += [dict(quantity="frontiers with a floor-chart null at their own fitted p_0",
                      value=float(len(Q))),
                 dict(quantity="fraction with T_obs above that null's p90",
                      value=float(Q.exceeds_own_null.mean())),
                 dict(quantity="median T_obs / T_null(own p_0)",
                      value=float(np.median(Q.T_obs / Q.T_null_ownp0_med))),
                 dict(quantity="median fitted floor share p_0", value=float(Q.p0_fitted.median()))]
        D = Q[Q.span >= 4.0]
        if len(D) > 4:
            rows += [dict(quantity="deep windows (span >= 4 nats of resource): n",
                          value=float(len(D))),
                     dict(quantity="deep windows: fraction above the own-p_0 null p90",
                          value=float(D.exceeds_own_null.mean())),
                     dict(quantity="deep windows: median T_obs / T_null(own p_0)",
                          value=float(np.median(D.T_obs / D.T_null_ownp0_med)))]
    # the one-sided floor diagnostic, calibrated
    m = P.merge(N[N.truth == "floor_p0_0.50"][["frontier", "dlogF_p90"]], on="frontier", how="inner")
    rows.append(dict(quantity="d lnF/dS above the floor-chart null p90 (fraction)",
                     value=float(np.mean(m.dlogF_obs > m.dlogF_p90))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(__doc__.split("Outputs")[0])
    P, N = run()
    S = summarise(P, N)
    print(S.to_string(index=False))
    print("\nby corpus (median):")
    print(P.groupby("corpus")[["T_obs", "beta_invariant", "beta_fit"]].median().to_string())
    save(P, "table_chart_test_per_frontier.csv")
    save(N, "table_chart_test_nulls.csv")
    save(P[["corpus", "frontier", "n", "span", "sigma", "T_obs",
            "beta_invariant", "beta_fit"]], "table_beta_from_invariant.csv")
    save(S, "table_chart_test_summary.csv")
