#!/usr/bin/env python3
"""THE KNEE CRITERION AGAINST A DESIGN-MATCHED NULL.
   (Section "Law II", Prop. 'knee' and the paragraph "What the archive can and
   cannot say about the knee"; Section "Reading a frontier without fitting it",
   the method rule.)

Script 91 reads W = max_x w directly off the archive and gets a median of 42,
which would mean a local floor share p_0 = W/(1+W) = 0.98.  That is the same
inflation script 82 found for the second-order invariant, and for the same
reason: W is a plug-in maximum of a ratio of local-quadratic derivatives, so it
inherits a bandwidth AND a maximum over nine grid points.  By the paper's own
method rule it must never be compared with its analytic value.

Here the comparison is done properly.  For every frontier we simulate, on THAT
frontier's x-grid and with its own estimated noise, two truths:

    floor + one channel at p_0(end) = 0.5    (W is infinite in truth)
    floorless shape law, beta = 0.25         (W is FINITE and small in truth)

and record the null distribution of the plug-in W.  Three readings follow:

  C1  the inflation factor: the null median W for the floorless truth, whose
      true W over the window is computable exactly, tells us how much of the
      observed W is bandwidth;
  C2  the calibrated criterion: a frontier has a knee in-window only if its
      observed W exceeds the p90 of the FLOORLESS null -- i.e. only if it is
      too peaked to be a frontier whose w never reaches 1;
  C3  the tangency test of script 91 repeated with the calibrated statistic.

Outputs: results/table_knee_null.csv
         results/table_knee_null_summary.csv
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save, all_frontiers

RES = os.path.join(os.path.dirname(HERE), "results")
RNG = np.random.default_rng(20261005)
HFRAC, NGRID, NREP = 0.40, 9, 120


def hat_ops(x, hfrac=HFRAC, ng=NGRID):
    Z = x[-1] - x[0]; h = Z * hfrac
    grid = np.linspace(x[0] + .15 * Z, x[-1] - .15 * Z, ng)
    H = []
    for g in grid:
        d = x - g; wq = np.exp(-.5 * (d / h) ** 2)
        M = np.column_stack([np.ones_like(d), d, .5 * d ** 2])
        A = M.T * wq
        H.append(np.linalg.solve(A @ M, A))
    return np.stack(H)


def W_from(H, Smat):
    B = np.einsum("gkn,rn->rgk", H, Smat)
    s, kap = B[..., 1], -B[..., 2]
    w = np.where((s > 0) & (kap > 0), kap / np.maximum(s, 1e-300) ** 2, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmax(w, axis=1)


def noise_level(x, S):
    Z = x[-1] - x[0]; h = max(Z * .35, 1e-9); r = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        d = x[m] - x[i]; wq = np.exp(-.5 * (d / h) ** 2)
        M = np.column_stack([np.ones_like(d), d, .5 * d ** 2])
        Wr = np.sqrt(wq)[:, None]
        b, *_ = np.linalg.lstsq(M * Wr, S[m] * Wr[:, 0], rcond=None)
        r.append(S[i] - b[0])
    r = np.array(r)
    return float(np.clip(1.4826 * np.median(np.abs(r - np.median(r))), 0.005, 0.25))


def truth_floor(x, S):
    """floor + one channel reproducing the window's S range with p0(end)=0.5."""
    S0, S1 = S.min(), S.max()
    e1 = np.exp(-S1); einf = 0.5 * e1
    c = (np.exp(-S0) - einf)
    al = (S1 - S0) / max(x[-1] - x[0], 1e-9)
    return -np.log(einf + c * np.exp(-al * (x - x[0])))


def truth_stable(x, S, beta):
    S0, S1 = max(S.min(), 1e-3), S.max()
    u0 = S0 ** (1 / beta); u1 = S1 ** (1 / beta)
    u = u0 + (u1 - u0) * (x - x[0]) / max(x[-1] - x[0], 1e-9)
    return u ** beta


def true_W_stable(S, beta):
    """exact W for the floorless chart over the window: w = (1-b)/(b S) is
    decreasing in S, so its maximum is at the shallow end."""
    return (1 - beta) / (beta * max(S.min(), 1e-6))


def run():
    K = pd.read_csv(os.path.join(RES, "table_knee_archive.csv"))
    obs = dict(zip(K.frontier, K.W)); gap = dict(zip(K.frontier, K.route_gap))
    rows = []
    for fr in all_frontiers():
        x, e = fr["x"], fr["e"]
        o = np.argsort(x); x, e = x[o], e[o]
        if len(x) < 7 or e.min() <= 0 or fr["name"] not in obs:
            continue
        S = -np.log(e)
        sig = noise_level(x, S)
        H = hat_ops(x)
        r = dict(corpus=fr["corpus"], frontier=fr["name"], n=len(x), sigma=sig,
                 W_obs=float(obs[fr["name"]]), route_gap=gap.get(fr["name"], np.nan))
        for tag, St, trueW in (("floor_p0_0.5", truth_floor(x, S), np.inf),
                               ("stable_0.25", truth_stable(x, S, 0.25),
                                true_W_stable(S, 0.25))):
            Sm = St[None, :] + RNG.normal(0, sig, (NREP, len(x)))
            Wn = W_from(H, Sm)
            Wn = Wn[np.isfinite(Wn)]
            if len(Wn) < 20:
                continue
            r[f"W_null_med_{tag}"] = float(np.median(Wn))
            r[f"W_null_p90_{tag}"] = float(np.percentile(Wn, 90))
            r[f"W_true_{tag}"] = float(trueW)
        rows.append(r)
    T = pd.DataFrame(rows)
    save(T, "table_knee_null.csv")

    st = T.dropna(subset=["W_null_med_stable_0.25"])
    infl = st["W_null_med_stable_0.25"] / st["W_true_stable_0.25"]
    above = st.W_obs > st["W_null_p90_stable_0.25"]
    G = st.dropna(subset=["route_gap"])
    calib = G.W_obs / G["W_null_med_stable_0.25"]
    rho_raw = spearmanr(G.W_obs, G.route_gap)
    rho_cal = spearmanr(calib, G.route_gap)
    rows2 = [
        dict(quantity="frontiers with a design-matched null", value=float(len(T))),
        dict(quantity="C1 median plug-in W under the FLOORLESS truth (beta=0.25)",
             value=float(st["W_null_med_stable_0.25"].median())),
        dict(quantity="C1 median TRUE W of that truth over the same window",
             value=float(st["W_true_stable_0.25"].median())),
        dict(quantity="C1 inflation factor of the plug-in maximum (median)",
             value=float(np.median(infl))),
        dict(quantity="C1 median observed W", value=float(st.W_obs.median())),
        dict(quantity="C1 median observed W / null median (floorless truth)",
             value=float(np.median(st.W_obs / st['W_null_med_stable_0.25']))),
        dict(quantity="C2 frontiers whose W exceeds the floorless null p90",
             value=float(above.sum())),
        dict(quantity="C2 fraction (nominal 0.10 if the floorless truth holds)",
             value=float(above.mean())),
        dict(quantity="C2 median plug-in W under the FLOOR truth (W true = infinity)",
             value=float(st["W_null_med_floor_p0_0.5"].median())),
        dict(quantity="C3 Spearman(raw W, route gap)", value=float(rho_raw.statistic)),
        dict(quantity="C3 p-value, raw", value=float(rho_raw.pvalue)),
        dict(quantity="C3 Spearman(null-calibrated W, route gap)", value=float(rho_cal.statistic)),
        dict(quantity="C3 p-value, calibrated", value=float(rho_cal.pvalue)),
        dict(quantity="C3 n", value=float(len(G))),
    ]
    S2 = pd.DataFrame(rows2)
    save(S2, "table_knee_null_summary.csv")
    print(S2.to_string(index=False))


if __name__ == "__main__":
    run()
