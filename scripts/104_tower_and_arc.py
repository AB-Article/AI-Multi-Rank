#!/usr/bin/env python3
"""THE TOWER, AND THE ARC AS A SQUARE ROOT OF DEPTH.
   (Section "The price is a frontier", Prop. 'tower', Prop. 'arcdepth'.)

Two structural claims of this revision, measured on the archive.

(A) THE TOWER.  Under the composition law every c_k = (-1)^{k+1} S^{(k)} is
itself a completely monotone frontier, so the invariant construction of Law IV
applies at EVERY level and closes:

    level k observable c_k,  "S" = -ln c_k,  "s" = c_{k+1}/c_k,
    "kappa" = (c_k c_{k+2} - c_{k+1}^2)/c_k^2,   hence   w^(k) = D_k

with  D_k = c_k c_{k+2}/c_{k+1}^2 - 1 >= 0  the log-convexity defect.  Read at
consecutive k this single inequality IS the whole apparatus:

    k = 0 :  DeltaS kappa >= s^2      <=>  N_eff <= DeltaS      (Thm. budget)
    k = 1 :  s(-kappa') >= kappa^2    <=>  sigma >= -1          (Prop. slope)
    k = 2 : -kappa S'''' >= kappa'^2                            (rung 3, simplified)

and level 1 reproduces Prop. 'recursion' exactly (w^(1) = 1 + sigma).  Measured
here: the ratio chain c_{k+1}/c_k must be NON-DECREASING in k, which is the same
statement; its first link is the budget theorem and its second the slope wall,
so the two walls of the previous revision are one wall at consecutive indices.
Also reported: the simplified rung 3 against the previous revision's
V <= 2w - w_2^2, on the frontiers where a fourth derivative is readable at all.

(B) THE ARC IS A SQUARE ROOT OF THE CHANNEL COUNT.  Law III is currently an
independent geometric fact with one formula per chart.  It is a consequence of
the budget.  From dphi = dS/sqrt(N_eff), whenever sigma = dN_eff/dS is constant
(which is exactly what "the charts are ordered by sigma alone" means),

    ####   phi  =  2 sqrt(N_eff) / |sigma|  +  const   ####

and this subsumes BOTH chart-specific arc laws, exactly:
    cure chart     sigma = -1          ->  L = 2 sqrt(DeltaS) = 2 sqrt(-ln eps_inf)
    shape law      sigma = b/(1-b)     ->  L = 2 sqrt(S (1-b)/b)
                                            = 2 sqrt((1-b)/b)(sqrt(S_max)-sqrt(S_min))
(verified to 40 digits on both, script 102.)  The test is therefore FLOOR-FREE
and needs no fitted chart: phi must be AFFINE IN sqrt(N_eff), with N_eff =
s^2/kappa read directly off the second derivative.  We compare it against phi
affine in N_eff and in S -- two parameters each, same data -- by R^2.

Outputs: results/table_tower_chain.csv       per frontier: the ratio chain
         results/table_tower_summary.csv     by corpus
         results/table_arc_depth.csv         per frontier: sqrt(S) vs S fits
         results/table_arc_depth_summary.csv by corpus
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                      # noqa: E402

HFRAC, NGRID = 0.40, 9


def jets(x, S, kmax=4, hfrac=HFRAC, ng=NGRID):
    """local polynomial of degree kmax: returns (grid, S, c_1..c_kmax) with
    c_k = (-1)^k (DeltaS)^{(k)}, which equals (-1)^{k+1} S^{(k)} for k >= 1.
    (The global convention: at k = 0 it gives c_0 = DeltaS, which the k = 0
    rung above needs, where (-1)^{k+1} S^{(k)} would give -S <= 0.  Round 13.)
    Same bandwidth convention as script 97."""
    Z = x[-1] - x[0]
    h = Z * hfrac
    grid = np.linspace(x[0] + 0.15 * Z, x[-1] - 0.15 * Z, ng)
    out = []
    for g in grid:
        d = x - g
        w = np.exp(-0.5 * (d / h) ** 2)
        fact = [1.0]
        for j in range(1, kmax + 1):
            fact.append(fact[-1] * j)
        M = np.column_stack([d ** j / fact[j] for j in range(kmax + 1)])
        Wr = np.sqrt(w)[:, None]
        b, *_ = np.linalg.lstsq(M * Wr, S * Wr[:, 0], rcond=None)
        ck = [(-1) ** (k + 1) * b[k] for k in range(1, kmax + 1)]
        out.append([g, b[0]] + ck)
    return np.array(out)


def run_tower():
    rows = []
    for fr in all_frontiers():
        x, eps = fr["x"], fr["e"]
        o = np.argsort(x)
        x, eps = x[o], eps[o]
        if len(x) < 7:
            continue
        S = -np.log(eps)
        J = jets(x, S, kmax=4)
        g, Sg = J[:, 0], J[:, 1]
        c1, c2, c3, c4 = J[:, 2], J[:, 3], J[:, 4], J[:, 5]
        good = (c1 > 0) & (c2 > 0) & np.isfinite(c3) & np.isfinite(c4)
        if good.sum() < 3:
            continue
        c1, c2, c3, c4, Sg = c1[good], c2[good], c3[good], c4[good], Sg[good]
        N = c1 ** 2 / c2                                  # N_eff
        # the ratio chain:  c2/c1 <= c3/c2 <= c4/c3   (non-decreasing)
        r21, r32, r43 = c2 / c1, c3 / c2, c4 / c3
        link1 = np.nanmedian(r32 - r21)                   # = sigma >= -1 link
        link2 = np.nanmedian(r43 - r32)                   # = rung 3 link
        D1 = np.nanmedian(c1 * c3 / c2 ** 2 - 1.0)        # = 1 + sigma
        D2 = np.nanmedian(c2 * c4 / c3 ** 2 - 1.0)
        # previous revision's rung 3, for comparison, in invariants
        w = c2 / c1 ** 2
        w2 = c3 / (c1 * c2)                               # dln kappa/dS
        V = c4 / (c1 ** 2 * c2)
        old3 = np.nanmedian((2 * w - w2 ** 2) - V)        # >= 0 required
        rows.append(dict(
            corpus=fr["corpus"], frontier=fr["name"], n=len(x),
            span=float(x[-1] - x[0]), N_eff_med=float(np.median(N)),
            chain_link1_med=float(link1), chain_link2_med=float(link2),
            chain_link1_nonneg=int(link1 >= 0), chain_link2_nonneg=int(link2 >= 0),
            D1_med=float(D1), D2_med=float(D2),
            D1_nonneg=int(D1 >= 0), D2_nonneg=int(D2 >= 0),
            old_rung3_slack_med=float(old3), old_rung3_ok=int(old3 >= 0)))
    T = pd.DataFrame(rows)
    out = []
    for corpus, gg in list(T.groupby("corpus")) + [("ALL", T)]:
        out.append(dict(
            corpus=corpus, frontiers=len(gg),
            median_span=round(float(gg.span.median()), 2),
            n_chain_link1_nonneg=int(gg.chain_link1_nonneg.sum()),
            n_chain_link2_nonneg=int(gg.chain_link2_nonneg.sum()),
            n_D1_nonneg=int(gg.D1_nonneg.sum()), n_D2_nonneg=int(gg.D2_nonneg.sum()),
            n_old_rung3_ok=int(gg.old_rung3_ok.sum()),
            median_D1=round(float(gg.D1_med.median()), 4),
            median_D2=round(float(gg.D2_med.median()), 4)))
    return T, pd.DataFrame(out)


def run_arc():
    """phi(x) = int sqrt(kappa) dx cumulative.  The arc-depth law says phi is
    AFFINE in sqrt(N_eff); the alternatives are affine in N_eff and in S.  Two
    parameters each, no floor, no chart fitted."""
    rows = []
    for fr in all_frontiers():
        x, eps = fr["x"], fr["e"]
        o = np.argsort(x)
        x, eps = x[o], eps[o]
        if len(x) < 7:
            continue
        S = -np.log(eps)
        J = jets(x, S, kmax=2)
        g, Sg, c1, c2 = J[:, 0], J[:, 1], J[:, 2], J[:, 3]
        good = (c1 > 0) & (c2 > 0) & np.isfinite(Sg) & (Sg > 0)
        if good.sum() < 4:
            continue
        g, Sg, c1, c2 = g[good], Sg[good], c1[good], c2[good]
        N = c1 ** 2 / c2
        phi = np.concatenate([[0.0], np.cumsum(np.sqrt(c2[:-1]) * np.diff(g))])

        def afit(z):
            A = np.column_stack([np.ones_like(z), z])
            b, *_ = np.linalg.lstsq(A, phi, rcond=None)
            pred = A @ b
            tt = np.sum((phi - phi.mean()) ** 2)
            r2 = 1.0 - np.sum((phi - pred) ** 2) / tt if tt > 0 else np.nan
            return float(b[1]), float(r2)

        sl_sq, r2_sq = afit(np.sqrt(N))
        sl_N, r2_N = afit(N)
        sl_S, r2_S = afit(Sg)
        # sigma from the arc law, and sigma measured directly as dN_eff/dS
        sig_arc = 2.0 / abs(sl_sq) if sl_sq != 0 else np.nan
        A = np.column_stack([np.ones_like(Sg), Sg])
        sig_dir = abs(float(np.linalg.lstsq(A, N, rcond=None)[0][1]))
        rows.append(dict(
            corpus=fr["corpus"], frontier=fr["name"], n=len(x),
            span=float(x[-1] - x[0]), arc=float(phi[-1]),
            N_eff_med=float(np.median(N)),
            slope_sqrtN=sl_sq, r2_sqrtN=r2_sq, r2_N=r2_N, r2_S=r2_S,
            sqrtN_wins=int((r2_sq >= r2_N) and (r2_sq >= r2_S)),
            sigma_from_arc=float(sig_arc), sigma_direct=sig_dir,
            sigma_ratio=float(sig_arc / sig_dir) if sig_dir > 0 else np.nan))
    T = pd.DataFrame(rows)
    out = []
    for corpus, gg in list(T.groupby("corpus")) + [("ALL", T)]:
        out.append(dict(
            corpus=corpus, frontiers=len(gg),
            n_sqrtN_wins=int(gg.sqrtN_wins.sum()),
            median_r2_sqrtN=round(float(gg.r2_sqrtN.median()), 4),
            median_r2_N=round(float(gg.r2_N.median()), 4),
            median_r2_S=round(float(gg.r2_S.median()), 4),
            median_sigma_ratio=round(float(gg.sigma_ratio.median()), 3)))
    return T, pd.DataFrame(out)


def run():
    T1, S1 = run_tower()
    T2, S2 = run_arc()
    save(T1, "table_tower_chain.csv")
    save(S1, "table_tower_summary.csv")
    save(T2, "table_arc_depth.csv")
    save(S2, "table_arc_depth_summary.csv")
    return T1, S1, T2, S2


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    T1, S1, T2, S2 = run()
    print("\n(A) THE TOWER: one inequality at consecutive indices\n")
    print(S1.to_string(index=False), "\n")
    n = len(T1)
    print(f"ratio chain c_{{k+1}}/c_k non-decreasing (median over the window):")
    print(f"   link 1  (c3/c2 >= c2/c1, i.e. sigma >= -1) : "
          f"{int(T1.chain_link1_nonneg.sum())}/{n}")
    print(f"   link 2  (c4/c3 >= c3/c2, i.e. rung 3)      : "
          f"{int(T1.chain_link2_nonneg.sum())}/{n}")
    print(f"the SIMPLIFIED rung 3 (-kappa S'''' >= kappa'^2) is readable on "
          f"all {n} frontiers and HOLDS on {int(T1.D2_nonneg.sum())};")
    print(f"the previous revision's rung 3 (V <= 2w - w_2^2) holds on "
          f"{int(T1.old_rung3_ok.sum())}/{n}.")
    print("\n(B) THE ARC IS A SQUARE ROOT OF THE CHANNEL COUNT\n")
    print(S2.to_string(index=False), "\n")
    m = len(T2)
    print("cumulative arc fitted as affine in sqrt(N_eff) / N_eff / S "
          "(two parameters each, no floor):")
    print(f"   sqrt(N_eff) wins on {int(T2.sqrtN_wins.sum())}/{m} frontiers; "
          f"median R^2 {T2.r2_sqrtN.median():.4f} against "
          f"{T2.r2_N.median():.4f} (N_eff) and {T2.r2_S.median():.4f} (S)")
    print(f"   sigma recovered from the arc slope vs measured directly: "
          f"median ratio {T2.sigma_ratio.median():.3f}")
    print("   -- one formula reproducing BOTH chart-specific arc laws "
          "(cure: sigma=-1; shape law: sigma=b/(1-b)).")
