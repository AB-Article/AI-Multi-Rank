#!/usr/bin/env python3
"""IS THE RESOURCE DIVISIBLE?  A THIRD-ORDER TEST WITH DERIVED TARGETS.
   (Section "The price is a frontier", Prop. 'slope' and the paragraph
    "Is the archive divisible".)

Prop. 'slope' gives the channel slope sigma = dN_eff/dS two NAMED targets
instead of one simulated null per candidate chart:

    composition law holds   ->   sigma >= -1          (tight: the cure chart)
    two-atom chart          ->   sigma == -(1+N_eff)  (an identity, not a fit)

so the statistic is the trend of the monotone functional

    G = N_eff + S = s^2/kappa - ln eps ,     dG/dS = 1 + sigma ,

which the composition law forces to be non-negative and the indivisible chart
forces to equal -N_eff.  Both ends are derived; nothing is fitted.

Per the paper's own method rule the plug-in trend of G inherits a bandwidth, so
it is never read against its analytic value.  For every frontier we simulate
each candidate truth on that frontier's own x-grid, its own S range and its own
estimated noise, and compare.  Truth families (all matched to the frontier's
right-edge floor share p0, so they differ only in structure):

    floor + 1 channel      indivisible, saturates the axiom    dG/dS = -N_eff
    floor + 2 channels     indivisible                          < 0
    CURE (Poisson ladder)  divisible, tight in Prop. 'slope'    dG/dS = 0
    floorless stable       divisible, beta in {0.15 ... 0.90}   dG/dS = (1+S*)/S* > 0

The test reports, per frontier, which truth's null the observed trend sits in,
and pooled, the fraction of frontiers each truth fails to cover -- the same
one-sided exceedance bookkeeping as script 82, so the two verdicts are directly
comparable.

Outputs: results/table_divisibility_per_frontier.csv
         results/table_divisibility_nulls.csv
         results/table_divisibility_summary.csv
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save, all_frontiers

RNG = np.random.default_rng(20261005)
HFRAC, NGRID, NREP = 0.40, 9, 200
P0_SET = (0.25, 0.50, 0.75, 0.90)
BETA_SET = (0.15, 0.25, 0.35, 0.50, 0.70, 0.90)


# ------------------------------------------------------------- estimators ---
def hat_ops(x, hfrac=HFRAC, ng=NGRID):
    Z = x[-1] - x[0]; h = Z * hfrac
    grid = np.linspace(x[0] + 0.15 * Z, x[-1] - 0.15 * Z, ng)
    H = []
    for g in grid:
        d = x - g; w = np.exp(-0.5 * (d / h) ** 2)
        M = np.column_stack([np.ones_like(d), d, 0.5 * d ** 2])
        A = M.T * w
        H.append(np.linalg.solve(A @ M, A))
    return np.stack(H)


def _slope(Xg, Yg, W):
    n = W.sum(1)
    with np.errstate(all="ignore"):
        mx = (Xg * W).sum(1) / n; my = (Yg * W).sum(1) / n
        dx = (Xg - mx[:, None]) * W; dy = (Yg - my[:, None]) * W
        den = (dx * dx).sum(1)
    out = np.full(len(n), np.nan)
    good = (n >= 5) & (den > 1e-12)
    out[good] = (dx * dy).sum(1)[good] / den[good]
    return out


def stats_from(H, Smat):
    """-> (trend of G against S, median N_eff) per replicate."""
    B = np.einsum("gkn,rn->rgk", H, Smat)
    Sg, s, kap = B[..., 0], B[..., 1], -B[..., 2]
    W = ((s > 0) & (kap > 0)).astype(float)
    with np.errstate(all="ignore"):
        N = np.where(W > 0, s ** 2 / np.maximum(kap, 1e-300), np.nan)
        G = N + Sg
    W = W * np.isfinite(G)
    return _slope(Sg, G, W), np.nanmedian(np.where(W > 0, N, np.nan), axis=1)


def noise_level(x, S):
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


def gen_cure(x, S, p0_end):
    """eps = exp{-lam (1 - e^{-c(x-x0)})}: the divisible chart WITH a floor.

    Matched to the frontier's own window exactly as gen_floor is: lam is set by
    the right-edge floor share, then c by the span.
    """
    Smin, Smax = S.min(), S.max(); Z = x[-1] - x[0]
    lam = Smax - np.log(p0_end)                       # eps_inf/eps(x_max) = p0_end
    u0, u1 = 1.0 - Smin / lam, 1.0 - Smax / lam
    if not (0 < u1 < u0 < 1):
        return None
    c = np.log(u0 / u1) / Z
    if not np.isfinite(c) or c <= 0:
        return None
    Sx = lam * (1.0 - u0 * np.exp(-c * (x - x[0])))
    return np.exp(-Sx) if np.all(Sx > 0) else None


def gen_stable(x, S, beta):
    Smin, Smax = S.min(), S.max(); Z = x[-1] - x[0]
    q = (Smax / max(Smin, 1e-9)) ** (1.0 / beta)
    if not np.isfinite(q) or q <= 1.0001 or q > 1e12:
        return None
    d = Z / (q - 1.0)
    return np.exp(-Smin * ((x - (x[0] - d)) / d) ** beta)


TRUTHS = ([(f"floor_p0_{p:.2f}", lambda x, S, p=p: gen_floor(x, S, p), 0) for p in P0_SET]
          + [("floor_2ch", lambda x, S: gen_floor(x, S, 0.5, k_ch=2), 0)]
          + [(f"cure_p0_{p:.2f}", lambda x, S, p=p: gen_cure(x, S, p), 1) for p in P0_SET]
          + [(f"stable_beta_{b:.2f}", lambda x, S, b=b: gen_stable(x, S, b), 1)
             for b in BETA_SET])


def null_dist(x, S, gen, sig, H, nrep=NREP):
    eps = gen(x, S)
    if eps is None:
        return np.array([]), np.array([])
    Sm = -np.log(eps)[None, :] + sig * RNG.standard_normal((nrep, len(x)))
    U, N = stats_from(H, Sm)
    m = np.isfinite(U)
    return U[m], N[m]


def fit_floor_share(x, eps):
    from scipy.optimize import least_squares
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


# --------------------------------------------------------------------- run ---
def run():
    per, nulls = [], []
    for fr in all_frontiers():
        x, eps = fr["x"], fr["e"]
        o = np.argsort(x); x, eps = x[o], eps[o]
        S = -np.log(eps)
        H = hat_ops(x)
        U, N = stats_from(H, S[None, :])
        U, N = float(U[0]), float(N[0])
        if not np.isfinite(U):
            continue
        sig = noise_level(x, S)
        p0f = fit_floor_share(x, eps)
        row = dict(corpus=fr["corpus"], frontier=fr["name"], n=len(x),
                   span=float(x[-1] - x[0]), sigma_noise=sig,
                   U_obs=U, N_eff_med=N, p0_fitted=p0f,
                   analytic_target_twoatom=-N, analytic_target_divisible=0.0)
        med, cov = {}, {}
        for name, gen, div in TRUTHS:
            Un, Nn = null_dist(x, S, gen, sig, H)
            if Un.size < 20:
                med[name] = np.nan; continue
            lo, hi = np.percentile(Un, 10), np.percentile(Un, 90)
            med[name] = float(np.median(Un))
            cov[name] = bool(lo <= U <= hi)
            nulls.append(dict(corpus=fr["corpus"], frontier=fr["name"], truth=name,
                              divisible=div, n_rep=int(Un.size),
                              U_med=float(np.median(Un)), U_p10=float(lo), U_p90=float(hi),
                              N_eff_null_med=float(np.nanmedian(Nn)),
                              covers_obs=int(cov[name]),
                              below_p10=int(U < lo), above_p90=int(U > hi)))
        cand = {k: v for k, v in med.items() if np.isfinite(v)}
        if cand:
            row["closest_truth"] = min(cand, key=lambda k: abs(cand[k] - U))
            row["closest_divisible"] = int(row["closest_truth"].startswith(("cure", "stable")))
            dv = {k: v for k, v in cand.items() if k.startswith(("cure", "stable"))}
            iv = {k: v for k, v in cand.items() if k.startswith("floor")}
            if dv:
                row["best_divisible_truth"] = min(dv, key=lambda k: abs(dv[k] - U))
                row["dist_divisible"] = min(abs(v - U) for v in dv.values())
            if iv:
                row["best_indivisible_truth"] = min(iv, key=lambda k: abs(iv[k] - U))
                row["dist_indivisible"] = min(abs(v - U) for v in iv.values())
            row["n_truths_covering"] = int(sum(cov.values()))
            row["any_divisible_covers"] = int(any(v for k, v in cov.items()
                                                  if k.startswith(("cure", "stable"))))
            row["any_indivisible_covers"] = int(any(v for k, v in cov.items()
                                                    if k.startswith("floor")))
        per.append(row)

    P = pd.DataFrame(per); Nu = pd.DataFrame(nulls)
    save(P, "table_divisibility_per_frontier.csv")
    save(Nu, "table_divisibility_nulls.csv")

    out = []
    for truth, g in Nu.groupby("truth"):
        out.append(dict(truth=truth, divisible=int(g.divisible.iloc[0]), n=len(g),
                        frac_covers_obs=round(float(g.covers_obs.mean()), 3),
                        frac_obs_below_p10=round(float(g.below_p10.mean()), 3),
                        frac_obs_above_p90=round(float(g.above_p90.mean()), 3),
                        median_U_null=round(float(g.U_med.median()), 4),
                        median_U_obs_over_null=round(
                            float(np.median(P.set_index("frontier")
                                            .loc[g.frontier].U_obs.values / g.U_med.values)), 3)))
    S2 = pd.DataFrame(out).sort_values(["divisible", "frac_covers_obs"], ascending=[True, False])
    save(S2, "table_divisibility_summary.csv")
    return P, Nu, S2


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    P, Nu, S2 = run()
    print(f"\n{len(P)} frontiers tested; statistic = trend of G = s^2/kappa - ln eps "
          f"against S.  Analytic targets: 0 (divisible, tight on the cure chart) and "
          f"-N_eff (two-atom).\n")
    print("RAW, before null calibration:")
    print(f"  observed trend non-negative on {int((P.U_obs >= 0).sum())}/{len(P)} frontiers; "
          f"median U_obs = {P.U_obs.median():.4f}, median N_eff = {P.N_eff_med.median():.4f}")
    print("\nNULL-CALIBRATED: does each truth's own design-matched 80% interval cover the "
          "observed trend?\n")
    print(S2.to_string(index=False), "\n")
    dv = S2[S2.divisible == 1].frac_covers_obs.max()
    iv = S2[S2.divisible == 0].frac_covers_obs.max()
    print(f"best divisible truth covers {dv:.3f} of frontiers; "
          f"best indivisible truth covers {iv:.3f}")
    print(f"a divisible truth covers the observation on "
          f"{int(P.any_divisible_covers.sum())}/{len(P)} frontiers; "
          f"an indivisible one on {int(P.any_indivisible_covers.sum())}/{len(P)}")
    print(f"closest truth is divisible on {int(P.closest_divisible.sum())}/{len(P)} frontiers")
    both = P[np.isfinite(P.dist_divisible) & np.isfinite(P.dist_indivisible)]
    print(f"of the {len(both)} frontiers where both classes are simulable, the divisible class "
          f"is closer on {int((both.dist_divisible < both.dist_indivisible).sum())}")
