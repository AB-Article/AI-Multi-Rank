#!/usr/bin/env python3
"""THE SHAPE INDEX IS THE COLLAPSE COORDINATE, AND THE ARC MEASURES IT.
   (Sections "Reading a frontier without fitting it", "The floor is not a
   parameter" and "Law III" of paper.tex.)

Proposition 'floor-free exponent and width' already gives the floor-free
collapse coordinate zeta = ln(kappa/s^2) = ln g, the pointwise floor
eps_inf = eps kappa/(s^2+kappa) and the canonical reference s^2 = kappa.
Script 69 separately fits a shape index beta to the stretched-exponential
correction.  Those are not two things.  With S = -ln eps,

    T1   beta = s^2/(s^2 + kappa S) = 1/(1 + S e^{zeta})

so beta is the collapse coordinate divided by the log-error -- origin-free,
floor-free, and already measured wherever zeta is.  Inverting,

    T2   p_0 = (1-beta)/((1-beta) + beta S)

which is the running floor in closed form, and

    T3   the arc of Law III measures beta with no pointwise ratio at all:
             L = 2 sqrt((1-beta)/beta) ( sqrt(S_max) - sqrt(S_min) )
         <=> beta = [1 + (L/2 Delta sqrt S)^2]^{-1} .
         This is the estimator to use, because L is an integral with a
         one-sided bound rather than a plug-in ratio with a bandwidth.

    T4   beta > 1 <=> kappa < 0 <=> INCREASING hazard <=> zero measurable arc.
         "Outside the completely monotone class" and "no measurable arc" are
         one diagnosis, and in the archive they coincide exactly, by corpus.

    T5   L/pi is a channel count that is a geometric invariant, by part (iii)
         of the path-length proposition (L -> K pi as atoms separate).  It
         replaces a selection rule.

    T6   the canonical point as a DEPTH: S* = (1-beta)/beta,
         eps* = exp(1 - 1/beta).  Three routes to it are compared on the same
         frontiers --- the archive's own local error shares, the fitted floor,
         and beta --- because the law holds on a window exactly to the extent
         that they agree.

Reads  results/table_path_per_frontier.csv   (script 34: L, g, s, eps, einf,
                                              and the local error shares)
       results/table_arc_per_frontier.csv    (script 58: the bandwidth-free
                                              three-point chord arc)
       results/frontiers/*.csv               (for the shape fit of T1)
Writes (results/):
    table_shape_from_arc.csv       T1-T5, one row per frontier
    table_shape_from_arc_summary.csv
    table_arc_channel_count.csv    T5
    table_canonical_routes.csv     T6, per frontier
    table_canonical_routes_summary.csv
"""
from __future__ import annotations
import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
FRONT = os.path.join(RESULTS, "frontiers")


def _best(fun, seeds):
    best = None
    for s0 in seeds:
        try:
            r = least_squares(fun, s0, method="lm", max_nfev=8000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    return best


def fit_shape(x, S):
    """Floorless shape law S = exp(lnA)(x-x0)^beta."""
    def res(th):
        x0 = x.min() - np.exp(th[0])
        return np.log(np.maximum(S, 1e-12)) - (np.exp(th[1]) * np.log(x - x0) + th[2])
    b = _best(res, [[a0, b0, 0.0] for a0 in (-2, 0, 1, 2.5)
                    for b0 in (-1.5, -0.7, 0.0, 0.5)])
    return np.exp(b.x[1]), b.x[2], x.min() - np.exp(b.x[0])


def beta_of_arc(L, S_lo, S_hi):
    d = np.sqrt(S_hi) - np.sqrt(S_lo)
    with np.errstate(all="ignore"):
        return np.where(d > 0, 1.0 / (1.0 + (L / (2 * d)) ** 2), np.nan)


def arc_of_beta(beta, S_lo, S_hi):
    with np.errstate(all="ignore"):
        return np.where((beta > 0) & (beta < 1),
                        2 * np.sqrt((1 - beta) / beta) * (np.sqrt(S_hi) - np.sqrt(S_lo)),
                        np.nan)


def shape_fits_by_frontier():
    """beta from a direct floorless fit, on the frontiers shipped point-wise."""
    out = {}
    for f in sorted(glob.glob(os.path.join(FRONT, "*.csv"))):
        key = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        res = [c for c in d.columns if c not in ("base", "img_size", "eps")]
        if len(res) != 1 or len(d) < 12:
            continue
        d = d.sort_values(res[0])
        x = np.log(d[res[0]].to_numpy(float)); eps = d.eps.to_numpy(float)
        beta, lnA, x0 = fit_shape(x, -np.log(eps))
        out[key] = beta
    return out


# --------------------------------------------------------------------- run
if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)

    P = pd.read_csv(os.path.join(RESULTS, "table_path_per_frontier.csv"))
    P["key"] = P.frontier.astype(str) + "|" + P.cost.astype(str)
    P["S_lo"] = -np.log(P.eps1)            # window start: larger error
    P["S_hi"] = -np.log(P.eps2)            # window end:   smaller error
    P["S_mid"] = 0.5 * (P.S_lo + P.S_hi)

    # ---- T1/T3: beta from the collapse coordinate and from the arc
    P["beta_zeta"] = 1.0 / (1.0 + P.S_mid * P.g_med)
    P["beta_arc"] = beta_of_arc(P.L.to_numpy(), P.S_lo.to_numpy(), P.S_hi.to_numpy())
    P["arc_from_beta_zeta"] = arc_of_beta(P.beta_zeta.to_numpy(),
                                          P.S_lo.to_numpy(), P.S_hi.to_numpy())

    # a third, independent route on the vision frontiers: the bandwidth-free chord
    A = pd.read_csv(os.path.join(RESULTS, "table_arc_per_frontier.csv"))
    A["vkey"] = A.eval_set + "__" + A.block
    sf = shape_fits_by_frontier()
    A["beta_fit"] = A.vkey.map(sf)
    fr = {}
    for f in sorted(glob.glob(os.path.join(FRONT, "*.csv"))):
        k = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        if "eps" in d:
            fr[k] = (-np.log(d.eps.max()), -np.log(d.eps.min()))
    A["S_lo"] = A.vkey.map(lambda k: fr.get(k, (np.nan, np.nan))[0])
    A["S_hi"] = A.vkey.map(lambda k: fr.get(k, (np.nan, np.nan))[1])
    A["beta_chord"] = beta_of_arc(A.arc.to_numpy(), A.S_lo.to_numpy(), A.S_hi.to_numpy())
    A["arc_pred"] = arc_of_beta(A.beta_fit.to_numpy(), A.S_lo.to_numpy(), A.S_hi.to_numpy())
    A["arc_rel_err"] = np.abs(A.arc_pred / A.arc.replace(0, np.nan) - 1)
    A["hazard"] = np.where(A.beta_fit < 1, "decreasing", "increasing")

    print("T1/T3  beta from the collapse coordinate, from the arc, and from a fit")
    ok = A[(A.beta_fit > 0) & (A.beta_fit < 1) & np.isfinite(A.arc_rel_err)]
    r1 = spearmanr(ok.beta_chord, ok.beta_fit)
    print(f"    chord arc -> beta  vs  direct shape fit : Spearman {r1.statistic:.3f}"
          f"  (n={len(ok)})")
    print(f"    arc predicted from beta vs measured chord arc: "
          f"median |rel. err| {ok.arc_rel_err.median():.3f}")
    okp = P[(P.beta_zeta > 0) & (P.beta_zeta < 1) & (P.beta_arc > 0) & (P.beta_arc < 1)]
    r2 = spearmanr(okp.beta_zeta, okp.beta_arc)
    print(f"    collapse coordinate -> beta  vs  integrated arc -> beta : "
          f"Spearman {r2.statistic:.3f}  (n={len(okp)})")
    print(f"    all 55 frontiers: median beta_arc = {P.beta_arc.median():.3f}, "
          f"median beta_zeta = {P.beta_zeta.median():.3f}")

    print("T4  increasing hazard and zero arc are one diagnosis")
    for h, grp in A.groupby("hazard"):
        print(f"    {h:10s} hazard: n={len(grp):2d}  median chord arc = "
              f"{grp.arc.median():.3f}  corpora: {', '.join(sorted(set(grp.eval_set)))}")

    print("T5  L/pi as a channel count")
    cc = P[["field", "frontier", "cost", "L", "frac_pi"]].copy()
    cc["channels_geometric"] = np.maximum(np.ceil(cc.frac_pi), 1)
    print(f"    L/pi < 1 on {int((cc.frac_pi < 1).sum())}/{len(cc)} frontiers;  "
          f"max L/pi = {cc.frac_pi.max():.3f};  median {cc.frac_pi.median():.3f}")
    cc.to_csv(os.path.join(RESULTS, "table_arc_channel_count.csv"), index=False)
    print("  -> results/table_arc_channel_count.csv")

    print("T6  three routes to the canonical point, on the same frontiers")
    # the archive's own local error shares: p_bot/p_top are the REDUCIBLE share
    # s/alpha_hat at the window start / end, so p_0 = 1 - p.
    P["p0_start"] = 1 - P.p_bot
    P["p0_end"] = 1 - P.p_top
    P["brackets_knee_local"] = (P.p0_start <= 0.5) & (P.p0_end >= 0.5)
    P["eps_floor_rule"] = 2 * P.einf_fit
    P["in_window_floor"] = (P.eps_floor_rule >= P.eps2) & (P.eps_floor_rule <= P.eps1)
    P["eps_shape_rule"] = np.where((P.beta_arc > 0) & (P.beta_arc < 1),
                                   np.exp(1 - 1 / P.beta_arc), np.nan)
    P["in_window_shape"] = (P.eps_shape_rule >= P.eps2) & (P.eps_shape_rule <= P.eps1)
    print(f"    local error shares bracket p_0 = 1/2 inside the window: "
          f"{int(P.brackets_knee_local.sum())}/{len(P)}")
    print(f"    fitted-floor rule eps = 2 eps_inf lands in the window:   "
          f"{int(P.in_window_floor.sum())}/{len(P)}")
    print(f"    shape rule eps* = e^(1-1/beta) lands in the window:      "
          f"{int(P.in_window_shape.sum())}/{len(P)}")
    both = P.dropna(subset=["eps_shape_rule"])
    both = both[both.eps_floor_rule > 1e-8]
    if len(both):
        lr = np.abs(np.log(both.eps_shape_rule / both.eps_floor_rule))
        print(f"    shape rule vs floor rule: median |log ratio| = {np.median(lr):.3f} nats"
              f"  (n={len(both)}); within the 0.05-nat band on "
              f"{int((lr <= 0.05889).sum())}/{len(both)}")
    P.to_csv(os.path.join(RESULTS, "table_canonical_routes.csv"), index=False)
    print("  -> results/table_canonical_routes.csv")

    A.to_csv(os.path.join(RESULTS, "table_shape_from_arc.csv"), index=False)
    summ = pd.DataFrame([dict(
        n_vision_frontiers=len(A),
        n_in_class=int(((A.beta_fit > 0) & (A.beta_fit < 1)).sum()),
        spearman_chord_vs_fit=float(r1.statistic),
        arc_median_rel_err=float(ok.arc_rel_err.median()),
        spearman_zeta_vs_arc=float(r2.statistic),
        n_path_frontiers=len(P),
        median_beta_arc=float(P.beta_arc.median()),
        median_beta_zeta=float(P.beta_zeta.median()),
        median_S_star=float(np.nanmedian((1 - P.beta_arc) / P.beta_arc)),
        median_eps_star=float(np.nanmedian(P.eps_shape_rule)),
        frontiers_L_lt_pi=int((P.frac_pi < 1).sum()),
        max_L_over_pi=float(P.frac_pi.max()),
        brackets_knee_local=int(P.brackets_knee_local.sum()),
        in_window_floor=int(P.in_window_floor.sum()),
        in_window_shape=int(P.in_window_shape.sum()))])
    summ.to_csv(os.path.join(RESULTS, "table_shape_from_arc_summary.csv"), index=False)
    summ.to_csv(os.path.join(RESULTS, "table_canonical_routes_summary.csv"), index=False)
    print("  -> results/table_shape_from_arc.csv, table_shape_from_arc_summary.csv")
    print("  -> results/table_canonical_routes_summary.csv")
