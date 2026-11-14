#!/usr/bin/env python3
"""WHICH MICROFOUNDATION?  A per-item test of the mechanism behind the law.
   (Section "What the item level says" of paper.tex.)

The spectral form of the law is a representation theorem (Bernstein): every
completely monotone frontier is SOME positive mixture of scale-free modes.
A representation is not a mechanism.  The paper's pairwise floor estimator,
however, uses a mechanism -- an item carries a rate a, a model at log-resource
x fails it with probability e^{-a x}, and failures are independent ACROSS
MODELS GIVEN THE ITEM -- and that mechanism makes a sharp prediction:

    (A)   Pr[ every model in S fails an item ] = h( sum_{i in S} x_i )

with h the raw failure-rate frontier.  Two rival mechanisms make different
predictions for a pair:

    (B)   failures are NESTED: the benchmark carries an item-difficulty order
          and a model fails an upper set of it, so Pr[both fail] = min(h_i,h_j)
    (C)   failures are INDEPENDENT outright, so Pr[both fail] = h_i h_j

(A), (B) and (C) are distinguishable from per-item correctness alone, with no
fitted law: (A) is tested against the measured marginal curve interpolated at
x_i + x_j, which lies inside the measured window whenever D_i D_j <= D_max.

The frontiers archived in this repository record aggregate accuracies only, so
this script uses the panel built by 63_cofailure_panel.py: 21 random-feature
ridge classifiers on Fashion-MNIST, resource = feature count, two families
(independent feature draws / shared nested draws), 10 000 test items with
per-item correctness.

Writes (results/):
    table_micro_summary.csv      one row per family: the three mechanisms scored
    table_micro_pairs.csv        every pair: observed rate and all predictions
    table_micro_nesting.csv      the nesting coefficient by resource gap
    table_micro_covfloor.csv     the pairwise floor estimator, per pair, vs the
                                 closed form it takes under nesting
    table_micro_floor.csv        floor bounds: single model, pair, all-m, and
                                 the chance-guessing correction
"""
import os, sys, itertools, argparse
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr, chi2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DAT, RES = os.path.join(ROOT, "data", "cofailure"), os.path.join(ROOT, "results")
FAMILIES = ["indep", "nested"]
Q0 = 0.10
TOL_SUM = 0.05


def load(fam):
    F = ~np.load(os.path.join(DAT, f"fmnist_{fam}_correct.npy"))
    fr = pd.read_csv(os.path.join(DAT, f"fmnist_{fam}_frontier.csv"))
    return F, fr["x"].to_numpy(), fr["err"].to_numpy()


def marginal(x, h):
    o = np.argsort(x)
    g = PchipInterpolator(x[o], np.log(h[o]), extrapolate=False)
    return lambda s: np.exp(g(s))


def run(fam):
    F, x, h = load(fam); M, m = F.shape
    g = marginal(x, h)
    rows = []
    for i, j in itertools.combinations(range(m), 2):
        p = F[:, [i, j]].all(1).mean()
        s = x[i] + x[j]
        inside = x.min() <= s <= x.max()
        hmin, hmax = min(h[i], h[j]), max(h[i], h[j])
        rows.append(dict(
            i=i, j=j, xi=x[i], xj=x[j], s=s, gap=abs(x[i] - x[j]),
            hi=h[i], hj=h[j], obs=p, se=np.sqrt(p * (1 - p) / M),
            A_spectral=float(g(s)) if inside else np.nan,
            B_nested=hmin, C_indep=h[i] * h[j],
            ratio_B=p / hmin,
            cov_floor=(p - h[i] * h[j]) / (1 - h[i] - h[j] + p)))
    P = pd.DataFrame(rows)
    ins = P.dropna(subset=["A_spectral"])

    # ---- (A) collision test: pairs with the same sum must agree
    Q = P.dropna(subset=["A_spectral"]).sort_values("s").reset_index(drop=True)
    gid, cur, last = [], 0, None
    for s in Q["s"]:
        if last is not None and s - last > TOL_SUM:
            cur += 1
        gid.append(cur); last = s
    Q = Q.assign(group=gid)
    c2, df_ = 0.0, 0
    for _, d in Q.groupby("group"):
        if len(d) < 2:
            continue
        w = 1 / d["se"] ** 2
        mu = (w * d["obs"]).sum() / w.sum()
        c2 += float((w * (d["obs"] - mu) ** 2).sum()); df_ += len(d) - 1

    # ---- the nesting coefficient kappa and its stability
    kap = P["ratio_B"]
    rho_gap = spearmanr(P["ratio_B"], P["gap"])

    # ---- what the pairwise floor estimator returns under nesting
    kappa = float(kap.median())
    P["cov_floor_nesting_prediction"] = (
        (kappa * P[["hi", "hj"]].min(1) - P.hi * P.hj) /
        (1 - P.hi - P.hj + kappa * P[["hi", "hj"]].min(1)))

    # ---- floor bounds
    o = np.argsort(h)                      # best model first
    allfail = F.all(1).mean()
    best2 = F[:, o[:2]].all(1).mean()
    bounds = []
    for k in [1, 2, 3, 5, 10, m]:
        r = F[:, o[:k]].all(1).mean()
        bounds.append(dict(family=fam, k_models=k, allfail_raw=r,
                           naive_norm=r / (1 - Q0),
                           guess_corrected=r / (1 - Q0) ** k,
                           note="eps_inf <= guess_corrected if impossible items"
                                " are answered at chance independently"))

    sumry = dict(
        family=fam, M=M, n_models=m, span=float(x.max() - x.min()),
        h_max=float(h.max()), h_min=float(h.min()),
        # mechanism scores: median |obs/pred - 1|
        err_A_spectral=float(np.median(np.abs(ins.obs / ins.A_spectral - 1))),
        err_B_nested=float(np.median(np.abs(P.obs / P.B_nested - 1))),
        err_C_indep=float(np.median(np.abs(P.obs / P.C_indep - 1))),
        n_pairs=len(P), n_pairs_inside=len(ins),
        A_median_z=float(np.median((ins.obs - ins.A_spectral) / ins.se)),
        A_collision_chi2=c2, A_collision_df=df_,
        A_collision_p=float(chi2.sf(c2, df_)) if df_ else np.nan,
        # the nesting law
        kappa_median=kappa, kappa_mean=float(kap.mean()), kappa_sd=float(kap.std()),
        kappa_min=float(kap.min()), kappa_max=float(kap.max()),
        kappa_rho_gap=float(rho_gap.statistic), kappa_p_gap=float(rho_gap.pvalue),
        # the pairwise floor estimator
        covfloor_median=float(P.cov_floor.median()),
        covfloor_iqr_over_median=float((P.cov_floor.quantile(.75) -
                                        P.cov_floor.quantile(.25)) / P.cov_floor.median()),
        covfloor_rho_s=float(spearmanr(P.cov_floor, P.s).statistic),
        covfloor_nesting_fit_medabs=float(
            np.median(np.abs(P.cov_floor / P.cov_floor_nesting_prediction - 1))),
        # floor
        allfail_raw=float(allfail), best2_raw=float(best2),
        single_bound_norm=float(h.min() / (1 - Q0)),
        pair_bound_guesscorr=float(best2 / (1 - Q0) ** 2),
        allm_bound_guesscorr=float(allfail / (1 - Q0) ** m),
    )
    return sumry, P.assign(family=fam), pd.DataFrame(bounds), kap, P


if __name__ == "__main__":
    S, Ps, Bs, Ns = [], [], [], []
    for fam in FAMILIES:
        s, P, B, kap, _ = run(fam)
        S.append(s); Ps.append(P); Bs.append(B)
        gaps = pd.cut(P["gap"], [0, 1, 2, 4, 8], include_lowest=True)
        Ns.append(P.groupby(gaps, observed=True)["ratio_B"]
                  .agg(["count", "median", "std"]).reset_index().assign(family=fam))
        print(f"\n=== {fam} " + "=" * 58)
        for k, v in s.items():
            print(f"   {k:<34} {v}")
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(S).to_csv(f"{RES}/table_micro_summary.csv", index=False)
    pd.concat(Ps).to_csv(f"{RES}/table_micro_pairs.csv", index=False)
    pd.concat(Ns).to_csv(f"{RES}/table_micro_nesting.csv", index=False)
    pd.concat(Ps)[["family", "i", "j", "s", "gap", "hi", "hj", "cov_floor",
                   "cov_floor_nesting_prediction"]].to_csv(
        f"{RES}/table_micro_covfloor.csv", index=False)
    pd.concat(Bs).to_csv(f"{RES}/table_micro_floor.csv", index=False)
    print("\nwrote 5 tables to results/")
