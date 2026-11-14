#!/usr/bin/env python3
"""T5b  The resolution model: theta_hat x A_t should be constant across tasks.

Resolution model (Proposition 'resolution model', main text).  If resource i
buys resolution h_i = t_i^{-1/b_i}, resolutions combine as
h = (sum_i h_i^{theta_r})^{1/theta_r}, and the task converts resolution into
error as eps - einf ∝ h^{A_t}, then the frontier is exactly a power mean with
    alpha_{t,i} = A_t / b_i,     theta_t = theta_r / A_t,
so the fitted interaction and the fitted exponents are inversely related:
theta_t * A_t = theta_r depends on the resources, not on the task.  With
A_t estimated by the geometric mean of the fitted exponents (b_i are absorbed
into the units), the product theta_hat * A_hat should be far more stable
across tasks evaluated on the same grid than either factor.

Writes results/table_resolution_model.csv.
"""
import os, numpy as np, pandas as pd
from _loaders import save, RES

RNG = np.random.default_rng(20260916)


def cv(v):
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1) / np.mean(v))


def main():
    L = pd.read_csv(os.path.join(RES, "table_factorial_N_test_lm.csv"))
    a = np.array([[float(z) for z in s.split(";")] for s in L.alpha_theta1])
    L["A_hat"] = np.exp(np.log(a).mean(1))                  # geometric mean exponent
    L["product"] = L.theta_hat * L.A_hat
    L["grid"] = [f.split("|")[0] for f in L.frontier]
    rows = []
    for grid, g in L.groupby("grid"):
        if len(g) < 3:
            continue
        # bootstrap the CV ratio across tasks of the same grid
        rat = []
        for _ in range(2000):
            i = RNG.choice(len(g), len(g), replace=True)
            if len(np.unique(i)) < 3:
                continue
            s = g.iloc[i]
            rat.append(cv(s["product"]) / cv(s.theta_hat))
        rows.append(dict(grid=grid, tasks=len(g),
                         theta_min=g.theta_hat.min(), theta_max=g.theta_hat.max(),
                         cv_theta=cv(g.theta_hat), cv_A=cv(g.A_hat),
                         cv_product=cv(g["product"]),
                         ratio_cv_product_over_theta=cv(g["product"]) / cv(g.theta_hat),
                         ratio_ci_lo=float(np.quantile(rat, .05)),
                         ratio_ci_hi=float(np.quantile(rat, .95)),
                         median_product=float(g["product"].median()),
                         corr_log_theta_log_A=float(np.corrcoef(np.log(g.theta_hat),
                                                                np.log(g.A_hat))[0, 1]),
                         slope_log_theta_on_log_A=float(np.polyfit(np.log(g.A_hat),
                                                                   np.log(g.theta_hat), 1)[0])))
    T = pd.DataFrame(rows); save(T, "table_resolution_model.csv")
    print(T.round(3).to_string(index=False))
    print("\nper-frontier products:")
    print(L[["frontier", "theta_hat", "A_hat", "product"]].round(3).to_string(index=False))
    # the model predicts a within-grid slope of -1 (theta_r is a property of
    # the resource pair, so pooling across grids is not a test of it)
    print("\nwithin-grid slope of ln theta on ln A (predicted -1):",
          ", ".join(f"{r.grid} {r.slope_log_theta_on_log_A:.2f}" for r in T.itertuples()))


if __name__ == "__main__":
    main()
