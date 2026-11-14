#!/usr/bin/env python3
"""T5  The budget law.

Theory (Proposition 'budget law', main text).  For the power-mean frontier
eps = einf + M_theta(g), g_i = exp(b_i - alpha_i x_i), and a budget
sum_i w_i x_i = ln C (w = 1: C = prod_i t_i, e.g. C = 6 N D), the optimal path satisfies
      d x_i / d ln C = q_i* / w_i,   q_i* = (w_i/alpha_i) / sum_k (w_k/alpha_k),
      eps - einf  ∝  C^{-alpha_eff},  1/alpha_eff = sum_k w_k / alpha_k,
for EVERY theta > 0: allocation exponents equal the elasticity shares at the
optimum and do not depend on the interaction.  For theta = 1, n = 2 this is
Hoffmann et al.'s a = beta/(alpha+beta).

Part 1 verifies the identity numerically (constrained optimisation) for
random (n, theta, alpha, b, w).
Part 2 tests it on the factorial LM grids: allocation exponent predicted from
the fitted exponents (theta_hat and theta = 1 fits) versus the model-free
iso-FLOP estimate (Chinchilla 'approach 2': per compute bin, the minimiser of
a quadratic in ln N; then a regression of ln N_opt on ln C).

Writes results/table_budget_verification.csv, table_budget_law.csv.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import minimize
from _loaders import factorial, save, RES, ROOT

RNG = np.random.default_rng(20260914)
fac = factorial()
powmean_eps, fit_theta = fac["powmean_eps"], fac["fit_theta"]


def optimal_path(alpha, b, theta, w, einf=0.05):
    n = len(alpha); xs = []
    lnC = np.linspace(-2, 6, 9)
    for L in lnC:
        def obj(z):
            x = np.r_[z, (L - w[:-1] @ z) / w[-1]]
            lg = b - alpha * x
            lm = lg.mean() if theta == 0 else (np.log(np.mean(np.exp(theta * (lg - lg.max())))) / theta + lg.max())
            return np.log(einf + np.exp(lm))
        r = minimize(obj, np.zeros(n - 1), method="BFGS", options=dict(gtol=1e-12))
        xs.append(np.r_[r.x, (L - w[:-1] @ r.x) / w[-1]])
    xs = np.array(xs)
    slopes = np.array([np.polyfit(lnC, xs[:, i], 1)[0] for i in range(n)])
    return slopes


def verify():
    rows = []
    for _ in range(30):
        n = int(RNG.integers(2, 5)); theta = float(RNG.choice([0.3, 1.0, 2.0, 5.0, 12.0]))
        alpha = RNG.uniform(0.15, 1.2, n); b = RNG.uniform(-1, 1, n); w = RNG.uniform(0.5, 2, n)
        sl = optimal_path(alpha, b, theta, w)
        q = (w / alpha) / np.sum(w / alpha)
        rows.append(dict(n=n, theta=theta, max_abs_dev=float(np.max(np.abs(sl - q / w)))))
    V = pd.DataFrame(rows)
    return pd.DataFrame([dict(result="dx_i/dlnC = q_i*/w_i, all theta",
                              cases=f"{len(V)} random (n<=4, theta in 0.3..12)",
                              max_deviation=V.max_abs_dev.max())])


def isoflop_slope(lnN, lnD, e, nbins=8):
    lnC = lnN + lnD
    edges = np.quantile(lnC, np.linspace(0, 1, nbins + 1))
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (lnC >= lo) & (lnC <= hi)
        if k.sum() < 5 or len(np.unique(np.round(lnN[k], 3))) < 3:
            continue
        c0 = lnC[k].mean(); z = lnN[k] - lnN[k].mean()
        M = np.column_stack([np.ones(k.sum()), z, z ** 2, lnC[k] - c0])
        beta, *_ = np.linalg.lstsq(M, e[k], rcond=None)
        if beta[2] <= 0:
            continue
        zs = -beta[1] / (2 * beta[2])
        if not (z.min() <= zs <= z.max()):
            continue
        pts.append((c0, zs + lnN[k].mean()))
    if len(pts) < 3:
        return np.nan, np.nan, len(pts)
    P = np.array(pts)
    A = np.column_stack([np.ones(len(P)), P[:, 0]])
    beta, res, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    r = P[:, 1] - A @ beta
    se = np.sqrt(np.sum(r ** 2) / max(len(P) - 2, 1) / np.sum((P[:, 0] - P[:, 0].mean()) ** 2))
    return float(beta[1]), float(se), len(P)


def main():
    V = verify(); save(V, "table_budget_verification.csv"); print(V.to_string(index=False))
    G = pd.read_csv(os.path.join(ROOT, "data", "grid_factorial.csv"))
    L = pd.read_csv(os.path.join(RES, "table_factorial_N_test_lm.csv"))
    th_hat = dict(zip(L.frontier, L.theta_hat))
    rows = []
    for name, s in G[G.field != "nas-widths"].groupby("frontier", sort=False):
        s = s[(s.eps > 0) & (s.eps < 1.5)]
        X = np.log(s[["cost_params", "cost_tokens"]].to_numpy(float)); Xc = X - X.mean(0)
        e = s.eps.to_numpy(float)
        rec = dict(frontier=name, field=s.field.iloc[0], m=len(e))
        for tag, th in [("theta1", 1.0), ("thetahat", float(th_hat.get(name, 1.0)))]:
            p, _ = fit_theta(Xc, e, th)
            a = p[3:5]
            rec[f"alpha_N_{tag}"], rec[f"alpha_D_{tag}"] = a
            rec[f"q_N_{tag}"] = (1 / a[0]) / (1 / a[0] + 1 / a[1]) if min(a) > 1e-3 else np.nan
            rec[f"alpha_eff_{tag}"] = 1 / (1 / a[0] + 1 / a[1]) if min(a) > 1e-3 else np.nan
        rec["theta_hat"] = th_hat.get(name, np.nan)
        sl, se, nb = isoflop_slope(X[:, 0], X[:, 1], e)
        rec.update(isoflop_slope=sl, isoflop_se=se, isoflop_bins=nb)
        rows.append(rec)
    T = pd.DataFrame(rows)
    T["dev_thetahat"] = T.isoflop_slope - T.q_N_thetahat
    T["z_thetahat"] = T.dev_thetahat / T.isoflop_se
    save(T, "table_budget_law.csv")
    cols = ["frontier", "m", "theta_hat", "alpha_N_thetahat", "alpha_D_thetahat", "q_N_thetahat",
            "q_N_theta1", "isoflop_slope", "isoflop_se", "isoflop_bins"]
    print(T[cols].round(3).to_string(index=False))
    ok = T.isoflop_slope.notna() & T.q_N_thetahat.notna()
    print(f"\n{ok.sum()} grids with an iso-FLOP estimate; median |slope - q*| = "
          f"{np.median(np.abs(T.dev_thetahat[ok])):.3f}; within 2 SE: {(np.abs(T.z_thetahat[ok]) <= 2).sum()}/{ok.sum()}; "
          f"corr = {np.corrcoef(T.isoflop_slope[ok], T.q_N_thetahat[ok])[0, 1]:.2f}")


if __name__ == "__main__":
    main()
