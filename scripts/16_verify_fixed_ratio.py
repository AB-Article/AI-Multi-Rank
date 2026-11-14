#!/usr/bin/env python3
"""Numerical check of Proposition (fixed-ratio selection is single-channel).

For eps = einf + M_theta(g), g_i = c_i t_i^-alpha_i, theta > 0, the maximiser of
S(x) - r*lam0.x over x, traced over the price LEVEL r at fixed price RATIOS lam0,
must (i) keep g_i/g_j constant, and (ii) satisfy eps - einf = C exp(-kappa s)
along the traced path, i.e. be a single channel with a floor."""
import os, numpy as np, pandas as pd
from scipy.optimize import minimize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)
rows = []
for theta in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
    for n in (2, 3, 5):
        a = rng.uniform(0.2, 1.2, n); b = rng.uniform(-1, 1, n); einf = 0.05
        lam0 = rng.uniform(0.2, 1.0, n); lam0 /= lam0.sum()
        def eps(x):
            lg = b - a * x
            z = theta * lg; zm = z.max()
            return einf + np.exp((zm + np.log(np.mean(np.exp(z - zm)))) / theta)
        X, E = [], []
        for r in np.linspace(0.05, 0.6, 8) * a.min():
            f = lambda x: np.log(eps(x)) + r * lam0 @ x
            best = min((minimize(f, rng.normal(0, 2, n), method="BFGS",
                                 options=dict(gtol=1e-12)) for _ in range(3)), key=lambda o: o.fun)
            X.append(best.x); E.append(eps(best.x))
        X, E = np.array(X), np.array(E)
        lg = b - a * X                          # log g_i along the path
        ratio_dev = np.ptp(lg - lg[:, [0]], axis=0).max()
        s = a[0] * X[:, 0]                      # path coordinate
        slope, icpt = np.polyfit(s, np.log(E - einf), 1)
        lin_dev = np.abs(np.log(E - einf) - (slope * s + icpt)).max()
        rows.append(dict(theta=theta, n=n, max_dev_log_ratio=ratio_dev,
                         max_dev_single_channel=lin_dev, slope=slope))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, "results", "fixed_ratio_verification.csv"), index=False)
print(R.to_string()); print("max deviations:", R.max_dev_log_ratio.max(), R.max_dev_single_channel.max())
