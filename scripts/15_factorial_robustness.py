#!/usr/bin/env python3
"""Robustness of theta_hat (Section factorial): (a) residuals on the log scale,
(b) Pythia restricted to checkpoints from step 33000 (23% of training)."""
import os, numpy as np, pandas as pd
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "14_test_N_equals_n.py")).read().split("# ----------------------------------------------------------------- run")[0]
exec(src)
_fit = fit_theta
def fit_theta_log(X, e, theta, warm=None):
    n = X.shape[1]
    lo = [0.0] + [-40.0]*n + [0.0]*n
    hi = [e.min()*0.999] + [10.0]*n + [6.0]*n
    best = None
    starts = ([warm] if warm is not None else []) + [np.array([f*e.min()] + [np.log(np.median(e)-f*e.min())]*n + [a]*n)
              for f in (0, .5, .9) for a in (.2, .6, 1.2)]
    for s in starts:
        s = np.clip(s, np.array(lo)+1e-12, np.array(hi)-1e-12)
        r = least_squares(lambda p: np.log(powmean_eps(p, X, theta)) - np.log(e), s, bounds=(lo, hi), max_nfev=20000)
        rss = float(np.sum(r.fun**2))
        if best is None or rss < best[1]: best = (r.x, rss)
    return best
D = pd.read_csv(os.path.join(ROOT, "data", "grid_factorial.csv"))
D = D[D.field.str.startswith("lm")]
rows = []
for variant in ("log-residuals", "pythia-late"):
    for (field, fr), d in D.groupby(["field", "frontier"], sort=False):
        if variant == "pythia-late":
            if not fr.startswith("pythia"): continue
            d = d[d.cost_tokens >= 33000 * 1024 * 2048]
        X = np.log(d[["cost_params", "cost_tokens"]].to_numpy(float)); X -= X.mean(0)
        e = d.eps.to_numpy(float)
        globals()["fit_theta"] = fit_theta_log if variant == "log-residuals" else _fit
        pr = profile(X, e)
        rows.append(dict(variant=variant, frontier=fr, m=len(e), theta_hat=pr["theta_hat"],
                         theta_lo=pr["ci_lo"], theta_hi=pr["ci_hi"],
                         dBIC_1_minus_0=pr["bic1"]-pr["bic0"]))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "table_factorial_robustness.csv"), index=False)
