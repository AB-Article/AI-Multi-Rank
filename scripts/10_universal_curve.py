#!/usr/bin/env python3
"""THE UNIVERSAL CURVE.

Model selection across every frontier prefers a SINGLE bottleneck channel plus
the floor, not one channel per resource.  With one channel the whole frontier
collapses onto a scalar reduced coordinate

    zeta = a . x - ln(c/eps_inf),        eps/eps_inf = 1 + exp(-zeta)

which is ONE dimensionless function with no shape parameter and no channel
count.  The entropic gap is then ln 2 -- one bit -- for ANY number of
resources, and the canonical rule is: buy until the reducible error equals the
irreducible error, eps = 2 eps_inf.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import pareto_mask, ROOT, save
warnings.filterwarnings("ignore")
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "09_all_fields.py")).read().split("rows_f, rows_law, rows_N")[0])


def fit_one_channel(X, e):
    n = X.shape[1]
    r = least_squares(lambda t: t[0] + t[1]*np.exp(-X @ t[2:2+n]) - e,
                      [0.05, 0.5] + [0.4]*n,
                      bounds=([0, 0]+[0]*n, [0.9, 1e3]+[6.0]*n), max_nfev=300000)
    m, k = len(e), 2+n
    rss = float(np.sum(r.fun**2))
    return dict(einf=r.x[0], c=r.x[1], a=r.x[2:2+n], rss=rss,
                bic=m*np.log(rss/m)+k*np.log(m))


def fit_per_resource(X, e):
    """One channel per resource: the law the earlier draft assumed."""
    n = X.shape[1]

    def resid(t):
        return t[0] + np.sum(t[1:1+n][None, :]*np.exp(-X*t[1+n:1+2*n][None, :]), 1) - e
    r = least_squares(resid, [0.05]+[0.5]*n+[0.4]*n,
                      bounds=([0]*(1+2*n), [0.9]+[1e3]*n+[6.0]*n), max_nfev=300000)
    m, k = len(e), 1+2*n
    rss = float(np.sum(r.fun**2))
    return dict(rss=rss, bic=m*np.log(rss/m)+k*np.log(m))


rows, pts = [], []
for field, name, X, e, costs in frontiers():
    one = fit_one_channel(X, e)
    per = fit_per_resource(X, e)
    einf, c, a = one["einf"], one["c"], one["a"]
    feas = 1e-3 < einf < 0.5
    rec = dict(field=field, frontier=name, n_res=X.shape[1], n_pareto=len(e),
               eps_inf=einf, bic_one_channel=one["bic"],
               bic_per_resource=per["bic"],
               dBIC=per["bic"]-one["bic"], feasible=feas)
    if feas:
        zeta = X @ a - np.log(c/einf)
        pred = einf*(1+np.exp(-zeta))
        resid = np.abs(e/pred - 1)
        rec.update(median_resid=float(np.median(resid)),
                   zeta_span=float(np.ptp(zeta)),
                   canonical_eps=2*einf,
                   canonical_in_range=bool(e.min() <= 2*einf <= e.max()))
        for k_ in range(len(e)):
            pts.append(dict(field=field, frontier=name, zeta=float(zeta[k_]),
                            eps_over_einf=float(e[k_]/einf),
                            master=float(1+np.exp(-zeta[k_])),
                            resid=float(resid[k_])))
    rows.append(rec)

R = pd.DataFrame(rows); P = pd.DataFrame(pts)
save(R, "table_universal_curve.csv"); save(P, "table_universal_points.csv")

print("=== one channel vs one-channel-per-resource (BIC; positive favours ONE) ===")
g = R.groupby("field").agg(n=("frontier", "size"),
                           one_wins=("dBIC", lambda s: int((s > 0).sum())),
                           median_dBIC=("dBIC", "median"))
print(g.round(1).to_string())
print(f"\noverall: one channel preferred on {int((R.dBIC>0).sum())}/{len(R)} frontiers, "
      f"median dBIC {R.dBIC.median():.1f}")

ok = R[R.feasible]
print("\n=== collapse onto  eps/eps_inf = 1 + exp(-zeta)  (zero shape parameters) ===")
print(ok.groupby("field").agg(frontiers=("frontier", "size"),
                              median_resid=("median_resid", "median"),
                              median_zeta_span=("zeta_span", "median")).round(4).to_string())
print(f"\nfeasible frontiers (eps_inf < 1/2): {len(ok)}/{len(R)}")
print(f"pooled points: {len(P)}")
print(f"  median within-frontier residual: {P.groupby('frontier').resid.median().median()*100:.2f}%")
print(f"  median pooled residual         : {P.resid.median()*100:.2f}%")
print(f"  ratio                          : "
      f"{P.resid.median()/P.groupby('frontier').resid.median().median():.2f}")
