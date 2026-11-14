#!/usr/bin/env python3
"""Worked example: apply the canonical selection rule to a real leaderboard.

Rule (Theorem: canonical point).  Fit the normal form, then pick the system at
xi = 0: every resource is bought until the error that resource still causes
equals the irreducible error.  No tuning constant is involved.
"""
import numpy as np, pandas as pd, warnings
from scipy.optimize import least_squares
from common import frontier, load_resources, save
warnings.filterwarnings("ignore")
B = load_resources()

def normal_form_fit(X, e):
    n = X.shape[1]
    f = lambda t: t[0]*(1.0+np.sum(np.exp(-t[1:1+n][None,:]*(X-t[1+n:1+2*n][None,:])),1)) - e
    r = least_squares(f, [0.05]+[0.5]*n+[0.0]*n,
                      bounds=([1e-4]+[1e-3]*n+[-20]*n, [0.6]+[6.0]*n+[20]*n),
                      max_nfev=300000)
    return r.x[0], r.x[1:1+n], r.x[1+n:1+2*n]

rows = []
CASES = [("ImageNet-1k", ["lat_rtx3090","param_count"], ["latency (us/img)","params"]),
         ("ImageNet-1k", ["lat_rtx3090","lat_cpu_i9"],  ["GPU lat (us/img)","CPU lat (us/img)"]),
         ("ImageNet-ReaL",["lat_rtx3090","param_count"], ["latency (us/img)","params"])]
for ev, cols, names in CASES:
    d, X, e = frontier(ev, cols, resources=B)
    einf, a, x0 = normal_form_fit(X, e)
    N = len(cols)
    t_star = np.exp(x0)                       # xi = 0  <=>  x = x0
    eps_star = einf*(N+1)                     # canonical error
    # nearest actually-existing architecture on the frontier, in xi-space
    xi = a[None,:]*(X-x0[None,:])
    k = int(np.argmin(np.linalg.norm(xi, axis=1)))
    sel = d.iloc[k]
    # what a "maximise accuracy" policy would pick instead
    kbest = int(np.argmin(e))
    best = d.iloc[kbest]
    rows.append(dict(eval_set=ev, resources=" x ".join(names), n_pareto=len(e),
        eps_inf=einf, alpha_1=a[0], alpha_2=a[1],
        canonical_t1=t_star[0], canonical_t2=t_star[1],
        canonical_eps=eps_star, canonical_top1=100*(1-eps_star*0.999),
        selected_model=sel["base"], selected_res=sel["img_size"],
        selected_top1=100*(1-sel["eps"]*0.999),
        selected_t1=sel[cols[0]], selected_t2=sel[cols[1]],
        accmax_model=best["base"], accmax_top1=100*(1-best["eps"]*0.999),
        accmax_t1=best[cols[0]], accmax_t2=best[cols[1]]))
    print(f"\n=== {ev}   resources: {' x '.join(names)}   ({len(e)} Pareto points)")
    print(f"  fitted floor eps_inf = {einf:.4f}   alphas = {np.round(a,3)}")
    print(f"  canonical point xi=0 : {names[0]}={t_star[0]:.2f}, {names[1]}={t_star[1]:.2f}"
          f"   predicted top-1 = {100*(1-eps_star*0.999):.1f}%")
    print(f"  -> SELECT {sel['base']} @ {int(sel['img_size'])}px : "
          f"top-1 {100*(1-sel['eps']*0.999):.2f}%, {names[0]}={sel[cols[0]]:.2f}, {names[1]}={sel[cols[1]]:.2f}")
    print(f"     accuracy-maximising pick would be {best['base']} @ {int(best['img_size'])}px : "
          f"top-1 {100*(1-best['eps']*0.999):.2f}%, {names[0]}={best[cols[0]]:.2f}, {names[1]}={best[cols[1]]:.2f}")
    print(f"     cost ratio {best[cols[0]]/sel[cols[0]]:.1f}x on {names[0]} "
          f"for {100*(best['eps']*0.999-sel['eps']*0.999)*-1:.2f} pp of top-1")
save(pd.DataFrame(rows), "table_selection_example.csv")
