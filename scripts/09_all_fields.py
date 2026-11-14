#!/usr/bin/env python3
"""Unified analysis over every corpus, in the standard schema.

Adds one experiment that is new to the theory: RECOVERING THE CHANNEL COUNT N.
The theory fixes the canonical error at (N+1) eps_inf and the entropic gap at
ln(N+1), but N has so far been *assumed* equal to the number of resources
priced.  Here N is estimated by model selection over the number of exponential
channels, which turns it from an assumption into an observable.

Writes:
    table_all_frontiers.csv      every frontier, screened against P1-P6
    table_all_lawtest.csv        additive vs multiplicative, per frontier
    table_channel_count.csv      recovered N per frontier

(An earlier revision also listed table_collapse_all.csv and
 table_collapse_points.csv here.  No save() in this script writes them and
 they are in no release of the archive: the cross-field collapse they served
 was withdrawn.  Removed from this list in round 13.)
"""
import os, glob, itertools, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from common import pareto_mask, ROOT, save

warnings.filterwarnings("ignore")
DATA = os.path.join(ROOT, "data")

# Cost columns to fit per corpus.  Collinear duplicates are dropped: fitting
# near-collinear costs together makes the exponentials unidentifiable and
# collapses eps_inf to the boundary (Corollary: only rank(A) prices exist).
COST_CHOICE = {
    "hpo":         ["cost_timetrain", "cost_timepredict"],
    "medicine":    ["cost_pixels", "cost_gmacs"],
    "vision-size": ["cost_flops_M", "cost_latency_s"],
}


# --------------------------------------------------------------- loading ----
def load_standard():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "field_*.csv"))):
        d = pd.read_csv(f)
        costs = COST_CHOICE.get(d.field.iloc[0],
                                [c for c in d.columns if c.startswith("cost_")])
        d = d.dropna(subset=["quality", "chance"] + costs)
        d = d[(d[costs] > 0).all(axis=1)]
        d["eps"] = (1 - d.quality) / (1 - d.chance)
        d = d[(d.eps > 0) & (d.eps <= 1.5)]
        out.append((d.field.iloc[0], d, costs))
    return out


def load_legacy():
    """The corpora that predate the standard schema."""
    out = []
    nb = pd.read_csv(os.path.join(DATA, "nb201_hwnas.csv"))
    for ds in nb.dataset.unique():
        for c1, c2 in [("edgegpu_latency", "eyeriss_latency"),
                       ("raspi4_latency", "fpga_latency"),
                       ("pixel3_latency", "edgetpu_latency")]:
            s = nb[nb.dataset == ds].copy()
            s["frontier"] = f"nb201|{ds}|{c1[:4]}+{c2[:4]}"
            out.append(("vision-nas", s, [c1, c2]))
    ta = pd.read_csv(os.path.join(DATA, "tabarena_metrics.csv"))
    ta = ta[ta.metric == "roc_auc"].copy()
    ta["eps"] = ta.metric_error / 0.5
    ta["frontier"] = "tabarena|" + ta.dataset.astype(str)
    out.append(("tabular", ta, ["time_train_s"]))
    mb = pd.read_csv(os.path.join(DATA, "matbench_models.csv"))
    for qn, q in [("F1", 1 - mb.F1), ("MAE", mb.MAE)]:
        s = mb.assign(eps=q, frontier=f"matbench|{qn}")
        out.append(("materials", s, ["params", "ntrain"]))
    from common import load_resources, frontier as timm_frontier, EVAL_FILES
    B = load_resources()
    for ev in ["ImageNet-1k", "ImageNet-ReaL", "ImageNet-V2"]:
        for cols in (["lat_rtx3090", "param_count"], ["lat_rtx3090", "lat_cpu_i9"]):
            d, X, e = timm_frontier(ev, cols, resources=B)
            d = d.assign(eps=e, frontier=f"timm|{ev}|{cols[0][:6]}+{cols[1][:6]}")
            out.append(("vision", d, cols))
    return out


def frontiers():
    """Yield (field, name, X=log-costs, e=eps) for every Pareto frontier."""
    for field, d, costs in load_standard() + load_legacy():
        for name, sub in d.groupby("frontier"):
            sub = sub.dropna(subset=["eps"] + costs)
            sub = sub[(sub[costs] > 0).all(axis=1)]
            if len(sub) < 8:
                continue
            M = sub[["eps"] + costs].to_numpy(float)
            P = sub[pareto_mask(M)]
            if len(P) < 8:
                continue
            yield field, str(name), np.log(P[costs].to_numpy(float)), \
                P.eps.to_numpy(float), costs


# ------------------------------------------------------------- fitting -----
def fit_channels(X, e, K, shared=True):
    """eps = einf + sum_{j=1..K} c_j exp(-a_j . x).

    shared=True gives one channel per resource (a_j = alpha_j e_j, K <= n);
    shared=False lets each channel respond to all resources (full a_j vector)."""
    n = X.shape[1]
    if shared:
        K = min(K, n)

        def resid(t):
            einf, c, a = t[0], t[1:1+K], t[1+K:1+2*K]
            return einf + np.sum(c[None, :] * np.exp(-X[:, :K] * a[None, :]), 1) - e
        p0 = [0.05] + [0.5]*K + [0.4]*K
        lo = [0.0] + [0.0]*K + [0.0]*K
        hi = [0.9] + [1e3]*K + [6.0]*K
    else:
        def resid(t):
            einf, c = t[0], t[1:1+K]
            A = t[1+K:1+K+K*n].reshape(K, n)
            return einf + np.sum(c[None, :] * np.exp(-X @ A.T), 1) - e
        p0 = [0.05] + [0.5]*K + [0.4]*(K*n)
        lo = [0.0] + [0.0]*K + [0.0]*(K*n)
        hi = [0.9] + [1e3]*K + [6.0]*(K*n)
    r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=300000)
    rss = float(np.sum(r.fun**2)); m = len(e); k = len(p0)
    return dict(theta=r.x, rss=rss, k=k,
                aic=m*np.log(rss/m) + 2*k, bic=m*np.log(rss/m) + k*np.log(m))


def fit_multiplicative(X, e):
    n = X.shape[1]
    r = least_squares(lambda t: t[0] + t[1]*np.exp(-X @ t[2:2+n]) - e,
                      [0.05, 0.5] + [0.4]*n,
                      bounds=([0, 0] + [0]*n, [0.9, 1e3] + [6.0]*n), max_nfev=300000)
    rss = float(np.sum(r.fun**2)); m = len(e); k = 2+n
    return dict(rss=rss, aic=m*np.log(rss/m)+2*k, bic=m*np.log(rss/m)+k*np.log(m))


def screen(X, e, einf, alpha):
    n = X.shape[1]
    rho = abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) if n > 1 else np.nan
    xi = np.array([alpha[i]*np.ptp(X[:, i]) for i in range(len(alpha))])
    return dict(P2_floor=bool(einf > 1e-3),
                P3_feasible=bool(einf < 1.0/(n+1)),
                P4_independent=bool((np.isnan(rho) or rho < 0.9) and min(alpha) > 0.05),
                P5_points=bool(len(e) >= 15),
                P6_resolved=bool(xi.min() > 1.0),
                corr=None if np.isnan(rho) else round(float(rho), 3),
                xi_span=float(xi.min()))


rows_f, rows_law, rows_N = [], [], []
print("scanning frontiers ...")
for field, name, X, e, costs in frontiers():
    n = X.shape[1]
    A = fit_channels(X, e, n)
    einf = A["theta"][0]; alpha = A["theta"][1+min(n, n):1+2*n]
    sc = screen(X, e, einf, alpha)
    rows_f.append(dict(field=field, frontier=name, n_res=n, n_pareto=len(e),
                       resources="+".join(c.replace("cost_", "") for c in costs),
                       eps_inf=einf, **sc,
                       verdict=all(v for k, v in sc.items() if k.startswith("P"))))
    M = fit_multiplicative(X, e)
    rows_law.append(dict(field=field, frontier=name, n_pareto=len(e),
                         corr=sc["corr"], dAIC_add_minus_mul=M["aic"]-A["aic"]))
    # ---- channel-count recovery ------------------------------------------
    best, fits = None, {}
    for K in range(1, 5):
        try:
            f = fit_channels(X, e, K, shared=False)
        except Exception:
            continue
        fits[K] = f["bic"]
        if best is None or f["bic"] < fits[best]:
            best = K
    if best is not None:
        rows_N.append(dict(field=field, frontier=name, n_res=n, n_pareto=len(e),
                           N_hat=best, eps_inf=einf,
                           **{f"bic_K{K}": v for K, v in fits.items()}))

F = pd.DataFrame(rows_f); L = pd.DataFrame(rows_law); NN = pd.DataFrame(rows_N)
save(F, "table_all_frontiers.csv"); save(L, "table_all_lawtest.csv")
save(NN, "table_channel_count.csv")

print("\n=== frontiers and screening ===")
print(F.groupby("field").agg(frontiers=("frontier", "size"),
                             pass_all=("verdict", "sum"),
                             floor=("P2_floor", "sum"),
                             median_pareto=("n_pareto", "median")).to_string())
print("\n=== additive vs multiplicative ===")
g = L.groupby("field").agg(n=("frontier", "size"),
                           add_wins=("dAIC_add_minus_mul", lambda s: int((s > 0).sum())),
                           median_dAIC=("dAIC_add_minus_mul", "median"),
                           median_corr=("corr", "median"))
print(g.round(2).to_string())
print(f"\noverall: additive favoured on {int((L.dAIC_add_minus_mul>0).sum())}/{len(L)} "
      f"frontiers, median dAIC {L.dAIC_add_minus_mul.median():.1f}")

print("\n=== recovered channel count N ===")
if len(NN):
    print(NN.groupby("field").N_hat.value_counts().unstack(fill_value=0).to_string())
    print(f"\nN_hat == n_resources on {int((NN.N_hat==NN.n_res).sum())}/{len(NN)}; "
          f"N_hat == 1 on {int((NN.N_hat==1).sum())}/{len(NN)}")
