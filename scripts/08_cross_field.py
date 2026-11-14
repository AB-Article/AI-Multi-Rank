#!/usr/bin/env python3
"""Cross-field collapse (O1).

Claim: after fitting only gauge (an error scale, and per resource an origin and
a unit), every saturating frontier lies on the single curve
    eps / eps_inf = 1 + sum_i exp(-xi_i)
with no shape parameter.  If frontiers from different fields land on that one
curve, a single dimensionless law governs the accuracy--cost trade-off wherever
the two assumptions hold, and only the units differ between fields.

Test statistic: the scatter of pooled residuals |eps/pred - 1| about the master
curve, compared with the within-frontier scatter.  A collapse means the pooled
scatter is not materially larger than the within-frontier scatter.

Fields: vision (timm/ImageNet), NAS (NAS-Bench-201 x HW-NAS-Bench),
        materials (Matbench Discovery), tabular (TabArena).
"""
import os, warnings
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from common import (pareto_mask, frontier as timm_frontier, load_resources,
                    EVAL_FILES, ROOT, RESULTS, save)
warnings.filterwarnings("ignore")

FLOOR_TOL = 1e-3


def nf_fit(X, e):
    """Fit gauge only: eps = einf * (1 + sum_i exp(-a_i (x_i - x0_i)))."""
    n = X.shape[1]
    f = lambda t: t[0] * (1 + np.sum(
        np.exp(-t[1:1+n][None, :] * (X - t[1+n:1+2*n][None, :])), 1)) - e
    r = least_squares(f, [0.05]+[0.5]*n+[0.0]*n,
                      bounds=([1e-5]+[1e-3]*n+[-25]*n, [0.9]+[8.0]*n+[25]*n),
                      max_nfev=400000)
    return r.x[0], r.x[1:1+n], r.x[1+n:1+2*n]


def pareto(df, eps, cols):
    d = df.dropna(subset=[eps]+list(cols))
    d = d[(d[list(cols)] > 0).all(axis=1)]
    M = d[[eps]+list(cols)].to_numpy(float)
    return d[pareto_mask(M)].reset_index(drop=True)


records, points = [], []


def add(field, name, X, e, cols):
    if len(e) < 11:
        return
    einf, a, x0 = nf_fit(X, e)
    if einf <= FLOOR_TOL:
        records.append(dict(field=field, frontier=name, n=len(e), eps_inf=einf,
                            floor_present=False, median_resid=np.nan,
                            resources=" x ".join(cols)))
        return
    xi = a[None, :] * (X - x0[None, :])
    pred = einf * (1 + np.sum(np.exp(-xi), 1))
    resid = np.abs(e/pred - 1)
    records.append(dict(field=field, frontier=name, n=len(e), eps_inf=einf,
                        floor_present=True, median_resid=float(np.median(resid)),
                        resources=" x ".join(cols)))
    for k in range(len(e)):
        points.append(dict(field=field, frontier=name,
                           xi_sum=float(np.sum(xi[k])),
                           eps_over_einf=float(e[k]/einf),
                           master=float(1 + np.sum(np.exp(-xi[k]))),
                           resid=float(resid[k])))


# ---------------------------------------------------------------- vision ---
B = load_resources()
for ev in ["ImageNet-1k", "ImageNet-ReaL", "ImageNet-V2"]:
    for cols in (["lat_rtx3090", "param_count"], ["lat_rtx3090", "lat_cpu_i9"]):
        try:
            d, X, e = timm_frontier(ev, cols, resources=B)
        except Exception:
            continue
        add("vision", f"{ev}|{'+'.join(cols)}", X, e, cols)

# ------------------------------------------------------------------- NAS ---
NB = pd.read_csv(os.path.join(ROOT, "data", "nb201_hwnas.csv"))
NASPAIRS = [["edgegpu_latency", "eyeriss_latency"],
            ["raspi4_latency", "fpga_latency"],
            ["edgegpu_latency", "fpga_latency"],
            ["pixel3_latency", "edgetpu_latency"]]
for ds in ["cifar10", "cifar100", "ImageNet16-120"]:
    sub = NB[NB.dataset == ds]
    for cols in NASPAIRS:
        d = pareto(sub, "eps", cols)
        add("NAS", f"{ds}|{'+'.join(c.replace('_latency','') for c in cols)}",
            np.log(d[cols].to_numpy(float)), d.eps.to_numpy(float), cols)

# ------------------------------------------------------------- materials ---
mb = pd.read_csv(os.path.join(ROOT, "data", "matbench_models.csv"))
for qn, q in [("1-F1", 1 - mb.F1), ("MAE", mb.MAE)]:
    for cols in (["params", "ntrain"],):
        d = pareto(mb.assign(eps=q), "eps", cols)
        add("materials", f"{qn}|{'+'.join(cols)}",
            np.log(d[cols].to_numpy(float)), d.eps.to_numpy(float), cols)

# --------------------------------------------------------------- tabular ---
TA = pd.read_csv(os.path.join(ROOT, "data", "tabarena_metrics.csv"))
ta = TA[TA.metric == "roc_auc"].copy(); ta["eps"] = ta.metric_error / 0.5
for ds, sub in ta.groupby("dataset"):
    d = pareto(sub, "eps", ["time_train_s"])
    add("tabular", f"{ds}|train", np.log(d[["time_train_s"]].to_numpy(float)),
        d.eps.to_numpy(float), ["time_train_s"])

rec = pd.DataFrame(records)
pts = pd.DataFrame(points)
save(rec, "table_cross_field_frontiers.csv")
save(pts, "table_cross_field_points.csv")

print("Frontiers entering the collapse test")
print(rec.groupby("field").agg(n_frontiers=("frontier", "size"),
                               with_floor=("floor_present", "sum")).to_string())
ok = rec[rec.floor_present]
print("\nPer-field median residual about the master curve (gauge-only fit):")
print(ok.groupby("field").median_resid.describe()[["count", "50%", "min", "max"]]
      .rename(columns={"50%": "median"}).round(4).to_string())

print("\nPooled points:", len(pts))
if len(pts):
    within = pts.groupby("frontier").resid.median()
    print(f"  median within-frontier residual : {within.median()*100:.2f}%")
    print(f"  median pooled residual          : {pts.resid.median()*100:.2f}%")
    print(f"  ratio (pooled / within)         : "
          f"{pts.resid.median()/within.median():.2f}   (1.0 = perfect collapse)")
    print("\n  per-field pooled residual:")
    print((pts.groupby("field").resid.median()*100).round(2).to_string())
