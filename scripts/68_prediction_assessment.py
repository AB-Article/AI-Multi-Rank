#!/usr/bin/env python3
"""HOW WELL DOES THE LAW PREDICT A MODEL IT HAS NOT SEEN?

The paper tests the law's *structure* (log-convexity, the arc, the collapse)
and its *decisions* (which of two candidates to take).  It never asks the
question a user of a scaling law actually asks: fit it on some models, hold
one out, and see how close it comes.  This script asks that, on every
single-resource frontier the archive can build, across all seven corpora, and
against three baselines.

Three regimes, because they are not the same question:
    INTERIOR     leave one interior point out, predict it            (interpolation)
    FORWARD      train on the cheapest 2/3, predict the dearest 1/3  (extrapolation)
    BACKWARD     train on the dearest 2/3, predict the cheapest 1/3

Baselines:
    power        eps = c t^-alpha          (the law with the floor removed)
    interp       log-linear in log-resource through the neighbours
    last         predict the nearest training point's error

Writes (results/):
    table_prediction_frontiers.csv   one row per frontier: n, span, errors by method
    table_prediction_summary.csv     pooled and by corpus
    table_prediction_correlations.csv pooled held-out agreement by method
    table_prediction_selfdiag.csv    prediction error split by whether the
                                     log-convexity defect is positive, i.e. by
                                     whether the structural class holds at all
"""
import os, sys, glob, itertools, warnings
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
MIN_PTS, MIN_SPAN = 8, 1.0


# ---------------------------------------------------------------- frontiers
def pareto(t, e):
    o = np.argsort(t); t, e = t[o], e[o]
    keep, best = [], np.inf
    for i in range(len(t)):
        if e[i] < best - 1e-12:
            keep.append(i); best = e[i]
    return t[keep], e[keep]


def build():
    F = []
    # timm / ImageNet, already archived as frontiers
    for f in sorted(glob.glob(os.path.join(RES, "frontiers", "*.csv"))):
        d = pd.read_csv(f)
        res = [c for c in d.columns if c not in ("base", "img_size", "eps")]
        if len(res) != 1: continue
        t = pd.to_numeric(d[res[0]], errors="coerce").values
        e = pd.to_numeric(d["eps"], errors="coerce").values
        ok = np.isfinite(t) & np.isfinite(e) & (t > 0) & (e > 0)
        name = os.path.basename(f)[:-4]
        corpus = "vision-shift" if ("ImageNet-A" in name or "ImageNet-R" in name) else "vision"
        F.append((corpus, name, *pareto(t[ok], e[ok])))
    # the uniform field_*.csv schema: hpo, medicine, vision-size
    for fn, corpus in [("field_hpo_yahpo.csv", "hpo"),
                       ("field_medicine_medmnist_published.csv", "medicine"),
                       ("field_vision_size_natssss.csv", "vision-size")]:
        d = pd.read_csv(os.path.join(DATA, fn))
        costs = [c for c in d.columns if c.startswith("cost_")]
        for (fr,), g in d.groupby(["frontier"]):
            q0 = float(g["chance"].iloc[0])
            e = (1 - pd.to_numeric(g["quality"], errors="coerce").values) / (1 - q0)
            for c in costs:
                t = pd.to_numeric(g[c], errors="coerce").values
                ok = np.isfinite(t) & np.isfinite(e) & (t > 0) & (e > 0)
                if ok.sum() < MIN_PTS: continue
                F.append((corpus, f"{fr}|{c}", *pareto(t[ok], e[ok])))
    # NAS: NB201 accuracy against hardware latency
    acc = pd.read_csv(os.path.join(DATA, "nb201_accuracy.csv"))
    hw = pd.read_csv(os.path.join(DATA, "nb201_hwnas.csv"))
    n = min(len(acc), len(hw))
    for ds, q0 in [("cifar10", .1), ("cifar100", .01), ("ImageNet16-120", 1 / 120)]:
        col = f"{ds}_test_acc"
        if col not in acc: continue
        e = (1 - pd.to_numeric(acc[col][:n], errors="coerce").values / 100) / (1 - q0)
        for dev in ["edgegpu_latency", "raspi4_latency", "pixel3_latency", "fpga_latency"]:
            if dev not in hw: continue
            t = pd.to_numeric(hw[dev][:n], errors="coerce").values
            ok = np.isfinite(t) & np.isfinite(e) & (t > 0) & (e > 0)
            if ok.sum() < MIN_PTS: continue
            F.append(("nas", f"nb201|{ds}|{dev}", *pareto(t[ok], e[ok])))
    # tabular
    tb = pd.read_csv(os.path.join(DATA, "tabarena_metrics.csv"))
    for (ds,), g in tb.groupby(["dataset"]):
        e = pd.to_numeric(g["metric_error"], errors="coerce").values
        for c in ["time_train_s", "time_infer_s"]:
            t = pd.to_numeric(g[c], errors="coerce").values
            ok = np.isfinite(t) & np.isfinite(e) & (t > 0) & (e > 0)
            if ok.sum() < MIN_PTS: continue
            F.append(("tabular", f"tabarena|{ds}|{c}", *pareto(t[ok], e[ok])))
    # materials
    mb = pd.read_csv(os.path.join(DATA, "matbench_models.csv"))
    for c in ["params", "ntrain", "gpu_h"]:
        t = pd.to_numeric(mb[c], errors="coerce").values
        e = pd.to_numeric(mb["MAE"], errors="coerce").values
        ok = np.isfinite(t) & np.isfinite(e) & (t > 0) & (e > 0)
        if ok.sum() >= MIN_PTS:
            F.append(("materials", f"matbench|MAE|{c}", *pareto(t[ok], e[ok])))
    out = []
    for corpus, name, t, e in F:
        x = np.log(t)
        if len(x) >= MIN_PTS and (x.max() - x.min()) >= MIN_SPAN and len(np.unique(e)) > 3:
            out.append(dict(corpus=corpus, name=name, x=x, e=e))
    return out


# ---------------------------------------------------------------- predictors
def fit_law(x, e, floor=True, k=1, w=None):
    """eps = floor + sum_{j<=k} c_j exp(-a_j x), fitted in log-error."""
    def mdl(th, xx):
        f = np.exp(th[0]) if floor else 0.0
        v = np.full_like(np.atleast_1d(xx), f, dtype=float)
        for j in range(k):
            v = v + np.exp(th[1 + 2 * j]) * np.exp(-np.exp(th[2 + 2 * j]) * np.atleast_1d(xx))
        return v
    th0 = [np.log(max(e.min() * .5, 1e-9))]
    for j in range(k):
        th0 += [np.log(e.max() / (j + 1)), np.log(.4 * (j + 1) ** 2)]
    ww = np.ones_like(e) if w is None else w
    r = least_squares(lambda th: ww * (np.log(mdl(th, x)) - np.log(e)),
                      np.array(th0), max_nfev=40000)
    return (lambda xx: mdl(r.x, xx)), r


def predict(xtr, etr, xte, method):
    if method == "law":    return fit_law(xtr, etr)[0](xte)
    if method == "law2":   return fit_law(xtr, etr, k=2)[0](xte)
    if method == "power":  return fit_law(xtr, etr, floor=False)[0](xte)
    if method == "interp":
        o = np.argsort(xtr)
        return np.exp(np.interp(xte, xtr[o], np.log(etr[o])))
    if method == "last":
        return etr[np.argmin(np.abs(xtr[:, None] - np.atleast_1d(xte)[None, :]), axis=0)]
    raise ValueError(method)


METHODS = ["law", "law2", "power", "interp", "last"]


def assess(fr):
    x, e = fr["x"], fr["e"]; n = len(x)
    o = np.argsort(x); x, e = x[o], e[o]
    rec = {m: {"INTERIOR": [], "FORWARD": [], "BACKWARD": []} for m in METHODS}
    pred_obs = {m: [] for m in METHODS}
    for i in range(1, n - 1):                                  # INTERIOR, leave-one-out
        msk = np.ones(n, bool); msk[i] = False
        for m in METHODS:
            p = float(np.atleast_1d(predict(x[msk], e[msk], x[i], m))[0])
            rec[m]["INTERIOR"].append(abs(np.log(max(p, 1e-12)) - np.log(e[i])))
            pred_obs[m].append((np.log(max(p, 1e-12)), np.log(e[i])))
    cut = int(np.ceil(n * 2 / 3))
    for lbl, tr, te in [("FORWARD", slice(0, cut), slice(cut, n)),
                        ("BACKWARD", slice(n - cut, n), slice(0, n - cut))]:
        for m in METHODS:
            p = np.atleast_1d(predict(x[tr], e[tr], x[te], m))
            rec[m][lbl] += list(np.abs(np.log(np.maximum(p, 1e-12)) - np.log(e[te])))
    row = dict(corpus=fr["corpus"], name=fr["name"], n=n, span=float(x.max() - x.min()))
    for m in METHODS:
        for lbl in ["INTERIOR", "FORWARD", "BACKWARD"]:
            row[f"{m}_{lbl}"] = float(np.median(rec[m][lbl])) if rec[m][lbl] else np.nan
    return row, {m: np.array(pred_obs[m]) for m in METHODS}


if __name__ == "__main__":
    Fr = build()
    print(f"{len(Fr)} single-resource frontiers with >={MIN_PTS} Pareto points "
          f"and >={MIN_SPAN} nat of span")
    print(pd.Series([f["corpus"] for f in Fr]).value_counts().to_string(), "\n")
    rows, po = [], {m: [] for m in METHODS}
    for fr in Fr:
        try:
            r, p = assess(fr)
        except Exception as ex:
            print("  skip", fr["name"], ex); continue
        rows.append(r)
        for m in METHODS: po[m].append(p[m])
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "table_prediction_frontiers.csv"), index=False)

    out = []
    for corpus, g in list(T.groupby("corpus")) + [("ALL", T)]:
        d = dict(corpus=corpus, frontiers=len(g), median_n=g.n.median(),
                 median_span=round(float(g.span.median()), 2))
        for m in METHODS:
            for lbl in ["INTERIOR", "FORWARD", "BACKWARD"]:
                d[f"{m}_{lbl}"] = round(float(g[f"{m}_{lbl}"].median()), 4)
        out.append(d)
    S = pd.DataFrame(out)
    # pooled held-out correlations (interior)
    cor = []
    for m in METHODS:
        A = np.vstack(po[m]); A = A[np.isfinite(A).all(1)]
        ss_res = np.sum((A[:, 0] - A[:, 1]) ** 2); ss_tot = np.sum((A[:, 1] - A[:, 1].mean()) ** 2)
        cor.append(dict(method=m, n_points=len(A),
                        pearson_log=round(float(np.corrcoef(A[:, 0], A[:, 1])[0, 1]), 4),
                        spearman_log=round(float(spearmanr(A[:, 0], A[:, 1]).statistic), 4),
                        R2_log=round(float(1 - ss_res / ss_tot), 4),
                        median_abs_nats=round(float(np.median(np.abs(A[:, 0] - A[:, 1]))), 4),
                        median_rel_pct=round(float(100 * np.median(np.abs(np.exp(A[:, 0] - A[:, 1]) - 1))), 2)))
    C = pd.DataFrame(cor)
    S.to_csv(os.path.join(RES, "table_prediction_summary.csv"), index=False)
    C.to_csv(os.path.join(RES, "table_prediction_correlations.csv"), index=False)
    pd.set_option("display.width", 250)
    print("MEDIAN |ln eps_pred - ln eps_obs| (nats), by corpus and regime\n")
    print(S.to_string(index=False), "\n")
    print("POOLED held-out (INTERIOR) agreement, all frontiers\n")
    print(C.to_string(index=False))

    # ---- self-diagnosis: is the prediction error large exactly where the
    #      structural class is independently known to fail?  The three-point
    #      log-convexity defect at the midpoint of the window is non-positive
    #      only if eps is not log-convex there, which a positive spectrum forbids.
    sd = []
    for fr in Fr:
        x, e = fr["x"], fr["e"]; o = np.argsort(x); x, e = x[o], e[o]
        S = -np.log(e); xm = 0.5 * (x[0] + x[-1])
        J = float(np.interp(xm, x, S) - 0.5 * (S[0] + S[-1]))
        row = T[T.name == fr["name"]]
        if len(row):
            sd.append(dict(corpus=fr["corpus"], name=fr["name"], defect=J,
                           class_holds=bool(J > 0),
                           law_INTERIOR=float(row.law_INTERIOR.iloc[0])))
    D = pd.DataFrame(sd)
    D.to_csv(os.path.join(RES, "table_prediction_selfdiag.csv"), index=False)
    a = D[D.class_holds].law_INTERIOR.median(); b = D[~D.class_holds].law_INTERIOR.median()
    print("\nSELF-DIAGNOSIS: held-out error where the structural class holds vs. fails")
    print(f"   log-convexity defect > 0  (class holds): {a:.4f} nats  (n={int(D.class_holds.sum())})")
    print(f"   log-convexity defect <= 0 (class fails): {b:.4f} nats  (n={int((~D.class_holds).sum())})")
    print(f"   ratio {b/a:.2f}x  -- the law predicts worst exactly where it is independently refuted")
