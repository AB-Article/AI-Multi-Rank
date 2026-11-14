#!/usr/bin/env python3
"""NAS-Bench-201 x HW-NAS-Bench, and TabArena.

Both corpora supply what the vision corpus could not: cost axes that genuinely
dissociate.  NB201 additionally gives an EXHAUSTIVE candidate set (all 15,625
architectures evaluated), so the Pareto front is exact rather than sampled.

Writes:
    table_nb201_pairs.csv        additive vs multiplicative on device pairs
    table_nb201_within_device.csv  latency-vs-energy on the same device (rank test)
    table_tabarena_pairs.csv     additive vs multiplicative, train x infer time
    table_new_corpora_extrap.csv held-out extrapolation on both corpora
    table_nb201_selection.csv    canonical selection on an exact Pareto front
"""
import os, itertools, warnings
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from common import (pareto_mask, fit_additive, fit_multiplicative,
                    fit_law, LAWS, RESULTS, save, ROOT)
warnings.filterwarnings("ignore")

NB = pd.read_csv(os.path.join(ROOT, "data", "nb201_hwnas.csv"))
TA = pd.read_csv(os.path.join(ROOT, "data", "tabarena_metrics.csv"))

DEVICES = ["edgegpu_latency", "raspi4_latency", "edgetpu_latency",
           "pixel3_latency", "eyeriss_latency", "fpga_latency"]
ENERGY = {"edgegpu_latency": "edgegpu_energy", "eyeriss_latency": "eyeriss_energy",
          "fpga_latency": "fpga_energy"}


def frontier(df, cols, eps="eps"):
    d = df.dropna(subset=[eps] + list(cols))
    d = d[(d[list(cols)] > 0).all(axis=1)]
    M = d[[eps] + list(cols)].to_numpy(float)
    d = d[pareto_mask(M)].reset_index(drop=True)
    return d, np.log(d[list(cols)].to_numpy(float)), d[eps].to_numpy(float)


def compare(X, e, bootstrap=200, seed=0):
    A = fit_additive(X, e); M = fit_multiplicative(X, e)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(bootstrap):
        i = rng.integers(0, len(e), len(e))
        try:
            bs.append(fit_multiplicative(X[i], e[i])["aic"] - fit_additive(X[i], e[i])["aic"])
        except Exception:
            pass
    bs = np.array(bs) if bs else np.array([np.nan])
    return A, M, bs


# ------------------------------------------------------ NB201: device pairs -
print("NB201 x HW-NAS-Bench: additive vs multiplicative on CROSS-DEVICE pairs")
rows = []
for ds in ["cifar10", "cifar100", "ImageNet16-120"]:
    sub = NB[NB.dataset == ds]
    for c1, c2 in itertools.combinations(DEVICES, 2):
        d, X, e = frontier(sub, [c1, c2])
        if len(e) < 20:
            continue
        rho = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
        A, M, bs = compare(X, e)
        rows.append(dict(corpus="nb201", dataset=ds, r1=c1, r2=c2, n_pareto=len(e),
                         corr_logres=rho, einf=A["einf"],
                         alpha_1=A["alpha"][0], alpha_2=A["alpha"][1],
                         dAIC_add_minus_mul=M["aic"] - A["aic"],
                         boot_p5=np.nanpercentile(bs, 5),
                         boot_p95=np.nanpercentile(bs, 95),
                         p_additive_wins=float(np.nanmean(bs > 0))))
nb_pairs = pd.DataFrame(rows)
save(nb_pairs, "table_nb201_pairs.csv")
print(f"  {len(nb_pairs)} pairs; additive favoured in "
      f"{(nb_pairs.dAIC_add_minus_mul>0).sum()}/{len(nb_pairs)}; "
      f"median dAIC {nb_pairs.dAIC_add_minus_mul.median():.1f}; "
      f"corr range {nb_pairs.corr_logres.min():.2f}-{nb_pairs.corr_logres.max():.2f}")

# ------------------------------------------- NB201: within-device rank test -
print("\nNB201: latency vs energy on the SAME device (Corollary: rank test)")
rows = []
for ds in ["cifar10", "cifar100", "ImageNet16-120"]:
    sub = NB[NB.dataset == ds]
    for lat, en in ENERGY.items():
        d, X, e = frontier(sub, [lat, en])
        if len(e) < 10:
            continue
        rho = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
        A, M, _ = compare(X, e, bootstrap=0)
        rows.append(dict(dataset=ds, device=lat.replace("_latency", ""),
                         n_pareto=len(e), corr_logres=rho,
                         dAIC_add_minus_mul=M["aic"] - A["aic"]))
wd = pd.DataFrame(rows)
save(wd, "table_nb201_within_device.csv")
print(wd.round(3).to_string(index=False))

# ---------------------------------------------------------------- TabArena -
print("\nTabArena: additive vs multiplicative, train time x inference time")
ta = TA[TA.metric == "roc_auc"].copy()          # bounded metric with a chance level
ta["eps"] = ta.metric_error / 0.5               # chance-corrected: AUC chance = 0.5
rows = []
for ds, sub in ta.groupby("dataset"):
    d, X, e = frontier(sub, ["time_train_s", "time_infer_s"])
    if len(e) < 12:
        continue
    rho = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    A, M, bs = compare(X, e, bootstrap=100)
    rows.append(dict(corpus="tabarena", dataset=ds, n_pareto=len(e), corr_logres=rho,
                     einf=A["einf"], alpha_train=A["alpha"][0], alpha_infer=A["alpha"][1],
                     dAIC_add_minus_mul=M["aic"] - A["aic"],
                     p_additive_wins=float(np.nanmean(bs > 0))))
ta_pairs = pd.DataFrame(rows)
save(ta_pairs, "table_tabarena_pairs.csv")
print(f"  {len(ta_pairs)} datasets; additive favoured in "
      f"{(ta_pairs.dAIC_add_minus_mul>0).sum()}/{len(ta_pairs)}; "
      f"median corr {ta_pairs.corr_logres.median():.2f}; "
      f"median dAIC {ta_pairs.dAIC_add_minus_mul.median():.1f}")

# ------------------------------------------------- held-out extrapolation ---
print("\nHeld-out extrapolation on the new corpora (fit gauge on cheap 60%)")


def split(X, e, frac=0.6):
    o = np.argsort(X.sum(1)); X, e = X[o], e[o]
    m = int(round(frac * len(e)))
    return X[:m], e[:m], X[m:], e[m:]


rows = []
CASES = []
for ds in ["cifar10", "cifar100", "ImageNet16-120"]:
    CASES.append(("nb201", ds, NB[NB.dataset == ds], ["edgegpu_latency", "eyeriss_latency"]))
    CASES.append(("nb201", ds, NB[NB.dataset == ds], ["raspi4_latency", "fpga_latency"]))
big = ta.groupby("dataset").size().sort_values(ascending=False).head(4).index
for ds in big:
    CASES.append(("tabarena", ds, ta[ta.dataset == ds], ["time_train_s", "time_infer_s"]))

for corpus, name, sub, cols in CASES:
    d, X, e = frontier(sub, cols)
    if len(e) < 30:
        continue
    Xtr, etr, Xte, ete = split(X, e)
    rec = dict(corpus=corpus, dataset=name, resources=" x ".join(cols),
               n_train=len(etr), n_test=len(ete))
    for nm in LAWS:
        th = fit_law(nm, Xtr, etr)
        pred = LAWS[nm][0](th, Xte)
        rec[f"mape_{nm.replace(' ', '_')}"] = float(np.mean(np.abs(pred - ete) / ete))
        if nm == "normal form":
            rec["einf_hat"] = th[0]
    rec["floor_present"] = rec["einf_hat"] > 1e-3
    rows.append(rec)
ex = pd.DataFrame(rows)
save(ex, "table_new_corpora_extrap.csv")
cols = [c for c in ex.columns if c.startswith("mape_")]
print(ex[["corpus", "dataset", "resources", "n_test", "floor_present"] + cols]
      .assign(**{c: (ex[c] * 100).round(1) for c in cols}).to_string(index=False))

# ------------------------------- canonical selection on an EXACT Pareto set -
print("\nCanonical selection on the exhaustive NB201 candidate set")


def nf_fit(X, e):
    n = X.shape[1]
    f = lambda t: t[0] * (1 + np.sum(np.exp(-t[1:1+n][None, :] * (X - t[1+n:1+2*n][None, :])), 1)) - e
    r = least_squares(f, [0.05]+[0.5]*n+[0.0]*n,
                      bounds=([1e-4]+[1e-3]*n+[-20]*n, [0.9]+[6.0]*n+[20]*n),
                      max_nfev=300000)
    return r.x[0], r.x[1:1+n], r.x[1+n:1+2*n]


rows = []
for ds in ["cifar10", "cifar100", "ImageNet16-120"]:
    sub = NB[NB.dataset == ds]
    cols = ["edgegpu_latency", "eyeriss_latency"]
    d, X, e = frontier(sub, cols)
    einf, a, x0 = nf_fit(X, e)
    xi = a[None, :] * (X - x0[None, :])
    k = int(np.argmin(np.linalg.norm(xi, axis=1)))
    kb = int(np.argmin(e))
    q0 = 1.0 / {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}[ds]
    acc = lambda ep: 100 * (1 - ep * (1 - q0))
    N = len(cols)
    # Feasibility of the canonical point (Diagnostic 3):
    #  (a) the canonical error must lie below chance:   (N+1) einf < 1
    #  (b) xi = 0 must lie inside the observed range of every reduced coordinate
    feas_a = (N + 1) * einf < 1.0
    feas_b = all(xi[:, i].min() <= 0 <= xi[:, i].max() for i in range(N))
    rows.append(dict(dataset=ds, n_candidates=len(sub), n_pareto=len(e),
                     canonical_eps=(N + 1) * einf,
                     feasible_below_chance=feas_a, feasible_in_range=feas_b,
                     feasible=bool(feas_a and feas_b),
                     eps_inf=einf, alpha_1=a[0], alpha_2=a[1],
                     canonical_t1=np.exp(x0[0]), canonical_t2=np.exp(x0[1]),
                     selected_acc=acc(d.eps.iloc[k]),
                     selected_t1=d[cols[0]].iloc[k], selected_t2=d[cols[1]].iloc[k],
                     accmax_acc=acc(d.eps.iloc[kb]),
                     accmax_t1=d[cols[0]].iloc[kb], accmax_t2=d[cols[1]].iloc[kb],
                     cost_ratio_t1=d[cols[0]].iloc[kb]/d[cols[0]].iloc[k],
                     acc_gain=acc(d.eps.iloc[kb])-acc(d.eps.iloc[k])))
sel = pd.DataFrame(rows)
save(sel, "table_nb201_selection.csv")
print(sel[["dataset", "n_pareto", "eps_inf", "canonical_eps",
           "feasible_below_chance", "feasible_in_range", "feasible",
           "selected_acc", "accmax_acc", "cost_ratio_t1",
           "acc_gain"]].round(3).to_string(index=False))
print("\nDiagnostic 3 rejects the canonical point wherever it is infeasible;\n"
      "the selection columns are meaningful only on rows with feasible=True.")
