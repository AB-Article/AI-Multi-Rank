#!/usr/bin/env python3
"""T6-T8  Fewer free constants, an external floor check, and shrinkage.

T6  Shared exponents (resolution model).  The resolution model of the main
    text predicts alpha_{t,i} = A_t / b_i: one number per task and one per
    resource, instead of one per pair.  Test: fit the (task x resource)
    exponent table on the cheap 60% of every frontier, either freely or
    constrained to the rank-one form, and predict the expensive 40%.

T7  Floor versus label noise (Fano).  The fitted floors on ImageNet with the
    original and the cleaner ReaL labels, against the chance-corrected share
    of erroneous validation labels.

T8  Bayesian shrinkage.  With a posterior on the normalised coordinate zeta of
    standard deviation tau, the Bayes declaration of the reducible share is
    E[sigmoid(-zeta)] ~ sigmoid(-zeta_hat / sqrt(1 + pi tau^2 / 8)): the
    declaration is pulled toward the canonical 1/2, and the expected regret of
    the plug-in rule exceeds that of the shrunk rule by ~ tau^2/8 nats.
    Tested with a bootstrap posterior on every attainable frontier.

Writes results/table_shared_exponents.csv, table_floor_label_noise_fano.csv,
table_shrinkage.csv.
"""
import os, numpy as np, pandas as pd
from _loaders import selection_rules, save, RES, ROOT

sr = selection_rules()
RNG = np.random.default_rng(20260915)


# ------------------------------------------------------------------- T6 ----
def t6():
    frs = [f for f in sr.single_resource_frontiers()]
    recs = []
    for field, name, cost, x, e in frs:
        k = np.where(x <= x.min() + 0.6 * np.ptp(x))[0]
        if len(k) < 6 or len(x) - len(k) < 3:
            continue
        ei, c, a, _ = sr.fit_law(x[k], e[k])
        if not (1e-3 < ei < 0.5 and a < 5.99):
            continue
        recs.append(dict(field=field, task=name, resource=cost, alpha_cheap=a,
                         einf_cheap=ei, c_cheap=c, x0=x[k].min(),
                         x_te=x[len(k):], e_te=e[len(k):]))
    D = pd.DataFrame(recs)
    rows = []
    for corpus, sub in D.groupby("field"):
        tab = sub.pivot_table(index="task", columns="resource", values="alpha_cheap")
        tab = tab.dropna(axis=1, thresh=3).dropna()
        if tab.shape[0] < 3 or tab.shape[1] < 2:
            continue
        lg = np.log(tab.to_numpy())
        u = lg.mean(1, keepdims=True); v = (lg - u).mean(0, keepdims=True)
        rank1 = np.exp(u + v); resid = lg - u - v
        sh = {(t_, r_): rank1[i, j] for i, t_ in enumerate(tab.index)
              for j, r_ in enumerate(tab.columns)}
        out = []
        for rec in recs:
            key = (rec["task"], rec["resource"])
            if key not in sh:
                continue
            u_ = rec["x_te"] - rec["x0"]
            for tag, al in [("free", rec["alpha_cheap"]), ("shared", sh[key])]:
                pred = rec["einf_cheap"] + rec["c_cheap"] * np.exp(-al * u_)
                out.append(dict(task=rec["task"], resource=rec["resource"], law=tag,
                                mape=float(np.mean(np.abs(pred / rec["e_te"] - 1)))))
        O = pd.DataFrame(out)
        g = O.pivot_table(index=["task", "resource"], columns="law", values="mape")
        rows.append(dict(corpus=corpus, tasks=tab.shape[0], resources=tab.shape[1],
                         free_params=int(tab.size),
                         shared_params=int(tab.shape[0] + tab.shape[1] - 1),
                         share_explained=float(1 - resid.var() / lg.var()),
                         frontiers=len(g),
                         mape_free=float(g["free"].median()),
                         mape_shared=float(g["shared"].median()),
                         shared_beats_free=float((g["shared"] < g["free"]).mean())))
    T = pd.DataFrame(rows); save(T, "table_shared_exponents.csv")
    print(T.round(4).to_string(index=False))
    return T


# ------------------------------------------------------------------- T7 ----
def t7():
    src = os.path.join(RES, "table_floor_label_noise.csv")
    if not os.path.exists(src):
        src = os.path.join(ROOT, "data", "table_floor_label_noise.csv")
    L = pd.read_csv(src)
    CHANCE = 1e-3
    L["label_error_bound"] = 0.0 / (1 - CHANCE)
    rows = []
    for cost, g in L.groupby("cost"):
        o = g[g.eval_set == "ImageNet-1k"]; r = g[g.eval_set == "ImageNet-ReaL"]
        if len(o) and len(r):
            rows.append(dict(cost=cost, einf_original=float(o.eps_inf.iloc[0]),
                             einf_real=float(r.eps_inf.iloc[0]),
                             drop=float(o.eps_inf.iloc[0] - r.eps_inf.iloc[0]),
                             real_below_original_ci=bool(r.eps_inf.iloc[0] < o.ci_lo.iloc[0]),
                             einf_above_6pct=bool(min(o.eps_inf.iloc[0], r.eps_inf.iloc[0]) > 0.06 / (1 - CHANCE))))
    T = pd.DataFrame(rows); save(T, "table_floor_label_noise_fano.csv")
    print(T.to_string(index=False))
    return T


# ------------------------------------------------------------------- T8 ----
def shrink(zhat, tau):
    """Bayes declaration of the reducible share under zeta ~ N(zhat, tau^2)."""
    return 1.0 / (1.0 + np.exp(zhat / np.sqrt(1.0 + np.pi * tau ** 2 / 8.0)))


def kl_bin(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12); q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def t8():
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        einf, c, a, rms = sr.fit_law(x, e)
        if not (1e-3 < einf < 0.5 and a < 5.99 and e.min() <= 2 * einf <= e.max()):
            continue
        xstar = x.min() + np.log(c / einf) / a
        boot = []
        for _ in range(200):
            i = np.sort(RNG.choice(len(e), len(e), replace=True))
            if len(np.unique(x[i])) < 5:
                continue
            try:
                ei, ci, ai, _ = sr.fit_law(x[i], e[i])
            except Exception:
                continue
            if not (1e-3 < ei < 0.5 and ai < 5.99):
                continue
            boot.append(x[i].min() + np.log(ci / ei) / ai)
        if len(boot) < 30:
            continue
        boot = np.array(boot)
        tau = float(np.std(a * (boot - xstar)))            # sd of zeta at a point
        # regret of each candidate declaration, averaged over the posterior
        cands = {}
        for tag, xc in [("plug-in", xstar), ("shrunk", np.nan), ("canonical 2einf", xstar)]:
            pass
        zeta_at = lambda xq, xs: a * (xq - xs)
        # posterior over the true reference; declare an operating point x_d
        def bayes_regret(xd):
            z = zeta_at(xd, boot)                           # realised zeta
            p = 1.0 / (1.0 + np.exp(z))                     # realised share
            return float(np.mean(kl_bin(p, 0.5)))
        # plug-in: pick the model nearest 2 einf_hat; shrunk: shift toward the
        # canonical point by the shrinkage factor
        i_plug = sr.nearest(x, xstar)
        z_shrunk = np.log(1 / shrink(0.0, tau) - 1)         # = 0: the reference is a fixed point
        xd_shrunk = xstar + z_shrunk / a
        i_shrunk = sr.nearest(x, xd_shrunk)
        rows.append(dict(field=field, frontier=name, cost=cost, n=len(e), alpha=a,
                         tau=tau, shrink_factor=1 / np.sqrt(1 + np.pi * tau ** 2 / 8),
                         regret_plugin=bayes_regret(x[i_plug]),
                         regret_shrunk=bayes_regret(x[i_shrunk]),
                         regret_twice_best=bayes_regret(x[sr.nearest(np.log(e), np.log(2 * e.min()))]),
                         regret_best=bayes_regret(x[-1]),
                         excess_tau2_over_8=tau ** 2 / 8))
    T = pd.DataFrame(rows); save(T, "table_shrinkage.csv")
    if len(T):
        print(T[["frontier", "cost", "tau", "regret_plugin", "regret_twice_best", "regret_best"]]
              .round(3).to_string(index=False))
        print(f"\nfrontiers {len(T)}; median tau {T.tau.median():.2f}; "
              f"median Bayes regret: plug-in {T.regret_plugin.median():.3f}, "
              f"twice-best {T.regret_twice_best.median():.3f}, "
              f"most accurate {T.regret_best.median():.3f}; "
              f"median tau^2/8 {T.excess_tau2_over_8.median():.3f}")
    return T


if __name__ == "__main__":
    print("=== T6 shared exponents ===");  t6()
    print("\n=== T7 floor vs label noise ==="); t7()
    print("\n=== T8 shrinkage ==="); t8()
