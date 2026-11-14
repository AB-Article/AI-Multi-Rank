#!/usr/bin/env python3
"""T2  Why so few channels?  The spectral resolution number.

Theory (Proposition 'spectral resolution', main text).  In the log-partition
representation, ln of the reducible error along the channel direction has the
cumulant expansion  -k1 u + k2 u^2/2 - k3 u^3/6 + ...  A second channel is
visible only through k2: over a window of Z reducible e-folds its largest
departure from the best single exponential is k2 Z^2 / 16 (in units k1 = 1).
Against m points with log-noise sigma, it is detectable when
      CV^2 * V >= c,     V = Z^2 sqrt(m) / (16 sigma),
with CV the coefficient of variation of the channel exponents.  A third
channel needs the third cumulant, V3 = Z^3 sqrt(m) / (192 sigma).

Part 1 calibrates P(N_hat >= 2 | V) by simulation (two channels with exponent
ratio 3; one channel; a continuous Gamma spectrum), using the same BIC
channel-count procedure as script 09.
Part 2 computes V for every frontier of Section 8.5 (script 09) and every
factorial LM grid (script 14) and relates it to the recovered N_hat.

Writes results/table_resolution_sim.csv, table_resolution_frontiers.csv,
table_resolution_summary.csv.
"""
import os, numpy as np, pandas as pd
from scipy import stats, optimize
from _loaders import all_fields, save, RES

RNG = np.random.default_rng(20260912)
ns = all_fields()
fit_channels, frontiers = ns["fit_channels"], ns["frontiers"]


def resolution(X, e, fit1, fit2):
    """Z = span of the single-channel coordinate a.x; sigma = log-rms of the
    two-channel fit (the noise estimate of the richer model)."""
    n = X.shape[1]
    t1 = fit1["theta"]; a = t1[2:2 + n]
    Z = float(np.ptp(X @ a))
    t2 = fit2["theta"]; K = 2
    c2, A2 = t2[1:1 + K], t2[1 + K:1 + K + K * n].reshape(K, n)
    pred = t2[0] + np.sum(c2[None, :] * np.exp(-X @ A2.T), 1)
    sig = float(np.sqrt(np.mean(np.log(e / np.maximum(pred, 1e-12)) ** 2)))
    sig = max(sig, 1e-4)
    m = len(e)
    return dict(Z=Z, sigma=sig, m=m, V=Z ** 2 * np.sqrt(m) / (16 * sig),
                V3=Z ** 3 * np.sqrt(m) / (192 * sig))


def nhat(X, e, Kmax=3):
    fits = {}
    for K in range(1, Kmax + 1):
        try:
            fits[K] = fit_channels(X, e, K, shared=False)
        except Exception:
            pass
    best = min(fits, key=lambda k: fits[k]["bic"])
    return best, fits



def fit_fast(u, e, K):
    """Fast 1-D K-channel fit used by the calibration simulations."""
    from scipy.optimize import least_squares
    best = None
    for seed in range(3):
        r0 = np.random.default_rng(seed)
        p0 = np.r_[0.5 * e.min(), np.full(K, np.log(np.median(e) / K)), r0.uniform(0.1, 2.0, K)]
        lo = np.r_[0.0, np.full(K, -30.0), np.zeros(K)]
        hi = np.r_[e.min(), np.full(K, 10.0), np.full(K, 6.0)]
        f = lambda p: p[0] + np.exp(p[1:1 + K][None, :] - u[:, None] * p[1 + K:][None, :]).sum(1) - e
        try:
            r = least_squares(f, np.clip(p0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if best is None or rss < best[0]:
            best = (rss, r.x)
    rss, p = best
    k = 1 + 2 * K
    return dict(theta=np.r_[p[0], np.exp(p[1:1 + K]), p[1 + K:]],
                rss=rss, bic=len(e) * np.log(rss / len(e)) + k * np.log(len(e)))


def nhat_fast(u, e, Kmax=2):
    fits = {K: fit_fast(u, e, K) for K in range(1, Kmax + 1)}
    return min(fits, key=lambda k: fits[k]["bic"]), fits

# ---------------------------------------------------------- Part 1: sims ---
def simulate():
    rows = []
    truths = {
        "two channels (ratio 3)": lambda u: 1.0 + 0.5 * np.exp(-0.5 * u) + 0.5 * np.exp(-1.5 * u),
        "one channel": lambda u: 1.0 + np.exp(-1.0 * u),
        "Gamma spectrum (shape 4)": lambda u: 1.0 + (1.0 + 1.0 * u / 4.0) ** -4.0,
    }
    for tname, f in truths.items():
        for W in (3.0, 6.0, 12.0):
            for m in (25,):
                for sig in (0.01, 0.05):
                    for rep in range(5):
                        u = np.sort(RNG.uniform(-W / 3, 2 * W / 3, m))
                        e = 0.05 * f(u) * np.exp(sig * RNG.standard_normal(m))
                        X = u[:, None]
                        try:
                            K, fits = nhat(X, e)
                            r = resolution(X, e, fits[1], fits[2])
                        except Exception:
                            continue
                        rows.append(dict(truth=tname, W=W, sigma_true=sig, N_hat=K, **r))
    return pd.DataFrame(rows)


def logistic(x, y):
    """P(y=1) = sigmoid(b0 + b1 x); returns (b0, b1, x at P=1/2, AUC)."""
    nll = lambda b: np.sum(np.logaddexp(0, -(2 * y - 1) * (b[0] + b[1] * x)))
    b = optimize.minimize(nll, [0.0, 1.0], method="Nelder-Mead").x
    auc = stats.mannwhitneyu(x[y == 1], x[y == 0]).statistic / (y.sum() * (1 - y).sum()) \
        if 0 < y.sum() < len(y) else np.nan
    return b[0], b[1], -b[0] / b[1] if b[1] != 0 else np.nan, auc


def main():
    print("Part 2: measured frontiers ...")
    NN = pd.read_csv(os.path.join(RES, "table_channel_count.csv")) \
        if os.path.exists(os.path.join(RES, "table_channel_count.csv")) else \
        pd.read_csv(os.path.join(os.path.dirname(RES), "data", "table_channel_count.csv"))
    nmap = {(r.field, str(r.frontier)): r.N_hat for r in NN.itertuples()}
    cache = os.path.join(RES, "table_resolution_frontiers.csv")
    if os.path.exists(cache):
        D = pd.read_csv(cache); print("  (reusing results/table_resolution_frontiers.csv)")
        rows = None
    else:
        rows = []
    for field, name, X, e, costs in (frontiers() if rows is not None else []):
        if (field, name) not in nmap:
            continue
        f1 = fit_channels(X, e, 1, shared=False); f2 = fit_channels(X, e, 2, shared=False)
        rows.append(dict(corpus="82 frontiers", field=field, frontier=name, n_res=X.shape[1],
                         N_hat=int(nmap[(field, name)]), **resolution(X, e, f1, f2)))
    # factorial LM grids
    import pandas as _pd
    G = _pd.read_csv(os.path.join(os.path.dirname(RES), "data", "grid_factorial.csv"))
    L = _pd.read_csv(os.path.join(RES, "table_factorial_N_test_lm.csv"))
    lm_n = dict(zip(L.frontier, L.N_hat))
    for name, s in (G[G.field != "nas-widths"].groupby("frontier") if rows is not None else []):
        X = np.log(s[["cost_params", "cost_tokens"]].to_numpy(float)); X = X - X.mean(0)
        e = s.eps.to_numpy(float)
        f1 = fit_channels(X, e, 1, shared=False); f2 = fit_channels(X, e, 2, shared=False)
        rows.append(dict(corpus="LM factorial", field=s.field.iloc[0], frontier=name, n_res=2,
                         N_hat=int(lm_n.get(name, np.nan)), **resolution(X, e, f1, f2)))
    if rows is not None:
        D = pd.DataFrame(rows); save(D, "table_resolution_frontiers.csv")
    D["logV"] = np.log10(D.V)

    print("\nPart 1: calibration simulations ...")
    S = simulate(); save(S, "table_resolution_sim.csv")
    S["logV"] = np.log10(S.V)
    S["bin"] = pd.cut(S.logV, [-9, 0, 0.5, 1, 1.5, 2, 2.5, 3, 9])
    cal = S.groupby(["truth", "bin"], observed=True).agg(
        n=("N_hat", "size"), p_detect=("N_hat", lambda v: float((v >= 2).mean())),
        ).reset_index()
    print(cal.to_string(index=False))
    two = S[S.truth.str.startswith("two")]
    b0, b1, v50, auc = logistic(two.logV.to_numpy(), (two.N_hat >= 2).to_numpy().astype(int))
    print(f"two-channel truth: P(detect)=1/2 at log10 V = {v50:.2f}  (V = {10**v50:.1f})")

    summ = []
    for corpus, g in [("82 frontiers", D[D.corpus == "82 frontiers"]), ("LM factorial", D[D.corpus == "LM factorial"]),
                      ("all", D)]:
        y = (g.N_hat >= 2).to_numpy().astype(int)
        for k in (1, 2, 3):
            h = g[g.N_hat == k]
            if len(h):
                summ.append(dict(corpus=corpus, N_hat=k, frontiers=len(h), median_Z=h.Z.median(),
                                 median_m=h.m.median(), median_sigma=h.sigma.median(),
                                 median_V=h.V.median(), median_V3=h.V3.median()))
        if 0 < y.sum() < len(y):
            p = stats.mannwhitneyu(g.logV[y == 1], g.logV[y == 0], alternative="greater").pvalue
            b0_, b1_, v50_, auc_ = logistic(g.logV.to_numpy(), y)
            summ.append(dict(corpus=corpus, N_hat=-1, frontiers=len(g), mw_p_greater=p, auc=auc_,
                             logistic_slope=b1_, log10V_at_half=v50_,
                             sim_log10V_at_half=v50, frac_N2_below_simhalf=float(
                                 (g.N_hat[g.logV < v50] >= 2).mean()) if (g.logV < v50).any() else np.nan,
                             frac_N2_above_simhalf=float((g.N_hat[g.logV >= v50] >= 2).mean())
                             if (g.logV >= v50).any() else np.nan))
    T = pd.DataFrame(summ); save(T, "table_resolution_summary.csv")
    print(T.round(3).to_string(index=False))
    print("\nlargest V3 (third-channel resolution):")
    print(D.sort_values("V3", ascending=False)[["corpus", "frontier", "N_hat", "m", "Z", "sigma", "V", "V3"]]
          .head(6).round(2).to_string(index=False))
    # where a third channel would become visible, per simulation of a third
    three = S[S.truth.str.startswith("Gamma")]
    print("\nGamma spectrum: fraction of fits with N_hat >= 3:", f"{(three.N_hat >= 3).mean():.3f}",
          " max N_hat:", int(three.N_hat.max()))


if __name__ == "__main__":
    main()
