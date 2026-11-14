#!/usr/bin/env python3
"""Does each resource have its own channel (N = n)?  A factorial test.

Model family (power mean of per-resource channels):
    eps = einf + M_theta(g),   g_i = exp(b_i - alpha_i * xc_i),
    M_theta(g) = ( mean_i g_i^theta )^(1/theta),   M_0 = geometric mean.
    theta = 1   : eps = einf + sum_i c_i t_i^-alpha_i      -> N = n  (diagonal law)
    theta = 0   : eps = einf + c prod_i t_i^-a_i           -> N = 1  (single channel)
    theta -> oo : eps = einf + max_i g_i                   -> Chebyshev limit (beta = 1/theta -> 0)
theta = 1/beta, so this is the beta-deformation of Section "Non-convex frontiers"
extended through beta = oo (the single channel).  xc = log-resource, centred
(a G2 gauge move, harmless).

Tests per frontier
  (1) profile likelihood in theta: theta_hat and 95% interval; N = n is theta = 1.
  (2) Tukey's one-degree-of-freedom test for non-additivity on the eps scale
      (additive main effects + gamma * yhat^2).  N = n  <=>  eps additively
      separable  <=>  gamma = 0;  N = 1 implies gamma > 0.  Law-free: main
      effects are level dummies (crossed grids) or cubic B-splines.
  (3) held-out extrapolation: fit on the cheapest 60% by total log-resource,
      predict the rest, theta = 0 vs theta = 1.
  (4) channel count by BIC over K = 1..4 free-vector channels (as Sec. N-obs).
  (5) crossed grids only: raw 2x2 interaction contrasts.
"""
import os, sys, warnings, itertools
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import BSpline
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
THETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0]


# --------------------------------------------------------------- the law ---
def powmean_eps(p, X, theta):
    n = X.shape[1]
    einf, b, a = p[0], p[1:1 + n], p[1 + n:1 + 2 * n]
    lg = b[None, :] - X * a[None, :]
    if theta == 0.0:
        lm = lg.mean(1)
    else:
        z = theta * lg
        zm = z.max(1, keepdims=True)
        lm = (zm[:, 0] + np.log(np.exp(z - zm).mean(1))) / theta
    return einf + np.exp(lm)


def fit_theta(X, e, theta, warm=None):
    n = X.shape[1]
    lo = [0.0] + [-40.0] * n + [0.0] * n
    hi = [max(e.min(), 1e-6) * 0.999 + 1e-9] + [10.0] * n + [6.0] * n
    starts = []
    if warm is not None:
        starts.append(np.clip(warm, np.array(lo) + 1e-12, np.array(hi) - 1e-12))
    grid = [(0.0, 0.2), (0.5, 0.6), (0.9, 1.2)] if (warm is not None and len(e) > 2000) \
        else list(itertools.product((0.0, 0.5, 0.9), (0.2, 0.6, 1.2)))
    for f0, a0 in grid:
            einf = f0 * e.min()
            b0 = np.log(max(np.median(e) - einf, 1e-6))
            starts.append(np.array([einf] + [b0] * n + [a0] * n))
    best = None
    for s in starts:
        s = np.clip(s, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
        try:
            r = least_squares(lambda p: powmean_eps(p, X, theta) - e, s,
                              bounds=(lo, hi), max_nfev=20000, x_scale="jac")
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if best is None or rss < best[1]:
            best = (r.x, rss)
    return best


def k_eff(n, theta):
    return 2 + n if theta == 0.0 else 1 + 2 * n


def bic(rss, m, k):
    return m * np.log(rss / m) + k * np.log(m)


def profile(X, e):
    m, n = len(e), X.shape[1]
    out, warm = {}, None
    for th in THETAS:
        p, rss = fit_theta(X, e, th, warm)
        out[th] = (p, rss)
        warm = p
    # refine around the minimum
    ths = np.array(THETAS)
    rs = np.array([out[t][1] for t in THETAS])
    i = int(np.argmin(rs))
    lo_t, hi_t = ths[max(i - 1, 0)], ths[min(i + 1, len(ths) - 1)]
    for th in np.linspace(lo_t, hi_t, 9):
        th = float(round(th, 4))
        if th not in out:
            out[th] = fit_theta(X, e, th, out[ths[i]][0])
    ths = np.array(sorted(out))
    rs = np.array([out[t][1] for t in ths])
    th_hat = float(ths[np.argmin(rs)])
    dev = m * np.log(rs / rs.min())
    ok = ths[dev <= 3.84]
    return dict(theta_hat=th_hat, ci_lo=float(ok.min()), ci_hi=float(ok.max()),
                dev_at_1=float(dev[ths == 1.0][0]), dev_at_0=float(dev[ths == 0.0][0]),
                rss0=out[0.0][1], rss1=out[1.0][1], p0=out[0.0][0], p1=out[1.0][0],
                bic0=bic(out[0.0][1], m, k_eff(n, 0.0)),
                bic1=bic(out[1.0][1], m, k_eff(n, 1.0)))


# ------------------------------------------------- Tukey non-additivity -----
def basis(x):
    u = np.unique(x)
    if len(u) <= 16:
        return (x[:, None] == u[None, 1:]).astype(float)
    k = 3
    inner = np.quantile(x, np.linspace(0, 1, 8)[1:-1])
    t = np.r_[[x.min()] * (k + 1), inner, [x.max()] * (k + 1)]
    B = BSpline.design_matrix(np.clip(x, x.min(), x.max()), t, k).toarray()
    return B[:, 1:]


def tukey(X, e):
    Z = np.column_stack([np.ones(len(e))] + [basis(X[:, i]) for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(Z, e, rcond=None)
    yh = Z @ beta
    rss_a = float(np.sum((e - yh) ** 2))
    q = (yh - yh.mean()) ** 2
    Z2 = np.column_stack([Z, q])
    b2, *_ = np.linalg.lstsq(Z2, e, rcond=None)
    rss_t = float(np.sum((e - Z2 @ b2) ** 2))
    df = len(e) - np.linalg.matrix_rank(Z2)
    F = (rss_a - rss_t) / (rss_t / df)
    return dict(gamma=float(b2[-1]), F=float(F), p=float(stats.f.sf(F, 1, df)),
                nonadd_share=float((rss_a - rss_t) / rss_a))


# --------------------------------------------------------- extrapolation ----
def extrap(X, e):
    order = np.argsort(X.sum(1))
    k = int(round(0.6 * len(e)))
    tr, te = order[:k], order[k:]
    res = {}
    for th in (0.0, 1.0):
        p, _ = fit_theta(X[tr], e[tr], th)
        pred = powmean_eps(p, X[te], th)
        res[th] = float(np.mean(np.abs(pred / e[te] - 1)) * 100)
    return res


# ----------------------------------------------------------- channels ------
def fit_K(X, e, K):
    n = X.shape[1]

    def f(t):
        einf, b = t[0], t[1:1 + K]
        A = t[1 + K:].reshape(K, n)
        return einf + np.exp(b[None, :] - X @ A.T).sum(1) - e
    best = None
    rng = np.random.default_rng(0)
    for s in range(6):
        p0 = np.r_[0.5 * e.min(), np.full(K, np.log(np.median(e) / K)),
                   rng.uniform(0.05, 1.0, K * n)]
        lo = np.r_[0.0, np.full(K, -40.0), np.zeros(K * n)]
        hi = np.r_[e.min(), np.full(K, 10.0), np.full(K * n, 6.0)]
        try:
            r = least_squares(f, np.clip(p0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                              max_nfev=20000)
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if best is None or rss < best:
            best = rss
    return bic(best, len(e), 1 + K + K * n)


def N_hat(X, e, Kmax=4):
    b = {K: fit_K(X, e, K) for K in range(1, Kmax + 1)}
    return min(b, key=b.get), b


# ------------------------------------------------------ raw contrasts ------
def contrasts(X, e, p0, p1):
    """All 2x2 rectangles in a fully crossed 2-resource grid."""
    df = pd.DataFrame(dict(a=X[:, 0].round(6), b=X[:, 1].round(6), e=e,
                           m0=powmean_eps(p0, X, 0.0), m1=powmean_eps(p1, X, 1.0)))
    A, B = np.sort(df.a.unique()), np.sort(df.b.unique())
    piv = {c: df.pivot_table(index="a", columns="b", values=c) for c in ("e", "m0", "m1")}
    if piv["e"].isna().values.any():
        return None
    out = {c: [] for c in piv}
    for i, j in itertools.combinations(range(len(A)), 2):
        for k, l in itertools.combinations(range(len(B)), 2):
            for c, P in piv.items():
                v = P.values
                tot = v[i, k] - v[j, l]
                if tot <= 0:
                    continue
                out[c].append((v[i, k] - v[j, k] - v[i, l] + v[j, l]) / tot)
    return {c: (float(np.median(v)), float(np.mean(np.array(v) > 0))) for c, v in out.items()}


# ----------------------------------------------------------------- run -----
D = pd.read_csv(os.path.join(ROOT, "data", "grid_factorial.csv"))
rows = []
ONLY = os.environ.get("ONLY")
for (field, fr), d in D.groupby(["field", "frontier"], sort=False):
    if ONLY and ONLY not in field:
        continue
    costs = [c for c in d.columns if c.startswith("cost_") and d[c].notna().all()]
    d = d.dropna(subset=["eps"] + costs)
    d = d[(d.eps > 0) & (d.eps < 1.5)]
    X = np.log(d[costs].to_numpy(float))
    X = X - X.mean(0)
    e = d.eps.to_numpy(float)
    n, m = X.shape[1], len(e)
    rho = np.corrcoef(X.T)[np.triu_indices(n, 1)]
    pr = profile(X, e)
    tk = tukey(X, e)
    ex = extrap(X, e)
    Nh, bics = N_hat(X, e) if m <= 400 else N_hat(X, e, Kmax=3)
    rec = dict(field=field, frontier=fr, n_res=n, m=m,
               max_abs_rho=float(np.abs(rho).max()),
               decades=";".join(f"{np.ptp(X[:, i]) / np.log(10):.2f}" for i in range(n)),
               theta_hat=pr["theta_hat"], theta_lo=pr["ci_lo"], theta_hi=pr["ci_hi"],
               dev_theta1=pr["dev_at_1"], dev_theta0=pr["dev_at_0"],
               dBIC_1_minus_0=pr["bic1"] - pr["bic0"],
               einf_theta1=float(pr["p1"][0]), alpha_theta1=";".join(
                   f"{a:.3f}" for a in pr["p1"][1 + n:1 + 2 * n]),
               tukey_gamma=tk["gamma"], tukey_p=tk["p"], nonadd_share=tk["nonadd_share"],
               extrap_mape_theta0=ex[0.0], extrap_mape_theta1=ex[1.0],
               N_hat=Nh, **{f"bic_K{K}": v for K, v in bics.items()})
    if n == 2 and field == "lm-downstream" and fr.startswith("pythia"):
        c = contrasts(X, e, pr["p0"], pr["p1"])
        if c:
            rec.update(contrast_med_obs=c["e"][0], contrast_pos_obs=c["e"][1],
                       contrast_med_N1=c["m0"][0], contrast_med_Nn=c["m1"][0])
    rows.append(rec)
    print(f"{fr:34s} m={m:5d} theta={pr['theta_hat']:.2f} "
          f"[{pr['ci_lo']:.2f},{pr['ci_hi']:.2f}]  dBIC(1-0)={rec['dBIC_1_minus_0']:8.1f} "
          f"tukey g={tk['gamma']:+.2f} p={tk['p']:.1e}  "
          f"extrap {ex[0.0]:.1f}/{ex[1.0]:.1f}  N^={Nh}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(RES, f"table_factorial_N_test{('_' + ONLY) if ONLY else ''}.csv"), index=False)
print("\nwritten results/table_factorial_N_test.csv")
