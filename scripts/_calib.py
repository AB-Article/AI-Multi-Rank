"""Shared machinery for the estimator-calibration scripts 40-44.

The width statistics of Proposition 'width coordinate' are identities in the
exact derivatives of S.  Evaluating them requires plug-in estimates of
s = S' and kappa = -S'', and any such estimate carries a bandwidth.  These
helpers reproduce the estimator of script 38 exactly, so that its null can be
simulated on the same grids rather than assumed to equal the analytic value.
"""
import numpy as np
from scipy import optimize

# --------------------------------------------- estimator, as in script 38 ---
def local(x, S, xc, h):
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
    return b[0], b[1], -b[2]


def width_fit(x, e, div=4.0, ng=41, trim=0.10):
    """Affine fit of zeta = ln(kappa/s^2) on x.  div=4 is the bandwidth of
    script 38 (h = span/4); other values probe the bandwidth dependence."""
    x = np.asarray(x, float); e = np.asarray(e, float)
    span = np.ptp(x)
    if span <= 0:
        return None
    S = -np.log(e); h = span / div
    gr = np.linspace(x.min() + trim * span, x.max() - trim * span, ng)
    o = np.array([local(x, S, xc, h) for xc in gr])
    s, kap = o[:, 1], o[:, 2]
    ok = (s > 1e-6) & (kap > 1e-12)
    if ok.sum() < 10:
        return None
    u, s, kap = gr[ok], s[ok], kap[ok]
    z = np.log(kap / s ** 2)
    b1, b0 = np.polyfit(u, z, 1)
    resid = z - (b1 * u + b0)
    ah = float(np.median(s + kap / s))
    if not (ah > 0):
        return None
    return dict(slope=float(b1), intercept=float(b0), alpha_hat=ah,
                r2=float(1.0 - resid.var() / z.var()) if z.var() > 0 else np.nan,
                R=float(b1 / ah), z_lo=float(z[0]), z_hi=float(z[-1]))


# ------------------------------------------- two-atom generator for a null ---
def fit3(x, e):
    """eps = einf + c exp(-a x), fitted jointly in log-error (as script 38's
    fit3, which returns only einf and a; c and the residual scale are needed
    to generate a matched null)."""
    x = np.asarray(x, float); e = np.asarray(e, float)
    f = lambda q, xx: np.exp(q[0]) + np.exp(q[1]) * np.exp(-np.exp(q[2]) * xx)
    r = lambda q: np.log(np.clip(f(q, x), 1e-12, None)) - np.log(e)
    best = None
    for a0 in (0.2, 0.5, 1.0, 2.0):
        try:
            sol = optimize.least_squares(
                r, [np.log(max(e.min() * .5, 1e-8)), np.log(e.max()), np.log(a0)])
            if best is None or sol.cost < best.cost:
                best = sol
        except Exception:
            pass
    if best is None:
        return None
    einf, c, a = (float(np.exp(v)) for v in best.x)
    res = np.log(np.clip(f(best.x, x), 1e-12, None)) - np.log(e)
    return dict(einf=einf, c=c, a=a, sd=float(max(res.std(ddof=1), 1e-4)))


def pareto1d(x, e):
    i = np.argsort(x); x, e = np.asarray(x)[i], np.asarray(e)[i]
    keep, best = [], np.inf
    for j in range(len(x)):
        if e[j] < best - 1e-15:
            keep.append(j); best = e[j]
    return x[keep], e[keep]


def simulate_null(x, einf, c, a, sd, rng, div=4.0, nsim=600, natoms=2,
                  a2_range=(2.0, 5.0), c2_range=(0.3, 3.0)):
    """R under a spectrum with `natoms` atoms, on the grid x, with log-error
    noise of scale sd and the same one-dimensional Pareto filter."""
    out = []
    for _ in range(nsim):
        et = einf + c * np.exp(-a * x)
        if natoms >= 3:
            et = et + (c * rng.uniform(*c2_range)) * np.exp(-a * rng.uniform(*a2_range) * x)
        en = et * np.exp(rng.normal(0, sd, len(x)))
        xs, es = pareto1d(x, en)
        if len(es) < 10:
            continue
        v = width_fit(xs, es, div=div)
        if v is not None and np.isfinite(v["R"]):
            out.append(v["R"])
    return np.array(out)
