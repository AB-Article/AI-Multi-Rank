#!/usr/bin/env python3
"""T12  The width coordinate: zeta = ln g is the normalized resource.

Theory (Proposition 'width coordinate' and Corollary 'affine width', main
text).  Along a resource ray write S = -ln eps, s = S', kappa = -S'',
kappa_3 = S''' (the first three cumulants of the exponent a under the tilted
measure p_x), g = kappa / s^2 and

      zeta(x) := ln g(x) = ln( kappa / s^2 ).                          (W1)

Then, for a spectrum with an atom at the origin and one further atom alpha
(one reducible channel and a floor),

  (i)   zeta = ln( p_0 / (1 - p_0) ), i.e. the log-width IS the normalized
        resource of the single-channel normal form, measured without the
        floor and without a fit;
  (ii)  d zeta / dx = alpha exactly, at every point, for every floor, every
        exponent and every scale: zeta is affine with slope the exponent;
  (iii) eps = 2 eps_inf exactly where zeta = 0, so the canonical reference
        point is the zero crossing of the log-width and the floor is half the
        error there;
  (iv)  in general d zeta / dx = 2 kappa / s - kappa_3 / kappa, so the
        two-atom case is the closure kappa_3 = kappa (kappa - s^2) / s;
  (v)   consequently R := (d zeta / dx) / alpha_hat, with
        alpha_hat = s + kappa/s the floor-free exponent, satisfies
            R = 0  for a gapless Gamma spectrum (zeta constant),
            R = 1  for one channel plus a floor,
            R > 1  when further structure is present,
        and R is fit-free, floor-free and unit-free.

Part 0 verifies (i)-(v) at 50-digit precision on random spectra.
Part 1 fits zeta affinely on every single-resource frontier of the corpus and
reports slope, R^2, R, the zero crossing and the half-error floor estimate.
Part 2 is the power study: R on synthetic two- and three-atom spectra at a
realistic benchmark design, with noise, and the resulting AUC.  This is the
statistic that replaces the geodesic deficit d/L, which Section 'path test'
shows has no power at present resolution.
Part 3 scores R against the BIC channel count on the corpus.

Writes results/table_width_verification.csv, table_width_per_frontier.csv,
table_width_summary.csv, table_width_power.csv.
"""
import numpy as np, pandas as pd, mpmath as mp, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from _loaders import selection_rules, all_fields, save

sr = selection_rules()
fit_channels = all_fields()["fit_channels"]
RNG = np.random.default_rng(20260938)
mp.mp.dps = 50


# ===================================================== Part 0: identities ==
def _S_factory(a, c):
    eps = lambda x: sum(cj * mp.e ** (-aj * x) for cj, aj in zip(c, a))
    S = lambda x: -mp.log(eps(x))
    return eps, S


def _zeta(S):
    return lambda t: mp.log(-mp.diff(S, t, 2) / mp.diff(S, t) ** 2)


def verify():
    d_norm = d_slope = d_gen = d_half = d_clos = 0.0
    d_gamma = 0.0
    r_three = []

    # (i) (ii) (iii): two atoms, random exponent / floor / scale
    for _ in range(25):
        alpha = mp.mpf(RNG.uniform(0.15, 3.0))
        einf = mp.mpf(RNG.uniform(0.01, 0.4))
        C = mp.mpf(RNG.uniform(0.1, 5.0))
        eps, S = _S_factory([mp.mpf(0), alpha], [einf, C])
        z = _zeta(S)
        x = mp.mpf(RNG.uniform(-2, 3))
        p0 = einf / eps(x)
        d_norm = max(d_norm, float(abs(z(x) - mp.log(p0 / (1 - p0)))))
        d_slope = max(d_slope, float(abs(mp.diff(z, x) - alpha)))
        xs = mp.findroot(lambda t: z(t), mp.log(C / einf) / alpha)
        d_half = max(d_half, float(abs(eps(xs) - 2 * einf)))

    # (iv) (v): general spectra
    for _ in range(20):
        K = int(RNG.integers(1, 4))
        a = [mp.mpf(0)] + [mp.mpf(RNG.uniform(0.2, 3.0)) for _ in range(K)]
        c = [mp.mpf(RNG.uniform(0.02, 0.4))] + [mp.mpf(RNG.uniform(0.1, 3)) for _ in range(K)]
        _, S = _S_factory(a, c)
        z = _zeta(S)
        x = mp.mpf(RNG.uniform(-1, 2))
        s = mp.diff(S, x); k = -mp.diff(S, x, 2); k3 = mp.diff(S, x, 3)
        d_gen = max(d_gen, float(abs(mp.diff(z, x) - (2 * k / s - k3 / k))))
        if K == 1:
            d_clos = max(d_clos, float(abs(k3 - k * (k - s ** 2) / s)))
        else:
            ah = s + k / s
            r_three.append(float(mp.diff(z, x) / ah))

    # Gamma spectrum: zeta constant, slope 0, R = 0
    for _ in range(10):
        k = mp.mpf(RNG.uniform(0.3, 6)); beta = mp.mpf(RNG.uniform(0.5, 4))
        S = lambda t: k * mp.log(1 + t / beta)
        z = _zeta(S)
        for x in mp.linspace(mp.mpf('0.1'), 6, 5):
            d_gamma = max(d_gamma, float(abs(mp.diff(z, x))))

    return pd.DataFrame([
        dict(result="zeta = ln g equals ln(p0/(1-p0))  (two atoms)", max_deviation=d_norm),
        dict(result="d zeta / dx = alpha  (two atoms, every x)", max_deviation=d_slope),
        dict(result="eps = 2 eps_inf at zeta = 0  (two atoms)", max_deviation=d_half),
        dict(result="d zeta / dx = 2 kappa/s - kappa_3/kappa  (any spectrum)", max_deviation=d_gen),
        dict(result="cumulant closure kappa_3 = kappa(kappa-s^2)/s  (two atoms)",
             max_deviation=d_clos),
        dict(result="Gamma spectrum: d zeta / dx = 0, hence R = 0", max_deviation=d_gamma),
        dict(result="min |R - 1| over >=3-atom spectra (must be > 0)",
             max_deviation=float(np.min(np.abs(np.array(r_three) - 1))) if r_three else np.nan)])


# ============================================ local estimator (as 24/32/34) =
def local(x, S, xc, h):
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
    return b[0], b[1], -b[2]                                   # S, s, kappa


def profile(x, S, ng=41, trim=0.10):
    span = np.ptp(x); h = span / 4
    gr = np.linspace(x.min() + trim * span, x.max() - trim * span, ng)
    o = np.array([local(x, S, xc, h) for xc in gr])
    return gr, o[:, 0], o[:, 1], o[:, 2]


def width_fit(x, e):
    """Affine fit of zeta = ln(kappa/s^2) on x.  Returns slope, R^2, R, x*."""
    S = -np.log(e)
    gr, Sf, s, kap = profile(x, S)
    ok = (s > 1e-6) & (kap > 1e-12)
    if ok.sum() < 10:
        return None
    u, Sf, s, kap = gr[ok], Sf[ok], s[ok], kap[ok]
    z = np.log(kap / s ** 2)
    b1, b0 = np.polyfit(u, z, 1)
    resid = z - (b1 * u + b0)
    r2 = 1.0 - resid.var() / z.var() if z.var() > 0 else np.nan
    ah = float(np.median(s + kap / s))
    xstar = -b0 / b1 if abs(b1) > 1e-9 else np.nan
    return dict(slope=float(b1), intercept=float(b0), r2=float(r2),
                alpha_hat=ah, R=float(b1 / ah) if ah > 0 else np.nan,
                xstar=float(xstar), z_lo=float(z[0]), z_hi=float(z[-1]),
                span=float(np.ptp(x)))


def half_floor(x, e, xstar):
    """eps(x*)/2 : the floor read at the zero crossing of the log-width."""
    if not np.isfinite(xstar):
        return np.nan
    xs = min(max(xstar, x.min()), x.max())
    S = -np.log(e)
    Ss, _, _ = local(x, S, xs, np.ptp(x) / 4)
    return float(np.exp(-Ss) / 2)


def fit3(x, e):
    from scipy import optimize
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
    return (np.exp(best.x[0]), np.exp(best.x[2])) if best else (np.nan, np.nan)


def nhat1(x, e):
    b = {}
    for K in (1, 2, 3):
        try:
            b[K] = fit_channels(x[:, None], e, K, shared=False)["bic"]
        except Exception:
            pass
    return min(b, key=b.get) if b else np.nan


def auc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ============================================== Part 1 + 3: real frontiers ==
def frontiers():
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        x = np.asarray(x, float); e = np.asarray(e, float)
        if len(e) < 15 or np.ptp(x) < 1.0:
            continue
        r = width_fit(x, e)
        if r is None:
            continue
        boot = []
        for _ in range(200):
            i = np.sort(RNG.choice(len(e), len(e), replace=True))
            if len(np.unique(x[i])) < 8:
                continue
            rr = width_fit(x[i], e[i])
            if rr and np.isfinite(rr["R"]):
                boot.append(rr["R"])
        se = float(np.std(boot, ddof=1)) if len(boot) > 20 else np.nan
        einf, alpha = fit3(x, e)
        r.update(field=field, frontier=name, cost=cost, n=len(e),
                 R_se=se, Z_R=((r["R"] - 1) / se) if se and se > 0 else np.nan,
                 inside=bool(np.isfinite(r["xstar"]) and x.min() <= r["xstar"] <= x.max()),
                 floor_half=half_floor(x, e, r["xstar"]),
                 einf_fit=einf, alpha_fit=alpha, N_hat=nhat1(x, e))
        rows.append(r)
    return pd.DataFrame(rows)


# ======================================================== Part 2: the power =
def synth(natoms, span, sigma, reps=400, npts=30):
    out = []
    for _ in range(reps):
        al = RNG.uniform(0.3, 1.5); ei = RNG.uniform(0.02, 0.30); C = 1.0
        a, c = [0.0, al], [ei, C]
        if natoms >= 3:
            a.append(al * RNG.uniform(2.0, 5.0)); c.append(RNG.uniform(0.3, 3.0))
        xc = np.log(C / ei) / al                       # centre on the crossover
        x = np.linspace(xc - span / 2, xc + span / 2, npts)
        e = sum(cj * np.exp(-aj * x) for cj, aj in zip(c, a))
        if sigma > 0:
            e = e * np.exp(RNG.normal(0, sigma, npts))
        r = width_fit(x, e)
        if r and np.isfinite(r["R"]):
            out.append(r["R"])
    return np.array(out)


def power():
    rows = []
    for span in (2.5, 4.7):
        for sigma in (0.0, 0.005, 0.01, 0.02):
            r2 = synth(2, span, sigma)
            r3 = synth(3, span, sigma)
            if len(r2) < 50 or len(r3) < 50:
                continue
            lab = np.r_[np.zeros(len(r2)), np.ones(len(r3))].astype(bool)
            rows.append(dict(
                span_nats=span, sigma=sigma,
                R_two_median=float(np.median(r2)),
                R_two_p05=float(np.percentile(r2, 5)),
                R_two_p95=float(np.percentile(r2, 95)),
                R_three_median=float(np.median(r3)),
                R_three_p05=float(np.percentile(r3, 5)),
                R_three_p95=float(np.percentile(r3, 95)),
                auc=auc(np.r_[r2, r3], lab),
                mwu_p=float(stats.mannwhitneyu(r3, r2, alternative="greater").pvalue)))
    return pd.DataFrame(rows)


# =============================================================== run =======
def main():
    V = verify(); save(V, "table_width_verification.csv")
    print(V.to_string(index=False))

    P = frontiers(); save(P, "table_width_per_frontier.csv")
    P["two"] = P.N_hat >= 2

    ok = np.isfinite(P.alpha_fit) & (P.alpha_fit > 0)
    mf = (P.inside & np.isfinite(P.floor_half) & (P.einf_fit > 1e-6)
          & np.isfinite(P.floor_half))
    lr = np.log(P.floor_half[mf] / P.einf_fit[mf])

    S = pd.DataFrame([
        dict(statistic="frontiers analysed", value=float(len(P))),
        dict(statistic="median R^2 of the affine fit of zeta on x",
             value=float(P.r2.median())),
        dict(statistic="fraction with R^2 >= 0.8", value=float((P.r2 >= 0.8).mean())),
        dict(statistic="median slope d zeta / dx (per nat)", value=float(P.slope.median())),
        dict(statistic="fraction with slope > 0", value=float((P.slope > 0).mean())),
        dict(statistic="median alpha_hat", value=float(P.alpha_hat.median())),
        dict(statistic="median fitted alpha (three-parameter)",
             value=float(P.alpha_fit[ok].median())),
        dict(statistic="corr(slope, fitted alpha)",
             value=float(np.corrcoef(P.slope[ok], P.alpha_fit[ok])[0, 1])),
        dict(statistic="median slope / fitted alpha",
             value=float(np.median(P.slope[ok] / P.alpha_fit[ok]))),
        dict(statistic="median R = slope / alpha_hat", value=float(P.R.median())),
        dict(statistic="fraction with R > 1", value=float((P.R > 1).mean())),
        dict(statistic="fraction with R > 1 by more than one bootstrap SE",
             value=float((P.Z_R > 1).mean())),
        dict(statistic="median R | N_hat = 1", value=float(P.loc[~P.two, "R"].median())),
        dict(statistic="median R | N_hat >= 2", value=float(P.loc[P.two, "R"].median())),
        dict(statistic="AUC  R -> N_hat >= 2", value=auc(P.R, P.two)),
        dict(statistic="Mann-Whitney p (R larger when N_hat >= 2)",
             value=float(stats.mannwhitneyu(P.loc[P.two, "R"].dropna(),
                                            P.loc[~P.two, "R"].dropna(),
                                            alternative="greater").pvalue)),
        dict(statistic="fraction whose window contains the zero crossing",
             value=float(P.inside.mean())),
        dict(statistic="half-error floor: n compared", value=float(mf.sum())),
        dict(statistic="half-error floor: median log-ratio to fitted floor",
             value=float(lr.median())),
        dict(statistic="half-error floor: corr on logs",
             value=float(np.corrcoef(np.log(P.floor_half[mf]),
                                     np.log(P.einf_fit[mf]))[0, 1]))])
    save(S, "table_width_summary.csv")
    print("\n" + S.to_string(index=False))

    W = power(); save(W, "table_width_power.csv")
    print("\npower of R (synthetic, 30 Pareto points)\n" + W.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
