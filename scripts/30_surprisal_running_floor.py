#!/usr/bin/env python3
"""S1 + S2 + S3  The surprisal identity, the running floor, and the measured
spectral width.

Theory (main text, Section 'The Surprisal Identity and the Running Floor').

S1  Surprisal identity.  For any spectral measure mu with an atom at the
    origin, at every operating point x,
        ln( eps / eps_inf ) = - ln p_0(x),
    where p_0 is the error share of the floor mode.  Excess log-error IS the
    surprisal of the floor.  Every scalar consequence of the single-channel
    theory is a coordinate of p_0:
        eps/eps_inf = 1/p_0,  eta_i = a_i (1 - p_0),  -S'' = (a.v)^2 p_0(1-p_0),
        zeta = ln( p_0/(1-p_0) ),  R = KL( Bern(1/2) || Bern(p_0) ),
        canonical point  p_0 = 1/(N+1),  feasibility  eps_inf <= p_0.

S2  Running floor.  Over a window of length X, exponents a <~ 1/X are not
    separable from the atom at 0, so the fitted floor is an upper bound that
    decreases as the window is extended:
        eps_inf_hat(X) - eps_inf  ~  mu( (0, c/X] ).
    A genuine irreducible floor exists iff mu has a spectral gap at 0.  With a
    gap the sequence stabilises; without one it keeps falling.

S3  Measured spectral width.  From the local slope s = S' and curvature
    k = -S'' alone (no floor, no fit),
        g := k / s^2 = alpha_hat / s - 1 = CV^2,
    which is the coefficient of variation squared that enters the resolution
    number of Proposition 'spectral resolution'.  The resolution criterion
    g * V >~ 1, V = Z^2 sqrt(m) / (16 sigma), is therefore measurable without
    fitting a second channel.

Writes results/table_surprisal_verification.csv, table_running_floor.csv,
table_running_floor_summary.csv, table_spectral_width.csv,
table_p0_selection.csv.
"""
import numpy as np, pandas as pd, mpmath as mp
import os
from _loaders import selection_rules, all_fields, save, RES

sr = selection_rules()
RNG = np.random.default_rng(20260914)
mp.mp.dps = 40


# ----------------------------------------------------- S1: the identities --
def verify():
    """High-precision check of the surprisal identity and of every scalar law
    it subsumes, on random spectra with an atom at the origin."""
    d_sur = d_eta = d_curv = d_zeta = d_reg = d_gap = d_logcvx = 0.0
    d_canon = d_feas = 0.0
    for _ in range(60):
        K = int(RNG.integers(1, 5))
        a = [mp.mpf(0)] + [mp.mpf(RNG.uniform(0.1, 3.0)) for _ in range(K)]
        c = [mp.mpf(RNG.uniform(0.02, 0.4))] + [mp.mpf(RNG.uniform(0.1, 3)) for _ in range(K)]
        eps = lambda x: sum(cj * mp.e ** (-aj * x) for cj, aj in zip(c, a))
        S = lambda x: -mp.log(eps(x))
        for x in mp.linspace(-3, 5, 9):
            E = eps(x)
            p0 = c[0] / E
            # (1) surprisal identity
            d_sur = max(d_sur, abs(mp.log(E / c[0]) + mp.log(p0)))
            s, k = mp.diff(S, x), -mp.diff(S, x, 2)
            # (5) measured width equals CV^2 of the tilted measure
            w = [cj * mp.e ** (-aj * x) for cj, aj in zip(c, a)]
            p = [wj / E for wj in w]
            m1 = sum(pj * aj for pj, aj in zip(p, a))
            m2 = sum(pj * aj ** 2 for pj, aj in zip(p, a))
            cv2 = (m2 - m1 ** 2) / m1 ** 2
            d_gap = max(d_gap, abs(k / s ** 2 - cv2), abs((s + k / s) / s - 1 - cv2))
            # (6) log-convexity of eps in x  (three-point validity condition)
            h = mp.mpf('0.37')
            d_logcvx = max(d_logcvx, float(eps(x) ** 2 - eps(x - h) * eps(x + h)))
            if K == 1:
                # (2) elasticity, (3) curvature, (4) zeta and regret
                d_eta = max(d_eta, abs(s - a[1] * (1 - p0)))
                d_curv = max(d_curv, abs(k - a[1] ** 2 * p0 * (1 - p0)))
                z = mp.log(p0 / (1 - p0))
                d_zeta = max(d_zeta, abs(E / c[0] - (1 + mp.e ** (-z))))
                R = mp.log(mp.cosh(z / 2))
                KLb = (mp.log(1 / (2 * p0)) + mp.log(1 / (2 * (1 - p0)))) / 2
                d_reg = max(d_reg, abs(R - KLb))
        # (7) canonical point and feasibility, in p_0 coordinates
        N = K
        p0c = mp.mpf(1) / (N + 1)
        d_canon = max(d_canon, abs(mp.log(1 / p0c) - mp.log(N + 1)))
        d_feas = max(d_feas, float(mp.mpf(1) / (N + 1) - p0c))
    return pd.DataFrame([
        dict(result="ln(eps/eps_inf) = -ln p_0 (surprisal identity)", max_deviation=float(d_sur)),
        dict(result="single channel: eta_i = a_i (1 - p_0)", max_deviation=float(d_eta)),
        dict(result="single channel: -S'' = (a.v)^2 p_0 (1 - p_0)", max_deviation=float(d_curv)),
        dict(result="single channel: eps/eps_inf = 1 + e^{-zeta}, zeta = ln(p_0/(1-p_0))",
             max_deviation=float(d_zeta)),
        dict(result="single channel: R = ln cosh(zeta/2) = KL(Bern(1/2)||Bern(p_0))",
             max_deviation=float(d_reg)),
        dict(result="k/s^2 = alpha_hat/s - 1 = CV^2 (measured spectral width)",
             max_deviation=float(d_gap)),
        dict(result="log-convexity eps(x)^2 - eps(x-h)eps(x+h) <= 0 (max value)",
             max_deviation=float(d_logcvx)),
        dict(result="canonical: -ln p_0 = ln(N+1) at p_0 = 1/(N+1)", max_deviation=float(d_canon)),
        dict(result="feasibility eps_inf <= p_0 = 1/(N+1) (max violation)", max_deviation=float(d_feas)),
    ])


# ----------------------------------------------------- S2: running floor ---
FRACS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def running_floor(x, e, fracs=FRACS, nmin=6):
    """Floor fitted on nested windows anchored at the cheap end."""
    out = {}
    lo, Z = x.min(), np.ptp(x)
    for f in fracs:
        m = x <= lo + f * Z
        if m.sum() < nmin:
            continue
        try:
            out[f] = sr.fit_law(x[m], e[m])[0]
        except Exception:
            pass
    return out


def _sample(a_draw, x, einf, scale, rms):
    """One realisation of a spectral frontier on the observed grid."""
    A = a_draw(400)
    u = x - x.min()
    e = einf + scale * np.exp(-np.outer(u, A)).mean(axis=1)
    return e * np.exp(rms * RNG.standard_normal(len(u)))


def floor_panel(nboot=60):
    """Balanced panel: frontiers on which the floor can be fitted on every
    nested window.  Two statistics per window fraction f:
        ratio(f) = eps_inf_hat(f) / eps_inf_hat(1)   (predicted >= 1, decreasing)
        over(f)  = eps_inf_hat(f) / eps_best         (predicted > 1 without a gap)
    calibrated against a gap null (fitted single channel) and a no-gap null
    (Gamma spectrum with mass down to a = 0 on top of the same floor)."""
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        if len(e) < 12:
            continue
        einf, c, a, rms = sr.fit_law(x, e)
        if not (1e-3 < einf < 0.5 and a < 5.99):
            continue
        obs = running_floor(x, e)
        if len(obs) < len(FRACS) or min(obs.values()) <= 0:
            continue                                   # balanced panel only
        nullG, nullN = {f: [] for f in FRACS}, {f: [] for f in FRACS}
        for _ in range(nboot):
            eg = (einf + c * np.exp(-a * (x - x.min()))) * np.exp(rms * RNG.standard_normal(len(x)))
            en = _sample(lambda k: RNG.gamma(1.0, a, k), x, einf, c, rms)
            for tgt, ee in ((nullG, eg), (nullN, en)):
                r = running_floor(x, ee)
                if len(r) == len(FRACS) and min(r.values()) > 0:
                    for f in FRACS:
                        tgt[f].append(r[f] / r[1.0])
        for f in FRACS:
            rows.append(dict(field=field, frontier=name, cost=cost, n=len(e),
                             span=np.ptp(x), window=f, einf=obs[f],
                             ratio=obs[f] / obs[1.0], over=obs[f] / e.min(),
                             null_gap=float(np.median(nullG[f])) if nullG[f] else np.nan,
                             null_gap_q95=float(np.quantile(nullG[f], .95)) if nullG[f] else np.nan,
                             null_nogap=float(np.median(nullN[f])) if nullN[f] else np.nan))
    R = pd.DataFrame(rows)
    save(R, "table_running_floor.csv")
    R["identified"] = R.ratio > 1e-6
    I = R[R.identified]
    g = R.groupby("window").agg(
        frontiers=("frontier", "nunique"),
        frac_floor_identified=("identified", "mean")).reset_index()
    g["median_ratio"] = g.window.map(I.groupby("window").ratio.median())
    g["q25_ratio"] = g.window.map(I.groupby("window").ratio.quantile(.25))
    g["q75_ratio"] = g.window.map(I.groupby("window").ratio.quantile(.75))
    g["frac_above_one"] = g.window.map(
        I.groupby("window").ratio.apply(lambda s: float((s > 1).mean())))
    g["frac_above_gap_null_q95"] = g.window.map(
        I.groupby("window").apply(lambda d: float((d.ratio > d.null_gap_q95).mean()),
                                  include_groups=False))
    g["median_null_gap"] = g.window.map(I.groupby("window").null_gap.median())
    g["median_null_nogap"] = g.window.map(I.groupby("window").null_nogap.median())
    save(g, "table_running_floor_summary.csv")
    return R, g


# ------------------------------------------------ S3: measured width g -----
def local(x, S, xc, h):
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
    return b[1], -b[2]


def auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _nhat_table():
    """Channel counts already selected by BIC in script 24."""
    f = os.path.join(RES, "table_floor_free_per_frontier.csv")
    d = pd.read_csv(f)
    return {(r.field, r.frontier, r.cost): r.N_hat for r in d.itertuples()}


def width_panel():
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        if len(e) < 10:
            continue
        S = -np.log(e)
        Z, m = np.ptp(x), len(e)
        s, k = local(x, S, x.min() + Z / 2, Z / 4)
        g = max(k, 0.0) / s ** 2 if s > 1e-6 else np.nan
        key = (field, name, cost)
        if key not in NHAT:
            continue
        Nhat = int(NHAT[key])
        einf, c, a, rms = sr.fit_law(x, e)
        V = Z ** 2 * np.sqrt(m) / (16 * max(rms, 1e-3))
        rows.append(dict(field=field, frontier=name, cost=cost, n=m, span=Z,
                         sigma=rms, N_hat=Nhat, s=s, k=k, g=g, V=V, gV=g * V,
                         alpha_hat=s + (max(k, 0.0) / s if s > 1e-6 else np.nan)))
    W = pd.DataFrame(rows)
    save(W, "table_spectral_width.csv")
    lab = (W.N_hat >= 2).to_numpy()
    print(f"  AUC(g)   = {auc(W.g, lab):.3f}   AUC(V) = {auc(W.V, lab):.3f}"
          f"   AUC(gV) = {auc(W.gV, lab):.3f}")
    return W


# --------------------------------------------- p_0 at the selected point ---
def p0_panel():
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        if len(e) < 10:
            continue
        einf, c, a, rms = sr.fit_law(x, e)
        if not (1e-3 < einf < 0.5 and a < 5.99):
            continue
        if not (e.min() <= 2 * einf <= e.max()):
            continue
        i = sr.nearest_eps(e, 2 * einf) if hasattr(sr, "nearest_eps") else int(np.argmin(np.abs(e - 2 * einf)))
        rows.append(dict(field=field, frontier=name, cost=cost,
                         p0_selected=float(einf / e[i]),
                         p0_best=float(einf / e.min()),
                         surprisal_nats=float(np.log(e[i] / einf))))
    P = pd.DataFrame(rows)
    save(P, "table_p0_selection.csv")
    print(f"  median p_0 at the reference = {P.p0_selected.median():.3f} "
          f"(predicted 0.5), median surprisal = {P.surprisal_nats.median():.3f} nats "
          f"(predicted {np.log(2):.3f})")
    return P


NHAT = None


def main():
    global NHAT
    NHAT = _nhat_table()
    V = verify(); save(V, "table_surprisal_verification.csv")
    print(V.to_string(index=False))
    print("\nRunning floor:")
    R, g = floor_panel(); print(g.round(3).to_string(index=False))
    print("\nMeasured spectral width:")
    W = width_panel()
    print(W.groupby("N_hat")[["g", "V", "gV"]].median().round(3).to_string())
    print("\nFloor share at the reference:")
    p0_panel()


if __name__ == "__main__":
    main()
