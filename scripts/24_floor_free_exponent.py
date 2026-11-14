#!/usr/bin/env python3
"""T3 + T4  The floor-free exponent and its drift.

Theory (Proposition 'floor-free exponent', main text).  Along any resource ray
of a frontier in the completely monotone class, with slope s = S' and
curvature k = -S'' of S = -ln eps in x = ln t,
      alpha_hat(x) := s + k/s = E_p[a^2] / E_p[a]
is (i) independent of the floor, (ii) identically alpha for one reducible
channel, (iii) non-increasing in x (Cauchy-Schwarz), and (iv) satisfies the
Bhatia-Davis bound k <= s (a_max - s), with equality iff one channel.
For one channel, the reference zeta = 0 is the point where s^2 = k, and the
local floor is eps * k / (s^2 + k).

Estimation: kernel-weighted local quadratic regression of S on x
(bandwidth h = span/4).  Tests
  T3  drift  D = alpha_hat(upper third) - alpha_hat(lower third) must be <= 0;
      counted as a violation when the 90% bootstrap interval lies above 0;
      calibrated against null A (fitted single channel + fitted noise).
  T4  alpha_hat(centre) and the three-point alpha versus the full-fit alpha;
      regret of the local rule 's^2 = k' against the full-fit reference.
Part 0 verifies (i)-(iv) at 50-digit precision on random spectra.

Writes results/table_floor_free_verification.csv,
table_floor_free_per_frontier.csv, table_floor_free_summary.csv.
"""
import numpy as np, pandas as pd, mpmath as mp
from _loaders import selection_rules, all_fields, save

sr = selection_rules()
fit_channels = all_fields()["fit_channels"]
RNG = np.random.default_rng(20260913)
mp.mp.dps = 50


# ------------------------------------------------------ Part 0: identities --
def verify():
    d_id = d_one = 0; worst_drift = -np.inf; worst_bd = -np.inf; d_ref = d_floor = 0
    for _ in range(40):
        K = int(RNG.integers(1, 5))
        a = [mp.mpf(0)] + [mp.mpf(RNG.uniform(0.1, 3.0)) for _ in range(K)]
        c = [mp.mpf(RNG.uniform(0.02, 0.4))] + [mp.mpf(RNG.uniform(0.1, 3)) for _ in range(K)]
        S = lambda x: -mp.log(sum(cj * mp.e ** (-aj * x) for cj, aj in zip(c, a)))
        for x in mp.linspace(-3, 5, 9):
            s, k = mp.diff(S, x), -mp.diff(S, x, 2)
            w = [cj * mp.e ** (-aj * x) for cj, aj in zip(c, a)]
            Z = sum(w); p = [wj / Z for wj in w]
            m1 = sum(pj * aj for pj, aj in zip(p, a)); m2 = sum(pj * aj ** 2 for pj, aj in zip(p, a))
            ah = s + k / s
            d_id = max(d_id, abs(ah - m2 / m1))                           # (i)
            worst_bd = max(worst_bd, float(k - s * (max(a) - s)))         # (iv) <= 0
            AH = lambda y: mp.diff(S, y) - mp.diff(S, y, 2) / mp.diff(S, y)
            dah = (AH(x + mp.mpf('1e-12')) - AH(x)) / mp.mpf('1e-12')
            worst_drift = max(worst_drift, float(dah))                    # (iii) <= 0
            if K == 1:
                d_one = max(d_one, abs(ah - a[1]))                        # (ii)
                eps = mp.e ** (-S(x))
                d_floor = max(d_floor, abs(eps * k / (s ** 2 + k) - c[0]))
        if K == 1:
            x0 = mp.log(c[1] / c[0]) / a[1]
            d_ref = max(d_ref, abs(mp.diff(S, x0) ** 2 + mp.diff(S, x0, 2)))
    return pd.DataFrame([
        dict(result="alpha_hat = E[a^2]/E[a] (floor-free)", max_deviation=float(d_id)),
        dict(result="alpha_hat = alpha for one channel", max_deviation=float(d_one)),
        dict(result="d alpha_hat / dx <= 0 (max value)", max_deviation=worst_drift),
        dict(result="Bhatia-Davis k - s(a_max - s) <= 0 (max value)", max_deviation=worst_bd),
        dict(result="one channel: s^2 = k at zeta = 0", max_deviation=float(d_ref)),
        dict(result="one channel: eps k/(s^2+k) = einf", max_deviation=float(d_floor))])


# ------------------------------------------------------ local estimator ----
def local(x, S, xc, h):
    w = np.exp(-0.5 * ((x - xc) / h) ** 2)
    M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
    return b[1], -b[2]                                                    # s, k


def alpha_hat(x, S, xc, h):
    s, k = local(x, S, xc, h)
    return s + max(k, 0.0) / s if s > 1e-6 else np.nan


def drift(x, S):
    h = np.ptp(x) / 4
    lo, hi = x.min() + np.ptp(x) / 3, x.min() + 2 * np.ptp(x) / 3
    return alpha_hat(x, S, hi, h) - alpha_hat(x, S, lo, h)


def s2k_pick(x, S):
    """First model after which s^2 - k turns negative (local reference)."""
    h = np.ptp(x) / 4
    grid = np.linspace(x.min() + 0.1 * np.ptp(x), x.max() - 0.1 * np.ptp(x), 60)
    g = []
    for xc in grid:
        s, k = local(x, S, xc, h); g.append(s * s - k)
    g = np.array(g)
    cross = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    if not len(cross):
        return None
    xs = grid[cross[0]] - g[cross[0]] * (grid[cross[0] + 1] - grid[cross[0]]) / (g[cross[0] + 1] - g[cross[0]])
    return sr.nearest(x, xs)


def nhat1(x, e):
    b = {}
    for K in (1, 2, 3):
        try:
            b[K] = fit_channels(x[:, None], e, K, shared=False)["bic"]
        except Exception:
            pass
    return min(b, key=b.get)


def main():
    V = verify(); save(V, "table_floor_free_verification.csv"); print(V.to_string(index=False))
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        if len(e) < 10:
            continue
        S = -np.log(e)
        einf, c, a, rms = sr.fit_law(x, e)
        D = drift(x, S)
        boot = []
        for _ in range(200):
            i = np.sort(RNG.choice(len(e), len(e), replace=True))
            if len(np.unique(x[i])) < 6:
                continue
            boot.append(drift(x[i], S[i]))
        boot = np.array([b for b in boot if np.isfinite(b)])
        # null A drift: the fitted single channel with the fitted noise
        u = x - x.min(); nullD = []
        for _ in range(100):
            es = (einf + c * np.exp(-a * u)) * np.exp(rms * RNG.standard_normal(len(u)))
            nullD.append(drift(x, -np.log(es)))
        nullD = np.array([v for v in nullD if np.isfinite(v)])
        floor = (1e-3 < einf < 0.5) and (a < 5.99)
        ac = alpha_hat(x, S, x.min() + np.ptp(x) / 2, np.ptp(x) / 4)
        full3 = np.array([0, sr.nearest(x, x.min() + np.ptp(x) / 2), len(x) - 1])
        e3, c3, a3 = sr.three_point(x[full3], e[full3])
        rec = dict(field=field, frontier=name, cost=cost, n=len(e), span=np.ptp(x),
                   N_hat=nhat1(x, e), floor=floor, alpha_fit=a, einf_fit=einf,
                   alpha_hat_centre=ac, alpha_3pt=a3, drift=D,
                   drift_lo=np.quantile(boot, .05) if len(boot) else np.nan,
                   drift_hi=np.quantile(boot, .95) if len(boot) else np.nan,
                   null_drift_median=np.median(nullD) if len(nullD) else np.nan,
                   null_drift_q95=np.quantile(nullD, .95) if len(nullD) else np.nan)
        if floor and e.min() <= 2 * einf <= e.max():
            i = s2k_pick(x, S)
            rec["s2k_regret"] = float(sr.regret_eps(e[i], einf)) if i is not None else np.nan
            rec["attainable"] = True
        else:
            rec["attainable"] = False
        rows.append(rec)
    P = pd.DataFrame(rows); save(P, "table_floor_free_per_frontier.csv")

    P["violation"] = P.drift_lo > 0
    P["decline"] = P.drift_hi < 0
    summ = []
    for lab, g in [("all", P), ("N_hat=1", P[P.N_hat == 1]), ("N_hat>=2", P[P.N_hat >= 2]),
                   ("floor", P[P.floor]), ("attainable", P[P.attainable])]:
        ok = g.alpha_hat_centre.notna() & g.floor
        summ.append(dict(subset=lab, frontiers=len(g),
                         median_drift=g.drift.median(), median_null_drift=g.null_drift_median.median(),
                         frac_drift_negative=float((g.drift < 0).mean()),
                         frac_significant_decline=float(g.decline.mean()),
                         frac_significant_increase_violation=float(g.violation.mean()),
                         frac_drift_below_null_q5=float((g.drift < g.null_drift_median - (g.null_drift_q95 - g.null_drift_median)).mean()),
                         median_rel_err_alpha_hat=float(np.median(np.abs(g.alpha_hat_centre[ok] / g.alpha_fit[ok] - 1))) if ok.any() else np.nan,
                         median_rel_err_alpha_3pt=float(np.nanmedian(np.abs(g.alpha_3pt[ok] / g.alpha_fit[ok] - 1))) if ok.any() else np.nan,
                         s2k_inband=float((g.s2k_regret <= sr.R_BAND).mean()) if "s2k_regret" in g and g.s2k_regret.notna().any() else np.nan,
                         s2k_median_regret=float(g.s2k_regret.median()) if "s2k_regret" in g else np.nan,
                         s2k_applicable=int(g.s2k_regret.notna().sum()) if "s2k_regret" in g else 0))
    T = pd.DataFrame(summ); save(T, "table_floor_free_summary.csv")
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
