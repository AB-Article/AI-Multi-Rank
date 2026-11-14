#!/usr/bin/env python3
"""T11  The chord form of the path: a three-point Fisher-Rao distance.

Theory (Proposition 'chord' and Corollary 'chord certificate', main text).
For a completely monotone frontier the tilted family p_x is a curve in the
simplex and the Bhattacharyya coefficient between two of its points is

    BC(x_1,x_2) = sum_i sqrt(p_i(x_1) p_i(x_2))
                = eps((x_1+x_2)/2) / sqrt(eps(x_1) eps(x_2))
                = exp(-J),     J = S(x_m) - [S(x_1)+S(x_2)]/2,

so the Fisher-Rao (geodesic) distance between the two ends of a window is

    d(x_1,x_2) = 2 arccos exp(-J)                                     (C1)

which needs THREE error values, no derivative, no smoother, no floor and no
fit.  Then

  (i)   d <= L always (a geodesic is no longer than the path);
  (ii)  two atoms (one channel + a floor)  =>  the path is a great-circle arc
        and d = L exactly;
  (iii) d < L certifies at least three spectral modes;
  (iv)  d > L with kappa >= 0 on the window certifies that mu is NOT a
        positive measure -- a strictly stronger falsification of complete
        monotonicity than the 2x2 Hankel minor J >= 0.

Part 0 verifies (i)-(ii) on random spectra.
Part 1 computes d and L on every single-resource frontier of the corpus and
compares them (agreement, bootstrap dispersion, arc fraction).
Part 2 is the power analysis: can d/L separate two atoms from three at a
realistic point count and noise level?
Part 3 is the refinement experiment: inscribed chord sums on raw Pareto data,
showing that arc length is only defined at coarse scale.

Writes results/table_chord_verification.csv, table_chord_per_frontier.csv,
table_chord_summary.csv, table_chord_power.csv, table_chord_refine.csv.
"""
import os, sys, warnings
import numpy as np, pandas as pd, mpmath as mp
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy import stats
from _loaders import selection_rules, save

sr = selection_rules()
RNG = np.random.default_rng(20260915)
mp.mp.dps = 40
NG = 801


# ===================================================== Part 0: identities ==
def verify():
    """(i) d <= L on random spectra; (ii) d = L for exactly two atoms."""
    d_two, worst_excess, min_deficit3 = 0.0, -mp.inf, mp.inf
    for _ in range(30):
        alpha = mp.mpf(RNG.uniform(0.15, 3.0))
        einf = mp.mpf(RNG.uniform(0.01, 0.4))
        C = mp.mpf(RNG.uniform(0.1, 5.0))
        eps = lambda x: einf + C * mp.e ** (-alpha * x)
        S = lambda x: -mp.log(eps(x))
        kap = lambda x: -mp.diff(S, x, 2)
        x1 = mp.mpf(RNG.uniform(-3, 0)); x2 = x1 + mp.mpf(RNG.uniform(0.5, 5))
        L = mp.quad(lambda u: mp.sqrt(abs(kap(u))), [x1, x2])
        J = S((x1 + x2) / 2) - (S(x1) + S(x2)) / 2
        d = 2 * mp.acos(mp.e ** (-J))
        d_two = max(d_two, float(abs(d - L)))

    # (i) and (iii): three and four atoms, random
    for _ in range(60):
        K = int(RNG.integers(2, 4))
        a = np.concatenate([[0.0], np.sort(RNG.uniform(0.2, 3.0, K))])
        c = np.concatenate([[RNG.uniform(.01, .4)], RNG.uniform(.1, 3, K)])
        am = [mp.mpf(v) for v in a]; cm = [mp.mpf(v) for v in c]
        eps = lambda x: sum(cj * mp.e ** (-aj * x) for cj, aj in zip(cm, am))
        S = lambda x: -mp.log(eps(x))
        kap = lambda x: -mp.diff(S, x, 2)
        x1 = mp.mpf(RNG.uniform(-2, 0)); x2 = x1 + mp.mpf(RNG.uniform(1, 4))
        L = mp.quad(lambda u: mp.sqrt(abs(kap(u))), [x1, x2])
        J = S((x1 + x2) / 2) - (S(x1) + S(x2)) / 2
        d = 2 * mp.acos(mp.e ** (-J))
        worst_excess = max(worst_excess, d - L)
        min_deficit3 = min(min_deficit3, L - d)

    # Bhattacharyya identity itself
    d_bc = 0.0
    for _ in range(20):
        K = int(RNG.integers(2, 5))
        a = np.sort(RNG.uniform(0, 3, K)); c = RNG.uniform(.05, 2, K)
        x1, x2 = RNG.uniform(-2, 0), RNG.uniform(0.5, 3)
        e = lambda x: float((c * np.exp(-a * x)).sum())
        p1 = c * np.exp(-a * x1) / e(x1); p2 = c * np.exp(-a * x2) / e(x2)
        bc = float(np.sqrt(p1 * p2).sum())
        d_bc = max(d_bc, abs(bc - e((x1 + x2) / 2) / np.sqrt(e(x1) * e(x2))))

    return pd.DataFrame([
        dict(result="BC = eps(midpoint)/sqrt(eps_1 eps_2)", max_deviation=d_bc),
        dict(result="d = L for exactly two atoms", max_deviation=d_two),
        dict(result="d - L for >=3 atoms (must be <= 0; max value)",
             max_deviation=float(worst_excess)),
        dict(result="L - d for >=3 atoms (strict deficit; min value)",
             max_deviation=float(min_deficit3))])


# ============================================== shared smoothed-curve engine
def _smooth(x, S, u, h):
    out = np.empty(len(u))
    for i, xc in enumerate(u):
        w = np.exp(-0.5 * ((x - xc) / h) ** 2)
        M = np.column_stack([np.ones_like(x), x - xc, 0.5 * (x - xc) ** 2])
        W = np.sqrt(w)[:, None]
        b, *_ = np.linalg.lstsq(M * W, S * W[:, 0], rcond=None)
        out[i] = b[0]
    return out


def chord_and_length(x, e, trim=0.10):
    """Both quantities from ONE smoothed curve, so D_k -> L is a numerics
    check and d/L is the physics."""
    o = np.argsort(x); x, e = np.asarray(x, float)[o], np.asarray(e, float)[o]
    S = -np.log(e); span = np.ptp(x)
    if span <= 0:
        return None
    u = np.linspace(x.min() + trim * span, x.max() - trim * span, NG)
    sp = CubicSpline(u, _smooth(x, S, u, span / 4))
    v = np.linspace(u[0], u[-1], 2001)
    s, kap = sp(v, 1), -sp(v, 2)
    if not np.all(s > 0):
        return None
    L = np.trapezoid(np.sqrt(np.clip(kap, 0, None)), v)
    x1, x2 = u[0], u[-1]

    def dpair(a, b):
        J = float(sp((a + b) / 2) - 0.5 * (sp(a) + sp(b)))
        return 2 * np.arccos(np.clip(np.exp(-max(J, 0.0)), -1, 1)), J

    d1, J1 = dpair(x1, x2)
    ed = np.linspace(x1, x2, 33)
    d32 = sum(dpair(ed[i], ed[i + 1])[0] for i in range(32))
    g2 = kap[-1] / s[-1] ** 2
    eps2 = float(np.exp(-sp(x2)))
    return dict(L=L, d=d1, J=J1, d32=d32, span=span, eps1=float(np.exp(-sp(x1))),
                eps2=eps2, einf_loc=eps2 * g2 / (1 + g2),
                frac_neg_kappa=float(np.mean(kap < 0)))


# =========================================== Part 1: chord vs length on data
def corpus(nboot=100):
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        x = np.asarray(x, float); e = np.asarray(e, float)
        if len(e) < 15 or np.ptp(x) < 1.0:
            continue
        r = chord_and_length(x, e)
        if r is None or r["L"] < 1e-6:
            continue
        bL, bd = [], []
        for _ in range(nboot):
            i = np.sort(RNG.choice(len(e), len(e), replace=True))
            if len(np.unique(x[i])) < 10:
                continue
            rr = chord_and_length(x[i], e[i])
            if rr and rr["L"] > 1e-6:
                bL.append(rr["L"]); bd.append(rr["d"])
        r.update(field=field, frontier=name, cost=cost, n=len(e),
                 ratio=r["d"] / r["L"], conv=r["d32"] / r["L"],
                 L_se=float(np.std(bL, ddof=1)) if len(bL) > 20 else np.nan,
                 d_se=float(np.std(bd, ddof=1)) if len(bd) > 20 else np.nan,
                 ratio_lo=float(np.quantile(np.array(bd) / np.array(bL), .05))
                 if len(bL) > 20 else np.nan,
                 ratio_hi=float(np.quantile(np.array(bd) / np.array(bL), .95))
                 if len(bL) > 20 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


# ================================================== Part 2: power analysis =
def power(sigmas=(0.0, 0.003, 0.01, 0.03), npts=30, span=2.5, ntrial=400):
    out = []
    for sig in sigmas:
        for K in (1, 2):
            vals = []
            for _ in range(ntrial):
                a = np.concatenate([[0.0], np.sort(RNG.uniform(0.3, 2.0, K))])
                c = np.concatenate([[RNG.uniform(.03, .3)], RNG.uniform(.2, 2, K)])
                xx = np.sort(RNG.uniform(0, span, npts))
                ee = (c[None] * np.exp(-np.outer(xx, a))).sum(1)
                ee = ee * np.exp(sig * RNG.standard_normal(npts))
                r = chord_and_length(xx, ee)
                if r and r["L"] > 0.05:
                    vals.append(r["d"] / r["L"])
            v = np.array(vals)
            out.append(dict(sigma=sig, atoms=K + 1, n=len(v),
                            med=float(np.median(v)),
                            q05=float(np.quantile(v, .05)),
                            q95=float(np.quantile(v, .95))))
    P = pd.DataFrame(out)
    # separation: |median gap| against the pooled 5-95 spread
    piv = P.pivot(index="sigma", columns="atoms", values="med")
    sprd = P.groupby("sigma").apply(lambda t: (t.q95 - t.q05).mean())
    P = P.merge(pd.DataFrame(dict(sigma=piv.index,
                                  med_gap=(piv[2] - piv[3]).values,
                                  spread=sprd.values)), on="sigma")
    P["separation"] = P.med_gap / P.spread
    return P


# ============================================ Part 3: refinement on raw data
def refine():
    """Inscribed chord sums D_k from raw Pareto points (monotone interpolation
    of S, no smoothing).  For a rectifiable curve D_1 <= D_2 <= ... <= L; on
    real data the fine sums diverge, so arc length exists only at coarse
    scale."""
    KS = (1, 2, 4, 8, 16)
    rows = []
    for field, name, cost, x, e in sr.single_resource_frontiers():
        x = np.asarray(x, float); e = np.asarray(e, float)
        if len(e) < 15 or np.ptp(x) < 1.0:
            continue
        o = np.argsort(x); x, e = x[o], e[o]
        xu, idx = np.unique(x, return_index=True)
        if len(xu) < 10:
            continue
        f = PchipInterpolator(xu, -np.log(e[idx]))
        span = xu[-1] - xu[0]
        x1, x2 = xu[0] + .05 * span, xu[-1] - .05 * span
        r = dict(field=field, frontier=name, cost=cost, n=len(e))
        for k in KS:
            ed = np.linspace(x1, x2, k + 1)
            tot = 0.0
            for i in range(k):
                J = float(f((ed[i] + ed[i + 1]) / 2)
                          - 0.5 * (f(ed[i]) + f(ed[i + 1])))
                tot += 2 * np.arccos(np.clip(np.exp(-max(J, 0.0)), -1, 1))
            r[f"D{k}"] = tot
        seq = [r[f"D{k}"] for k in KS]
        r["monotone"] = all(seq[i] <= seq[i + 1] + 1e-12 for i in range(len(seq) - 1))
        r["D16_over_pi"] = seq[-1] / np.pi
        rows.append(r)
    return pd.DataFrame(rows)


# =================================================================== run ==
if __name__ == "__main__":
    print("T11  chord form of the path")
    V = verify(); save(V, "table_chord_verification.csv")
    print(V.to_string(index=False))

    C = corpus(); save(C, "table_chord_per_frontier.csv")
    r_p = stats.pearsonr(C.d, C.L); r_s = stats.spearmanr(C.d, C.L)
    S = pd.DataFrame([
        dict(statistic="frontiers analysed", value=float(len(C))),
        dict(statistic="numerics check: median D32/L", value=float(C.conv.median())),
        dict(statistic="numerics check: max |D32/L - 1|",
             value=float((C.conv - 1).abs().max())),
        dict(statistic="Pearson r(d, L)", value=float(r_p[0])),
        dict(statistic="Pearson p", value=float(r_p[1])),
        dict(statistic="Spearman rho(d, L)", value=float(r_s[0])),
        dict(statistic="median d/L", value=float(C.ratio.median())),
        dict(statistic="q25 d/L", value=float(C.ratio.quantile(.25))),
        dict(statistic="q75 d/L", value=float(C.ratio.quantile(.75))),
        dict(statistic="median bootstrap SE of d", value=float(C.d_se.median())),
        dict(statistic="median bootstrap SE of L", value=float(C.L_se.median())),
        dict(statistic="frontiers with d >= pi", value=float((C.d >= np.pi).sum())),
        dict(statistic="median d/pi", value=float((C.d / np.pi).median())),
        dict(statistic="max d/pi", value=float((C.d / np.pi).max())),
        dict(statistic="frontiers with d/L significantly < 0.97",
             value=float((C.ratio_hi < .97).sum())),
        dict(statistic="frontiers with d/L significantly > 1",
             value=float((C.ratio_lo > 1.0).sum()))])
    save(S, "table_chord_summary.csv"); print(S.to_string(index=False))

    P = power(); save(P, "table_chord_power.csv"); print(P.to_string(index=False))
    R = refine(); save(R, "table_chord_refine.csv")
    print("refinement monotone on raw data: %d / %d;  max D16/pi = %.2f"
          % (int(R.monotone.sum()), len(R), R.D16_over_pi.max()))
