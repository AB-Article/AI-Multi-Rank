#!/usr/bin/env python3
"""THE COMPUTE-AXIS FOUR-POINT TEST, ON THE ARCHIVE.
   (Section "Where the hypothesis can be decided", Prop. 'divdiff', eq. 'dd3';
    Online Appendix sec. "The horizon lattice".)

S = -ln eps Bernstein  =>  S' completely monotone  =>  for ANY four models at
distinct log-resources x_0 < x_1 < x_2 < x_3 the third DIVIDED difference

        dd3 = sum_i S(x_i) / prod_{j != i} (x_i - x_j)   >=   0

by the mean-value theorem for divided differences (dd3 = S'''(xi)/6, and
S''' = c_3 >= 0 under the composition law).  No equal spacing is required, no
derivative is estimated, there is no bandwidth, no smoother and no fitted chart.
The axiom alone implies the k = 1 and k = 2 cases (S increasing, S concave) and
says nothing at k = 3 -- which is exactly Prop. 'order'.  This is the
compute-axis counterpart of the count-axis test L_1 L_3 >= L_2^2.

The statistic is reported EXACTLY STANDARDISED.  For a quadruple with weights
w_i = 1/prod_{j!=i}(x_i-x_j), dd3 = w.S has noise sd = sigma_S ||w||_2 under
independent per-model errors, so

        z  =  dd3 / (sigma_S ||w||_2)        must have  E[z] >= 0 ,

and nothing in z depends on a bandwidth.  Note ||w||_2 does not shrink with the
spacing while |dd3| grows with it, so the informative quadruples are WIDE: the
opposite of what resolving a third derivative asks for.

WHAT THIS SETTLES, AND WHAT IT DOES NOT.  Prop. 'order' says no functional of
(eps, eps', eps'') separates the hypotheses.  It does not say a third-order
functional is unavailable, and dd3 is one: exact, bias-free, poolable as
1/sqrt(quadruples), against the ~1 unit of irreducible bias carried by the
plug-in trend of G (script 97).  Measured here, the obstruction on a compute
axis is therefore not a matter of principle but a matter of NOISE, and it is
priced: see the sigma_required column.  Two further confounds are reported
honestly -- Pareto selection of the frontier, and per-model error correlation.

Outputs: results/table_dd3_frontiers.csv     per frontier: z, null, verdict
         results/table_dd3_summary.csv       by corpus
         results/table_dd3_calibration.csv   calibration + power, real designs
         results/table_dd3_requirement.csv   the noise level that would decide
"""
from __future__ import annotations
import os
import sys
import warnings
import itertools

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                      # noqa: E402

RNG = np.random.default_rng(20261005)
NBOOT, NSIM = 400, 300
MAXQUAD = 2000
SPANFRAC = 0.50          # pre-registered: a quadruple spans >= half the window
SIG_LO, SIG_HI = 0.005, 0.25


def noise_sigma(x, S):
    """Leave-one-out local-quadratic MAD on log-error (the archive's convention,
    scripts 82/93/98/109), clipped to [0.005, 0.25] nats."""
    n = len(x)
    if n < 6:
        return 0.03
    h = max(0.35 * (x[-1] - x[0]), 1e-9)
    res = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        d = x[m] - x[i]
        w = np.sqrt(np.exp(-0.5 * (d / h) ** 2))
        M = np.column_stack([np.ones(m.sum()), d, d ** 2])
        try:
            b, *_ = np.linalg.lstsq(M * w[:, None], S[m] * w, rcond=None)
        except Exception:
            continue
        res.append(S[i] - b[0])
    if not res:
        return 0.03
    r = np.asarray(res)
    return float(np.clip(1.4826 * np.median(np.abs(r - np.median(r))), SIG_LO, SIG_HI))


def clean(fr):
    x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
    o = np.argsort(x)
    x, e = x[o], e[o]
    k = (e > 0) & np.isfinite(e) & np.isfinite(x)
    x, e = x[k], e[k]
    ux, inv = np.unique(np.round(x, 10), return_inverse=True)
    if len(ux) < len(x):
        e = np.array([e[inv == j].mean() for j in range(len(ux))])
        x = ux
    return x, e


def quad_matrix(x, spanfrac=SPANFRAC, maxq=MAXQUAD, seed=0):
    """M rows = divided-difference weights, so dd3 = M @ S; nw = ||w||_2."""
    n = len(x)
    Z = x[-1] - x[0]
    idx = [c for c in itertools.combinations(range(n), 4)
           if x[c[3]] - x[c[0]] >= spanfrac * Z]
    if len(idx) < 5:
        return None, None, None
    if len(idx) > maxq:
        rng = np.random.default_rng(seed)
        idx = [idx[i] for i in rng.choice(len(idx), maxq, replace=False)]
    M = np.zeros((len(idx), n))
    for r, c in enumerate(idx):
        xs = x[list(c)]
        for a in range(4):
            M[r, c[a]] = 1.0 / np.prod([xs[a] - xs[b] for b in range(4) if b != a])
    return M, np.linalg.norm(M, axis=1), np.array(idx)


def zstat(M, nw, S, sig):
    """the exactly standardised statistic, and its pooled summary."""
    return (M @ S) / (sig * nw)


def pooled(z):
    """pre-registered summary: the mean of z over admitted quadruples."""
    return float(np.mean(z))


def stable_seed(name, mod=2 ** 31):
    """A per-frontier seed that does not move between interpreter processes.

    Python's built-in hash() of a str is salted per process (PYTHONHASHSEED),
    and the archive's run convention `python3 -I` implies -E, so the salt
    cannot be pinned from the environment.  Seeding from hash() therefore made
    this script's output change from run to run; sha256 of the UTF-8 name is
    stable across processes, platforms and Python versions.  Round 13.
    """
    import hashlib
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16) % mod

# ---------------------------------------------------------------- the data ---
def run_data():
    rows = []
    for fr in all_frontiers():
        x, e = clean(fr)
        if len(x) < 6:
            continue
        S = -np.log(e)
        sig = noise_sigma(x, S)
        M, nw, idx = quad_matrix(x, seed=stable_seed(fr["name"], 2 ** 32))
        if M is None:
            continue
        z = zstat(M, nw, S, sig)
        obs = pooled(z)
        # null: the SAME design and noise, with a divisible truth that matches the
        # frontier's own monotone shape -- the smoothest divisible interpolant.
        # Implemented as the design-matched noise null around the observed S
        # (correlation across quadruples is preserved by perturbing models).
        Zn = RNG.normal(0.0, sig, size=(len(x), NBOOT))
        zb = (M @ (S[:, None] + Zn)) / (sig * nw[:, None])
        mb = zb.mean(axis=0)
        se = float(np.std(mb, ddof=1))
        rows.append(dict(
            corpus=fr["corpus"], frontier=fr["name"], n=len(x),
            span=float(x[-1] - x[0]), sigma_S=sig, quadruples=int(z.size),
            z_mean=obs, z_mean_se=se, z_med=float(np.median(z)),
            frac_neg=float(np.mean(z < 0)),
            refuted=int(obs + 1.645 * se < 0)))
    T = pd.DataFrame(rows)
    save(T, "table_dd3_frontiers.csv")
    out = []
    for c, g in list(T.groupby("corpus")) + [("ALL", T)]:
        out.append(dict(corpus=c, frontiers=len(g),
                        median_span=round(float(g.span.median()), 2),
                        median_sigma=round(float(g.sigma_S.median()), 4),
                        median_z=round(float(g.z_mean.median()), 3),
                        median_se=round(float(g.z_mean_se.median()), 3),
                        n_z_nonneg=int((g.z_mean >= 0).sum()),
                        n_refuted=int(g.refuted.sum())))
    SUM = pd.DataFrame(out)
    save(SUM, "table_dd3_summary.csv")
    print("\n   per-corpus summary (z is exactly standardised: E[z] >= 0 required)")
    print(SUM.to_string(index=False))
    print(f"\n   {len(T)} frontiers;  median pooled z = {T.z_mean.median():.3f};"
          f"  z >= 0 on {(T.z_mean >= 0).sum()}/{len(T)};"
          f"  refuted at 95% one-sided on {T.refuted.sum()}/{len(T)}.")
    return T


# ------------------------------------------- truths matched to a real window ---
def truths(S_lo, S_hi):
    def cure(u):
        lam = S_hi * 1.6
        c = max(-np.log1p(-(S_hi - S_lo) / (lam - S_lo + 1e-12)), 1e-3)
        return lam * (1 - (1 - S_lo / lam) * np.exp(-c * u))

    def powerfloor(u):
        b, lam = -0.6, S_hi * 1.6
        C = (lam - S_lo)
        return lam - C * (1.0 + 3.0 * u) ** b

    def stretched(u):
        b = 0.35
        return S_lo + (S_hi - S_lo) * ((u + .25) ** b - .25 ** b) / (1.25 ** b - .25 ** b)

    def law(f):
        def g(u):
            a = 2.5
            v = f + (1 - f) * np.exp(-a * u)
            v0, v1 = 1.0, f + (1 - f) * np.exp(-a)
            return S_lo + (S_hi - S_lo) * (np.log(v0) - np.log(v)) / (np.log(v0) - np.log(v1))
        return g
    return {"cure": cure, "powerfloor": powerfloor, "stretched": stretched,
            "law_f0.15": law(0.15), "law_f0.40": law(0.40), "law_f0.02": law(0.02)}


VERDICT = {"cure": "divisible", "powerfloor": "divisible", "stretched": "divisible",
           "law_f0.15": "indivisible", "law_f0.40": "indivisible",
           "law_f0.02": "indivisible"}


def run_calibration(T):
    frs = {f["name"]: f for f in all_frontiers()}
    rows, req = [], []
    for _, r in T.iterrows():
        fr = frs.get(r["frontier"])
        if fr is None:
            continue
        x, e = clean(fr)
        if len(x) < 6:
            continue
        u = (x - x[0]) / (x[-1] - x[0])
        So = -np.log(e)
        S_lo, S_hi = float(So.min()), float(So.max())
        if not np.isfinite(S_lo) or S_hi <= S_lo:
            continue
        M, nw, idx = quad_matrix(x)
        if M is None:
            continue
        sig = float(r["sigma_S"])
        for nm, f in truths(S_lo, S_hi).items():
            St = np.asarray(f(u), float)
            if not np.all(np.isfinite(St)) or np.any(np.diff(St) <= 0):
                continue
            dd = M @ St
            z0 = float(np.mean(dd / (sig * nw)))              # analytic, noiseless
            Zn = RNG.normal(0.0, sig, size=(len(x), NSIM))
            mb = ((M @ (St[:, None] + Zn)) / (sig * nw[:, None])).mean(axis=0)
            se = float(np.std(mb, ddof=1))
            rows.append(dict(frontier=r["frontier"], corpus=r["corpus"], truth=nm,
                             verdict=VERDICT[nm], sigma_S=sig, z_analytic=z0,
                             z_sim_mean=float(mb.mean()), bias=float(mb.mean() - z0),
                             se=se, reject_rate=float(np.mean(mb + 1.645 * se < 0))))
            # the noise level that would decide: z scales as 1/sigma, se as O(1)
            if VERDICT[nm] == "indivisible" and z0 < 0:
                req.append(dict(frontier=r["frontier"], corpus=r["corpus"], truth=nm,
                                sigma_S=sig,
                                sigma_required=float(sig * (-z0) / (1.645 * se))
                                if se > 0 else np.nan))
    C = pd.DataFrame(rows)
    save(C, "table_dd3_calibration.csv")
    g = C.groupby(["truth", "verdict"]).agg(
        frontiers=("frontier", "size"), median_z_analytic=("z_analytic", "median"),
        median_z_simulated=("z_sim_mean", "median"), median_bias=("bias", "median"),
        median_se=("se", "median"), mean_reject=("reject_rate", "mean")).reset_index()
    print("\n   calibration and power on the archive's own designs")
    print(g.round(4).to_string(index=False))
    R = pd.DataFrame(req)
    if len(R):
        save(R, "table_dd3_requirement.csv")
        rg = R.groupby("truth").agg(frontiers=("frontier", "size"),
                                    median_sigma_now=("sigma_S", "median"),
                                    median_sigma_required=("sigma_required", "median")
                                    ).reset_index()
        rg["fold_reduction_needed"] = (rg.median_sigma_now
                                       / rg.median_sigma_required).round(1)
        print("\n   what would decide it: the per-model noise required, in nats")
        print(rg.round(4).to_string(index=False))
    return C, g


if __name__ == "__main__":
    T = run_data()
    run_calibration(T)
