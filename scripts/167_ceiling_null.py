#!/usr/bin/env python3
"""167_ceiling_null.py -- the admissible-ceiling test, read against a
design-matched null.

WHY THIS SCRIPT EXISTS
----------------------
Script 129(C) reports that the archive's fitted three-parameter floors exceed
the axiom's ceiling eps_inf <= eps/(1+N_eff) on 89 of 117 frontiers.  That
count was read against no null, and the paper's own method law forbids it: the
ceiling is evaluated through a plug-in,

    N_eff = (l[+] - l[-])^2 / (4 (l[-] - 2 l[0] + l[+])),      l = ln eps,

whose denominator is a second difference, so N_eff -> +infinity as the measured
curvature kappa -> 0+.  Script 129 then takes the MINIMUM of eps/(1+N_eff) over
every admissible triple of the frontier -- a median of 55 and up to 16 719 of
them -- so the reported ceiling is the single most favourable of thousands of
heavy-tailed draws.  On 16 of 117 frontiers the companion wall returns exactly
0, which no frontier with a floor can have: the estimator is noise-dominated
and the minimum is a selection, not a bound.

WHAT IS COMPUTED HERE
---------------------
(1) THE NULL IS THE INCUMBENT FORM ITSELF.  Form (4), eps = eps_inf + c e^{-a x},
    is the two-atom chart, which SATURATES the axiom wall: N_eff = rho - 1 and
    eps_inf = eps/(1+N_eff) identically, at every point and for every parameter.
    A frontier of form (4) can therefore never violate its own ceiling, and the
    ceiling test is exactly a goodness-of-fit test of (4) directed at its floor.
    Its null is (4): per frontier we fit (4) in log space, take its residual
    scale sigma-hat, and draw replicates on the frontier's OWN nodes.

(2) THE INCUMBENT STATISTIC'S LEVEL.  On each replicate we re-run script 129's
    pipeline verbatim -- refit (4), enumerate the same triples, take the same
    minimum -- and record whether it declares a violation.  That rate is the
    false-violation rate of the published count.

(3) A PRE-REGISTERED CALIBRATED STATISTIC.  Taking a minimum is what fails, so
    the location is fixed in advance instead of searched:

      g_med  = median over the frontier's CONSECUTIVE triples of
               ln[ eps_inf-hat (1 + N_eff-hat) / eps ]                (> 0 = violation)
      g_wide = the same quantity at the single widest triple
               (first, middle, last node), chosen before seeing the data.

    Both are read against the per-frontier null of (1) as a Monte-Carlo
    p-value, so the level is exact by construction and no analytic null is used.

(4) POWER, AGAINST A TRUTH THAT GENUINELY VIOLATES.  Because (4) saturates the
    wall, an alternative must leave the two-atom shape.  We use the
    one-parameter deformation of the wall itself,

        N_eff(dS) = theta (e^{dS} - 1),        theta >= 1,

    which Proposition 'plane'(ii) integrates in closed form to

        s(dS) = alpha (1 - e^{-dS})^{1/theta},   x = x0 - int dS / s,

    giving theta = 1 exactly form (4) and theta > 1 a frontier whose floor
    exceeds its ceiling by ln[(1 + theta(rho-1))/rho] -> ln theta at depth.
    The two free constants (alpha, x0) are used to put the truth on the
    frontier's own node range and error range, as that proposition says they
    must be.  theta > 1 is a spectrum broader than two atoms, which is what
    Section 10's forecasting residual independently reports.

Outputs: results/table_ceiling_null_per_frontier.csv
         results/table_ceiling_null_summary.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save, all_frontiers                        # noqa: E402

WIDE = 0.667          # script 129's definition of a wide triple
R_NULL = 300          # replicates per frontier under the null
R_POWER = 200         # replicates per frontier per alternative
THETAS = (1.25, 1.5, 2.0)
ALPHA_LEVEL = 0.05
MAX_TRIPLES = 20000
SEED = 20261114


# --------------------------------------------------------------- the chart (4)
def mdl(th, xx):
    return np.exp(th[0]) + np.exp(th[1]) * np.exp(-np.exp(th[2]) * np.atleast_1d(xx))


def fit4(x, e):
    """Form (4) in log space, exactly as scripts 109/122/125/129 fit it.
    Returns (eps_inf, c, alpha, sigma_resid) or None."""
    th0 = [np.log(max(e.min() * .5, 1e-9)), np.log(e.max()), np.log(.4)]
    try:
        r = least_squares(lambda th: np.log(mdl(th, x)) - np.log(e), th0, max_nfev=40000)
        dof = max(len(x) - 3, 1)
        return (float(np.exp(r.x[0])), float(np.exp(r.x[1])), float(np.exp(r.x[2])),
                float(np.sqrt(np.sum(r.fun ** 2) / dof)))
    except Exception:
        return None


# ------------------------------------------------- the plug-in, vectorised
def triple_index(x, cap=MAX_TRIPLES, rng=None):
    """Script 129's triple set: every consecutive triple, plus every triple
    spanning at least WIDE of the window.  Returned as index arrays so the
    plug-in can be evaluated for many replicates at once."""
    n = len(x)
    Z = x[-1] - x[0]
    cons = [(j, j + 1, j + 2) for j in range(n - 2)]
    wide = [(i, j, m)
            for i in range(n) for j in range(i + 1, n - 1) for m in range(j + 1, n)
            if x[m] - x[i] >= WIDE * Z]
    if len(wide) > cap:
        rng = rng or np.random.default_rng(0)
        wide = [wide[k] for k in rng.choice(len(wide), cap, replace=False)]
    return np.array(cons, dtype=int), np.array(cons + wide, dtype=int)


def neff(L, idx, x):
    """N_eff on each triple of idx, for L of shape (R, n) or (n,).
    s is the central first difference, kappa twice the second divided
    difference; the spacing cancels in neither separately but the ratio is
    exactly script 129's estimator."""
    L = np.atleast_2d(L)
    i0, i1, i2 = idx[:, 0], idx[:, 1], idx[:, 2]
    x0, x1, x2 = x[i0], x[i1], x[i2]
    l0, l1, l2 = L[:, i0], L[:, i1], L[:, i2]
    s = -(l2 - l0) / (x2 - x0)
    kap = 2.0 * (l0 / ((x0 - x1) * (x0 - x2))
                 + l1 / ((x1 - x0) * (x1 - x2))
                 + l2 / ((x2 - x0) * (x2 - x1)))
    good = np.isfinite(s) & np.isfinite(kap) & (kap > 0) & (s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        N = np.where(good, s * s / kap, np.nan)
    return N, good


def gaps(L, idx, x, floor):
    """g = ln[ floor (1 + N_eff) / eps ] at every triple: > 0 is a violation of
    the ceiling eps_inf <= eps/(1+N_eff).  eps is read at the middle node, as
    the plug-in's own evaluation point."""
    N, _ = neff(L, idx, x)
    L = np.atleast_2d(L)
    lmid = L[:, idx[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(floor).reshape(-1, 1)) + np.log1p(N) - lmid


def statistics(L, cons_idx, all_idx, x, floor):
    """The three readings, for each row of L.  floor is per row."""
    g_all = gaps(L, all_idx, x, floor)
    g_con = gaps(L, cons_idx, x, floor) if len(cons_idx) else np.full((len(np.atleast_2d(L)), 1), np.nan)
    # the incumbent: min over every triple of eps/(1+N) == max over triples of g
    g_min = np.nanmax(g_all, axis=1)
    g_med = np.nanmedian(g_con, axis=1)
    # the widest triple, fixed in advance: first, middle, last node
    n = len(x)
    w = np.array([[0, n // 2, n - 1]], dtype=int)
    g_wide = gaps(L, w, x, floor)[:, 0]
    return g_min, g_med, g_wide


# ------------------------------------------------- the violating truth, theta>1
def theta_curve(theta, dS_lo, dS_hi, x_lo, x_hi, ngrid=4000):
    """Integrate N_eff = theta(e^{dS}-1) once, then use the two integration
    constants of Proposition 'plane'(ii) to put the curve on [x_lo, x_hi].
    Returns a function x -> dS on that range."""
    dS = np.linspace(dS_lo, dS_hi, ngrid)
    s = (1.0 - np.exp(-dS)) ** (1.0 / theta)          # alpha = 1 for now
    u = np.concatenate([[0.0], np.cumsum(np.diff(dS) / (0.5 * (s[1:] + s[:-1])))])
    # dS increases as x decreases; match both endpoints
    span = u[-1] - u[0]
    if not np.isfinite(span) or span <= 0:
        return None
    xs = x_hi - (u - u[0]) * (x_hi - x_lo) / span      # xs[0]=x_hi, xs[-1]=x_lo
    order = np.argsort(xs)
    return xs[order], dS[order]


def truth_theta(x, e, floor, theta):
    """A frontier with floor `floor` whose channel count is theta times the
    two-atom one at every depth, sampled at the observed nodes x."""
    if not (np.isfinite(floor) and floor > 0 and floor < e.min()):
        return None
    dS_hi, dS_lo = np.log(e.max() / floor), np.log(e.min() / floor)
    if not (np.isfinite(dS_hi) and dS_lo > 1e-6 and dS_hi > dS_lo):
        return None
    cur = theta_curve(theta, dS_lo, dS_hi, x[0], x[-1])
    if cur is None:
        return None
    xs, dS = cur
    return floor * np.exp(np.interp(x, xs, dS))



# ------------------------------------------------- the precision that would decide
SIGMAS = (0.002, 0.005, 0.010, 0.020)
R_SWEEP = 150
THETA_SWEEP = 1.5


def sigma_sweep(x, e, eps_inf, c, alpha, cons_idx, all_idx, rng):
    """Level and power of each statistic at a sequence of per-model precisions,
    the archive's own residual scale replaced by a declared sigma.  This is the
    ceiling test's analogue of (53): what a measurement would have to deliver
    for the test to decide anything."""
    out = {}
    truth0 = mdl([np.log(eps_inf), np.log(c), np.log(alpha)], x)
    truth1 = truth_theta(x, e, eps_inf, THETA_SWEEP)
    n = len(x)
    for sg in SIGMAS:
        def rep(truth, R):
            lt = np.log(truth)[None, :]
            Lr = lt + sg * rng.standard_normal((R, n))
            fl = np.empty(R)
            for r in range(R):
                f_ = fit4(x, np.exp(Lr[r]))
                fl[r] = f_[0] if f_ is not None else np.nan
            return statistics(Lr, cons_idx, all_idx, x, fl)
        gm0, gd0, gw0 = rep(truth0, R_SWEEP)
        out[f"s{sg}_false_min"] = float(np.mean(gm0[np.isfinite(gm0)] > 0)) if np.isfinite(gm0).any() else np.nan
        for tag, nl in (("med", gd0), ("wide", gw0)):
            nl = nl[np.isfinite(nl)]
            q = float(np.nanquantile(nl, 1 - ALPHA_LEVEL)) if len(nl) >= 50 else np.nan
            out[f"s{sg}_q95_{tag}"] = q
            if truth1 is None or not np.isfinite(q):
                out[f"s{sg}_power_{tag}"] = np.nan
                continue
            _, gd1, gw1 = rep(truth1, R_SWEEP)
            alt = (gd1 if tag == "med" else gw1)
            alt = alt[np.isfinite(alt)]
            out[f"s{sg}_power_{tag}"] = float(np.mean(alt > q)) if len(alt) else np.nan
    return out

# ----------------------------------------------------------------------- run
def run():
    rng = np.random.default_rng(SEED)
    rows = []
    for fr in all_frontiers():
        x, e = np.asarray(fr["x"], float), np.asarray(fr["e"], float)
        o = np.argsort(x)
        x, e = x[o], e[o]
        k = (e > 0) & np.isfinite(e)
        x, e = x[k], e[k]
        if len(x) < 5:
            continue
        n = len(x)
        f = fit4(x, e)
        if f is None:
            continue
        eps_inf, c, alpha, sig = f
        cons_idx, all_idx = triple_index(x, rng=rng)
        if len(all_idx) == 0:
            continue

        # ---------------- observed ------------------------------------------
        L = np.log(e)[None, :]
        g_min_o, g_med_o, g_wide_o = statistics(L, cons_idx, all_idx, x, [eps_inf])
        g_min_o, g_med_o, g_wide_o = g_min_o[0], g_med_o[0], g_wide_o[0]

        # ---------------- the null: the fitted form (4) itself ---------------
        def replicate(truth, R):
            lt = np.log(truth)[None, :]
            Lr = lt + sig * rng.standard_normal((R, n))
            fl = np.empty(R)
            for r in range(R):
                fr_ = fit4(x, np.exp(Lr[r]))
                fl[r] = fr_[0] if fr_ is not None else np.nan
            return statistics(Lr, cons_idx, all_idx, x, fl)

        truth0 = mdl([np.log(eps_inf), np.log(c), np.log(alpha)], x)
        gm0, gd0, gw0 = replicate(truth0, R_NULL)

        def pval(obs, null):
            null = null[np.isfinite(null)]
            if not np.isfinite(obs) or len(null) < 50:
                return np.nan, np.nan
            return ((1 + np.sum(null >= obs)) / (len(null) + 1),
                    float(np.nanquantile(null, 1 - ALPHA_LEVEL)))

        p_min, q_min = pval(g_min_o, gm0)
        p_med, q_med = pval(g_med_o, gd0)
        p_wide, q_wide = pval(g_wide_o, gw0)

        rec = dict(
            corpus=fr["corpus"], name=fr["name"], n=n, sigma=sig,
            n_triples_all=len(all_idx), n_triples_cons=len(cons_idx),
            eps_inf_fit=eps_inf, eps_min=float(e.min()),
            g_min=g_min_o, g_med=g_med_o, g_wide=g_wide_o,
            incumbent_violates=int(g_min_o > 0),
            null_rate_min=float(np.mean(gm0[np.isfinite(gm0)] > 0)) if np.isfinite(gm0).any() else np.nan,
            null_rate_med=float(np.mean(gd0[np.isfinite(gd0)] > 0)) if np.isfinite(gd0).any() else np.nan,
            null_rate_wide=float(np.mean(gw0[np.isfinite(gw0)] > 0)) if np.isfinite(gw0).any() else np.nan,
            p_min=p_min, p_med=p_med, p_wide=p_wide,
            q95_min=q_min, q95_med=q_med, q95_wide=q_wide,
        )

        # ---------------- power, against theta > 1 ---------------------------
        for th in THETAS:
            tr = truth_theta(x, e, eps_inf, th)
            if tr is None:
                rec[f"power_med_t{th}"] = np.nan
                rec[f"power_wide_t{th}"] = np.nan
                rec[f"power_min_t{th}"] = np.nan
                continue
            gm1, gd1, gw1 = replicate(tr, R_POWER)
            for tag, alt, q in (("min", gm1, q_min), ("med", gd1, q_med), ("wide", gw1, q_wide)):
                alt = alt[np.isfinite(alt)]
                rec[f"power_{tag}_t{th}"] = (float(np.mean(alt > q))
                                             if (len(alt) and np.isfinite(q)) else np.nan)
        rec.update(sigma_sweep(x, e, eps_inf, c, alpha, cons_idx, all_idx, rng))
        rows.append(rec)
        print(f"  {fr['corpus']:<13s} {fr['name'][:34]:<34s} "
              f"n={n:3d} sig={sig:.3f}  incumbent={rec['incumbent_violates']} "
              f"null(min)={rec['null_rate_min']:.2f} p_med={p_med if np.isfinite(p_med) else float('nan'):.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    save(df, "table_ceiling_null_per_frontier.csv")

    # ------------------------------------------------------------- report
    N = len(df)
    print("\n" + "=" * 78)
    print("THE ADMISSIBLE-CEILING TEST, READ AGAINST A DESIGN-MATCHED NULL")
    print("=" * 78)
    print(f"\nfrontiers .......................................... {N}")
    inc = int(df.incumbent_violates.sum())
    nr = df.null_rate_min.dropna()
    print(f"\n(1) THE INCUMBENT STATISTIC (min over all triples)")
    print(f"  observed violations .............................. {inc} of {N}")
    print(f"  expected under the null (sum of per-frontier rates) {nr.sum():.1f}")
    print(f"  median per-frontier false-violation rate ......... {nr.median():.3f}")
    print(f"  frontiers whose null rate exceeds 0.50 ........... {int((nr > .5).sum())} of {len(nr)}")
    print(f"  frontiers rejecting at {ALPHA_LEVEL} once calibrated ...... "
          f"{int((df.p_min < ALPHA_LEVEL).sum())} of {int(df.p_min.notna().sum())}")

    for tag, lab in (("med", "median over consecutive triples"),
                     ("wide", "single widest triple, fixed in advance")):
        r = df[f"null_rate_{tag}"].dropna()
        p = df[f"p_{tag}"].dropna()
        rej = int((p < ALPHA_LEVEL).sum())
        print(f"\n({2 if tag == 'med' else 3}) CALIBRATED: {lab}")
        print(f"  raw violations (g > 0) .......................... "
              f"{int((df[f'g_{tag}'] > 0).sum())} of {N}")
        print(f"  median per-frontier false-violation rate ........ {r.median():.3f}")
        print(f"  rejects at {ALPHA_LEVEL} against its own null .......... {rej} of {len(p)}"
              f"   (expected {ALPHA_LEVEL * len(p):.1f})")
        if len(p):
            q = np.sort(p.values)
            m = len(q)
            bh = q <= ALPHA_LEVEL * np.arange(1, m + 1) / m
            print(f"  Benjamini-Hochberg at FDR {ALPHA_LEVEL} ................. "
                  f"{int(np.max(np.where(bh)[0]) + 1) if bh.any() else 0} of {m}")
        for th in THETAS:
            pw = df[f"power_{tag}_t{th}"].dropna()
            if len(pw):
                print(f"  power at theta={th} ({np.log(th):.2f} nats of violation) ..... "
                      f"median {pw.median():.3f}   >=0.8 on {int((pw >= .8).sum())} of {len(pw)}")

    print(f"\n(4) THE PRECISION THAT WOULD DECIDE IT"
          f"   (level and power against theta={THETA_SWEEP}, "
          f"{np.log(THETA_SWEEP):.2f} nats of violation)")
    print("    sigma      false-violation rate      power, median triple   power, widest triple")
    for sg in SIGMAS:
        fm = df[f"s{sg}_false_min"].dropna()
        pm = df[f"s{sg}_power_med"].dropna()
        pw = df[f"s{sg}_power_wide"].dropna()
        print(f"    {sg*1000:5.0f} mn   {fm.median():.3f} (incumbent)       "
              f"{pm.median():.3f}                  {pw.median():.3f}")
    fmr = df.null_rate_min.dropna()
    print(f"    {df.sigma.median()*1000:5.0f} mn   {fmr.median():.3f} (archive's own)   "
          f"{df[f'power_med_t1.5'].dropna().median():.3f}                  "
          f"{df[f'power_wide_t1.5'].dropna().median():.3f}")

    summ = dict(n_frontiers=N, median_sigma=float(df.sigma.median()), incumbent_violations=inc,
                incumbent_expected_under_null=float(nr.sum()),
                incumbent_median_null_rate=float(nr.median()),
                incumbent_calibrated_rejections=int((df.p_min < ALPHA_LEVEL).sum()),
                alpha=ALPHA_LEVEL, r_null=R_NULL, r_power=R_POWER)
    for tag in ("med", "wide"):
        p = df[f"p_{tag}"].dropna()
        q = np.sort(p.values)
        m = len(q)
        bh = q <= ALPHA_LEVEL * np.arange(1, m + 1) / m if m else np.array([False])
        summ[f"{tag}_raw_violations"] = int((df[f"g_{tag}"] > 0).sum())
        summ[f"{tag}_median_null_rate"] = float(df[f"null_rate_{tag}"].dropna().median())
        summ[f"{tag}_rejections"] = int((p < ALPHA_LEVEL).sum())
        summ[f"{tag}_n_tested"] = m
        summ[f"{tag}_bh_rejections"] = int(np.max(np.where(bh)[0]) + 1) if bh.any() else 0
        for th in THETAS:
            pw = df[f"power_{tag}_t{th}"].dropna()
            summ[f"{tag}_power_t{th}"] = float(pw.median()) if len(pw) else np.nan
        for sg in SIGMAS:
            summ[f"{tag}_power_sigma{sg}"] = float(df[f"s{sg}_power_{tag}"].dropna().median())
    for sg in SIGMAS:
        summ[f"false_min_sigma{sg}"] = float(df[f"s{sg}_false_min"].dropna().median())
    save(pd.DataFrame([summ]), "table_ceiling_null_summary.csv")


if __name__ == "__main__":
    run()
