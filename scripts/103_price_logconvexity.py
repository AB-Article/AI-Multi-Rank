#!/usr/bin/env python3
"""THE FOUR-POINT TEST: IS THE PRICE OF THE NEXT ATTEMPT LOG-CONVEX?
   (Section "The price is a frontier", Prop. 'logconvex', Table 'fourpoint'.)

Script 98 could not decide the composition law on the compute axis, and said so.
The reason is structural, not statistical, and it is Prop. 'order':

    the axiom and the composition law are IDENTICAL at every order <= 2
    and first differ at order 3.

A window that resolves two derivatives cannot separate them at all -- at that
order they are the same hypothesis -- so no reanalysis of a compute-axis corpus
can settle it.  Prop. 'relocate' says where to go instead: with eps = P(m(x))
and m Bernstein, S = (-ln P) o m, and Bernstein o Bernstein = Bernstein, so

    S is Bernstein on EVERY admissible compute axis
      <=>  -ln P is Bernstein
      <=>  the law of per-item log-difficulty b = -ln g is infinitely divisible.

The composition law is a property of the TASK, not of any axis; a compute-axis
test is a joint test of the task and the unmeasured clock.  A COUNT axis has no
clock freedom, and on it every order is an exact finite difference of
integer-indexed observables: no bandwidth, no smoother, no null simulation.

Write the price of the next attempt  L_m = ln( P(m)/P(m+1) )  (so L_1 =
-ln kappa_nest, which script 71 already measures).  Rung 1 of the log-convexity
ladder C_k C_{k+2} >= C_{k+1}^2, with C_k = (-1)^{k+1} Delta^k S, collapses
algebraically to

    ####   L_1 L_3  >=  L_2^2   ####        the price sequence is log-convex

needing P(1..4) and nothing else.  We report it in the well-conditioned form

    T  =  ln L_1 + ln L_3 - 2 ln L_2   >=  0 ,

a smooth function of three positive, directly measured prices.  (The raw
cumulant ratio C_1C_3/C_2^2 is algebraically identical but divides by a
difference that can cross zero under resampling, so it is unusable as a plug-in
-- the archive's own method law, one level up.)

Tested here on the purpose-built panel of script 63 (21 models, 10^4 items, two
families), with the item bootstrap as the only source of uncertainty:

  T1  T and its one-sided bootstrap interval on the best-model chain;
  T2  T on all C(20,3) nested four-model chains: sign agreement and median;
  T3  the sign conditions (-1)^{k+1} Delta^k S >= 0 (the weaker Bernstein form)
      and the Hausdorff conditions (-1)^k Delta^k P >= 0 (the AXIOM), for
      contrast;
  T4  CALIBRATION on truths with a known verdict: divisible (cure, compound
      Poisson, Gamma, stable) and indivisible (the archive's own incumbent
      chart, floor + one channel).  T is exactly signed on all of them;
  T5  POWER: the item count needed to refute the incumbent chart at 95%.  This
      replaces 'we need a deeper corpus' with a number of items.

Reads  data/cofailure/fmnist_{indep,nested}_correct.npy
Writes (results/):
    table_fourpoint_panel.csv        per family: P(m), L_m, T, CIs, chains
    table_fourpoint_chains.csv       T2: every nested four-model chain
    table_fourpoint_calibration.csv  T4: known truths, exact and simulated
    table_fourpoint_power.csv        T5: power against item count
    table_fourpoint_summary.csv      the verdict, one row per family
"""
from __future__ import annotations
import itertools
import os
import sys
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "cofailure")
from _loaders import save                                     # noqa: E402

Q0 = 0.10                      # Fashion-MNIST: 10 classes
NBOOT = 4000
RNG = np.random.default_rng(20261005)
FAMILIES = ("indep", "nested")
MMAX = 8


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------
def prices(P):
    """L_m = ln(P(m)/P(m+1)) >= 0: nats of log-error bought by attempt m+1."""
    P = np.asarray(P, float)
    return np.log(P[:-1] / P[1:])


def defect(P, m0=1):
    """T = ln L_{m0} + ln L_{m0+2} - 2 ln L_{m0+1}.

    T >= 0 is exactly rung 1 of the log-convexity ladder at m0, hence exactly
    the composition law read at third order on the count axis.  Needs
    P(m0..m0+3).  nan if any price is non-positive (P not strictly decreasing)."""
    L = prices(np.asarray(P, float)[m0 - 1:m0 + 3])
    if len(L) < 3 or np.any(L <= 0):
        return np.nan
    return float(np.log(L[0]) + np.log(L[2]) - 2.0 * np.log(L[1]))


def cumulants(P, m0=1, kmax=3):
    """C_k = (-1)^{k+1} Delta^k S, S = -ln P: moments of one positive measure
    under the composition law, hence non-negative and log-convex."""
    S = -np.log(np.asarray(P, float))
    i = m0 - 1
    return np.array([(-1) ** (k + 1) * sum((-1) ** (k - j) * comb(k, j) * S[i + j]
                                           for j in range(k + 1))
                     for k in range(1, kmax + 1)], float)


def hausdorff_violations(P, kmax=6):
    """(-1)^k Delta^k P >= 0: the AXIOM on the count axis (Hausdorff)."""
    D, out = np.asarray(P, float).copy(), []
    for k in range(1, kmax + 1):
        D = np.diff(D)
        out.append(int(np.sum((-1) ** k * D < 0)))
    return out


def bernstein_violations(P, kmax=5):
    """(-1)^{k+1} Delta^k S >= 0: the sign-only (weaker) form of the law."""
    S = -np.log(np.asarray(P, float))
    D, out = S.copy(), []
    for k in range(1, kmax + 1):
        D = np.diff(D)
        out.append(int(np.sum((-1) ** (k + 1) * D < 0)))
    return out


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------
def chain_P(Fail, order, mmax):
    """P(m) over the m BEST models (script 71's chain: the hardest member is
    fixed, so P is a pure moment sequence of one and the same nu)."""
    return np.array([Fail[:, order[:m]].all(1).mean() for m in range(1, mmax + 1)])


def load_panel(family):
    C = np.load(os.path.join(DATA, f"fmnist_{family}_correct.npy"))
    Fail = ~C.astype(bool)
    order = np.argsort(Fail.mean(0))                           # best model first
    return Fail, order


def all_chains(Fail, order, nmax=None):
    """Every nested four-model chain sharing the best model as hardest member:
    {a, b, c, d} with a the best model and b<c<d by failure rate.  Each is a
    valid P(1..4), so each gives one T."""
    a, rest = order[0], list(order[1:])
    rows = []
    combos = list(itertools.combinations(rest, 3))
    if nmax is not None and len(combos) > nmax:
        idx = RNG.choice(len(combos), nmax, replace=False)
        combos = [combos[i] for i in idx]
    for trio in combos:
        ids = [a] + list(trio)
        P = np.array([Fail[:, ids[:m]].all(1).mean() for m in (1, 2, 3, 4)])
        if np.any(P <= 0) or np.any(np.diff(P) >= 0):
            continue
        rows.append(dict(models="|".join(map(str, ids)), P1=P[0], P2=P[1],
                         P3=P[2], P4=P[3], T=defect(P)))
    return pd.DataFrame(rows)


def run_panel():
    rows, summ, chains = [], [], []
    for fam in FAMILIES:
        Fail, order = load_panel(fam)
        n_items, n_models = Fail.shape
        P = chain_P(Fail, order, MMAX)
        L = prices(P)
        C = cumulants(P, 1, 3)
        T_obs = defect(P)

        # item bootstrap.  The four all-fail events are nested, so resampling
        # items is exactly a multinomial draw over the five nested cells --
        # identical in law to re-drawing item indices, and far cheaper.
        boot = simulate_T(P[:4], n_items, NBOOT, RNG)
        ok = np.isfinite(boot)
        se = float(np.std(boot[ok]))
        lo, hi = np.percentile(boot[ok], [2.5, 97.5])
        lo95 = float(np.percentile(boot[ok], 5.0))             # one-sided
        frac_neg = float(np.mean(boot[ok] < 0.0))

        ch = all_chains(Fail, order, nmax=400)
        ch.insert(0, "family", fam)
        chains.append(ch)

        for m in range(1, MMAX):
            rows.append(dict(family=fam, m=m, P_m=P[m - 1], P_m1=P[m], L_m=L[m - 1]))

        hv, bv = hausdorff_violations(P), bernstein_violations(P)
        summ.append(dict(
            family=fam, n_items=n_items, n_models=n_models, q0=Q0,
            P1=P[0], P2=P[1], P3=P[2], P4=P[3],
            L1=L[0], L2=L[1], L3=L[2], C1=C[0], C2=C[1], C3=C[2],
            T_obs=T_obs, T_se=se, T_ci_lo=float(lo), T_ci_hi=float(hi),
            T_onesided_lo95=lo95, boot_frac_negative=frac_neg,
            ratio_L1L3_over_L2sq=float(np.exp(T_obs)),
            verdict=("REFUTED" if hi < 0 else
                     ("not refuted (one-sided p>0.05)" if lo95 < 0 <= T_obs
                      else ("not refuted" if lo >= 0 else "inconclusive"))),
            chains_n=len(ch), chains_frac_T_nonneg=float((ch["T"] >= 0).mean()),
            chains_T_median=float(ch["T"].median()),
            kappa_nest=P[1] / P[0], price_attempt_nats=L[0],
            hausdorff_violations=";".join(map(str, hv)),
            bernstein_sign_violations=";".join(map(str, bv))))
    return (pd.DataFrame(rows), pd.concat(chains, ignore_index=True),
            pd.DataFrame(summ))


# ---------------------------------------------------------------------------
# calibration and power
# ---------------------------------------------------------------------------
TRUTHS = {
    "cure (1 obstacle type)":
        (lambda m: np.exp(-1.3 * (1 - np.exp(-0.9 * m))), True),
    "compound Poisson, 3 types":
        (lambda m: np.exp(-(0.9 * (1 - np.exp(-0.4 * m))
                            + 0.5 * (1 - np.exp(-1.2 * m))
                            + 0.3 * (1 - np.exp(-3.0 * m)))), True),
    "Gamma b (shape 2, rate 1.5)":
        (lambda m: (1.5 / (1.5 + m)) ** 2, True),
    "one-sided stable b (beta=0.5)":
        (lambda m: np.exp(-1.1 * np.sqrt(m)), True),
    "floor + 1 channel, f=0.15":
        (lambda m: 0.15 + 0.85 * np.exp(-0.7 * m), False),
    "floor + 1 channel, f=0.40":
        (lambda m: 0.40 + 0.60 * np.exp(-0.9 * m), False),
    "floor + 1 channel, f=0.02":
        (lambda m: 0.02 + 0.98 * np.exp(-0.5 * m), False),
    "2 atoms in g, no floor":
        (lambda m: 0.6 * np.exp(-0.25 * m) + 0.4 * np.exp(-1.8 * m), False),
    "4 atoms in g, no floor":
        (lambda m: sum(0.25 * np.exp(-b * m) for b in (0.3, 0.8, 1.4, 2.2)), False),
}


def simulate_T(Ptrue, n_items, nrep, rng):
    """P(m) are NESTED all-fail probabilities, so the four counts are determined
    by one multinomial over the five nested cells
        {fails>=4}, {fails exactly 3}, {2}, {1}, {0},
    with probabilities P4, P3-P4, P2-P3, P1-P2, 1-P1.  Drawing that multinomial
    is exact and costs O(1) in the item count, which is what makes the power
    curve below computable out to 10^9 items."""
    P = np.asarray(Ptrue, float)
    cells = np.array([P[3], P[2] - P[3], P[1] - P[2], P[0] - P[1], 1.0 - P[0]])
    cells = np.clip(cells, 0.0, None)
    cells = cells / cells.sum()
    draws = rng.multinomial(int(n_items), cells, size=nrep).astype(float) / n_items
    # rebuild the nested P-hat: P4 = c0, P3 = c0+c1, P2 = +c2, P1 = +c3
    Ph = np.cumsum(draws[:, :4], axis=1)[:, ::-1]        # columns P1..P4
    out = np.full(nrep, np.nan)
    good = np.all(Ph > 0, axis=1) & np.all(np.diff(Ph, axis=1) < 0, axis=1)
    if good.any():
        L = np.log(Ph[good, :-1] / Ph[good, 1:])
        ok2 = np.all(L > 0, axis=1)
        idx = np.where(good)[0][ok2]
        Lg = L[ok2]
        out[idx] = np.log(Lg[:, 0]) + np.log(Lg[:, 2]) - 2.0 * np.log(Lg[:, 1])
    return out[np.isfinite(out)]


def calibration(n_items=10_000, nrep=4000):
    rng = np.random.default_rng(20261005)
    rows = []
    for name, (Pf, div) in TRUTHS.items():
        Ptrue = np.array([Pf(m) for m in range(1, 5)])
        T_exact = defect(Ptrue)
        sim = simulate_T(Ptrue, n_items, nrep, rng)
        rows.append(dict(
            truth=name, divisible=int(div), T_exact=T_exact,
            ratio_exact=float(np.exp(T_exact)),
            T_sim_median=float(np.median(sim)), T_sim_se=float(np.std(sim)),
            T_sim_lo=float(np.percentile(sim, 2.5)),
            T_sim_hi=float(np.percentile(sim, 97.5)),
            exact_sign_correct=int((T_exact >= 0) == div
                                   or (not div and T_exact < 0)),
            frac_sim_negative=float(np.mean(sim < 0)),
            power_at_n=float(np.mean(np.array([
                np.percentile(sim, 95)]) < 0))))
    return pd.DataFrame(rows)


def power(nrep=4000, ns=(1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8)):
    """How many items refute an indivisible truth at the 95% one-sided level?
    The statistic has no tuning parameter, so this is pure sampling power."""
    rng = np.random.default_rng(20261005)
    rows = []
    for name, (Pf, div) in TRUTHS.items():
        Ptrue = np.array([Pf(m) for m in range(1, 5)])
        T_exact = defect(Ptrue)
        for n in ns:
            sim = simulate_T(Ptrue, int(n), nrep, rng)
            if len(sim) < 20:
                continue
            # one-sided test of H0: T >= 0.  Reject when the upper 95% point < 0.
            rows.append(dict(truth=name, divisible=int(div), n_items=int(n),
                             T_exact=T_exact, T_sim_median=float(np.median(sim)),
                             T_sim_se=float(np.std(sim)),
                             frac_reject=float(np.mean(
                                 sim + 1.645 * np.std(sim) < 0))))
    return pd.DataFrame(rows)


def run():
    P, chains, summ = run_panel()
    cal = calibration()
    pw = power()
    save(P, "table_fourpoint_panel.csv")
    save(chains, "table_fourpoint_chains.csv")
    save(cal, "table_fourpoint_calibration.csv")
    save(pw, "table_fourpoint_power.csv")
    save(summ, "table_fourpoint_summary.csv")
    return P, chains, cal, pw, summ


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    P, chains, cal, pw, summ = run()
    print("\nTHE FOUR-POINT TEST ON THE PANEL   T = ln L1 + ln L3 - 2 ln L2 >= 0\n")
    print(summ[["family", "P1", "P2", "P3", "P4", "L1", "L2", "L3",
                "ratio_L1L3_over_L2sq", "T_obs", "T_se", "T_onesided_lo95",
                "verdict"]].to_string(index=False), "\n")
    print(summ[["family", "chains_n", "chains_frac_T_nonneg", "chains_T_median",
                "hausdorff_violations", "bernstein_sign_violations"]].to_string(index=False), "\n")
    print("CALIBRATION: the statistic is EXACTLY signed on truths with a known verdict\n")
    print(cal[["truth", "divisible", "T_exact", "ratio_exact", "T_sim_median",
               "T_sim_se", "exact_sign_correct"]].to_string(index=False), "\n")
    nd, ni = cal[cal.divisible == 1], cal[cal.divisible == 0]
    nf = ni[ni.truth.str.startswith("floor")]
    print(f"  divisible truths with T >= 0 exactly       : {int((nd.T_exact >= 0).sum())}/{len(nd)}")
    print(f"  floor+1-channel truths with T < 0 exactly  : {int((nf.T_exact < 0).sum())}/{len(nf)}")
    print("  the test is ONE-SIDED: finite mixtures of exponentials pass at every")
    print("  measurable order although their nu cannot be infinitely divisible.\n")
    print("POWER: items needed to refute an indivisible truth (one-sided, 95%)\n")
    piv = pw[pw.divisible == 0].pivot_table(index="truth", columns="n_items",
                                            values="frac_reject")
    print(piv.to_string(), "\n")
    for t, g in pw[pw.divisible == 0].groupby("truth"):
        hit = g[g.frac_reject >= 0.95].n_items.min()
        print(f"  {t:28s} 95% power at n_items >= "
              f"{('%.0e' % hit) if np.isfinite(hit) else '> 1e7'}")
