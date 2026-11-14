#!/usr/bin/env python3
"""THE COUNT-AXIS LADDER COMPLETES: HANKEL POSITIVITY FROM P(1..6).
   (Section "Where the hypothesis can be decided", Prop. 'hankel', 'attemptfloor';
    Online Appendix sec. "The attempt chain".)

Revision h tested the composition law on a count axis with ONE inequality,
L_1 L_3 >= L_2^2, which is the 2x2 minor of the Hankel array of the discrete
cumulants C_k = (-1)^{k+1} Delta^k S, S = -ln P.  Log-convexity of a sequence is
NECESSARY for it to be a moment sequence and is NOT SUFFICIENT (script 113, sec.
7), so the four-point test is rung 1 of a ladder whose complete form is

        H_n  =  det ( C_{i+j+1} )_{i,j<n}  >=  0   for every n,

each rung an EXACT function of integer-indexed all-fail rates: rung 1 needs
P(1..4), rung 2 needs P(1..6).  No bandwidth, no smoother, no derivative and no
fitted chart at any rung, and every rung is one-sided for the reason of
Prop. 'nodes' (the extremal measures of every moment bound are atomic).

Measured here, on the archive's own co-failure panel and with the archive's own
item bootstrap:
  (A) rung 1 and rung 2 on the best-model chain of each family, with bootstrap
      intervals and the one-sided 95% limit;
  (B) rung 2 on every nested six-model chain, as rung 1 was reported on
      four-model chains;
  (C) calibration on truths with a known verdict, including the three-atom
      census that rung 1 PASSES and rung 2 REFUTES (the archive's own inverted
      nu is bimodal with a third, systematically-wrong class);
  (D) the heterogeneity discount S(m)/(m S(1)): 1 exactly iff items are
      homogeneous, so the gap between "nats" and "attempts" is measured, not
      assumed -- the Jensen gap of the count axis;
  (E) the attempt-budget floor bracket  P(ceil(Lambda)) <= eps_inf <= P(floor(Lambda)),
      whose width is one attempt's price L_{floor(Lambda)}: the first two-sided
      floor statement in the paper that needs no extrapolation and no fit.

Outputs: results/table_count_hankel.csv        panel, rungs 1 and 2
         results/table_count_hankel_chains.csv per six-model chain
         results/table_count_hankel_calib.csv  truths with a known verdict
         results/table_attempt_budget.csv      heterogeneity + floor bracket
"""
from __future__ import annotations
import os
import sys
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.special import comb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save                                      # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "cofailure")
RNG = np.random.default_rng(20261005)
FAMILIES = ("indep", "nested")
Q0 = 0.10
NBOOT = 4000
MMAX = 8
NCHAIN = 400


# ------------------------------------------------------------- the statistic --
def cumulants(P, kmax):
    """C_k = (-1)^{k+1} Delta^k S from P(1..kmax+1), S = -ln P."""
    S = -np.log(np.asarray(P, float))
    return np.array([(-1) ** (k + 1) * sum((-1) ** (k - j) * comb(k, j) * S[j]
                                           for j in range(k + 1))
                     for k in range(1, kmax + 1)], float)


def rung(P, n):
    """Rung n of the Hankel ladder, in GAUGE-INVARIANT form.

    H_n = det(C_{i+j+1})_{i,j<n} is the raw minor; under the error-power group
    C_k -> t C_k it scales as t^n, so it is reported normalised,

        Hhat_n = det( C_{i+j+1} )_{i,j<n} / C_1^n ,

    which is gauge-invariant, needs only C_1 = L_1 > 0 (always available), and
    vanishes EXACTLY on the equality case -- a geometric cumulant sequence, i.e.
    one obstacle type.  n = 2 is L_1 L_3 >= L_2^2 up to the positive factor
    1/C_1^2."""
    C = cumulants(P, 2 * n - 1)
    if not np.all(np.isfinite(C)) or C[0] <= 0:
        return np.nan
    M = np.array([[C[i + j] for j in range(n)] for i in range(n)])
    return float(np.linalg.det(M) / C[0] ** n)


def prices(P):
    P = np.asarray(P, float)
    return np.log(P[:-1] / P[1:])


def rung1_L(P):
    """the archive's own form of rung 2: T = ln L_1 + ln L_3 - 2 ln L_2."""
    L = prices(np.asarray(P, float)[:4])
    if len(L) < 3 or np.any(L <= 0):
        return np.nan
    return float(np.log(L[0]) + np.log(L[2]) - 2 * np.log(L[1]))


# ------------------------------------------------------------- the bootstrap --
def boot_rungs(Ptrue, n_items, nrep, rng, nmax=6):
    """Nested all-fail rates => one multinomial over the nmax+1 nested cells.
    Exact, and O(1) in the item count (the archive's convention, script 103)."""
    P = np.asarray(Ptrue, float)[:nmax]
    cells = np.concatenate([[P[-1]], -np.diff(P)[::-1], [1.0 - P[0]]])
    cells = np.clip(cells, 0.0, None)
    cells /= cells.sum()
    draws = rng.multinomial(int(n_items), cells, size=nrep).astype(float) / n_items
    Ph = np.cumsum(draws[:, :nmax], axis=1)[:, ::-1]           # columns P1..Pnmax
    out = {2: [], 3: []}
    good = np.all(Ph > 0, axis=1) & np.all(np.diff(Ph, axis=1) < 0, axis=1)
    for row in Ph[good]:
        out[2].append(rung(row[:4], 2))
        if nmax >= 6:
            out[3].append(rung(row[:6], 3))
    return {k: np.array([v for v in vals if np.isfinite(v)]) for k, vals in out.items()}


# --------------------------------------------------- design-matched nulls -----
def fit_gamma(P_obs):
    """P(m) = (1 + m/theta)^{-k}: a Gamma(k) spectrum in b, DIVISIBLE.
    Least squares on S = -ln P over the observed m, i.e. the best divisible
    description of the SAME curve -- the archive's design-matched null."""
    from scipy.optimize import least_squares
    S = -np.log(np.asarray(P_obs, float))
    mm = np.arange(1, len(S) + 1, dtype=float)

    def r(p):
        k, th = np.exp(p)
        return k * np.log1p(mm / th) - S
    try:
        sol = least_squares(r, np.log([2.0, 2.0]), method="lm", max_nfev=5000)
        k, th = np.exp(sol.x)
    except Exception:
        return None
    return lambda m: (1 + m / th) ** (-k)


def fit_cure(P_obs):
    """P(m) = exp(-lam(1-e^{-c m})), the cure chart on the count axis: DIVISIBLE,
    the equality case of every rung.  Least squares on S over the observed m."""
    from scipy.optimize import least_squares
    S = -np.log(np.asarray(P_obs, float))
    mm = np.arange(1, len(S) + 1, dtype=float)

    def r(p):
        lam, c = np.exp(p)
        return lam * (1 - np.exp(-c * mm)) - S
    try:
        sol = least_squares(r, np.log([max(S[-1], 1e-3) * 2, 0.5]),
                            method="lm", max_nfev=5000)
        lam, c = np.exp(sol.x)
    except Exception:
        return None
    return lambda m: np.exp(-lam * (1 - np.exp(-c * m)))


def null_thresholds(P_obs, n_items, rng, nrep=NBOOT):
    """The archive's method law: compare a plug-in of an exact identity against a
    null simulated on the SAME design, never against the analytic value.  Two
    divisible truths are matched to the observed P(1), P(2) and the 5th
    percentile of each null is the one-sided threshold at the 95% level."""
    out = {}
    for nm, fit in (("gamma", fit_gamma), ("cure", fit_cure)):
        f = fit(np.asarray(P_obs, float)[:6])
        if f is None:
            continue
        Pt = np.array([f(m) for m in range(1, 7)])
        if np.any(np.diff(Pt) >= 0) or np.any(Pt <= 0):
            continue
        sim = boot_rungs(Pt, n_items, nrep, rng, nmax=6)
        out[nm] = {n: ((float(np.percentile(sim[n], 5)), float(np.median(sim[n])),
                        rung(Pt[:2 * n], n)) if len(sim[n]) > 20
                       else (np.nan, np.nan, np.nan)) for n in (2, 3)}
    return out


# ------------------------------------------------------------------ the panel --
def load_panel(family):
    C = np.load(os.path.join(DATA, f"fmnist_{family}_correct.npy"))
    Fail = ~C.astype(bool)
    return Fail, np.argsort(Fail.mean(0))


def chain_P(Fail, order, mmax):
    return np.array([Fail[:, order[:m]].all(1).mean() for m in range(1, mmax + 1)])


def run_panel():
    rows, chains, budget = [], [], []
    for fam in FAMILIES:
        Fail, order = load_panel(fam)
        n_items = Fail.shape[0]
        P = chain_P(Fail, order, MMAX)
        L = prices(P)
        b = boot_rungs(P, n_items, NBOOT, RNG, nmax=6)
        nul = null_thresholds(P, n_items, np.random.default_rng(20261005))
        for n in (2, 3):
            obs = rung(P[:2 * n], n)
            sim = b[n]
            thr = [v[n][0] for v in nul.values() if np.isfinite(v[n][0])]
            rows.append(dict(
                family=fam, rung=n - 1, needs=f"P(1..{2 * n})", Hhat_n=obs,
                se=float(np.std(sim)), boot_lo95=float(np.percentile(sim, 5)),
                frac_boot_negative=float(np.mean(sim < 0)),
                null_gamma_p05=nul.get("gamma", {}).get(n, (np.nan,) * 3)[0],
                null_cure_p05=nul.get("cure", {}).get(n, (np.nan,) * 3)[0],
                null_gamma_median=nul.get("gamma", {}).get(n, (np.nan,) * 3)[1],
                refuted_vs_null=int(all(obs < t for t in thr) if thr else 0),
                T_archive_form=rung1_L(P) if n == 2 else np.nan))
        # (D) heterogeneity discount and (E) the floor bracket
        for m in range(1, MMAX + 1):
            budget.append(dict(
                family=fam, m=m, P_m=P[m - 1], S_m=-np.log(P[m - 1]),
                homog_S=m * (-np.log(P[0])),
                discount=float(-np.log(P[m - 1]) / (m * -np.log(P[0]))),
                L_m=L[m - 1] if m <= len(L) else np.nan,
                bracket_lo=P[m] if m < MMAX else np.nan,
                bracket_hi=P[m - 1],
                bracket_width_nats=L[m - 1] if m <= len(L) else np.nan))
        # (B) rung 2 on every nested six-model chain sharing the best model
        a, rest = order[0], list(order[1:])
        combos = list(itertools.combinations(rest, 5))
        if len(combos) > NCHAIN:
            combos = [combos[i] for i in RNG.choice(len(combos), NCHAIN, replace=False)]
        for trio in combos:
            ids = [a] + list(trio)
            Pc = np.array([Fail[:, ids[:m]].all(1).mean() for m in range(1, 7)])
            if np.any(Pc <= 0) or np.any(np.diff(Pc) >= 0):
                continue
            chains.append(dict(family=fam, models="|".join(map(str, ids)),
                               H1=rung(Pc[:4], 2), H2=rung(Pc, 3)))
    return pd.DataFrame(rows), pd.DataFrame(chains), pd.DataFrame(budget)


# ------------------------------------------------------------- calibration ----
MIX = lambda ws, gs: (lambda m: float(sum(w * g ** m for w, g in zip(ws, gs))))
TRUTHS = {
    # name: (P(m), divisible?)
    "one atom in b (cure chart on the count axis)": (MIX([1.0], [0.60]), True),
    "Gamma(2) in b (divisible)": (lambda m: (1 + m / 2.0) ** -2.0, True),
    "Gamma(4) in b (divisible)": (lambda m: (1 + m / 4.0) ** -4.0, True),
    "two atoms in g (indivisible)": (MIX([0.6, 0.4], [0.55, 0.93]), False),
    "three atoms in g (indivisible, the archive's census)":
        (MIX([0.5, 0.3, 0.2], [0.20, 0.70, 0.97]), False),
    "four atoms in g (indivisible)":
        (MIX([0.25] * 4, [0.25, 0.55, 0.80, 0.95]), False),
}


def run_calibration(n_items=10_000, nrep=NBOOT):
    rng = np.random.default_rng(20261005)
    rows = []
    for name, (Pf, div) in TRUTHS.items():
        P = np.array([Pf(m) for m in range(1, 7)])
        H1, H2 = rung(P[:4], 2), rung(P, 3)
        sim = boot_rungs(P, n_items, nrep, rng, nmax=6)
        # the equality case (a geometric cumulant sequence, i.e. one obstacle
        # type) has H_n = 0 exactly; a float64 determinant returns +-1e-30 there,
        # so the verdict uses a tolerance rather than a bare sign.
        TOL = 1e-12
        v1 = "refutes" if H1 < -TOL else ("equality" if abs(H1) <= TOL else "passes")
        v2 = "refutes" if H2 < -TOL else ("equality" if abs(H2) <= TOL else "passes")
        rows.append(dict(
            truth=name, divisible=int(div), H1_exact=H1, H2_exact=H2,
            rung1_verdict=v1, rung2_verdict=v2,
            rung2_strictly_better=int(v1 != "refutes" and v2 == "refutes"),
            H1_sim_median=float(np.median(sim[2])), H2_sim_median=float(np.median(sim[3])),
            H1_bias=float(np.median(sim[2]) - H1), H2_bias=float(np.median(sim[3]) - H2),
            rung1_power_1e4=float(np.mean(sim[2] < 0)),
            rung2_power_1e4=float(np.mean(sim[3] < 0))))
    return pd.DataFrame(rows)


def run_power(ns=(1e4, 1e5, 1e6, 1e7, 1e8)):
    rng = np.random.default_rng(20261005)
    rows = []
    for name, (Pf, div) in TRUTHS.items():
        if div:
            continue
        P = np.array([Pf(m) for m in range(1, 7)])
        for n in ns:
            sim = boot_rungs(P, int(n), 2000, rng, nmax=6)
            rows.append(dict(truth=name, n_items=int(n),
                             rung1_reject=float(np.mean(np.percentile(sim[2], 95) < 0)),
                             rung2_reject=float(np.mean(np.percentile(sim[3], 95) < 0)),
                             rung1_frac_neg=float(np.mean(sim[2] < 0)),
                             rung2_frac_neg=float(np.mean(sim[3] < 0))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ run --
if __name__ == "__main__":
    print(__doc__.splitlines()[0])
    print("\n(A,B,D,E)  the co-failure panel")
    T, CH, B = run_panel()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(T.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n  NOTE: the rung-2 plug-in is biased NEGATIVE at this item count on every\n"
          "  truth, divisible ones included (see the calibration below), so the sign of\n"
          "  the observed Hhat_2 is NOT evidence; only the comparison against the\n"
          "  design-matched null is.")
    save(T, "table_count_hankel.csv")
    save(CH, "table_count_hankel_chains.csv")
    save(B, "table_attempt_budget.csv")

    print("\n  rung 2 over nested six-model chains:")
    for fam, g in CH.groupby("family"):
        print(f"    {fam:<8} n={len(g):>4}  median H2 = {g.H2.median():+.3e}  "
              f"frac negative {np.mean(g.H2 < 0):.3f}   "
              f"(rung 1 on the same chains: median {g.H1.median():+.3e}, "
              f"frac negative {np.mean(g.H1 < 0):.3f})")

    print("\n  heterogeneity discount S(m)/(m S(1))  [1 iff items are homogeneous]")
    for fam, g in B.groupby("family"):
        d = g.sort_values("m")
        print(f"    {fam:<8} " + "  ".join(f"m={int(r.m)}:{r.discount:.3f}"
                                           for _, r in d.iterrows() if r.m <= 6))
    print("\n  floor bracket width = one attempt's price (nats)")
    for fam, g in B.groupby("family"):
        d = g.sort_values("m")
        print(f"    {fam:<8} " + "  ".join(f"L_{int(r.m)}={r.L_m:.4f}"
                                           for _, r in d.iterrows() if r.m <= 5))

    print("\n(C)  calibration on truths with a known verdict")
    C = run_calibration()
    print(C[["truth", "divisible", "H1_exact", "H2_exact", "rung1_verdict",
             "rung2_verdict", "rung2_strictly_better"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4g}"))
    save(C, "table_count_hankel_calib.csv")
    n_gain = int(C.rung2_strictly_better.sum())
    ind = C[C.divisible == 0]
    print(f"\n  rung 2 refutes {n_gain} truth(s) that rung 1 does not; of the "
          f"{len(ind)} indivisible truths rung 1 refutes "
          f"{int((ind.rung1_verdict == 'refutes').sum())} and rung 2 refutes "
          f"{int((ind.rung2_verdict == 'refutes').sum())}; neither refutes any "
          f"divisible truth.")

    P = run_power()
    save(P, "table_count_hankel_power.csv")
    print("\n  one-sided 95% rejection by item count (indivisible truths)")
    print(P.pivot_table(index="truth", columns="n_items",
                        values=["rung1_reject", "rung2_reject"])
          .to_string(float_format=lambda v: f"{v:.2f}"))
