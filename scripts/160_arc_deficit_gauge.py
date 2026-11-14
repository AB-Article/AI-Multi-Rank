#!/usr/bin/env python3
"""The arc deficit eq:arcweights: the shares must be read at the crossover.

Why this script exists
----------------------
Revision 2026-10-09 found that Proposition "path"(iii)'s closed-form arc
deficit was stated with the RAW spectral weights,

    L(-inf, inf) = pi + sum_{k=2}^{N} 2 arcsin sqrt(p_k / P_k),
    p_k = c_k / sum_j c_j,      P_k = p_1 + ... + p_k,

and that this cannot be right, for a reason that needs no numerics: the arc
length L is invariant under the reparameterisation x -> x + Delta, which sends
c_j -> c_j exp(-a_j Delta), while p_k / P_k is not.  The right-hand side is
therefore not a functional of the frontier at all.

The repair is to read each share AT ITS OWN CROSSOVER.  Order the atoms by
increasing rate, let x_k solve

    c_{k+1} exp(-a_{k+1} x) = sum_{j<=k} c_j exp(-a_j x)

(the k-th crossover, where atom k+1 and the accumulated slower group carry
equal error), and put

    q_k = c_k exp(-a_k x_k) / sum_{j<=k} c_j exp(-a_j x_k).

Then  L = pi + sum_{k=2}^{N} 2 arcsin sqrt(q_k)  in the separated-rate limit
a_{k+1}/a_k -> infinity, and q_k is gauge-invariant by construction.

What is tested
--------------
T1  gauge dependence of the raw-weight form, and gauge invariance of q_k:
    one frontier, five shifts Delta.
T2  the raw form's error where it is largest: weights that move the crossovers
    far from the origin at fixed, well-separated rates.
T3  the separated-rate limit, with equal weights and exponents (0, 1, A):
    the A-sequence the Online Appendix quotes.
T4  the crossover form against direct quadrature on random 3- and 4-atom
    spectra with consecutive rates separated by >= 1e4.
T5  eight three-atom weight vectors at a rate ratio of 1e8, to confirm that
    once the rates are that separated the crossovers sit at the origin, where
    q_k = p_k/P_k and the two forms agree to 4e-8.  (These are NOT script 89's
    eight cases, which have K = 2..5 and rates separated by 1e4 to 1e6; on
    those the closed form's absolute discrepancy against quadrature is up to
    8.7e-4, reported in table_arc_weights.csv.  Both sets sit in the regime
    where the two readings coincide, which is why script 89 did not catch the
    gauge-dependence; T1 and T2 above are the configurations that do.)

Writes results/table_arc_deficit_gauge.csv
       results/table_arc_deficit_gauge_summary.csv
"""
import os, sys
import mpmath as mp
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import save                                          # noqa: E402

mp.mp.dps = 30
RNG = mp.mpf  # no randomness beyond the fixed lists below


# --------------------------------------------------------------- the three maps
def arc_over_pi(cs, as_):
    """L / pi by direct quadrature of int sqrt(-S'') dx, breakpointed at every
    pairwise crossover so the quadrature sees each kink."""
    eps = lambda x: sum(c * mp.e ** (-a * x) for c, a in zip(cs, as_))
    S = lambda x: -mp.log(eps(x))
    kap = lambda x: -mp.diff(S, x, 2)
    bps = set()
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if as_[j] != as_[i]:
                bps.add(mp.log(cs[j] / cs[i]) / (as_[j] - as_[i]))
    pts = sorted(bps)
    lo, hi = pts[0] - 120, pts[-1] + 120
    return mp.quad(lambda x: mp.sqrt(max(kap(x), mp.mpf(0))),
                   [lo] + pts + [hi]) / mp.pi


def raw_over_pi(cs, as_):
    """pi + sum 2 arcsin sqrt(p_k / P_k), the retired raw-weight form."""
    z = sorted(zip(as_, cs))
    p = [c for _, c in z]
    tot = sum(p)
    p = [v / tot for v in p]
    P, s = [], mp.mpf(0)
    for v in p:
        s += v
        P.append(s)
    N = len(p) - 1
    return (mp.pi + sum(2 * mp.asin(mp.sqrt(p[k - 1] / P[k - 1]))
                        for k in range(2, N + 1))) / mp.pi


def crossover_over_pi(cs, as_):
    """pi + sum 2 arcsin sqrt(q_k), the shares read at their own crossover."""
    z = sorted(zip(as_, cs))
    a = [x[0] for x in z]
    c = [x[1] for x in z]
    K = len(c)
    acc = lambda x, k: sum(c[j] * mp.e ** (-a[j] * x) for j in range(k + 1))
    out = mp.pi
    for k in range(1, K - 1):
        f = lambda x: mp.log(c[k + 1]) - a[k + 1] * x - mp.log(acc(x, k))
        xk = mp.findroot(f, mp.mpf(0))
        out += 2 * mp.asin(mp.sqrt(c[k] * mp.e ** (-a[k] * xk) / acc(xk, k)))
    return out / mp.pi


def row(label, test, cs, as_):
    q = arc_over_pi(cs, as_)
    r = raw_over_pi(cs, as_)
    x = crossover_over_pi(cs, as_)
    return dict(test=test, case=label,
                quadrature_L_over_pi=float(q),
                raw_weight_form=float(r),
                crossover_form=float(x),
                rel_err_raw=float(abs(q - r) / q),
                rel_err_crossover=float(abs(q - x) / q))


def main():
    rows = []

    # T1 -- one frontier, five gauges
    a = [mp.mpf(0), mp.mpf(1), mp.mpf(1000)]
    for D in [0, 1, 2, -2, 5]:
        cs = [mp.e ** (-v * D) for v in a]
        rows.append(row(f"a=(0,1,1000), x -> x+{D}", "T1 gauge", cs, a))

    # T2 -- crossovers driven far from the origin at fixed separated rates
    for cs, lab in [([mp.mpf(1), mp.mpf(1), mp.e ** (-5000)], "c=(1,1,e^-5000)"),
                    ([mp.mpf(1), mp.mpf(1), mp.e ** (5000)], "c=(1,1,e^+5000)"),
                    ([mp.mpf(1), mp.mpf(100), mp.mpf(1)], "c=(1,100,1)")]:
        rows.append(row(f"a=(0,1,1000), {lab}", "T2 weights", cs, a))

    # T3 -- the separated-rate limit, equal weights
    for A in [2, 10, 100, 1000, 10 ** 6]:
        aA = [mp.mpf(0), mp.mpf(1), mp.mpf(A)]
        rows.append(row(f"a=(0,1,{A}), equal weights", "T3 rate limit",
                        [mp.mpf(1)] * 3, aA))

    # T4 -- fixed pseudo-random separated spectra, 3 and 4 atoms
    quads = [([0, 1, 1e4], [1e-2, 1e1, 1e3]),
             ([0, 1, 1e5], [1e3, 1e-1, 1e2]),
             ([0, 1, 1e4, 1e8], [1, 1e2, 1e-2, 1e1]),
             ([0, 1, 1e5, 1e10], [1e-3, 1, 1e3, 1e-1]),
             ([0, 1, 1e4, 1e9], [1e2, 1e-1, 1e1, 1e-3])]
    for i, (aa, cc) in enumerate(quads):
        rows.append(row(f"separated spectrum #{i + 1}", "T4 separated",
                        [mp.mpf(v) for v in cc], [mp.mpf(v) for v in aa]))

    # T5 -- eight three-atom vectors deep in the separated-rate regime
    #        (script 89's own eight are different: K = 2..5, ratios 1e4-1e6)
    for w in [(1, 1, 1), (1, 3, 1), (2, 1, 1), (1, 1, 3),
              (3, 1, 1), (1, 2, 2), (2, 2, 1), (1, 1, 2)]:
        a8 = [mp.mpf(0), mp.mpf(1), mp.mpf(10) ** 8]
        rows.append(row(f"weights {w}, a=(0,1,1e8)", "T5 separated weights",
                        [mp.mpf(v) for v in w], a8))

    T = pd.DataFrame(rows)
    save(T.round(10), "table_arc_deficit_gauge.csv")

    g1 = T[T.test == "T1 gauge"]
    summ = [
        ("T1  distinct L from quadrature across five gauges (should be 1)",
         float(g1.quadrature_L_over_pi.round(9).nunique())),
        ("T1  distinct raw-weight values across the same five gauges",
         float(g1.raw_weight_form.round(9).nunique())),
        ("T1  distinct crossover values across the same five gauges",
         float(g1.crossover_form.round(9).nunique())),
        ("T1  max relative error, raw-weight form", float(g1.rel_err_raw.max())),
        ("T1  max relative error, crossover form", float(g1.rel_err_crossover.max())),
        ("T2  max relative error, raw-weight form", float(T[T.test == "T2 weights"].rel_err_raw.max())),
        ("T2  max relative error, crossover form", float(T[T.test == "T2 weights"].rel_err_crossover.max())),
        ("T3  relative error of the closed form at A = 2 (equal weights)",
         float(T[T.case == "a=(0,1,2), equal weights"].rel_err_crossover.iloc[0])),
        ("T3  relative error of the closed form at A = 1e6 (equal weights)",
         float(T[T.case == "a=(0,1,1000000), equal weights"].rel_err_crossover.iloc[0])),
        ("T4  max relative error, crossover form, separated spectra",
         float(T[T.test == "T4 separated"].rel_err_crossover.max())),
        ("T5  max relative error, raw-weight form, separated weights",
         float(T[T.test == "T5 separated weights"].rel_err_raw.max())),
        ("overall max relative error, crossover form, separated cases only",
         float(T[T.test.isin(["T1 gauge", "T2 weights", "T4 separated",
                              "T5 separated weights"])].rel_err_crossover.max())),
    ]
    S = pd.DataFrame([dict(quantity=k, value=v) for k, v in summ])
    save(S.round(10), "table_arc_deficit_gauge_summary.csv")

    print("=" * 78)
    print("THE ARC DEFICIT: THE SHARES MUST BE READ AT THE CROSSOVER")
    print()
    hdr = f"  {'case':38s} {'quadrature':>12s} {'raw':>12s} {'crossover':>12s}"
    for test in ["T1 gauge", "T2 weights", "T3 rate limit", "T4 separated",
                 "T5 separated weights"]:
        print(f"{test}")
        print(hdr)
        for _, r in T[T.test == test].iterrows():
            print(f"  {r.case:38s} {r.quadrature_L_over_pi:12.7f}"
                  f" {r.raw_weight_form:12.7f} {r.crossover_form:12.7f}")
        print()
    print("=" * 78)
    for k, v in summ:
        print(f"  {k:62s} {v:.3e}" if abs(v) < 1 else f"  {k:62s} {v:.6f}")
    print()
    print("  Read T1 first: the quadrature returns ONE value across the five")
    print("  gauges, the crossover form returns one, and the raw-weight form")
    print("  returns five.  T3 is the separated-rate limit the proposition is")
    print("  stated in; T5 shows that deep in that regime the two forms agree,")
    print("  which is why script 89's own check did not catch the gauge problem.")


if __name__ == "__main__":
    main()
