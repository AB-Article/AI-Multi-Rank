#!/usr/bin/env python3
"""THE INVARIANT IS AN INVERSE CHANNEL COUNT.
   (Section "Law III", Prop. 'arcweights'; Section "Reading a frontier without
   fitting it", Prop. 'neff' and the operating rule.)

The gauge-invariant w = kappa/s^2 is the squared coefficient of variation of the
frailty posterior.  For a Gamma(k) spectrum -- the difficulty of a task built
from k independent exponential subtasks -- it is 1/k at EVERY point, so

    N_eff := 1/w = s^2/kappa

is the effective number of independent difficulty channels still active, and
every statement of Laws I-III is a statement about it:

  N1  N_eff = k exactly for Gamma(k).
  N2  R = ln[(1+w)/(2 sqrt w)] = ln cosh( (1/2) ln N_eff ) -- so the regret
      depends on |ln N_eff| alone, and the operating rule is
          OPERATE WHERE |ln N_eff| IS SMALLEST,
      which reduces to w=1 when w=1 is attainable and remains well posed when
      it is not (script 90).  It also identifies the collapse coordinate of the
      covering law as ln w: one doubling costs ln cosh(ln2 / 2) = 0.05889 nats.
  N3  The floorless chart conserves S w = S*, so N_eff = S/S* : ONE NEW
      EFFECTIVE CHANNEL PER S* = (1-beta)/beta NATS of log-error.  S* is the
      reportable form of the shape index: it is the depth of the knee.
  N4  Karamata: if mu[0,a] ~ C a^theta then eps ~ C Gamma(theta+1) x^{-theta}
      and N_eff -> theta.  Hence the complete trichotomy of late behaviour:
          atom at 0        -> floor,            N_eff -> 0
          edge index theta -> eps ~ x^{-theta}, N_eff -> theta
          gap a_min > 0    -> eps ~ e^{-a_min x}, N_eff -> infinity
      so the scaling exponent of a power-law frontier IS an effective channel
      count, and the floor question is the theta = 0 case of an edge exponent.
  N5  The arc with its weights.  d phi = sqrt(w) dS = sqrt(kappa) dx =
      d theta / sqrt(theta(1-theta)), so one COMPLETE two-atom transition is
      exactly pi for any separation and any weights; for K separated atoms
      ordered by increasing rate, with P_k = p_1+...+p_k,

          L = pi + sum_{k=2}^{K-1} 2 arcsin sqrt( q_k )  <=  (K-1) pi ,

      where q_k is the share atom k holds within the accumulated slower group
      AT THE k-th CROSSOVER.  The eight cases below have rates separated by
      1e4 to 1e6, which puts every crossover at the origin, so there
      q_k = p_k / P_k with p_k = c_k / sum_j c_j and P_k = p_1+...+p_k, and
      that is the form this script implements.  It is NOT the general form:
      p_k/P_k is not gauge-invariant (x -> x+Delta leaves L alone and moves it),
      so away from this regime the crossover shares must be used.
      Script 160 separates the two readings.  This makes Prop. 'path'(iii)'s
      qualitative "the corner is cut" exact in the separated-rate limit.
      Consequences: the number of modes obeys K >= 1 + L/pi, a LOWER bound; and
      L < pi certifies that exactly two modes are resolved, since the K=3
      minimum exceeds pi.

Outputs: results/table_neff_identities.csv
         results/table_arc_weights.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save


def jets(a, p, xs):
    a = np.asarray(a, float); lp = np.log(np.asarray(p, float))
    l = lp[None, :] - a[None, :] * np.asarray(xs, float)[:, None]
    Z = logsumexp(l, axis=1); wt = np.exp(l - Z[:, None])
    s = (wt * a).sum(1); kap = (wt * (a - s[:, None]) ** 2).sum(1)
    return -Z, s, kap


def arc_numeric(a, p, n=900000):
    """int sqrt(kappa) dx over the whole axis, log grid on both sides."""
    a = np.asarray(a, float); hi = 1e7 / a[a > 0].min(); lo = 1e-7 / a.max()
    xs = np.concatenate([-np.logspace(np.log10(hi), np.log10(lo), n // 2),
                         np.logspace(np.log10(lo), np.log10(hi), n // 2)])
    _, _, kap = jets(a, p, xs)
    return float(np.trapezoid(np.sqrt(np.maximum(kap, 0)), xs))


def arc_formula(p):
    p = np.asarray(p, float); P = np.cumsum(p)
    return float(np.pi + sum(2 * np.arcsin(np.sqrt(p[k] / P[k])) for k in range(1, len(p) - 1)))


def run():
    rows = []
    # ------------------------------------------------------------- N1 ------
    for k in (1, 2, 3, 5, 8):
        th = 0.7
        x = np.array([0.3, 2.0, 9.0, 40.0])
        s = k * th / (1 + th * x); kap = k * th ** 2 / (1 + th * x) ** 2
        rows.append(dict(block="N1 Gamma", test=f"Gamma(k={k}): N_eff = s^2/kappa at four points",
                         value=float(np.max(np.abs(s ** 2 / kap - k)))))
    # ------------------------------------------------------------- N2 ------
    w = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0])
    R1 = np.log((1 + w) / (2 * np.sqrt(w))); R2 = np.log(np.cosh(0.5 * np.log(w)))
    rows.append(dict(block="N2 regret", test="max |ln[(1+w)/(2 sqrt w)] - ln cosh(ln w / 2)|",
                     value=float(np.max(np.abs(R1 - R2)))))
    rows.append(dict(block="N2 regret", test="R at one doubling of w (= the covering tolerance)",
                     value=float(np.log(np.cosh(0.5 * np.log(2))))))
    rows.append(dict(block="N2 regret", test="R is even in ln w: max |R(w) - R(1/w)|",
                     value=float(np.max(np.abs(R1 - np.log((1 + 1 / w) / (2 * np.sqrt(1 / w))))))))
    # ------------------------------------------------------------- N3 ------
    for beta in (0.15, 0.25, 0.5, 0.9):
        al = 1.3; x = np.array([0.5, 2.0, 8.0, 40.0, 500.0])
        S = (al * x) ** beta
        s = beta * al * (al * x) ** (beta - 1)
        kap = beta * (1 - beta) * al ** 2 * (al * x) ** (beta - 2)
        rows.append(dict(block="N3 conserved",
                         test=f"beta={beta}: max |S w - (1-beta)/beta|  (S* = nats per channel)",
                         value=float(np.max(np.abs(S * kap / s ** 2 - (1 - beta) / beta)))))
        rows.append(dict(block="N3 conserved", test=f"beta={beta}: S* = (1-beta)/beta nats",
                         value=float((1 - beta) / beta)))
    # ------------------------------------------------------------- N4 ------
    for th in (0.5, 1.0, 2.0, 3.0):
        # exact discretisation of mu[0,a] = a^theta on a log grid
        edge = np.logspace(-12, 0, 300000)
        aa = np.sqrt(edge[1:] * edge[:-1]); dens = np.diff(edge ** th); dens = dens / dens.sum()
        _, s, kap = jets(aa, dens, [2e4])
        rows.append(dict(block="N4 Karamata",
                         test=f"edge index theta={th}: N_eff at x=2e4 (predicted {th})",
                         value=float(s[0] ** 2 / kap[0])))
    for name, a, p in [("atom at 0 {0,1}", [0., 1.], [.3, .7]),
                       ("gap {0.3,1}", [.3, 1.], [.5, .5])]:
        _, s, kap = jets(a, p, [1., 10., 100.])
        rows.append(dict(block="N4 Karamata", test=f"{name}: N_eff at x=1,10,100",
                         value=float(s[-1] ** 2 / kap[-1])))
    save(pd.DataFrame(rows), "table_neff_identities.csv")
    print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------- N5 ------
    cases = [("K=2 equal", [1., 1e6], [.5, .5]),
             ("K=2 skew .1/.9", [1., 1e6], [.1, .9]),
             ("K=3 equal", [1., 1e6, 1e12], [1 / 3.] * 3),
             ("K=3 p=(.6,.3,.1)", [1., 1e6, 1e12], [.6, .3, .1]),
             ("K=3 p=(.05,.9,.05)", [1., 1e6, 1e12], [.05, .9, .05]),
             ("K=4 equal", [1., 1e5, 1e10, 1e15], [.25] * 4),
             ("K=4 geometric", [1., 1e5, 1e10, 1e15], [.01, .09, .3, .6]),
             ("K=5 equal", [1., 1e4, 1e8, 1e12, 1e16], [.2] * 5)]
    arows = []
    for name, a, p in cases:
        Ln, Lf = arc_numeric(a, p), arc_formula(p)
        arows.append(dict(case=name, K=len(a), L_numeric_over_pi=Ln / np.pi,
                          L_formula_over_pi=Lf / np.pi, bound_K_minus_1=len(a) - 1,
                          modes_lower_bound=1 + Ln / np.pi))
    A = pd.DataFrame(arows)
    save(A, "table_arc_weights.csv")
    print()
    print(A.to_string(index=False))


if __name__ == "__main__":
    run()
