#!/usr/bin/env python3
"""THE GAUGE PRINCIPLE AND THE INVARIANT CALCULUS.
   (Sections "Reading a frontier without fitting it" and "The floor is not a
   parameter" of paper.tex; Prop. 'gauge', Prop. 'chartode', Thm. 'monofloor'.)

A resource axis is defined only up to its unit and its origin, x -> lam x + c.
The error is not: eps = 1 is chance level, so S = -ln eps is absolute.  Under
the gauge group,

    S -> S ,   s -> s/lam ,   kappa -> kappa/lam^2 ,   kappa' -> kappa'/lam^3 ,

so the gauge-invariant local data of a frontier is

    S          (nats of log-error, absolute)
    w  = kappa / s^2                 = Var(a|X>x)/E[a|X>x]^2  = squared CV
    w2 = dln kappa / dS = kappa'/(s kappa)
    dw/dS = w (w2 + 2 w)   ... and so on.

alpha is the UNIT of the axis.  It is therefore not gauge-invariant and cannot
appear in any invariant statement: there is no derivation of alpha from the
axioms (T0).  The same argument explains why pi is universal and the spectral
width is not: int sqrt(kappa) dx = int sqrt(w) dS is invariant, int kappa dx
= a_max - a_min is the unit itself.

In the invariant coordinates each chart of the theory is a FIRST-ORDER ODE for
one function w(S), and "three parameters" = two gauge + one integration
constant:

    T1  floor + one reducible channel :  w2 = 1 - w     <=>  dw/dS = w(1+w)
                                          integrates to p_0 = eps_inf/eps
    T2  floorless stable (Hougaard)   :  w2 = -2w - 1/S <=>  dw/dS = -w/S
                                          integrates to w = (1-beta)/(beta S)

and the universal law that bounds every frontier is

    T3  kappa' = -mu_3, the third central moment of the frailty posterior;
        among measures on [0,inf) with given mean and variance mu_3 is
        minimised by the lower principal representation (two atoms, one pinned
        at a = 0), so for EVERY completely monotone frontier

            w2 <= 1 - w   <=>   dw/dS <= w(1+w)   <=>   d/dx [ kappa eps /
            (s^2 + kappa) ] <= 0 ,

        with equality iff the frontier is floor + exactly one channel.  The
        two-atom chart is the ENVELOPE of the admissible region, which is why a
        fitted floor can only err on one side.

    T4  Corollary (ordering): at the canonical point w = 1 the bound gives
        kappa' <= 0, so the curvature maximum never comes AFTER the canonical
        point, and the two coincide iff the frontier is two-atom.

    T5  Corollary (bound): eps_inf <= kappa eps/(s^2+kappa) pointwise, and the
        right-hand side is non-increasing in resource, so the floor is the
        infimum of a measured monotone sequence.  Script 81 measures how much
        tighter this is than eps(x_0+X).

Outputs: results/table_invariant_identities.csv
         results/table_inequality_stress.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save

RNG = np.random.default_rng(20261005)
mp.mp.dps = 60


# --------------------------------------------------------------- exact jets --
def jets(a, c, x):
    """(S, eps, s, kappa, kappa') from the spectrum, by posterior moments."""
    M = -np.outer(x, a) + np.log(c)[None, :]
    mx = M.max(1, keepdims=True)
    W = np.exp(M - mx); Z = W.sum(1, keepdims=True); p = W / Z
    eps = np.exp(mx[:, 0]) * Z[:, 0]
    m1 = p @ a
    d = a[None, :] - m1[:, None]
    var = (p * d ** 2).sum(1); mu3 = (p * d ** 3).sum(1)
    return -np.log(eps), eps, m1, var, -mu3           # kappa' = -mu_3


def jets_mp(a, c, x):
    a = [mp.mpf(v) for v in a]; c = [mp.mpf(v) for v in c]; x = mp.mpf(x)
    w = [ci * mp.e ** (-ai * x) for ai, ci in zip(a, c)]
    Z = sum(w); p = [wi / Z for wi in w]
    m1 = sum(pi * ai for pi, ai in zip(p, a))
    var = sum(pi * (ai - m1) ** 2 for pi, ai in zip(p, a))
    mu3 = sum(pi * (ai - m1) ** 3 for pi, ai in zip(p, a))
    return m1, var, -mu3


# ------------------------------------------------------------- the two ODEs --
def chart_odes():
    """T1, T2 to machine precision on exact curves."""
    rows = []
    # T1: floor + one channel
    d1 = d1b = 0.0
    for _ in range(200):
        ei = RNG.uniform(1e-3, 0.3); c = RNG.uniform(0.05, 3.0)
        al = RNG.uniform(0.2, 2.5); x = RNG.uniform(-2, 8, 7)
        a = np.array([0.0, al]); cc = np.array([ei, c])
        S, eps, s, k, k1 = jets(a, cc, x)
        w = k / s ** 2; w2 = k1 / (s * k)
        d1 = max(d1, np.abs(w2 - (1 - w)).max())
        d1b = max(d1b, (np.abs(w * (w2 + 2 * w) - w * (1 + w))
                        / np.maximum(np.abs(w * (1 + w)), 1e-30)).max())
    rows += [dict(result="T1  floor+1 channel: w2 = 1 - w", max_deviation=d1),
             dict(result="T1' same, as dw/dS = w(1+w) [relative]", max_deviation=d1b)]
    # T2: floorless stable, by exact differentiation of exp(-(alpha u)^beta)
    d2 = d2b = 0.0
    for _ in range(200):
        be = RNG.uniform(0.1, 0.95); al = RNG.uniform(0.1, 3.0)
        u = RNG.uniform(0.2, 12.0, 7)
        A = (al * u) ** be                       # S
        s = be * A / u
        k = be * (1 - be) * A / u ** 2
        k1 = -be * (1 - be) * (2 - be) * A / u ** 3
        w = k / s ** 2; w2 = k1 / (s * k)
        d2 = max(d2, np.abs(w2 + 2 * w + 1 / A).max())
        d2b = max(d2b, (np.abs(w * (w2 + 2 * w) + w / A)
                        / np.maximum(np.abs(w / A), 1e-30)).max())
    rows += [dict(result="T2  floorless stable: w2 = -2w - 1/S", max_deviation=d2),
             dict(result="T2' same, as dw/dS = -w/S [relative]", max_deviation=d2b)]
    # the integrated forms
    d3 = d4 = 0.0
    for _ in range(200):
        ei = RNG.uniform(1e-3, 0.3); c = RNG.uniform(0.05, 3.0)
        al = RNG.uniform(0.2, 2.5); x = RNG.uniform(-2, 8, 7)
        S, eps, s, k, _ = jets(np.array([0., al]), np.array([ei, c]), x)
        w = k / s ** 2
        d3 = max(d3, np.abs(w / (1 + w) - ei / eps).max())
        be = RNG.uniform(0.1, 0.95); al = RNG.uniform(0.1, 3.0)
        u = RNG.uniform(0.2, 12.0, 7); A = (al * u) ** be
        s = be * A / u; k = be * (1 - be) * A / u ** 2
        d4 = max(d4, np.abs(k / s ** 2 - (1 - be) / (be * A)).max())
    rows += [dict(result="T1'' integral of T1 is p_0 = eps_inf/eps", max_deviation=d3),
             dict(result="T2'' integral of T2 is w = (1-beta)/(beta S)", max_deviation=d4)]
    # gauge invariance of (S, w, w2) and non-invariance of alpha
    dg = 0.0; dng = 0.0
    for _ in range(200):
        a = np.array([0.0, RNG.uniform(0.2, 2.5)]); c = np.array([RNG.uniform(1e-3, .3), 1.0])
        x = RNG.uniform(0, 6, 5); lam = RNG.uniform(0.2, 5.0); sh = RNG.uniform(-3, 3)
        S, e, s, k, k1 = jets(a, c, x)
        # x' = lam x + sh  is the same curve re-labelled: a' = a/lam, c' = c e^{a sh/lam}
        a2 = a / lam; c2 = c * np.exp(a * sh / lam); x2 = lam * x + sh
        S2, e2, s2, k2, k12 = jets(a2, c2, x2)
        dg = max(dg, abs(S - S2).max(),
                 abs(k / s ** 2 - k2 / s2 ** 2).max(),
                 abs(k1 / (s * k) - k12 / (s2 * k2)).max())
        dng = max(dng, abs(a[1] - a2[1]))
    rows += [dict(result="gauge: (S, w, w2) invariant under x -> lam x + c", max_deviation=dg),
             dict(result="gauge: alpha NOT invariant (max change, must be > 0)",
                  max_deviation=dng)]
    # T4: ordering at the canonical point
    viol = -np.inf
    for _ in range(300):
        K = int(RNG.integers(1, 5))
        a = np.sort(np.r_[0.0, RNG.uniform(0.1, 3.0, K)])
        c = RNG.random(K + 1) + 1e-3; c /= c.sum()
        x = np.linspace(0.01, 60, 4000)
        S, e, s, k, k1 = jets(a, c, x)
        ok = (s > 0) & (k > 0)
        w = k[ok] / s[ok] ** 2
        ww2 = k1[ok] / (s[ok] * k[ok])
        j = np.where(np.diff(np.sign(w - 1)) != 0)[0]
        for i in j:                      # linear interpolation to the w = 1 crossing
            f = (1 - w[i]) / (w[i + 1] - w[i])
            viol = max(viol, float(ww2[i] + f * (ww2[i + 1] - ww2[i])))
    rows.append(dict(result="T4  kappa'/(s kappa) at w=1 (must be <= 0)", max_deviation=viol))
    return pd.DataFrame(rows)


# --------------------------------------------- the universal inequality T3 --
def inequality_stress(n_trials=40000):
    """Stress the bound w2 <= 1 - w on random positive spectra."""
    worst = -np.inf; worst_case = None; npts = 0
    for _ in range(n_trials):
        K = int(RNG.integers(1, 8))
        a = np.sort(np.abs(RNG.standard_normal(K)) * RNG.choice([0.1, 1.0, 10.0]))
        if RNG.random() < 0.5:
            a[0] = 0.0
        c = RNG.random(K) + 1e-4; c /= c.sum()
        x = np.sort(RNG.random(8)) * RNG.choice([0.5, 3.0, 20.0, 200.0])
        S, eps, s, k, k1 = jets(a, c, x)
        ok = np.isfinite(k) & (k > 1e-10 * np.maximum(s, 1e-30) ** 2) & (s > 1e-12)
        if not ok.any():
            continue
        npts += int(ok.sum())
        w = k[ok] / s[ok] ** 2; w2 = k1[ok] / (s[ok] * k[ok])
        v = (w2 - (1 - w))
        i = int(np.argmax(v))
        if v[i] > worst:
            worst = float(v[i]); worst_case = (a.copy(), c.copy(), float(x[ok][i]))
    # recompute the worst float64 case at 60 digits
    a, c, xv = worst_case
    m1, var, k1 = jets_mp(a, c, xv)
    w = var / m1 ** 2; w2 = k1 / (m1 * var)
    exact = float(w2 - (1 - w))
    rows = [dict(test="random spectra points tested", value=float(npts)),
            dict(test="worst float64 value of w2-(1-w)", value=worst),
            dict(test="same case recomputed at 60 digits", value=exact),
            dict(test="=> float64 excess is cancellation (0 = equality case)",
                 value=float(abs(exact) < 1e-20))]
    # equality is exactly 'two atoms, one at 0'
    for name, a, c in [("{0,alpha}: floor + 1 channel", np.array([0., 1.3]), np.array([.1, .9])),
                       ("{0,a1,a2}: floor + 2 channels", np.array([0., .7, 2.]), np.array([.1, .4, .5])),
                       ("{a1,a2}: two atoms, no floor", np.array([.5, 2.]), np.array([.4, .6])),
                       ("Gamma(2) continuum, no floor", None, None),
                       ("Gamma(2) + floor", None, None)]:
        if a is None:
            ar = np.linspace(.01, 6, 500); cr = ar * np.exp(-2 * ar); cr /= cr.sum()
            if "floor" in name:
                a = np.r_[0., ar]; c = np.r_[.1, .9 * cr]
            else:
                a, c = ar, cr
        x = np.linspace(.3, 8, 6)
        S, eps, s, k, k1 = jets(a, c, x)
        w = k / s ** 2; w2 = k1 / (s * k); sl = (1 - w) - w2
        rows.append(dict(test=f"slack (1-w)-w2, min over window: {name}", value=float(sl.min())))
    # the equivalent monotonicity, tested directly
    bad = 0; tot = 0
    for _ in range(20000):
        K = int(RNG.integers(1, 8))
        a = np.sort(np.abs(RNG.standard_normal(K)) * RNG.choice([0.1, 1.0, 10.0]))
        if RNG.random() < 0.5:
            a[0] = 0.0
        c = RNG.random(K) + 1e-4; c /= c.sum()
        x = np.linspace(0, 1, 40) * RNG.choice([0.5, 3.0, 20.0])
        S, eps, s, k, k1 = jets(a, c, x)
        ok = np.isfinite(k) & (k > 1e-10 * np.maximum(s, 1e-30) ** 2) & (s > 1e-12)
        F = k[ok] * eps[ok] / (s[ok] ** 2 + k[ok])
        if F.size < 3:
            continue
        tot += 1
        if np.diff(F).max() > 1e-12 * max(F.max(), 1e-30):
            bad += 1
    rows += [dict(test="random spectra whose local floor ever increases", value=float(bad)),
             dict(test="random spectra tested for monotonicity", value=float(tot))]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(__doc__.split("Outputs")[0])
    print("== chart ODEs, gauge invariance, ordering ==")
    I = chart_odes(); print(I.to_string(index=False)); save(I, "table_invariant_identities.csv")
    print("\n== the universal inequality  w2 <= 1 - w ==")
    J = inequality_stress(); print(J.to_string(index=False)); save(J, "table_inequality_stress.csv")
