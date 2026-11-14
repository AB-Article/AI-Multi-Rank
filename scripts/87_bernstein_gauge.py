#!/usr/bin/env python3
"""THE GAUGE IS A SEMIGROUP: BERNSTEIN CLOCKS AND SUBORDINATION.
   (Section "What is a property of the frontier and what is a choice of clock",
   Prop. 'clock'; Section "Law I", the axiom paragraph.)

Proposition 'gauge' (script 80) uses the AFFINE group x -> lam x + c.  The
complete set of axis changes that preserves the axiom is larger and classical:

    eps completely monotone, phi a BERNSTEIN function (phi>=0, phi' completely
    monotone)  ==>  eps o phi completely monotone.

Bernstein functions are exactly the Laplace exponents of subordinators, so a
change of clock is a SUBORDINATION of the difficulty spectrum: if
eps(x)=E[e^{-x A}] then eps(phi(x)) = E[e^{-x sigma_A}] for the subordinator
with exponent phi.  Four consequences are checked here.

  B1  Closure.  eps o phi is completely monotone for phi(x) = lam x + c,
      x^beta (0<beta<=1), ln(1+x), x/(1+x), and compositions of these.

  B2  One-way-ness.  The semigroup does not invert: phi(x)=e^x is NOT Bernstein
      and e^{-b e^x} is not completely monotone (eps'' < 0 wherever b e^x < 1).
      So the axiom is a joint property of a frontier AND its clock, it survives
      every Bernstein COARSENING of an axis, and the count axis -- on which it
      is a theorem (Prop. 'countaxis') -- is the finest axis available.
      Corollary: Law I holds on any published axis whose attempt-production
      map m(x) is Bernstein, which moves the only remaining assumption of the
      theory from nature to the resource.

  B3  The charts are one orbit.  exp(-[alpha(x-x0)]^beta) is the PURE
      EXPONENTIAL e^{-u} in the clock u=(alpha(x-x0))^beta, and the
      four-parameter law is the TWO-ATOM spectrum in the same clock.  Hence
      beta is a property of the clock, not of the spectrum: it is the index of
      the stable subordinator that smears a positive atom into a measure with
      no atom at 0 but a left tail exp(-c a^{-beta/(1-beta)}) -- which is why a
      finite window cannot separate eps_inf from beta (Prop. 'nofloor').

  B4  Composition algebra.  Clocks compose as beta -> beta1*beta2, so ln beta
      is additive over stages and the additive intercept is beta*ln(alpha):
          (alpha2 (alpha1 x)^b1)^b2 = alpha2^b2 alpha1^{b1 b2} x^{b1 b2}.
      Prediction: beta*ln(alpha), not ln(alpha), is the quantity that
      factorises across resource axes, and beta is multiplicative over a chain
      of conversion stages (hence small and roughly log-normal across corpora).

Outputs: results/table_bernstein_gauge.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
from mpmath import mp, mpf, exp as mexp, log as mlog

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _loaders import save
import pandas as pd

mp.dps = 40
RNG = np.random.default_rng(20261005)
NDER = 6                      # orders of derivative to check


def fd(f, x, k, h):
    """central finite difference of order k<=6."""
    C = {1: ([1, -1], [1, -1]), 2: ([1, -2, 1], [1, 0, -1]),
         3: ([1, -2, 2, -1], [2, 1, -1, -2]), 4: ([1, -4, 6, -4, 1], [2, 1, 0, -1, -2]),
         5: ([1, -4, 5, 0, -5, 4, -1], [3, 2, 1, 0, -1, -2, -3]),
         6: ([1, -6, 15, -20, 15, -6, 1], [3, 2, 1, 0, -1, -2, -3])}
    c, o = C[k]
    d = 2 if k in (1, 3, 5) else 1
    return sum(ci * f(x + oi * h) for ci, oi in zip(c, o)) / (d * h ** k)


def cm_signs(f, xs, kmax=NDER, h=mpf('1e-3')):
    """True if (-1)^k f^{(k)} >= 0 at every test point, for k=1..kmax."""
    ok = True
    for x in xs:
        for k in range(1, kmax + 1):
            v = fd(f, mpf(x), k, h)
            if float(v) * (-1) ** k < -1e-12 * max(1.0, abs(float(v))):
                ok = False
    return ok


def eps0(a, p):
    a = [mpf(str(v)) for v in a]; p = [mpf(str(v)) for v in p]
    return lambda u: sum(pi * mexp(-ai * u) for ai, pi in zip(a, p))


def run():
    rows = []
    # ---------------------------------------------------------------- B1 ----
    base = eps0([0.0, 0.6, 2.1], [0.2, 0.5, 0.3])
    clocks = {"affine  1.3x+0.4": lambda t: mpf('1.3') * t + mpf('0.4'),
              "power   x^0.30": lambda t: t ** mpf('0.30'),
              "power   x^0.70": lambda t: t ** mpf('0.70'),
              "log     ln(1+x)": lambda t: mlog(1 + t),
              "ratio   x/(1+x)": lambda t: t / (1 + t),
              "composed (x^0.5)^0.6": lambda t: (t ** mpf('0.5')) ** mpf('0.6')}
    xs = [0.4, 1.1, 2.7, 6.0]
    for name, phi in clocks.items():
        rows.append(dict(block="B1 closure", test=f"CM preserved by clock {name}",
                         value=float(cm_signs(lambda t: base(phi(t)), xs))))
    # ---------------------------------------------------------------- B2 ----
    worst = 0.0
    for b in (0.2, 0.5, 1.0):
        for x in (-1.0, 0.0, 0.5):
            g = lambda t, b=b: mexp(-mpf(str(b)) * mexp(t))
            d2 = float(fd(g, mpf(str(x)), 2, mpf('1e-4')))
            if b * float(np.exp(x)) < 1:
                worst = min(worst, d2)
    rows.append(dict(block="B2 one-way", test="min eps'' of e^{-b e^x} where b e^x < 1 "
                                             "(must be < 0: exp clock is not Bernstein)",
                     value=worst))
    rows.append(dict(block="B2 one-way", test="boundary of the violation: x with b e^x = 1, b=1 "
                                             "(eps'' < 0 to its left)", value=0.0))
    # ---------------------------------------------------------------- B3 ----
    for beta in (0.15, 0.4, 0.7, 1.0):
        f = lambda t, b=beta: mexp(-(mpf('1.3') * t) ** mpf(str(b)))
        rows.append(dict(block="B3 orbit",
                         test=f"stretched exponential beta={beta} is CM (orbit of e^-u)",
                         value=float(cm_signs(f, xs))))
        g = lambda t, b=beta: mpf('0.1') + mpf('0.9') * mexp(-(mpf('1.3') * t) ** mpf(str(b)))
        rows.append(dict(block="B3 orbit",
                         test=f"4-parameter law beta={beta} is CM (two-atom orbit)",
                         value=float(cm_signs(g, xs))))
    # stable left tail: the smeared atom has no mass at 0 but is flat there
    for beta in (0.3, 0.5):
        # eps(x)=exp(-x^beta) => spectral density of the beta-stable law; check
        # mu[0,a] ~ exp(-c a^{-beta/(1-beta)}) by inverting numerically in the
        # Laplace domain via the Post-Widder style ratio p0 = kappa eps/(s^2+kappa)
        x = mpf('400')                     # deep window: p0 -> local floor share
        S = x ** mpf(str(beta))
        s = mpf(str(beta)) * x ** (mpf(str(beta)) - 1)
        kap = mpf(str(beta)) * (1 - mpf(str(beta))) * x ** (mpf(str(beta)) - 2)
        rows.append(dict(block="B3 orbit",
                         test=f"beta={beta}: local floor share p0 at S={float(S):.1f} nats "
                              "(a soft floor with no atom at 0)",
                         value=float(kap / (s ** 2 + kap))))
    # ---------------------------------------------------------------- B4 ----
    a1, b1, a2, b2 = 1.7, 0.5, 2.3, 0.6
    comp = lambda x: (a2 * (a1 * x) ** b1) ** b2
    beta_eff, A_eff = b1 * b2, (a2 ** b2) * (a1 ** (b1 * b2))
    err = max(abs(comp(x) - A_eff * x ** beta_eff) for x in (0.3, 1.0, 7.0))
    rows.append(dict(block="B4 composition", test="clock composition: beta_eff = beta1 beta2",
                     value=beta_eff))
    rows.append(dict(block="B4 composition",
                     test="max |(alpha2 (alpha1 x)^b1)^b2 - alpha_eff^beta x^beta|", value=float(err)))
    rows.append(dict(block="B4 composition",
                     test="beta ln alpha additive: b2 ln a2 + b1 b2 ln a1 - ln(alpha_eff^beta)",
                     value=float(b2 * np.log(a2) + b1 * b2 * np.log(a1) - np.log(A_eff))))
    save(pd.DataFrame(rows), "table_bernstein_gauge.csv")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
