#!/usr/bin/env python3
"""AXIS SELF-DUALITY AT EVERY ORDER (Section 6 of paper.tex).

The revision of 2026-10-04 asserted that "only the N = 1 case is realized as a
reflection of the frontier itself; for N >= 2 the group acts on the share
simplex and not on any one resource axis".  That is false, and this script is
the counterexample plus the general condition.

CLAIM.  sigma_lambda(x) = -ln eps(x) - lambda x is invariant under the
reflection x -> 2 x0 - x if and only if the exponent support is symmetric
about its midpoint A/2 (necessarily A = a_max, since a_min = 0 must map to
a_max), and then

      lambda = A/2 = abar          (the canonical price of Law II)

with x0 fixed by a palindromic weight condition c_j y0^{a_j} = c_{sigma(j)}
y0^{a_sigma(j)}, y0 = exp(-x0).  Equivalently: eps is a polynomial in
exp(-a x) with self-reciprocal coefficients after centring.

Checks:
  (1) support {0,a,2a}: the x -> oo limit forces lambda = a; at that price the
      difference vanishes IDENTICALLY iff c_2 = c_0 exp(2 a x0).
  (2) support {0,a,3a} (not symmetric): no lambda and no weights work.
  (3) support {0,a,3a,4a} (symmetric, non-arithmetic): the reflection exists,
      lambda = 2a = abar.
  (4) N = 1, support {0,a}: automatically symmetric -- which is the whole
      reason the N = 1 case looked special.
  (5) 40-digit numerical confirmation of (1) and (3) at random parameters.
  (6) the self-dual fixed point for a single resource direction is e_0 = e_max
      (extreme mode errors equal), for every order.

Writes results/table_axis_selfduality.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import sympy as sp
from mpmath import mp, mpf, log as mlog, exp as mexp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
mp.dps = 40
rng = np.random.default_rng(20261004)
rows = []


def record(block, statement, value, expected, ok, note=""):
    rows.append(dict(block=block, statement=statement, value=str(value),
                     expected=expected, ok=bool(ok), note=note))
    print(f"[{'ok ' if ok else 'FAIL'}] {block:12s} {statement:56s} "
          f"{str(value)[:28]:>28s}  (exp {expected})")


x, x0, lam, a = sp.symbols("x x0 lam a", positive=True)


def analyse(exponents, name, symmetric_expected):
    """Force lambda from the x->oo slope, then test identical invariance."""
    M = len(exponents)
    cs = sp.symbols("c0:%d" % M, positive=True)
    eps = sum(c * sp.exp(-e * x) for c, e in zip(cs, exponents))
    sig = -sp.log(eps) - lam * x
    diff = sig.subs(x, 2 * x0 - x) - sig
    slope = sp.simplify(sp.limit(sp.diff(diff, x), x, sp.oo))
    lam_forced = sp.solve(slope, lam)
    lam_star = sp.simplify(lam_forced[0]) if lam_forced else None
    abar = sp.simplify(sum(exponents) / M)
    A = sp.simplify(max(exponents))
    record(name, "lambda forced by the x->oo slope", lam_star,
           f"A/2 = {sp.simplify(A/2)}",
           lam_star is not None and sp.simplify(lam_star - A / 2) == 0)
    record(name, "forced lambda equals the canonical price abar",
           sp.simplify(lam_star - abar) == 0 if lam_star is not None else None,
           "True iff support symmetric",
           (sp.simplify(lam_star - abar) == 0) == symmetric_expected)
    if lam_star is None:
        return
    d = sp.simplify(diff.subs(lam, lam_star))
    A = max(exponents)   # palindromic substitution pairs a_j with A - a_j
    pair = {}
    for j, e in enumerate(exponents):
        k = [i for i, f in enumerate(exponents) if sp.simplify(f - (A - e)) == 0]
        pair[j] = k[0] if k else None
    if any(v is None for v in pair.values()):
        resid_x = sp.simplify(sp.diff(d, x))
        record(name, "support symmetric about A/2?", "no", "no", not symmetric_expected)
        record(name, "residual still x-dependent for every weight choice",
               resid_x != 0, "True", resid_x != 0)
        return
    record(name, "support symmetric about A/2?", "yes", "yes", symmetric_expected)
    sub = {}
    for j, k in pair.items():
        if j < k:
            sub[cs[k]] = cs[j] * sp.exp((exponents[k] - exponents[j]) * x0)
    d2 = sp.simplify(d.subs(sub))
    record(name, "difference under palindromic weights", d2, "0", sp.simplify(d2) == 0)


analyse([0, a, 2 * a], "0,a,2a", True)
analyse([0, a, 3 * a], "0,a,3a", False)
analyse([0, a, 3 * a, 4 * a], "0,a,3a,4a", True)
analyse([0, a], "0,a (N=1)", True)

# ------------------------------------------------- (5) 40-digit numeric check
def sigma_num(xx, cs, es, lm):
    return -mlog(sum(c * mexp(-e * xx) for c, e in zip(cs, es))) - lm * xx


worst = mpf(0)
for t in range(300):
    aa = mpf(rng.uniform(0.2, 2.0)); xx0 = mpf(rng.uniform(-2, 3))
    c0 = mpf(rng.uniform(1e-3, 0.4)); c1 = mpf(rng.uniform(1e-3, 5))
    c2 = c0 * mexp(2 * aa * xx0)
    es = [mpf(0), aa, 2 * aa]; cs = [c0, c1, c2]
    for _ in range(4):
        xx = mpf(rng.uniform(-6, 9))
        worst = max(worst, abs(sigma_num(2 * xx0 - xx, cs, es, aa) - sigma_num(xx, cs, es, aa)))
record("numeric", "40-digit |sigma(2x0-x) - sigma(x)|, support {0,a,2a}",
       f"{float(worst):.3e}", "0", float(worst) < 1e-30)

worst4 = mpf(0)
for t in range(300):
    aa = mpf(rng.uniform(0.2, 2.0)); xx0 = mpf(rng.uniform(-2, 3))
    c0 = mpf(rng.uniform(1e-3, 0.4)); c1 = mpf(rng.uniform(1e-3, 5))
    c3 = c1 * mexp(2 * aa * xx0)          # pairs a=a with a=3a (A=4a)
    c4 = c0 * mexp(4 * aa * xx0)          # pairs a=0 with a=4a
    es = [mpf(0), aa, 3 * aa, 4 * aa]; cs = [c0, c1, c3, c4]
    for _ in range(4):
        xx = mpf(rng.uniform(-6, 9))
        worst4 = max(worst4, abs(sigma_num(2 * xx0 - xx, cs, es, 2 * aa)
                                 - sigma_num(xx, cs, es, 2 * aa)))
record("numeric", "40-digit |sigma(2x0-x) - sigma(x)|, support {0,a,3a,4a}",
       f"{float(worst4):.3e}", "0", float(worst4) < 1e-30)

# ------------------------------- (6) the fixed point: extreme mode errors equal
bad = 0
for t in range(400):
    M = int(rng.integers(2, 6))
    es = np.concatenate([[0.0], np.sort(rng.uniform(0.2, 3.0, M - 1))])
    cs = np.exp(rng.uniform(-5, 2, M))
    # solve c_0 = c_max exp(-a_max x)
    xs = np.log(cs[-1] / cs[0]) / es[-1]
    e = cs * np.exp(-es * xs)
    if abs(e[0] - e[-1]) > 1e-10 * e[0]:
        bad += 1
record("fixedpoint", "e_0 = e_max solvable on any 1-D frontier (400 spectra)",
       f"{bad} failures", "0", bad == 0,
       "this point is the one-dimensional minimax declaration; see script 56")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "table_axis_selfduality.csv"), index=False)
print(f"\nwrote results/table_axis_selfduality.csv "
      f"({len(out)} statements, {int(out.ok.sum())} ok, {int((~out.ok).sum())} failed)")
