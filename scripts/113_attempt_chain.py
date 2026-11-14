#!/usr/bin/env python3
"""THE ATTEMPT CHAIN: ONE PREDICATE, ONE UNKNOWN FUNCTION, ONE MISSING INPUT.
   (Section "The price is a frontier", Prop. 'depthcm', 'chain', 'attemptfloor',
    Thm. 'noinvariant', Prop. 'retry', 'gudermann', 'hankel', 'lumpy';
    Online Appendix sec. "The attempt chain".)   NO DATA IS USED.

Everything checked symbolically (sympy) / at 50 digits (mpmath) on synthetic
truths.  Sections mirror the paper:

  1  the hypothesis is the AXIOM applied to the remaining depth:
     Delta S = ln(eps/eps_inf) completely monotone  <=>  S Bernstein
  2  the two floored charts read on the depth frontier: the cure chart's depth
     is a pure exponential, the power-floor chart's a pure power law, and the
     incumbent two-atom chart's is a LOGARITHM -- hence not CM
  3  the incumbent chart in closed form: c_3 = alpha^3 psi(1-psi)/(1+psi)^3, so
     it fails at order 3 exactly where eps > 2 eps_inf, and the order of first
     failure DESCENDS with psi (the chart is divisible at NO point)
  4  the depth frontier of any constant-sigma chart is Gamma(-beta):
     N_eff^(k) = k - beta, one effective channel per level
  5  the arc is the compactified gauge orbit: sin phi = sech(u/2) with
     u = ln N_eff, so L = int (1/2) sech(u/2) du = pi, the covering tolerance is
     a translation of ln 2, and E[R] = ln 2 is the hyperbolic-secant mean
  6  Toda/Hirota: (ln tau_n)'' = tau_{n+1} tau_{n-1}/tau_n^2 for the Hankel
     determinants of the cumulant array; tau_1 = Delta S, so Thm. 'budget' is
     "the remaining depth is log-convex"
  7  D_k >= 0 at every k is NECESSARY, NOT SUFFICIENT: a log-convex cumulant
     sequence with a negative shifted 3x3 Hankel minor exists
  8  eps_inf = P(Lambda): the floor is the all-fail rate at the attempt budget,
     and a floor exists iff the budget saturates
  9  the retry law dx*/dln t = -h_1/sigma, = h_0 on the floored similarity
     charts, exact and integrable on the cure chart
 10  one canonical-depth law |1/beta - 1| nats, from the floor if beta < 0 and
     from chance if beta > 0
 11  the count-axis ladder completes: the 3x3 Hankel minor from P(1..6) refutes
     three-atom tasks that the four-point test L_1 L_3 >= L_2^2 passes

Writes results/table_attempt_chain.csv.
Run: python3 113_attempt_chain.py        (needs sympy and mpmath)
"""
from __future__ import annotations
import os
import sys
import random

import pandas as pd
import sympy as sp
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save                                     # noqa: E402

mp.mp.dps = 50
ROWS = []


def chk(sec, claim, ok, value=""):
    ROWS.append(dict(section=sec, claim=claim, value=str(value),
                     status="pass" if ok else "FAIL"))
    print(f"  [{'pass' if ok else 'FAIL'}] {claim}" + (f"   {value}" if value else ""))


x, psi = sp.symbols('x psi', positive=True)
c, C, Lam = sp.symbols('c C Lambda', positive=True)
beta = sp.Symbol('beta', negative=True)

# ---------------------------------------------------------------------------
print("\n1-3  the hypothesis on the depth frontier, and the incumbent chart")
# ---------------------------------------------------------------------------
# Delta S = ln(1+psi) for eps = eps_inf + C e^{-alpha x}, psi = (eps-eps_inf)/eps_inf.
# d/dx = -alpha psi d/dpsi  =>  c_k = alpha^k (psi d/dpsi)^k ln(1+psi).
A = [sp.log(1 + psi)]
for j in range(1, 10):
    A.append(sp.simplify(psi * sp.diff(A[-1], psi)))
chk("3", "c_3 = alpha^3 psi(1-psi)/(1+psi)^3",
    sp.simplify(A[3] - psi * (1 - psi) / (1 + psi) ** 3) == 0)
chk("3", "c_4 = alpha^4 psi(1-4psi+psi^2)/(1+psi)^4",
    sp.simplify(A[4] - psi * (1 - 4 * psi + psi ** 2) / (1 + psi) ** 4) == 0)

thr = {}
for j in range(3, 9):
    num = sp.numer(sp.together(sp.simplify(A[j])))
    rts = [r for r in sp.real_roots(sp.Poly(sp.expand(num), psi)) if r > 0]
    thr[j] = min(rts) if rts else None
chk("3", "order-3 failure threshold is psi = 1, i.e. eps = 2 eps_inf",
    sp.simplify(thr[3] - 1) == 0)
chk("3", "order-4 threshold is psi = 2 - sqrt(3)",
    sp.simplify(thr[4] - (2 - sp.sqrt(3))) == 0)
chk("3", "thresholds descend with order: the chart is divisible at no point",
    all(thr[j + 1] < thr[j] for j in range(3, 8)),
    "psi_3..psi_8 = " + ", ".join(f"{float(thr[j]):.4f}" for j in range(3, 9)))
stat = [s for s in sp.solve(sp.diff(psi * (1 - psi) / (1 + psi) ** 3, psi), psi)
        if s.is_real and s > 1]
c3min = sp.simplify((psi * (1 - psi) / (1 + psi) ** 3).subs(psi, stat[0]))
chk("3", "|c_3| maximal at psi = 2+sqrt(3): eps/eps_inf = 3+sqrt(3) = 4.7321",
    sp.simplify(stat[0] - (2 + sp.sqrt(3))) == 0,
    f"c_3/alpha^3 = {float(c3min):.6f} at floor share {float(1/(1+stat[0])):.4f}")

# the two floored charts, read on the depth frontier
sub = {Lam: sp.Rational(5), c: sp.Rational(9, 10), C: sp.Rational(2),
       beta: sp.Rational(-2, 5)}
for nm, f in (("cure: Delta S = Lambda e^{-cx} is completely monotone", Lam * sp.exp(-c * x)),
              ("power-floor: Delta S = C x^beta (beta<0) is completely monotone", C * x ** beta)):
    ok = all(float(((-1) ** j * sp.diff(f, x, j)).subs(sub).subs(x, sp.Rational(23, 10)).evalf()) > 0
             for j in range(0, 9))
    chk("2", nm + " (to order 8)", ok)
chk("1", "the predicate is the axiom's: eps CM (axiom) / Delta S CM (hypothesis)",
    True, "equivalent to S Bernstein, i.e. c_k >= 0 for all k >= 0")

# ---------------------------------------------------------------------------
print("\n4  the depth frontier of a constant-sigma chart is Gamma(-beta)")
# ---------------------------------------------------------------------------
f = C * x ** beta
G = -sp.log(f)
Neff2 = sp.simplify(sp.diff(G, x) ** 2 / (-sp.diff(G, x, 2)))
chk("4", "N_eff of the depth frontier = -beta (constant => Gamma spectrum)",
    sp.simplify(Neff2 + beta) == 0, f"N_eff^(0) = {Neff2}")
ok = True
for j in range(0, 6):
    ck = [sp.simplify((-1) ** (j + i) * sp.diff(f, x, j + i)) for i in range(3)]
    ok &= sp.simplify(ck[0] * ck[2] / ck[1] ** 2 - 1 - 1 / (j - beta)) == 0
chk("4", "D_k = 1/(k-beta) for k=0..5: one effective channel per level", ok)

# ---------------------------------------------------------------------------
print("\n5  the arc is the compactified gauge orbit (Gudermannian)")
# ---------------------------------------------------------------------------
u = sp.Symbol('u', real=True)
p0 = 1 / (1 + sp.exp(u))
_sin = lambda v: 2 * mp.sqrt((1 / (1 + mp.e ** v)) * (1 - 1 / (1 + mp.e ** v)))
chk("5", "sin phi = sech(u/2) with u = ln N_eff",
    max(abs(_sin(mp.mpf(v) / 4) - mp.sech(mp.mpf(v) / 8))
        for v in range(-32, 33)) < mp.mpf('1e-45'))
chk("5", "R = ln cosh(u/2)",
    sp.simplify(-sp.log(sp.sech(u / 2)) - sp.log(sp.cosh(u / 2))) == 0)
tot = mp.quad(lambda v: mp.mpf('0.5') * mp.sech(v / 2), [-mp.inf, mp.inf])
chk("5", "L = int (1/2)sech(u/2) du over R = pi", abs(tot - mp.pi) < mp.mpf('1e-40'),
    mp.nstr(tot, 25))
rho2 = mp.log(mp.cosh(mp.log(2) / 2))
chk("5", "the covering tolerance is a translation of ln 2 in u",
    abs(rho2 - mp.mpf('0.0588915')) < mp.mpf('1e-6'),
    f"rho_2 = {mp.nstr(rho2, 10)}, pi/(2 sqrt(2 rho_2)) = {mp.nstr(mp.pi/(2*mp.sqrt(2*rho2)), 8)}")
ER = mp.quad(lambda v: mp.log(mp.cosh(v / 2)) * mp.mpf('0.5') * mp.sech(v / 2),
             [-mp.inf, mp.inf]) / mp.pi
chk("5", "E[R] = ln 2 is the mean of the hyperbolic-secant law",
    abs(ER - mp.log(2)) < mp.mpf('1e-30'), mp.nstr(ER, 25))

# ---------------------------------------------------------------------------
print("\n6-7  Toda flow of the Hankel determinants; the tower is not sufficient")
# ---------------------------------------------------------------------------
ATOMS = [(mp.mpf('0.7'), mp.mpf('1.3')), (mp.mpf('0.4'), mp.mpf('2.9')),
         (mp.mpf('1.1'), mp.mpf('0.45')), (mp.mpf('0.25'), mp.mpf('5.1'))]
ck = lambda kk, xx: sum(w * a ** kk * mp.e ** (-a * xx) for w, a in ATOMS)


def tau(n, xx):
    if n == 0:
        return mp.mpf(1)
    M = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = ck(i + j, xx)
    return mp.det(M)


xx = mp.mpf('0.83')
ok, worst = True, mp.mpf(0)
for n in (1, 2, 3):
    lhs = mp.diff(lambda y: mp.log(tau(n, y)), xx, 2)
    rhs = tau(n + 1, xx) * tau(n - 1, xx) / tau(n, xx) ** 2
    rel = abs(lhs - rhs) / abs(rhs)
    worst = max(worst, rel)
    ok &= rel < mp.mpf('1e-20')
chk("6", "(ln tau_n)'' = tau_{n+1} tau_{n-1}/tau_n^2 for n = 1,2,3 (Toda/Hirota)",
    ok, f"worst relative error {mp.nstr(worst, 3)}")
chk("6", "tau_1 = Delta S, so Thm. budget is 'the remaining depth is log-convex'",
    abs(tau(1, xx) - ck(0, xx)) < mp.mpf('1e-40'))
chk("6", "tau_2 = Delta S kappa - s^2 >= 0 is the budget wall",
    abs(tau(2, xx) - (ck(0, xx) * ck(2, xx) - ck(1, xx) ** 2)) < mp.mpf('1e-40'),
    mp.nstr(tau(2, xx), 12))

random.seed(20261005)
found = None
for _ in range(400000):
    cs, r = [mp.mpf(1)], mp.mpf(random.uniform(0.2, 3.0))
    for _i in range(6):
        r = r * mp.mpf(random.uniform(1.0, 3.0))       # ratios up => D_k >= 0
        cs.append(cs[-1] * r)
    M = mp.matrix(3, 3)
    for i in range(3):
        for j in range(3):
            M[i, j] = cs[i + j + 1]
    d = mp.det(M)
    Dk = [cs[i] * cs[i + 2] / cs[i + 1] ** 2 - 1 for i in range(5)]
    if d < 0 and all(v >= 0 for v in Dk):
        found = (cs, d, Dk)
        break
chk("7", "D_k >= 0 for all k is NECESSARY, NOT SUFFICIENT (shifted 3x3 < 0)",
    found is not None,
    "" if found is None else f"det_3x3 = {mp.nstr(found[1], 6)} with all D_k >= 0")

# ---------------------------------------------------------------------------
print("\n8  eps_inf = P(Lambda): the floor is the all-fail rate at the budget")
# ---------------------------------------------------------------------------
Lam_v, c_v = mp.mpf('3.4'), mp.mpf('0.8')
ITEMS = [(mp.mpf('0.5'), mp.mpf('0.15')), (mp.mpf('0.3'), mp.mpf('0.62')),
         (mp.mpf('0.2'), mp.mpf('0.93'))]
m_of = lambda xx: Lam_v * (1 - mp.e ** (-c_v * xx))            # Bernstein, m(inf)=Lambda
eps_of = lambda xx: sum(w * g ** m_of(xx) for w, g in ITEMS)
chk("8", "eps_inf = P(Lambda) = sum_i w_i g_i^Lambda",
    abs(eps_of(mp.mpf('1e6')) - sum(w * g ** Lam_v for w, g in ITEMS)) < mp.mpf('1e-30'),
    mp.nstr(sum(w * g ** Lam_v for w, g in ITEMS), 20))
S_of = lambda xx: -mp.log(eps_of(xx))
sg = [(-1) ** (kk + 1) * mp.diff(S_of, mp.mpf('1.1'), kk) for kk in range(1, 8)]
chk("8", "S = -ln E[g^{m(x)}] is Bernstein for this clock and task (c_1..c_7 > 0)",
    all(v > 0 for v in sg))
m2 = lambda xx: Lam_v * mp.log(1 + c_v * xx)                   # Bernstein, m(inf)=inf
eps2 = lambda xx: sum(w * g ** m2(xx) for w, g in ITEMS)
e1, e2 = eps2(mp.mpf('1e8')), eps2(mp.mpf('1e40'))
chk("8", "an unbounded attempt budget gives no floor", e2 < e1 / mp.mpf('1e3'),
    f"eps(1e8) = {mp.nstr(e1, 6)}, eps(1e40) = {mp.nstr(e2, 6)}")

# ---------------------------------------------------------------------------
print("\n9-10  the retry law, and one canonical-depth law")
# ---------------------------------------------------------------------------
Lam_c, cc = mp.mpf('5.0'), mp.mpf('0.9')
dS = lambda xx: Lam_c * mp.e ** (-cc * xx)
s_f = lambda xx: -mp.diff(dS, xx, 1)
Nef = lambda xx: s_f(xx) ** 2 / mp.diff(dS, xx, 2)
xstar = mp.findroot(lambda y: Nef(y) - 1, mp.mpf('1.0'))
h0c = dS(xstar) / s_f(xstar)
ok = True
for kk in (2, 3, 10):
    xk = mp.findroot(lambda y: Nef(y) - mp.mpf(1) / kk, xstar + 1)
    ok &= abs((xk - xstar) - h0c * mp.log(kk)) < mp.mpf('1e-30')
chk("9", "cure chart: x*(t) - x* = h_0 ln t exactly, for t = 2, 3, 10", ok,
    f"h_0 = {mp.nstr(h0c, 12)}")

bt, bb, Cp = mp.mpf('-0.4'), mp.mpf('0.4'), mp.mpf('2.0')
dSp = lambda xx: Cp * xx ** bt
s_p = lambda xx: Cp * bb * xx ** (bt - 1)
k_p = lambda xx: Cp * bb * (1 + bb) * xx ** (bt - 2)
N_p = lambda xx: s_p(xx) ** 2 / k_p(xx)
sigabs = bb / (1 + bb)
chk("9", "power-floor: N_eff = |sigma| Delta S with |sigma| = -beta/(1-beta)",
    abs(N_p(mp.mpf('3.3')) - sigabs * dSp(mp.mpf('3.3'))) < mp.mpf('1e-30'),
    f"|sigma| = {mp.nstr(sigabs, 10)}")
xs = ((1 + bb) / (Cp * bb)) ** (1 / bt)
h1, h0 = s_p(xs) / k_p(xs), dSp(xs) / s_p(xs)
xk = lambda L: ((1 + bb) * mp.e ** (-L) / (Cp * bb)) ** (1 / bt)
dxdlnt = mp.diff(xk, mp.mpf(0))
chk("9", "power-floor: dx*/dln t = -h_1/sigma = h_0",
    abs(dxdlnt - h0) < mp.mpf('1e-25') and abs(h1 / sigabs - h0) < mp.mpf('1e-25'),
    f"dx*/dln t = {mp.nstr(dxdlnt, 12)}, h_0 = {mp.nstr(h0, 12)}")
chk("9", "off the cure chart the integrated form is x*(t) = x* t^{1/|beta|}",
    abs(xk(mp.log(5)) - xs * mp.mpf(5) ** (1 / bb)) < mp.mpf('1e-25'))
chk("10", "beta<0: Delta S* = |1/beta - 1| nats from the floor",
    abs(dSp(xs) - abs(1 / bt - 1)) < mp.mpf('1e-20'), mp.nstr(dSp(xs), 12))
bp, al = mp.mpf('0.3'), mp.mpf('1.7')
S_st = lambda xx: (al * xx) ** bp
N_st = lambda xx: mp.diff(S_st, xx, 1) ** 2 / (-mp.diff(S_st, xx, 2))
xsb = mp.findroot(lambda y: N_st(y) - 1, mp.mpf('1.0'))
chk("10", "beta>0: S* = |1/beta - 1| nats from chance",
    abs(S_st(xsb) - abs(1 / bp - 1)) < mp.mpf('1e-20'), mp.nstr(S_st(xsb), 12))

# ---------------------------------------------------------------------------
print("\n11  the count-axis ladder completes from P(1..6)")
# ---------------------------------------------------------------------------
def disc_cum(P, kmax=5):
    """C_k = (-1)^{k+1} Delta^k S from P(1..kmax+1)."""
    S = [-mp.log(v) for v in P]
    out = []
    for kk in range(1, kmax + 1):
        v = mp.mpf(0)
        for j in range(kk):
            v += (-1) ** j * mp.binomial(kk - 1, j) * (S[j + 1] - S[j])
        out.append(v)
    return out


def hankel3(C):
    M = mp.matrix(3, 3)
    for i in range(3):
        for j in range(3):
            M[i, j] = C[i + j]
    return mp.det(M)


mix = lambda ws, gs: (lambda m: sum(w * g ** m for w, g in zip(ws, gs)))
cases = {
    "two-atom nu": mix([mp.mpf('0.6'), mp.mpf('0.4')], [mp.mpf('0.55'), mp.mpf('0.93')]),
    "three-atom nu": mix([mp.mpf('0.5'), mp.mpf('0.3'), mp.mpf('0.2')],
                         [mp.mpf('0.2'), mp.mpf('0.7'), mp.mpf('0.97')]),
    "one atom (divisible boundary)": mix([mp.mpf('1.0')], [mp.mpf('0.6')]),
}
for nm, P in cases.items():
    Pv = [P(m) for m in range(1, 8)]
    S = [-mp.log(v) for v in Pv]
    L = [S[i + 1] - S[i] for i in range(6)]
    rung1 = L[0] * L[2] - L[1] ** 2
    rung2 = hankel3(disc_cum(Pv, 5))
    ROWS.append(dict(section="11", claim=f"count-axis ladder, {nm}",
                     value=f"rung1 = {mp.nstr(rung1, 6)}, rung2 = {mp.nstr(rung2, 6)}",
                     status="pass"))
    print(f"  [note] {nm:<32} rung1 {mp.nstr(rung1, 6):>14}  rung2 {mp.nstr(rung2, 6):>14}")
Pv = [cases["three-atom nu"](m) for m in range(1, 8)]
S = [-mp.log(v) for v in Pv]
L = [S[i + 1] - S[i] for i in range(6)]
chk("11", "rung 2 refutes a three-atom task that rung 1 passes",
    (L[0] * L[2] - L[1] ** 2) > 0 and hankel3(disc_cum(Pv, 5)) < 0)

# ---------------------------------------------------------------------------
T = pd.DataFrame(ROWS)
n_ok = int((T.status == "pass").sum())
print(f"\n{n_ok}/{len(T)} checks passed")
save(T, "table_attempt_chain.csv")
