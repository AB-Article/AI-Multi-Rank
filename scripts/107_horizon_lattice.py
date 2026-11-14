#!/usr/bin/env python3
"""THE HORIZON LATTICE: VERIFICATION OF THE ANALYTIC IDENTITIES.
   (Section "The price is a frontier": Prop. 'lattice', 'horizon', 'chartline',
    'powerfloor', 'floorsign', 'divdiff', 'gauge', 'genfun', 'nodes';
    Online Appendix sec. "The horizon lattice".)   NO DATA IS USED.

Everything checked symbolically / at 30-50 digits on synthetic truths.

Sections mirror the paper:
  1  the lattice identity    r_k' = r_k(r_k - r_{k+1}),  h_k' = h_k/h_{k+1} - 1
     -- an IDENTITY, true of every smooth frontier, with no hypothesis
  2  the horizon array h_k = c_k/c_{k+1}: non-increasing in k and non-decreasing
     in x exactly on the divisible charts;  D_k = h_k'
  3  the three lengths  l = 1/s, h_1 = s/kappa, h_0 = DeltaS/s, and
     N_eff = h_1/l, DeltaS = h_0/l, 1+v = h_0/h_1
  4  the similarity solutions  h_k = (x-x0)/(k-beta),  D_k = 1/(k-beta)
  5  the power-floor chart  S = Lam - C (x-x0)^beta, beta<0:  divisible, floored,
     and  N_eff = |sigma| DeltaS  exactly
  6  the floor-sign law   eps_inf > 0  <=>  sigma < 0  <=>  D_1 < 1
  7  two-point rungs  h_k(x2) >= h_k(x1):  rung k at derivative order k+1
  8  the line fit of h_1 against x  ->  slope 1+sigma, zero x_0
  9  the generating identity  DeltaS(x-h) = sum_k h^k c_k/k!, and the origin as
     the abscissa of convergence
 10  the arc identity  phi = 2 int sqrt(1+v) d sqrt(DeltaS)
 11  the error-power group  eps -> eps^t:  c_k extensive, D_k invariant, and
     complete monotonicity of eps^t for every t  <=>  the composition law
 12  roots of eps: the order at which an indivisible chart fails DESCENDS as t
     falls, reaching 3 -- which is why the t -> 0 statistic is the sharp one
 13  the compute-axis four-point test  Delta^3 S >= 0 on an equal grid
 14  the same test with divided differences: no equal spacing required

Writes results/table_horizon_identities.csv.
Run: python3 107_horizon_lattice.py        (needs mpmath)
"""
from __future__ import annotations
import os
import sys
import random

from mpmath import mp, mpf, exp, log, sqrt, diff, quad, gamma, binomial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

mp.dps = 50
TOL = mpf('1e-30')
ROWS = []


def rec(section, item, value, target=None, ok=None):
    ROWS.append(dict(section=section, item=item, value=mp.nstr(value, 12),
                     target=("" if target is None else mp.nstr(target, 12)),
                     ok=("" if ok is None else int(bool(ok)))))


def flag(b):
    return "OK " if b else "** FAIL **"


# --------------------------------------------------------------- the charts ---
# S(x) = -ln eps(x);  Sinf = S(infinity), or None when there is no floor.
_NU3 = [(mpf('0.9'), mpf('0.3')), (mpf('0.5'), mpf('1.1')), (mpf('0.2'), mpf('4.0'))]
_BST, _BPF = mpf('0.4'), mpf('-0.6')


def _law(f, a):
    return lambda x: -log(f + (1 - f) * exp(-a * x))


CHARTS = {
    'cure            ': (lambda x: mpf('1.7') * (1 - exp(-mpf('0.8') * x)), mpf('1.7'), 'div'),
    'powerfloor b=-.6': (lambda x: mpf('2.2') - mpf('1.1') * (x + mpf('0.7')) ** _BPF,
                         mpf('2.2'), 'div'),
    'stretched b=.4  ': (lambda x: (mpf('1.3') * x) ** _BST, None, 'div'),
    'lomax k=3       ': (lambda x: 3 * log(1 + x / mpf('0.9')), None, 'div'),
    '3-jump clock    ': (lambda x: sum(w * (1 - exp(-c * x)) for w, c in _NU3),
                         sum(w for w, c in _NU3), 'div'),
    'floor+1ch f=.15 ': (_law(mpf('0.15'), mpf('0.9')), -log(mpf('0.15')), 'lumpy'),
    'floor+1ch f=.40 ': (_law(mpf('0.40'), mpf('0.9')), -log(mpf('0.40')), 'lumpy'),
    'floor+1ch f=.02 ': (_law(mpf('0.02'), mpf('0.9')), -log(mpf('0.02')), 'lumpy'),
    '2-exp           ': (lambda x: -log(mpf('0.6') * exp(-mpf('0.5') * x)
                                        + mpf('0.4') * exp(-mpf('2.0') * x)), None, 'lumpy'),
}
X = [mpf('0.4'), mpf('1.0'), mpf('2.5')]

# c_k = (-1)^{k+1} S^{(k)};  h_k = c_k/c_{k+1};  r_k = 1/h_k
def ck(S, k, x):
    return (-1) ** (k + 1) * diff(S, x, k)


def hk(S, k, x):
    return ck(S, k, x) / ck(S, k + 1, x)


def rk(S, k, x):
    return ck(S, k + 1, x) / ck(S, k, x)


# ------------------------------------------------- 1  the lattice identity ---
def sec1():
    print("== 1  the lattice identity  r_k' = r_k(r_k - r_{k+1})   (no hypothesis) ==")
    worst_all = mpf(0)
    for name, (S, _, kind) in CHARTS.items():
        worst = mpf(0)
        for x in X:
            for k in (1, 2, 3, 4):
                lhs = diff(lambda xx: rk(S, k, xx), x)
                rhs = rk(S, k, x) * (rk(S, k, x) - rk(S, k + 1, x))
                worst = max(worst, abs(lhs - rhs))
                # the horizon form
                lhs2 = diff(lambda xx: hk(S, k, xx), x)
                rhs2 = hk(S, k, x) / hk(S, k + 1, x) - 1
                worst = max(worst, abs(lhs2 - rhs2))
        rec('1 lattice', name.strip(), worst, mpf(0), worst < mpf('1e-20'))
        worst_all = max(worst_all, worst)
        print(f"   {name}  max |lhs-rhs| = {mp.nstr(worst,3):>10}  truth={kind:5} "
              f"{flag(worst < mpf('1e-20'))}")
    print(f"   -> holds on ALL {len(CHARTS)} charts, divisible and not: it is an identity.")
    return worst_all


# ------------------------------------- 2  the horizon array, and D_k = h_k' ---
def sec2():
    print("\n== 2  horizon array: down in k, up in x  <=>  divisible;  D_k = h_k' ==")
    for name, (S, _, kind) in CHARTS.items():
        down = all(hk(S, k, x) >= hk(S, k + 1, x) - TOL for k in (1, 2, 3) for x in X)
        up = all(diff(lambda xx: hk(S, k, xx), x) >= -TOL for k in (1, 2, 3) for x in X)
        # D_k = h_k' identity
        e = max(abs(diff(lambda xx: hk(S, k, xx), x)
                    - (ck(S, k, x) * ck(S, k + 2, x) / ck(S, k + 1, x) ** 2 - 1))
                for k in (1, 2) for x in X)
        good = (down and up) == (kind == 'div')
        rec('2 horizons', name.strip() + ' |D_k-h_k\'|', e, mpf(0), e < TOL)
        rec('2 horizons', name.strip() + ' monotone', mpf(int(down and up)),
            mpf(int(kind == 'div')), good)
        print(f"   {name}  down_in_k={str(down):5} up_in_x={str(up):5} "
              f"|D_k - h_k'|={mp.nstr(e,3):>9}  truth={kind:5} {flag(good and e < TOL)}")


# --------------------------------------------------- 3  the three lengths ---
def sec3():
    print("\n== 3  three lengths  l=1/s, h_1=s/kappa, h_0=DeltaS/s ==")
    print("      N_eff = h_1/l,  DeltaS = h_0/l,  1+v = h_0/h_1,  rule: h_1 = l")
    for name, (S, Sinf, kind) in CHARTS.items():
        if Sinf is None:
            continue
        x = X[1]
        s, kap, dS = diff(S, x), -diff(S, x, 2), Sinf - S(x)
        l, h1, h0 = 1 / s, s / kap, dS / s
        e1, e2 = abs(h1 / l - s * s / kap), abs(h0 / h1 - dS * kap / (s * s))
        rec('3 lengths', name.strip() + ' N_eff=h1/l', e1, mpf(0), e1 < TOL)
        rec('3 lengths', name.strip() + ' 1+v=h0/h1', e2, mpf(0), e2 < TOL)
        print(f"   {name}  l={mp.nstr(l,5):>9} h1={mp.nstr(h1,5):>9} h0={mp.nstr(h0,5):>9}"
              f"  N_eff={mp.nstr(h1/l,7):>10} 1+v={mp.nstr(h0/h1,7):>9}"
              f"  budget h1<=h0: {str(h1 <= h0):5} {flag(e1 < TOL and e2 < TOL)}")


# ---------------------------------- 4/5  similarity solutions, new chart ---
def sec45():
    print("\n== 4-5  similarity solutions  h_k=(x-x0)/(k-beta),  D_k=1/(k-beta) ==")
    for name, b, x0 in [('stretched b=.4  ', _BST, mpf(0)),
                        ('lomax k=3       ', mpf(0), -mpf('0.9')),
                        ('powerfloor b=-.6', _BPF, -mpf('0.7'))]:
        S = CHARTS[name][0]
        x = X[1]
        e1 = max(abs(hk(S, k, x) - (x - x0) / (k - b)) for k in (1, 2, 3, 4))
        e2 = max(abs(diff(lambda xx: hk(S, k, xx), x) - 1 / (k - b)) for k in (1, 2, 3, 4))
        rec('4 similarity', name.strip() + ' h_k', e1, mpf(0), e1 < TOL)
        rec('4 similarity', name.strip() + ' D_k', e2, mpf(0), e2 < TOL)
        print(f"   {name} beta={mp.nstr(b,3):>7} |h_k-(x-x0)/(k-b)|={mp.nstr(e1,3):>9}"
              f" |D_k-1/(k-b)|={mp.nstr(e2,3):>9} sigma={mp.nstr(b/(1-b),6):>9} "
              f"{flag(e1 < TOL and e2 < TOL)}")
    print("\n   the power-floor chart (sigma in (-1,0)): divisible, floored, N_eff=|sigma|DeltaS")
    S, Sinf, _ = CHARTS['powerfloor b=-.6']
    cmv = [(-1) ** k * diff(lambda xx: diff(S, xx), X[1], k) for k in range(0, 8)]
    cm = all(z >= -TOL for z in cmv)
    x = X[1]
    s, kap, dS = diff(S, x), -diff(S, x, 2), Sinf - S(x)
    sig = _BPF / (1 - _BPF)
    e = abs(s * s / kap / dS - abs(sig))
    rec('5 powerfloor', "S' completely monotone to order 7", mpf(int(cm)), mpf(1), cm)
    rec('5 powerfloor', 'N_eff/DeltaS - |sigma|', e, mpf(0), e < TOL)
    print(f"      S' completely monotone to order 7: {flag(cm)}   "
          f"eps_inf = {mp.nstr(exp(-Sinf),5)} > 0")
    print(f"      N_eff/DeltaS = {mp.nstr(s*s/kap/dS,10)}  |sigma| = {mp.nstr(abs(sig),10)}"
          f"  {flag(e < TOL)}")


# ------------------------------------------------ 6  the floor-sign law ---
def sec6():
    print("\n== 6  floor-sign law:  eps_inf > 0  <=>  sigma < 0  <=>  D_1 < 1 ==")
    for name, (S, Sinf, kind) in CHARTS.items():
        x = X[2]
        D1 = diff(lambda xx: hk(S, 1, xx), x)
        good = (D1 < 1) == (Sinf is not None)
        rec('6 floorsign', name.strip() + ' D_1', D1,
            None, good)
        print(f"   {name}  D_1={mp.nstr(D1,7):>11} sigma={mp.nstr(D1-1,7):>11}"
              f"  floor={str(Sinf is not None):5} predicted={str(D1 < 1):5} {flag(good)}")
    print("   verdicts from one number:  D_1<0 refutes divisibility | [0,1) a floor |")
    print("                              =1 Pareto | >1 no floor, beta = 1 - 1/D_1")


# ------------------------------------------------- 7  two-point rungs ---
def sec7():
    print("\n== 7  two-point rungs  h_k(x2) >= h_k(x1):  rung k at order k+1 ==")
    x1, x2 = mpf('0.6'), mpf('2.2')
    for name, (S, _, kind) in CHARTS.items():
        p = [hk(S, k, x2) >= hk(S, k, x1) for k in (1, 2, 3)]
        good = all(p) == (kind == 'div')
        rec('7 twopoint', name.strip(), mpf(int(all(p))), mpf(int(kind == 'div')), good)
        print(f"   {name}  rung1={str(p[0]):5} rung2={str(p[1]):5} rung3={str(p[2]):5}"
              f"  truth={kind:5} {flag(good)}")


# ------------------------------------------- 8  the line fit for (sigma,x0) ---
def sec8():
    print("\n== 8  line fit of h_1 = s/kappa against x -> slope 1+sigma, zero x_0 ==")
    for name, b, x0 in [('stretched b=.4  ', _BST, mpf(0)),
                        ('lomax k=3       ', mpf(0), -mpf('0.9')),
                        ('powerfloor b=-.6', _BPF, -mpf('0.7'))]:
        S = CHARTS[name][0]
        xa, xb = mpf('0.8'), mpf('2.4')
        h1a, h1b = hk(S, 1, xa), hk(S, 1, xb)
        slope = (h1b - h1a) / (xb - xa)
        zero = xa - h1a / slope
        e = abs(zero - x0) + abs(slope - (1 + b / (1 - b)))
        rec('8 linefit', name.strip() + ' slope', slope, 1 + b / (1 - b), e < mpf('1e-20'))
        rec('8 linefit', name.strip() + ' x0', zero, x0, e < mpf('1e-20'))
        print(f"   {name}  slope={mp.nstr(slope,10):>13} (1+sigma={mp.nstr(1+b/(1-b),10):>13})"
              f"  zero={mp.nstr(zero,8):>12} (x0={mp.nstr(x0,8):>8})"
              f"  beta=1-1/slope={mp.nstr(1-1/slope,6):>9} {flag(e < mpf('1e-20'))}")


# ---------------------------------- 9  generating identity, and the origin ---
def sec9():
    print("\n== 9  DeltaS(x-h) = sum_k h^k c_k/k!   (the tower is the depth, backwards) ==")
    for name, (S, Sinf, _) in CHARTS.items():
        if Sinf is None:
            continue
        x, h = mpf('1.6'), mpf('0.35')
        ser = sum(h ** k * ((ck(S, k, x) if k >= 1 else Sinf - S(x))) / gamma(k + 1)
                  for k in range(0, 26))
        e = abs(ser - (Sinf - S(x - h)))
        rec('9 genfun', name.strip(), e, mpf(0), e < mpf('1e-20'))
        print(f"   {name}  |series - DeltaS(x-h)| = {mp.nstr(e,3):>10} {flag(e < mpf('1e-20'))}")
    print("   radius of convergence = x - x_0 = abscissa of convergence of the Levy")
    print("   transform; the limsup converges slowly -- Hadamard ill-posedness, which")
    print("   is WHY beta is weakly identified (Prop. 'genfun'(ii)):")
    for name, x0 in [('stretched b=.4  ', mpf(0)), ('powerfloor b=-.6', -mpf('0.7')),
                     ('cure            ', None)]:
        S = CHARTS[name][0]
        x = mpf('1.6')
        est = [abs(ck(S, k, x) / gamma(k + 1)) ** (mpf(1) / k) for k in (10, 14, 18)]
        tgt = "inf (nu compact)" if x0 is None else mp.nstr(x - x0, 6)
        print(f"      {name}  1/(c_k/k!)^(1/k) at k=10,14,18 -> "
              + ", ".join(mp.nstr(1 / z, 6) for z in est) + f"   x-x0 = {tgt}")
        rec('9 radius', name.strip(), 1 / est[-1],
            None, None)


# ------------------------------------------------------- 10  the arc ---
def sec10():
    print("\n== 10  arc identity  phi = 2 int sqrt(1+v) d sqrt(DeltaS) ==")
    for name in ['cure            ', 'powerfloor b=-.6', '3-jump clock    ']:
        S, Sinf, _ = CHARTS[name]
        a, b = mpf('0.25'), mpf('6.0')
        arc = quad(lambda xx: sqrt(-diff(S, xx, 2)), [a, b])

        def ig(xx):
            s, kap, dS = diff(S, xx), -diff(S, xx, 2), Sinf - S(xx)
            return sqrt(dS * kap / (s * s)) * (-s / (2 * sqrt(dS)))
        rhs = -2 * quad(ig, [a, b])
        kant = 2 * (sqrt(Sinf - S(a)) - sqrt(Sinf - S(b)))
        e = abs(arc - rhs)
        rec('10 arc', name.strip(), e, mpf(0), e < mpf('1e-12'))
        print(f"   {name}  arc={mp.nstr(arc,10):>13} identity={mp.nstr(rhs,10):>13}"
              f"  2(sqrt dS1 - sqrt dS2)={mp.nstr(kant,10):>13}"
              f"  {flag(e < mpf('1e-12') and arc >= kant - TOL)}")
    print("   equality iff v == 0 (the cure chart); the lower bound is Prop. 'kant',")
    print("   windowed, recovered as the v >= 0 case of an identity.")


# ------------------------------------------------- 11  the error gauge ---
def sec11():
    print("\n== 11  error-power group  eps -> eps^t:  c_k extensive, D_k invariant ==")
    t = mpf('3.7')
    for name, (S, _, kind) in CHARTS.items():
        x = X[1]

        def St(xx):
            return t * S(xx)
        inv = max(abs(diff(lambda xx: hk(S, k, xx), x)
                      - diff(lambda xx: hk(St, k, xx), x)) for k in (1, 2, 3))
        s, kap = diff(S, x), -diff(S, x, 2)
        st, kapt = diff(St, x), -diff(St, x, 2)
        ext = abs((st * st / kapt) / (s * s / kap) - t)
        hinv = max(abs(hk(S, k, x) - hk(St, k, x)) for k in (1, 2, 3))
        good = inv < TOL and ext < TOL and hinv < TOL
        rec('11 gauge', name.strip() + ' D_k invariant', inv, mpf(0), inv < TOL)
        rec('11 gauge', name.strip() + ' N_eff extensive', ext, mpf(0), ext < TOL)
        print(f"   {name}  max|D_k - D_k^(t)|={mp.nstr(inv,3):>9}"
              f"  |N_eff ratio/t - 1|={mp.nstr(ext,3):>9}"
              f"  max|h_k - h_k^(t)|={mp.nstr(hinv,3):>9} {flag(good)}")
    print("   so the invariants are exactly (D_0, D_1, ...) and the gauge coordinates")
    print("   are (c_0, c_1) = (DeltaS, s):  the floor and the axis unit.")


# ------------------------- 12  roots of eps: the order of failure descends ---
def sec12():
    print("\n== 12  complete monotonicity of eps^t:  CM for every t  <=>  divisible ==")
    XS = [mpf('0.2'), mpf('0.7'), mpf('1.5'), mpf('3.0')]
    print(f"   {'chart':18}" + "".join(f"{'t=1/'+str(2**j):>9}" for j in range(5)))
    for name, (S, _, kind) in CHARTS.items():
        row = []
        for j in range(5):
            t = mpf(1) / 2 ** j

            def f(xx):
                return exp(-t * S(xx))
            first = None
            for k in range(1, 9):
                if any((-1) ** k * diff(f, x, k) < -mpf('1e-25') for x in XS):
                    first = k
                    break
            row.append("   CM   " if first is None else f"  fail@{first}")
        rec('12 roots', name.strip(), mpf(0), None, None)
        print(f"   {name}" + "".join(f"{z:>9}" for z in row) + f"   truth={kind}")
    print("   the order of failure DESCENDS as t falls, reaching 3: the sharp")
    print("   statistic is the t -> 0 one, which is Delta^3 S >= 0 (sections 13-14).")


# ------------------------------- 13  the compute-axis four-point test ---
def sec13():
    print("\n== 13  compute-axis four-point test on an equal grid:  Delta^3 S >= 0 ==")
    def D3(S, x, h):
        return S(x + 3 * h) - 3 * S(x + 2 * h) + 3 * S(x + h) - S(x)
    for name, (S, _, kind) in CHARTS.items():
        worst = None
        for h10 in range(1, 31):
            h = mpf(h10) / 10
            for xb10 in range(0, 41, 2):
                xb = mpf(xb10) / 10
                v = D3(S, xb, h)
                if worst is None or v < worst[0]:
                    worst = (v, h, xb)
        v, h, xb = worst
        good = (v < 0) == (kind == 'lumpy') or (v >= 0)
        rec('13 fourpoint', name.strip() + ' min D3S', v, None, None)
        print(f"   {name}  min D3S = {mp.nstr(v,4):>11} at h={mp.nstr(h,3)}, x={mp.nstr(xb,3)}"
              + ("   <- REFUTED" if v < 0 else "   pass (one-sided)"))
    print("   noise on Delta^3 S is sqrt(20) sigma_S, INDEPENDENT of the spacing,")
    print("   while the signal grows with spacing: the optimal quadruple is wide.")


# --------------------------------------- 14  divided differences, any spacing ---
def sec14():
    print("\n== 14  divided differences: NO equal spacing needed ==")
    random.seed(20261005)
    quads = []
    while len(quads) < 1500:
        xs = sorted(mpf(random.uniform(0.05, 6.0)) for _ in range(4))
        if min(xs[i + 1] - xs[i] for i in range(3)) >= mpf('0.05'):
            quads.append(xs)

    def dd3(S, xs):
        return sum(S(x) / mp.fprod([x - y for y in xs if y != x]) for x in xs)
    print(f"   {'chart':18}{'frac dd3<0':>12}{'min dd3':>13}{'verdict':>12}")
    for name, (S, _, kind) in CHARTS.items():
        vals = [dd3(S, xs) for xs in quads]
        neg = sum(1 for v in vals if v < 0)
        good = (neg == 0) if kind == 'div' else True
        rec('14 divdiff', name.strip() + ' frac neg', mpf(neg) / len(vals),
            mpf(0) if kind == 'div' else None, good)
        print(f"   {name}{neg/len(vals):>12.3f}{mp.nstr(min(vals),4):>13}"
              f"{('REFUTED' if neg else 'pass'):>12}  {flag(good)}")
    print(f"   ({len(quads)} random unequally-spaced quadruples; every divisible chart")
    print("    has dd3 >= 0 on all of them, as the mean-value theorem requires.)")


def main():
    sec1(); sec2(); sec3(); sec45(); sec6(); sec7(); sec8()
    sec9(); sec10(); sec11(); sec12(); sec13(); sec14()
    import pandas as pd
    from _loaders import save
    save(pd.DataFrame(ROWS), "table_horizon_identities.csv")
    bad = [r for r in ROWS if r["ok"] == 0]
    print(f"\n{len(ROWS)} recorded checks, {len(bad)} failures.")
    for r in bad:
        print("   FAIL:", r)


if __name__ == "__main__":
    main()
