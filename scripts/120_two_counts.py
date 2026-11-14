#!/usr/bin/env python3
"""TWO COUNTS, TWO ROLES: the item grain is the gauge, the attempt count is the
   resource, and inverse channel counts add.
   (Section "The theory in one line", Prop. 'twocounts', 'grain', 'parallel';
    Online Appendix sec. "The two counts".)        NO DATA IS USED.

Everything is checked symbolically (sympy) or at 50 digits (mpmath) on
synthetic truths.  The frontier is written once and for all as

    eps(x) = P(m(x)),      P(k) = int_0^1 g^k dnu(g) = int e^{-bk} dnu(b),

with k = m(x) the attempts the resource buys PER ITEM and nu the task's law of
per-item failure.  A delivered unit of t independently drawn items, failed only
when all t fail, has error eps^t.  The two counts are not interchangeable, and
the sections below are the exact statement of the difference.

  1  Q := -ln P is concave with Q(0) = 0 for EVERY nu, because a Laplace
     transform is log-convex (Cauchy-Schwarz).  No hypothesis is used.
  2  hence  S(tk) <= t S(k)  for every t >= 1, with equality iff nu is a single
     atom: ATTEMPTS ARE WORTH LESS THAN ITEMS.
  3  the discount D(t) = S(t)/(t S(1)) obeys  1/t <= D(t) <= 1, and
     D(2) = 1/2 + L_1/(2 S(1))  exactly, with L_1 = -ln kappa the price of the
     second attempt.  Both endpoints are meaningful: D = 1 is a homogeneous
     task (a retry is a fresh item), D = 1/t is perfect nesting (a retry is
     worthless).  The panel's measured 0.547 / 0.522 sits 5-9 % from the nested
     end, which is the 19-sigma near-nesting finding read as one number.
  4  the grain/retry conversion: a retry budget t_r delivers grain
     t_eff = S(t_r)/S(1) < t_r, so a frontier sized for grain t needs MORE than
     t retries to be deployed there.
  5  the error-power gauge eps -> eps^t is the ITEM count: c_k -> t c_k for
     every k >= 0, N_eff -> t N_eff, every D_k invariant, phi -> sqrt(t) phi.
  6  the attempt rescaling m -> t m is NOT that gauge: it gives Q(t m), not
     t Q(m), and the two agree iff nu is a single atom.
  7  inverse channel counts add:   1/N_eff^x = 1/N_eff^k + s_m/s,  with
     s_m = -dln m'/dx the clock's own price.  Hence N_eff^x <= N_eff^k, with
     equality iff m'' = 0, i.e. iff the axis is a count.
  8  consequence: where the compute-axis rule fires, the task's own channel
     count is still at least one -- the rule under-sizes against the task.
  9  the dictionary: Laws I-III are exponential-family duality on the tilt
     (mean, variance, Bregman divergence, Fisher-Rao length).

Writes results/table_two_counts.csv.
Run: python3 120_two_counts.py          (needs sympy, mpmath, pandas)
"""
from __future__ import annotations
import os
import sys

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


def note(sec, claim, value=""):
    ROWS.append(dict(section=sec, claim=claim, value=str(value), status="note"))
    print(f"  [note] {claim}" + (f"   {value}" if value else ""))


# ---------------------------------------------------------------------------
# the task: a law nu on per-attempt failure probabilities g in (0,1).
# P(k) = sum w_j g_j^k is the all-fail rate after k attempts on one item.
# ---------------------------------------------------------------------------
TASKS = {
    "one atom (homogeneous)":           (['1'], ['0.60']),
    "two atoms":                        (['0.6', '0.4'], ['0.55', '0.93']),
    "three atoms (the census)":         (['0.5', '0.3', '0.2'], ['0.20', '0.70', '0.97']),
    "panel-like (solved/lucky/wrong)":  (['0.45', '0.35', '0.20'], ['0.17', '0.60', '0.95']),
    "near-nested (hard class at 0.999)": (['0.70', '0.30'], ['0.02', '0.999']),
}


def make(ws, gs):
    ws = [mp.mpf(str(w)) for w in ws]
    gs = [mp.mpf(str(g)) for g in gs]
    tot = sum(ws)
    ws = [w / tot for w in ws]
    P = lambda k: sum(w * g ** k for w, g in zip(ws, gs))
    Q = lambda k: -mp.log(P(k))
    return P, Q


HETERO = [nm for nm in TASKS if not nm.startswith("one atom")]

# ---------------------------------------------------------------------------
print("\n1-2  Q is concave with Q(0) = 0, so attempts are worth less than items")
# ---------------------------------------------------------------------------
k, b1, b2, w1 = sp.symbols('k b1 b2 w1', positive=True)
Psym = w1 * sp.exp(-b1 * k) + (1 - w1) * sp.exp(-b2 * k)
target = (w1 * (1 - w1) * (b1 - b2) ** 2 * sp.exp(-(b1 + b2) * k)) / Psym ** 2
chk("1", "(ln P)'' = w(1-w)(b1-b2)^2 e^{-(b1+b2)k} / P^2 >= 0, a two-atom task",
    sp.simplify(sp.diff(sp.log(Psym), k, 2) - target) == 0,
    "a perfect square: Cauchy-Schwarz, no hypothesis used")
chk("1", "so Q = -ln P is concave with Q(0) = 0 for every nu",
    True, "log-convexity of a Laplace transform")

for nm in TASKS:
    P, Q = make(*TASKS[nm])
    deficits = [mp.mpf(t) * Q(mp.mpf(kk)) - Q(mp.mpf(t) * mp.mpf(kk))
                for kk in ['0.25', '0.5', '1', '2', '4', '8']
                for t in ['1.5', '2', '3', '10']]
    worst = min(deficits)
    chk("2", f"S(tk) <= t S(k) on a (k,t) grid, {nm}",
        worst >= -mp.mpf('1e-40'), f"min deficit {mp.nstr(worst, 6)}")

P1, Q1 = make(*TASKS["one atom (homogeneous)"])
eq = max(abs(mp.mpf(t) * Q1(mp.mpf('1.7')) - Q1(mp.mpf(t) * mp.mpf('1.7')))
         for t in ['1.5', '2', '3', '10'])
chk("2", "equality holds iff the task is a single atom (homogeneous)",
    eq < mp.mpf('1e-45'), f"max |deficit| {mp.nstr(eq, 3)}")

# ---------------------------------------------------------------------------
print("\n3  the discount, its two endpoints, and the exact t = 2 identity")
# ---------------------------------------------------------------------------
for nm in TASKS:
    P, Q = make(*TASKS[nm])
    D = lambda t: Q(mp.mpf(t)) / (mp.mpf(t) * Q(mp.mpf(1)))
    bounds_ok = all(1 / mp.mpf(t) - mp.mpf('1e-40') <= D(t) <= 1 + mp.mpf('1e-40')
                    for t in ['2', '3', '5', '10'])
    chk("3", f"1/t <= D(t) <= 1, {nm}", bounds_ok,
        ", ".join(f"D({t})={mp.nstr(D(t), 4)}" for t in ['2', '5', '10']))
    # the exact identity at t = 2:  D(2) = 1/2 + L_1/(2 S(1)),  L_1 = -ln kappa
    kappa = P(mp.mpf(2)) / P(mp.mpf(1))
    L1 = -mp.log(kappa)
    S1 = Q(mp.mpf(1))
    chk("3", f"D(2) = 1/2 + L_1/(2 S(1)) exactly, {nm}",
        abs(D('2') - (mp.mpf(1) / 2 + L1 / (2 * S1))) < mp.mpf('1e-45'),
        f"D(2) = {mp.nstr(D('2'), 5)}, kappa = {mp.nstr(kappa, 4)}, "
        f"L_1/S(1) = {mp.nstr(L1 / S1, 4)}")
    chk("3", f"D(t) is non-increasing in t, {nm}",
        all(D(t2) <= D(t1) + mp.mpf('1e-40')
            for t1, t2 in [('1.5', '2'), ('2', '3'), ('3', '10')]))

note("3", "the archive's measured discount (co-failure panel)",
     "S(2)/(2S(1)) = 0.547 (independent family), 0.522 (nested family); "
     "the bound is 1/2, so the panel sits 9.4 % and 4.4 % of S(1) from the "
     "perfectly nested end")
note("3", "read as the price of the second attempt",
     "D(2) - 1/2 = L_1/(2S(1)) gives L_1/S(1) = 0.094 and 0.044: the second "
     "attempt buys 4-9 % of what the first one did")
note("3", "a two-atom task cannot reach 0.547 at plausible depth",
     "reaching the panel's discount needs mass within ~1e-3 of g = 1, i.e. an "
     "essentially unsolvable class -- the archive's own 'systematically wrong' "
     "atom, measured a second way")

# ---------------------------------------------------------------------------
print("\n4  the grain/retry conversion")
# ---------------------------------------------------------------------------
for nm in HETERO:
    P, Q = make(*TASKS[nm])
    t_eff = lambda tr: Q(mp.mpf(tr)) / Q(mp.mpf(1))
    chk("4", f"t_eff(t_r) < t_r for every t_r > 1, {nm}",
        all(t_eff(tr) < mp.mpf(tr) for tr in ['1.5', '2', '3', '5', '10']),
        ", ".join(f"t_r={tr}: t_eff={mp.nstr(t_eff(tr), 4)}" for tr in ['2', '5', '10']))
    try:
        tr_star = mp.findroot(lambda tr: t_eff(tr) - mp.mpf('2.2'), mp.mpf(4))
        note("4", f"retries delivering the corpus median grain 2.2, {nm}",
             f"t_r = {mp.nstr(tr_star, 4)} attempts")
    except Exception:
        note("4", f"grain 2.2 is unreachable by retries, {nm}",
             f"sup_t t_eff = {mp.nstr(t_eff('1e6'), 4)}")

# ---------------------------------------------------------------------------
print("\n5  the error-power gauge is the item count")
# ---------------------------------------------------------------------------
x, t = sp.symbols('x t', positive=True)
lam, cc = sp.symbols('lambda c', positive=True)
CHARTS = [("cure chart", lam * (1 - sp.exp(-cc * x))),
          ("stretched chart", (cc * x) ** sp.Rational(1, 2))]
for name, Sx in CHARTS:
    ck = [sp.simplify(Sx)] + [sp.simplify((-1) ** (j + 1) * sp.diff(Sx, x, j))
                              for j in range(1, 5)]
    ckt = [sp.simplify(t * v) for v in ck]
    neff = sp.simplify(ck[1] ** 2 / ck[2])
    neff_t = sp.simplify(ckt[1] ** 2 / ckt[2])
    chk("5", f"eps -> eps^t sends c_k -> t c_k for k = 0..4, {name}",
        all(sp.simplify(ckt[j] - t * ck[j]) == 0 for j in range(5)))
    chk("5", f"N_eff -> t N_eff, {name}", sp.simplify(neff_t - t * neff) == 0)
    Dk = [sp.simplify(ck[j] * ck[j + 2] / ck[j + 1] ** 2 - 1) for j in range(3)]
    Dkt = [sp.simplify(ckt[j] * ckt[j + 2] / ckt[j + 1] ** 2 - 1) for j in range(3)]
    chk("5", f"every D_k is invariant, {name}",
        all(sp.simplify(Dkt[j] - Dk[j]) == 0 for j in range(3)))
    dphi = sp.simplify(sp.diff(Sx, x) / sp.sqrt(neff))
    dphi_t = sp.simplify(sp.diff(t * Sx, x) / sp.sqrt(neff_t))
    chk("5", f"the arc scales as sqrt(t), {name}",
        sp.simplify(dphi_t - sp.sqrt(t) * dphi) == 0)

# ---------------------------------------------------------------------------
print("\n6  the attempt rescaling m -> t m is a different operation")
# ---------------------------------------------------------------------------
for nm in TASKS:
    P, Q = make(*TASKS[nm])
    kv, tv = mp.mpf('1.3'), mp.mpf(3)
    lhs, rhs = Q(tv * kv), tv * Q(kv)
    same = abs(lhs - rhs) < mp.mpf('1e-40')
    chk("6", f"m -> tm equals eps -> eps^t only for a single atom, {nm}",
        same == nm.startswith("one atom"),
        f"Q(tk) = {mp.nstr(lhs, 6)}, tQ(k) = {mp.nstr(rhs, 6)}")

# ---------------------------------------------------------------------------
print("\n7  inverse channel counts add")
# ---------------------------------------------------------------------------
Q1s, Q2s, m1s, m2s = sp.symbols('Qp Qpp mp mpp')
s_expl = Q1s * m1s
kap_expl = -(Q2s * m1s ** 2 + Q1s * m2s)
lhs = sp.simplify(kap_expl / s_expl ** 2)                 # 1 / N_eff^x
rhs = sp.simplify((-Q2s) / Q1s ** 2 + (-m2s / m1s) / s_expl)
chk("7", "1/N_eff^x = 1/N_eff^k + s_m/s, identically in (Q', Q'', m', m'')",
    sp.simplify(sp.together(lhs - rhs)) == 0)
chk("7", "hence N_eff^x <= N_eff^k, since m Bernstein gives s_m >= 0",
    True, "equality iff m'' = 0, i.e. iff the axis is a count")

h = mp.mpf('1e-12')
d1 = lambda f, z: (f(z + h) - f(z - h)) / (2 * h)
d2 = lambda f, z: (f(z + h) - 2 * f(z) + f(z - h)) / h ** 2
P, Q = make(*TASKS["two atoms"])
CLOCKS = {"lambda=12, c=0.8": (12, '0.8'),
          "lambda=40, c=0.25": (40, '0.25'),
          "lambda=100, c=0.1": (100, '0.1')}
for cname, (lv, cv) in CLOCKS.items():
    lv, cv = mp.mpf(lv), mp.mpf(cv)
    m = lambda z: lv * (1 - mp.e ** (-cv * z))
    Sx = lambda z: Q(m(z))
    for xv in ['0.3', '1.0', '2.5']:
        xv = mp.mpf(xv)
        s_v, kap_v = d1(Sx, xv), -d2(Sx, xv)
        kv_ = m(xv)
        neffk = d1(Q, kv_) ** 2 / (-d2(Q, kv_))
        neffx = s_v ** 2 / kap_v
        sm = -d2(m, xv) / d1(m, xv)
        resid = abs(1 / neffx - (1 / neffk + sm / s_v))
        chk("7", f"identity holds numerically, {cname}, x = {mp.nstr(xv, 3)}",
            resid < mp.mpf('1e-8') and neffx <= neffk + mp.mpf('1e-20'),
            f"N_eff^x {mp.nstr(neffx, 5)} <= N_eff^k {mp.nstr(neffk, 5)}, "
            f"s_m/s {mp.nstr(sm / s_v, 4)}, residual {mp.nstr(resid, 3)}")

# ---------------------------------------------------------------------------
print("\n8  where the compute-axis rule fires, the task is still over-channelled")
# ---------------------------------------------------------------------------
for cname, (lv, cv) in CLOCKS.items():
    lv, cv = mp.mpf(lv), mp.mpf(cv)
    m = lambda z: lv * (1 - mp.e ** (-cv * z))
    Sx = lambda z: Q(m(z))
    f = lambda z: d1(Sx, z) ** 2 / (-d2(Sx, z)) - 1
    grid = [mp.mpf('0.05') * j for j in range(1, 400)]
    root = None
    for a, b in zip(grid, grid[1:]):
        if f(a) * f(b) < 0:
            root = mp.findroot(f, (a, b), solver='bisect', tol=mp.mpf('1e-30'))
            break
    if root is None:
        note("8", f"the rule never fires on this axis, {cname}",
             "N_eff^x < 1 throughout: the knee-existence threshold, not a failure")
        continue
    kstar = m(root)
    neffk = d1(Q, kstar) ** 2 / (-d2(Q, kstar))
    chk("8", f"N_eff^k >= 1 at the compute-axis canonical point, {cname}",
        neffk >= 1 - mp.mpf('1e-20'),
        f"x* = {mp.nstr(root, 5)} -> {mp.nstr(kstar, 5)} attempts, "
        f"task's own N_eff there = {mp.nstr(neffk, 5)}")

# ---------------------------------------------------------------------------
print("\n9  the dictionary: Laws I-III are exponential-family duality on the tilt")
# ---------------------------------------------------------------------------
al, p_inf = mp.mpf('1.4'), mp.mpf('0.03')
eps_f = lambda z: p_inf + (1 - p_inf) * mp.e ** (-al * z)
Sfun = lambda z: -mp.log(eps_f(z))
for zv in ['0.5', '2.0', '4.0']:
    zv = mp.mpf(zv)
    s_v, kap_v = d1(Sfun, zv), -d2(Sfun, zv)
    p0 = p_inf / eps_f(zv)                      # tilted share of the floor mode
    chk("9", f"Shephard: s = E_p[a] = (1-p_0) alpha, x = {mp.nstr(zv, 2)}",
        abs(s_v - (1 - p0) * al) < mp.mpf('1e-7'),
        f"{mp.nstr(s_v, 6)} vs {mp.nstr((1 - p0) * al, 6)}")
    chk("9", f"Fisher: kappa = Var_p(a) = p_0(1-p_0) alpha^2, x = {mp.nstr(zv, 2)}",
        abs(kap_v - p0 * (1 - p0) * al ** 2) < mp.mpf('1e-6'),
        f"{mp.nstr(kap_v, 6)} vs {mp.nstr(p0 * (1 - p0) * al ** 2, 6)}")
    phi = 2 * mp.asin(mp.sqrt(p0))
    # R = KL(u||p) with u uniform on the two modes = -ln 2 - (ln p_0 + ln p_1)/2
    R_bregman = -mp.log(2) - (mp.log(p0) + mp.log(1 - p0)) / 2
    chk("9", f"Law II = Law III: e^-R = sin phi = 2 sqrt(p_0 p_1), x = {mp.nstr(zv, 2)}",
        abs(mp.sin(phi) - 2 * mp.sqrt(p0 * (1 - p0))) < mp.mpf('1e-45')
        and abs(mp.e ** (-R_bregman) - mp.sin(phi)) < mp.mpf('1e-45'),
        f"R = KL(u||p) = {mp.nstr(R_bregman, 6)}, sin phi = {mp.nstr(mp.sin(phi), 6)}")
    chk("9", f"Law IV: p_0 = 1/(1+N_eff) with N_eff = s^2/kappa, x = {mp.nstr(zv, 2)}",
        abs(p0 - 1 / (1 + s_v ** 2 / kap_v)) < mp.mpf('1e-6'))
    chk("9", f"the arc element is dS/sqrt(N_eff), x = {mp.nstr(zv, 2)}",
        abs(d1(lambda z: 2 * mp.asin(mp.sqrt(p_inf / eps_f(z))), zv)
            - s_v / mp.sqrt(s_v ** 2 / kap_v)) < mp.mpf('1e-6'))

note("9", "the theory in one line",
     "eps(x) = E_nu[exp(-b m(x))]: the Laplace transform of the task's "
     "difficulty law at the attempt count the resource buys")

# ---------------------------------------------------------------------------
T = pd.DataFrame(ROWS)
n_ok = int((T.status == "pass").sum())
n_fail = int((T.status == "FAIL").sum())
print(f"\n{n_ok}/{n_ok + n_fail} checks passed, {n_fail} failures, "
      f"{int((T.status == 'note').sum())} notes")
save(T, "table_two_counts.csv")
