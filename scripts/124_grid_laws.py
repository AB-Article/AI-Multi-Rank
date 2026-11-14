#!/usr/bin/env python3
"""124_grid_laws.py -- the laws that need no bandwidth.

Pure mathematics: no data, no corpus, no fit.  Every statement below is a
consequence of material already in paper.tex / supplement.tex pushed against
the 1006b propositions (unbounded, elbow, notestable, wallrefutes) and against
Cor. prony.  mpmath, 40 digits.

  (A) THE OPERATING DEPTH AND THE TWO WALLS
      At N_eff = 1/t:  Delta S* >= ln(1+1/t)   (axiom; equality on two atoms)
                       Delta S* >= 1/t         (hypothesis; equality on the cure chart)
      band g(t) = 1/t - ln(1+1/t), maximal 1-ln2 = 0.30685 at t = 1.
      At t = 1 the operating point IS the elbow (c_3 = 0): the unique grain at
      which the recommendation is its own test point.
  (B) THE CANONICAL PRICE AT ANY GRAIN
      lambda* = alpha/(1+t),  eps* = eps_inf (1+t)/t,  = sqrt(kappa/t) at the point.
  (C) MINIMAX OVER AN UNKNOWN GRAIN
      t in [t1,t2]  ->  N_eff = 1/sqrt(t1 t2),  worst-case regret ln cosh((1/4)ln(t2/t1)).
  (D) THE LADDER WITHOUT A BANDWIDTH
      Axiom: the sampled error is a Hausdorff moment sequence.
      Hypothesis: the sampled depth ln(eps/eps_inf) is one.
      3 accuracies give the first minor, 5 the second, and the second is stronger.
  (E) THE FLOOR, BOUNDED TWICE FROM THREE ACCURACIES
      eps_A (Aitken)  = largest floor compatible with the axiom     (exact on two atoms)
      eps_W           = largest floor compatible with divisibility  (exact on the cure chart)
      A floor measured above eps_W refutes the hypothesis; above eps_A, the axiom.
  (F) RANK
      Hankel rank of Delta(eps) = number of reducible atoms (Cor. prony); finite rank
      => bounded spectrum => NOT divisible (Prop. unbounded).  Bandwidth-free.
  (G) SUM RULE IN THE ARC GAUGE
      int R dphi = pi ln 2 -- one bit per radian.
"""
import os
import random
import sys

import mpmath as mp
import pandas as pd

mp.mp.dps = 40
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save                                       # noqa: E402
rows = []
def chk(block, name, cond, detail=""):
    rows.append((block, name, bool(cond), detail))

# ---------- exact machinery for finite atom mixtures ----------
def mom(ws, as_, x):
    e = [w*mp.e**(-a*x) for w, a in zip(ws, as_)]
    Z = sum(e); p = [ei/Z for ei in e]
    m1 = sum(pi*a for pi, a in zip(p, as_)); m2 = sum(pi*a*a for pi, a in zip(p, as_))
    return Z, m1, m2 - m1*m1                      # eps, s, kappa   (exact)
cure = lambda lam, c: (lambda x: mp.e**(-lam*(1-mp.e**(-c*x))))
pfl  = lambda Lam, C, b, x0: (lambda x: mp.e**(-Lam + C*(x-x0)**b))
lomaxf = lambda ei, b, k: (lambda x: ei + (1-ei)*(1+x/b)**(-k))
def hdet(seq, k):
    M = mp.matrix(k, k)
    for i in range(k):
        for j in range(k): M[i, j] = seq[i+j]
    return mp.det(M)

# ================= (A) =================
for t in [1, 2, 3, 10, 100]:
    for (ei, c, a) in [(mp.mpf('0.01'), mp.mpf('0.9'), mp.mpf('1.3')),
                       (mp.mpf('0.001'), mp.mpf('0.5'), mp.mpf('0.4'))]:
        ws, as_ = [ei, c], [mp.mpf(0), a]
        f = lambda x: ei + c*mp.e**(-a*x)
        # operating point: N_eff = 1/t  <=>  psi = (eps-eps_inf)/eps_inf = 1/t
        x = (mp.log(c) - mp.log(ei/mp.mpf(t)))/a
        eps, s, k = mom(ws, as_, x)
        dS = mp.log(eps/ei)
        chk("A", f"two-atom dS*=ln(1+1/t)  t={t}", abs(dS-mp.log(1+mp.mpf(1)/t)) < mp.mpf('1e-30'))
        chk("A", f"two-atom eps*=eps_inf(1+t)/t  t={t}", abs(eps-ei*(1+mp.mpf(t))/t) < mp.mpf('1e-30'))
        chk("B", f"two-atom lambda*=alpha/(1+t)  t={t}", abs(s-a/(1+mp.mpf(t))) < mp.mpf('1e-28'), f"{mp.nstr(s,8)}")
        chk("B", f"two-atom sqrt(kappa/t)=lambda*  t={t}", abs(mp.sqrt(k/t)-s) < mp.mpf('1e-28'))
for t in [1, 2, 5, 20]:
    for (lam, cc) in [(mp.mpf(5), mp.mpf('0.45')), (mp.mpf(2), mp.mpf('1.1'))]:
        # cure chart: S = lam(1-e^{-cx}), dS = lam e^{-cx}, N_eff = dS  -> dS* = 1/t
        x = mp.log(lam*mp.mpf(t))/cc
        dS = lam*mp.e**(-cc*x)
        chk("A", f"cure dS*=1/t  t={t}", abs(dS-mp.mpf(1)/t) < mp.mpf('1e-30'))
g = lambda t: mp.mpf(1)/t - mp.log(1+mp.mpf(1)/t)
chk("A", "band g(1) = 1 - ln 2 = 0.30685", abs(g(1)-(1-mp.log(2))) < mp.mpf('1e-35'), f"{mp.nstr(g(1),8)}")
chk("A", "band decreasing in the grain", all(g(t) > g(2*t) for t in [mp.mpf(1), mp.mpf(3), mp.mpf(30)]))
chk("A", "band at t=1 exceeds the count-axis bracket (0.20 / 0.094 nats)", g(1) > mp.mpf('0.20'))
chk("A", "band at t=2 already below it", g(2) < mp.mpf('0.20'), f"g(2)={mp.nstr(g(2),6)}")
# universality on finite-atom spectra: no violation of either wall
random.seed(11); bad = tot = 0
for _ in range(400):
    n = random.randint(1, 5)
    as_ = [mp.mpf(0)]+[mp.mpf(random.uniform(0.02, 5)) for _ in range(n)]
    ws = [mp.mpf(random.uniform(1e-4, 0.3))]+[mp.mpf(random.uniform(0.02, 1)) for _ in range(n)]
    tw = sum(ws); ws = [w/tw for w in ws]; ei = ws[0]
    for i in range(120):
        x = mp.mpf(i)/3
        eps, s, k = mom(ws, as_, x)
        if k == 0: continue
        tot += 1
        if s*s/k > mp.e**mp.log(eps/ei)-1+mp.mpf('1e-30'): bad += 1
chk("A", "axiom wall N_eff <= e^{dS}-1 (so dS* >= ln(1+1/t))", bad == 0, f"{bad} violations / {tot} points")
viol = found = 0
for _ in range(300):
    n = random.randint(1, 5)
    as_ = [mp.mpf(0)]+[mp.mpf(random.uniform(0.02, 5)) for _ in range(n)]
    ws = [mp.mpf(random.uniform(1e-4, 0.3))]+[mp.mpf(random.uniform(0.02, 1)) for _ in range(n)]
    tw = sum(ws); ws = [w/tw for w in ws]; ei = ws[0]
    for t in [1, 2, 7, 50]:
        target = mp.mpf(1)/t; prev = None
        for i in range(400):
            x = mp.mpf(i)/5
            eps, s, k = mom(ws, as_, x)
            if k == 0: break
            N = s*s/k
            if prev and (prev[1]-target)*(N-target) < 0:
                xr = mp.findroot(lambda y: (lambda e, s2, k2: s2*s2/k2)(*mom(ws, as_, y))-target,
                                 (prev[0], x), solver='bisect', tol=mp.mpf('1e-35'))
                eps, s, k = mom(ws, as_, xr); found += 1
                if mp.log(eps/ei) < mp.log(1+target)-mp.mpf('1e-25'): viol += 1
            prev = (x, N)
chk("A", "dS* >= ln(1+1/t) at every located operating point", viol == 0, f"{viol} / {found} operating points")
# t = 1: operating point = elbow
ei, c, a = mp.mpf('0.01'), mp.mpf('0.9'), mp.mpf('1.3')
f = lambda x: ei + c*mp.e**(-a*x)
x_el = mp.log(c/ei)/a                       # eps = 2 eps_inf
x_op = (mp.log(c) - mp.log(ei))/a           # psi = 1  (t = 1)
chk("A", "t=1 operating point IS the elbow", abs(x_el-x_op) < mp.mpf('1e-30'))
h = mp.mpf('1e-5'); S = lambda y: -mp.log(f(y))
c3 = -(S(x_el+2*h)-2*S(x_el+h)+2*S(x_el-h)-S(x_el-2*h))/(2*h**3)
chk("A", "c_3 = 0 there (the divisibility boundary)", abs(c3) < mp.mpf('1e-6'), f"c3={mp.nstr(c3,3)}")

# ================= (C) =================
R = lambda N: mp.log(mp.cosh(mp.log(N)/2))
for (t1, t2) in [(mp.mpf(1), mp.mpf(4)), (mp.mpf(1), mp.mpf(11)), (mp.mpf(2), mp.mpf(100))]:
    N0 = 1/mp.sqrt(t1*t2); val = max(R(t1*N0), R(t2*N0))
    better = sum(1 for i in range(-300, 300)
                 if max(R(t1*N0*mp.e**(mp.mpf(i)/100)), R(t2*N0*mp.e**(mp.mpf(i)/100))) < val-mp.mpf('1e-30'))
    chk("C", f"N=1/sqrt(t1t2) is minimax on [{t1},{t2}]", better == 0)
    chk("C", f"value = ln cosh((1/4)ln(t2/t1)) on [{t1},{t2}]",
        abs(val-mp.log(mp.cosh(mp.log(t2/t1)/4))) < mp.mpf('1e-30'), f"{mp.nstr(val,8)} nats")
chk("C", "grain ignorance is unbounded in regret", R(mp.mpf('1e9')) > 9, f"R(1e9)={mp.nstr(R(mp.mpf('1e9')),6)}")
chk("C", "a 10x grain uncertainty costs 0.157 nats", abs(mp.log(mp.cosh(mp.log(10)/4))-mp.mpf('0.1573')) < mp.mpf('1e-3'))

# ================= (D) =================
bad = tot = 0
random.seed(5)
for _ in range(40):
    n = random.randint(1, 4)
    as_ = [mp.mpf(0)]+[mp.mpf(random.uniform(0.05, 3)) for _ in range(n)]
    ws = [mp.mpf(random.uniform(1e-3, 0.3))]+[mp.mpf(random.uniform(0.05, 1)) for _ in range(n)]
    tw = sum(ws); ws = [w/tw for w in ws]
    F = lambda x: sum(w*mp.e**(-aa*x) for w, aa in zip(ws, as_))
    h = mp.mpf(random.uniform(0.1, 1.5)); x0 = mp.mpf(random.uniform(0, 3))
    e = [F(x0+j*h) for j in range(7)]
    de = [-(e[j+1]-e[j]) for j in range(6)]
    for k in (1, 2, 3):
        tot += 2
        if hdet(e, k) < -mp.mpf('1e-30'): bad += 1
        if hdet(de, k) < -mp.mpf('1e-30'): bad += 1
chk("D", "AXIOM: sampled error and -Delta(error) are Hausdorff moment sequences", bad == 0, f"{bad}/{tot} minors negative")
bad = tot = 0
for nm, F, ei in [("cure", cure(mp.mpf(5), mp.mpf('0.45')), mp.e**(-mp.mpf(5))),
                  ("power-floor", pfl(mp.mpf(3), mp.mpf(2), mp.mpf('-0.5'), mp.mpf(0)), mp.e**(-mp.mpf(3))),
                  ("lomax+floor", lomaxf(mp.mpf('0.02'), mp.mpf(3), mp.mpf('1.2')), mp.mpf('0.02'))]:
    for x0 in [mp.mpf('0.5'), mp.mpf(2), mp.mpf(6)]:
        for h in [mp.mpf('0.2'), mp.mpf('0.9')]:
            d = [mp.log(F(x0+j*h)/ei) for j in range(7)]
            for k in (1, 2, 3):
                tot += 1
                if hdet(d, k) < -mp.mpf('1e-35'): bad += 1
chk("D", "HYPOTHESIS: divisible charts -- every minor of the sampled depth >= 0", bad == 0, f"{bad}/{tot}")
ei = mp.e**(-mp.mpf(5)); F = cure(mp.mpf(5), mp.mpf('0.45'))
w = max(abs((lambda d: d[0]*d[2]-d[1]**2)([mp.log(F(x0+j*h)/ei) for j in range(3)]))
        for x0 in [mp.mpf('0.1'), mp.mpf(1), mp.mpf(5)] for h in [mp.mpf('0.03'), mp.mpf('0.7'), mp.mpf(3)])
chk("D", "the cure chart satisfies the discrete wall with EQUALITY at every grid", w < mp.mpf('1e-38'), f"max|defect|={mp.nstr(w,3)}")
x0, h = mp.mpf(4), mp.mpf('1e-4'); F = pfl(mp.mpf(3), mp.mpf(2), mp.mpf('-0.5'), mp.mpf(0)); ei = mp.e**(-mp.mpf(3))
d = [mp.log(F(x0+j*h)/ei) for j in range(3)]; v = d[0]*d[2]-d[1]**2
dS = 2*(x0+h)**mp.mpf('-0.5'); s = (x0+h)**mp.mpf('-1.5'); k = mp.mpf('1.5')*(x0+h)**mp.mpf('-2.5')
chk("D", "h->0: (d0d2-d1^2)/h^2 -> kappa*DeltaS - s^2  (i.e. the wall)", abs(v/h**2-(k*dS-s*s)) < mp.mpf('1e-9'))
c2 = c3 = 0
random.seed(3)
for _ in range(300):
    n = random.randint(1, 3)
    as_ = [mp.mpf(0)]+[mp.mpf(random.uniform(0.05, 3)) for _ in range(n)]
    ws = [mp.mpf(random.uniform(1e-3, 0.2))]+[mp.mpf(random.uniform(0.05, 1)) for _ in range(n)]
    tw = sum(ws); ws = [w/tw for w in ws]; ei = ws[0]
    F = lambda x: sum(w*mp.e**(-aa*x) for w, aa in zip(ws, as_))
    x0 = mp.mpf(random.uniform(0, 3)); h = mp.mpf(random.uniform(0.2, 1.2))
    d = [mp.log(F(x0+j*h)/ei) for j in range(5)]
    c2 += hdet(d, 2) < -mp.mpf('1e-35'); c3 += hdet(d, 3) < -mp.mpf('1e-35')
chk("D", "5 accuracies refute strictly more indivisible frontiers than 3", c3 > c2, f"3-point {c2}/300, 5-point {c3}/300")

# ================= (E) =================
def eps_A(p):                                   # Aitken / Prony, any spacing
    (x0, e0), (x1, e1), (x2, e2) = p
    u, v = x1-x0, x2-x0
    if not (e0 > e1 > e2): return None
    tgt = (e0-e1)/(e1-e2)
    if tgt <= u/(v-u): return None
    fa = lambda a: (1-mp.e**(-a*u))/(mp.e**(-a*u)-mp.e**(-a*v))-tgt
    lo = hi = mp.mpf('1e-9')
    while fa(hi) < 0 and hi < mp.mpf('1e8'): hi *= 2
    for _ in range(250):
        m = (lo+hi)/2
        if fa(m) >= 0: hi = m
        else: lo = m
    a = (lo+hi)/2; c = (e0-e1)/(mp.e**(-a*x0)-mp.e**(-a*x1))
    return e0-c*mp.e**(-a*x0), a
def eps_W(p):                                   # largest floor compatible with divisibility
    (x0, e0), (x1, e1), (x2, e2) = p
    l = [mp.log(e0), mp.log(e1), mp.log(e2)]; h1, h2 = x1-x0, x2-x1
    def defect(cc):
        d = [li-cc for li in l]
        if min(d) <= 0: return mp.mpf(-1)
        return (mp.log(d[2])-mp.log(d[1]))/h2-(mp.log(d[1])-mp.log(d[0]))/h1
    hi = min(l)-mp.mpf('1e-12'); lo = min(l)-mp.mpf(60)
    if defect(lo) < 0: return None
    for _ in range(250):
        m = (lo+hi)/2
        if defect(m) >= 0: lo = m
        else: hi = m
    return mp.e**((lo+hi)/2)
charts = [("cure", cure(mp.mpf(5), mp.mpf('0.45')), mp.e**(-mp.mpf(5)), True),
          ("cure2", cure(mp.mpf(2), mp.mpf('1.1')), mp.e**(-mp.mpf(2)), True),
          ("power-floor", pfl(mp.mpf(3), mp.mpf(2), mp.mpf('-0.5'), mp.mpf(-1)), mp.e**(-mp.mpf(3)), True),
          ("lomax+floor", lomaxf(mp.mpf('0.02'), mp.mpf(3), mp.mpf('1.2')), mp.mpf('0.02'), True),
          ("two-atom", lambda x: mp.mpf('0.01')+mp.mpf('0.9')*mp.e**(-mp.mpf('1.3')*x), mp.mpf('0.01'), False),
          ("three-atom", lambda x: mp.mpf('0.005')+mp.mpf('0.5')*mp.e**(-mp.mpf('0.6')*x)
                                   + mp.mpf('0.495')*mp.e**(-mp.mpf('2.2')*x), mp.mpf('0.005'), False)]
unsoundA = unsoundW = n = 0; margins = []
for nm, F, true, divisible in charts:
    for x0 in [mp.mpf('0.4'), mp.mpf(2), mp.mpf(5)]:
        for h in [mp.mpf('0.4'), mp.mpf(1), mp.mpf(2)]:
            p = [(x0+j*h, F(x0+j*h)) for j in range(3)]
            A, W = eps_A(p), eps_W(p)
            if A is None or W is None: continue
            n += 1
            if true > A[0]*(1+mp.mpf('1e-20')): unsoundA += 1
            if divisible and true > W*(1+mp.mpf('1e-20')): unsoundW += 1
            if not divisible and W < true: margins.append(mp.log(true/W))
chk("E", "eps_A is an upper bound on the floor of every CM frontier", unsoundA == 0, f"{unsoundA}/{n}")
chk("E", "eps_W is an upper bound on the floor of every DIVISIBLE frontier", unsoundW == 0, f"{unsoundW}/{n} (so a floor above eps_W refutes)")
p = [(mp.mpf(2)+j*mp.mpf('0.7'), cure(mp.mpf(5), mp.mpf('0.45'))(mp.mpf(2)+j*mp.mpf('0.7'))) for j in range(3)]
chk("E", "eps_W is EXACT on the cure chart", abs(eps_W(p)/mp.e**(-mp.mpf(5))-1) < mp.mpf('1e-20'))
f2 = lambda x: mp.mpf('0.01')+mp.mpf('0.9')*mp.e**(-mp.mpf('1.3')*x)
p = [(mp.mpf(2)+j*mp.mpf('0.7'), f2(mp.mpf(2)+j*mp.mpf('0.7'))) for j in range(3)]
chk("E", "eps_A is EXACT on the two-atom chart", abs(eps_A(p)[0]/mp.mpf('0.01')-1) < mp.mpf('1e-25'))
mm = sorted(margins)
chk("E", "refutation margin on indivisible truths (median, in nats of floor)",
    mm[len(mm)//2] > mp.mpf('0.9'),
    f"median {mp.nstr(mm[len(mm)//2],4)} nats, range [{mp.nstr(mm[0],3)}, {mp.nstr(mm[-1],4)}] -- window-dependent, as Prop. elbow requires")
# and the window dependence IS the elbow law, in nats of floor precision
ei_, c_, a_ = mp.mpf('0.01'), mp.mpf('0.9'), mp.mpf('1.3')
F_ = lambda x: ei_ + c_*mp.e**(-a_*x); x_el_ = mp.log(c_/ei_)/a_
tab = []
for off in [-3, -2, -1, 0, 1, 2, 4]:
    xc = x_el_ + off; hh = mp.mpf('0.8')
    W = eps_W([(xc-hh+j*hh, F_(xc-hh+j*hh)) for j in range(3)])
    tab.append((off, mp.log(ei_/W) if W else mp.mpf(0)))
chk("E", "required floor precision falls monotonically through the elbow",
    all(tab[i][1] > tab[i+1][1] for i in range(len(tab)-1)),
    "nats needed at e-folds from the elbow: " + ", ".join(f"{o:+d}:{mp.nstr(v,3)}" for o, v in tab))
chk("E", "left of the elbow the count-axis bracket (0.094-0.20 nats) suffices",
    tab[2][1] > mp.mpf('0.20') and tab[4][1] < mp.mpf('0.094'),
    "1 e-fold left needs 1.94 nats of floor; 1 e-fold right needs 0.030 -- the test lives left of the elbow")

# ================= (F) =================
def rank_delta(F, x0, h, maxk=5, tol=mp.mpf('1e-25')):
    e = [F(x0+j*h) for j in range(2*maxk+2)]
    de = [e[j+1]-e[j] for j in range(2*maxk+1)]
    r = 0
    for k in range(1, maxk+1):
        if abs(hdet(de, k)) > tol: r = k
    return r
for nm, ws, as_, exp in [("2 atoms", [mp.mpf('0.01'), mp.mpf('0.99')], [mp.mpf(0), mp.mpf('1.3')], 1),
                         ("3 atoms", [mp.mpf('0.01'), mp.mpf('0.5'), mp.mpf('0.49')],
                          [mp.mpf(0), mp.mpf('0.7'), mp.mpf('2.1')], 2),
                         ("4 atoms", [mp.mpf('0.01'), mp.mpf('0.3'), mp.mpf('0.3'), mp.mpf('0.39')],
                          [mp.mpf(0), mp.mpf('0.4'), mp.mpf('1.1'), mp.mpf('2.6')], 3)]:
    F = lambda x: sum(w*mp.e**(-a*x) for w, a in zip(ws, as_))
    chk("F", f"Hankel rank of Delta(eps) counts the reducible atoms ({nm})",
        rank_delta(F, mp.mpf('0.3'), mp.mpf('0.7')) == exp)
F = cure(mp.mpf(5), mp.mpf('0.45'))
de = [F(mp.mpf('0.3')+(j+1)*mp.mpf('0.7'))-F(mp.mpf('0.3')+j*mp.mpf('0.7')) for j in range(12)]
chk("F", "a divisible chart has NO finite rank (every minor nonzero)",
    all(abs(hdet(de, k)) > mp.mpf('1e-30') for k in (1, 2, 3, 4, 5)),
    "|minors| " + ", ".join(mp.nstr(abs(hdet(de, k)), 3) for k in (1, 2, 3, 4, 5)))
F2 = lambda x: f2(x)**2
chk("F", "bundling k=2 takes rank 1 -> rank 2 (arithmetic, not information)",
    rank_delta(F2, mp.mpf('0.3'), mp.mpf('0.7')) == 2)
chk("F", "integer error gauges are theorems: eps^k is CM for every CM eps",
    all(hdet([f2(mp.mpf('0.2')+j*mp.mpf('0.6'))**kk for j in range(5)], 2) >= -mp.mpf('1e-35')
        for kk in (1, 2, 3, 4)), "so the hypothesis has content only at non-integer grain")

# ================= (G) =================
IR = mp.quad(lambda u: mp.log(mp.cosh(u/2))*mp.sech(u/2)/2, [-mp.inf, mp.inf])
chk("G", "int R dphi = pi ln 2 (one bit per radian)", abs(IR-mp.pi*mp.log(2)) < mp.mpf('1e-20'), mp.nstr(IR, 18))
L = mp.quad(lambda u: mp.sech(u/2)/2, [-mp.inf, mp.inf])
chk("G", "int dphi = pi (the total arc, Prop. gudermann)", abs(L-mp.pi) < mp.mpf('1e-25'))

# ---------------- report ----------------
print(__doc__)
cur = None
for b, nm, c, d in rows:
    if b != cur:
        print(f"\n--- block {b} " + "-"*(60-len(b))); cur = b
    print(("  PASS  " if c else "  FAIL  ") + nm + (("   | " + str(d)) if d else ""))
p = sum(1 for *_, c, _ in rows if c)
print(f"\n{p} checks, {len(rows)-p} failures")

save(pd.DataFrame([dict(block=b, check=nm, passed=int(c), detail=str(d)) for b, nm, c, d in rows]),
     "table_grid_laws.csv")
save(pd.DataFrame([dict(efolds_from_elbow=o, nats_of_floor_required=float(v),
                        decided_by_count_axis_bracket_020=int(float(v) > 0.20),
                        decided_by_count_axis_bracket_0094=int(float(v) > 0.094))
                   for o, v in tab]), "table_floor_precision.csv")
save(pd.DataFrame([dict(grain=t, operating_depth_axiom=float(mp.log(1+mp.mpf(1)/t)),
                        operating_depth_divisible=float(mp.mpf(1)/t),
                        band_nats=float(g(t)),
                        beats_bracket_020=int(g(t) > mp.mpf('0.20')),
                        beats_bracket_0094=int(g(t) > mp.mpf('0.094')))
                   for t in [1, 1.5, 2, 3, 5, 10, 30, 100]], ), "table_operating_band.csv")
save(pd.DataFrame([dict(grain_ratio=r, minimax_N_eff=float(1/mp.sqrt(mp.mpf(r))),
                        worst_case_regret_nats=float(mp.log(mp.cosh(mp.log(mp.mpf(r))/4))),
                        inside_005_band=int(mp.log(mp.cosh(mp.log(mp.mpf(r))/4)) < mp.mpf('0.05')))
                   for r in [1.5, 2, 3, 4, 7.3, 10, 30, 100]]), "table_grain_minimax.csv")
