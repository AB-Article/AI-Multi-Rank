#!/usr/bin/env python3
"""THE PRICE IS CONVEX: FIVE LAWS DERIVED FROM THE AXIOM AND THE WALLS.
   (Section "The price is a frontier": Prop. 'unbounded', 'elbow', 'notestable',
    'wallrefutes'; Section "The two counts": Prop. 'exchange'.
    Online Appendix sec. "The price is convex".)

Nothing here is measured.  Every statement is a consequence of Law I (complete
monotonicity), Thm. 'budget', Thm. 'monofloor' and Prop. 'twocounts', which are
already in the paper.  No corpus, no fit, no data -- the archive's 117 frontiers
enter only in script 122.

The five statements, in the order they are checked:

 A  SUM RULE.  For a bounded exponent spectrum the tilt concentrates on a_max as
    x -> -inf and on a_min as x -> +inf, so kappa = Var_p(a) vanishes at BOTH
    ends.  Hence

        int kappa dx = a_max - a_min        and      int c_k dx = 0, k >= 3.

    The spectral width is the only non-vanishing charge of the resource flow.

 B  THE ELBOW IS THE GATE.  int c_3 dx = 0 while divisibility demands c_3 >= 0,
    so c_3 must change sign: kappa rises, peaks, falls.  The sign change is the
    curvature maximum.  Divisibility fails strictly to its LEFT.  Equivalently
    (Prop. 'unbounded'): s positive, decreasing, convex and non-constant on R
    forces s(-inf) = +inf, so

        divisible  ==>  a_max = infinity  (no steepest mode, no hardest gate),
        bounded spectrum  ==>  NOT divisible.

 C  NO TEST AT THE OPERATING POINT.  Thm. 'monofloor' is d ln kappa/dS <= 1 -
    kappa/s^2.  At an operating point kappa/s^2 = t, so c_3 >= kappa*s*(t-1) >= 0
    for every grain t >= 1, on every completely monotone frontier.  The
    four-point test is powerless exactly where the theory recommends operating.
    On a two-atom chart the gap to the elbow is ln(2t/(1+t)) < ln 2: the grain
    can move the recommendation by at most ONE BIT, ever.

 D  THE WALL REFUTES, AT SECOND ORDER.  Thm. 'budget' (N_eff <= Delta_S) is
    proved on the Levy measure, which exists only under divisibility, so its
    violation refutes.  On a floored two-atom chart, with rho = eps/eps_inf,

        N_eff = rho - 1,   Delta_S = ln rho,   N_eff - Delta_S = (rho-1) - ln rho > 0

    at EVERY rho > 1 -- the incumbent chart is refuted on its whole range, one
    derivative cheaper than Prop. 'lumpy'.  The axiom allows N_eff <= rho-1, the
    hypothesis only N_eff <= ln rho; the gap is a refutation band of width
    N_eff - ln(1+N_eff) nats, and the archive's reported "tightening factor"
    (1+N)e^{-N} is the same function.

 E  THE EXCHANGE RATE.  theta(k) = d ln S / d ln k in [0,1] is the local worth of
    one attempt measured in items (Prop. 'twocounts' in differential form), with
    ln D(t) = -int_0^{ln t} (1 - theta) du.  theta is NOT monotone: it tends to 1
    at both ends, so the retry discount is a transient.  theta is the task index
    of Prop. 'chain', so beta_clock = beta/theta_task and theta_task >= beta.

Outputs: results/table_derived_laws.csv
         results/table_refutation_band.csv
"""
from __future__ import annotations
import os
import sys
import random

import mpmath as mp
import pandas as pd

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _loaders import save                                       # noqa: E402

ROWS: list[dict] = []
OK = FAIL = 0


def chk(law: str, name: str, cond: bool, info: str = "") -> None:
    global OK, FAIL
    cond = bool(cond)
    OK, FAIL = OK + cond, FAIL + (not cond)
    ROWS.append(dict(law=law, check=name, passed=int(cond), detail=info))
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {info}")


def hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)


def finite(atoms):
    """eps(x) = sum_j w_j exp(-a_j x): an arbitrary bounded exponent spectrum."""
    eps = lambda x: mp.fsum([w * mp.e ** (-a * x) for w, a in atoms])
    return (lambda x: -mp.log(eps(x))), eps


BOUNDED = {
    "two-atom floored":   [(0.30, 0.0), (0.70, 1.0)],
    "two-atom no floor":  [(0.40, 0.35), (0.60, 2.10)],
    "three-atom floored": [(0.15, 0.0), (0.45, 0.7), (0.40, 2.3)],
    "four-atom floored":  [(0.10, 0.0), (0.25, 0.4), (0.35, 1.1), (0.30, 3.0)],
    "five-atom no floor": [(0.2, 0.2), (0.2, 0.6), (0.2, 1.3), (0.2, 2.2), (0.2, 4.0)],
}
DIVISIBLE = {
    "cure       S=L(1-e^{-cx})":     (lambda x: 5 * (1 - mp.e ** (-mp.mpf("0.45") * x)), -8, 30),
    "stretched  S=(ax)^b, b=0.5":    (lambda x: (mp.mpf("0.8") * x) ** mp.mpf("0.5"), 1e-3, 80),
    "power-floor S=L-Cx^b, b=-0.7":  (lambda x: 6 - 3 * x ** mp.mpf("-0.7"), 1e-2, 300),
    "Lomax      S=k ln(1+x/c)":      (lambda x: 2 * mp.log(1 + x / mp.mpf("1.5")), 1e-3, 300),
}

# ------------------------------------------------------------------ A: sum rule
hdr("A.  THE SUM RULE.  int kappa dx = a_max - a_min ;  int c_k dx = 0 for k >= 3")
for nm, at in BOUNDED.items():
    S, _ = finite(at)
    kap = lambda x: -mp.diff(S, x, 2)
    amax = max(a for w, a in at)
    amin = min(a for w, a in at)
    P = [-60, -5, 0, 5, 20, 80]
    chk("A", f"[{nm}] s(-inf)=a_max and s(+inf)=a_min",
        abs(mp.diff(S, -60, 1) - amax) < 1e-12 and abs(mp.diff(S, 80, 1) - amin) < 1e-12)
    chk("A", f"[{nm}] kappa vanishes at both ends",
        abs(kap(-60)) < 1e-12 and abs(kap(80)) < 1e-12)
    chk("A", f"[{nm}] int kappa dx = a_max - a_min",
        abs(mp.quad(kap, P) - (amax - amin)) < 1e-12)
    chk("A", f"[{nm}] int c_3 dx = 0", abs(mp.quad(lambda x: mp.diff(S, x, 3), P)) < 1e-12)
    chk("A", f"[{nm}] int c_4 dx = 0", abs(mp.quad(lambda x: -mp.diff(S, x, 4), P)) < 1e-12)

# -------------------------------------------------------------------- B: elbow
hdr("B.  THE ELBOW IS THE GATE.  c_3 < 0 near the low-resource end, >= 0 near the high")
print("     (the ordering is EXACT only when kappa is unimodal, which L < pi certifies;")
print("      with well-separated atoms each channel contributes its own bump -- checked below)")
for nm, at in BOUNDED.items():
    S, _ = finite(at)
    xs = [mp.mpf(j) / 4 for j in range(-120, 180)]
    sg = [(x, mp.sign(mp.diff(S, x, 3))) for x in xs]
    neg = [x for x, v in sg if v < 0]
    pos = [x for x, v in sg if v > 0]
    chk("B", f"[{nm}] c_3 < 0 near -inf and >= 0 near +inf",
        bool(neg) and bool(pos) and min(neg) < min(pos) and max(pos) > max(neg))
    chk("B", f"[{nm}] exactly one sign change (kappa unimodal)",
        bool(neg) and bool(pos) and max(neg) < min(pos),
        f" elbow in ({mp.nstr(max(neg), 4)},{mp.nstr(min(pos), 4)})")

# the scope of the sharp form: well-separated atoms break the single sign change
hdr("B*. SCOPE.  the sharp 'left of THE elbow' form needs kappa unimodal")
for nm, at in {"3-atom well separated": [(1/3, 0.0), (1/3, 1.0), (1/3, 60.0)],
               "3-atom clustered": [(0.15, 0.0), (0.45, 0.7), (0.40, 2.3)]}.items():
    S, _ = finite(at)
    aa = sorted(a for w, a in at)
    amax = aa[-1]
    amin2 = min(a for a in aa if a > 0)
    # the asymptotics live on the spectrum's OWN scales: 1/a_max at the left end,
    # 1/a_min at the right.  A fixed grid misses them whenever the atoms are far
    # apart, which is what makes the single-sign-change form chart-dependent.
    xL = -mp.mpf(25) / amax
    xR = mp.mpf(40) / amin2
    cL, cR = mp.diff(S, xL, 3), mp.diff(S, xR, 3)
    xs = [xL + (xR - xL) * mp.mpf(j) / 400 for j in range(1, 400)]
    sg = [mp.sign(mp.diff(S, x, 3)) for x in xs]
    neg = [x for x, v in zip(xs, sg) if v < 0]
    pos = [x for x, v in zip(xs, sg) if v > 0]
    one = bool(neg) and bool(pos) and max(neg) < min(pos)
    print(f"  NOTE  [{nm}] single sign change on a uniform grid = {one}"
          f"   (sharp form needs kappa unimodal)")
    chk("B", f"[{nm}] GENERAL form: c_3 < 0 at the low end and > 0 at the high end",
        cL < 0 and cR > 0, f" c_3({mp.nstr(xL,3)})={mp.nstr(cL,3)}, c_3({mp.nstr(xR,3)})={mp.nstr(cR,3)}")

hdr("B'. no DIVISIBLE chart has an elbow: c_3 >= 0 and kappa strictly decreasing")
for nm, (S, lo, hi) in DIVISIBLE.items():
    xs = [lo + (hi - lo) * mp.mpf(j) / 120 for j in range(1, 120)]
    kap = lambda x: -mp.diff(S, x, 2)
    chk("B", f"[{nm}] c_3 >= 0 everywhere (the price is convex)",
        min(mp.diff(S, x, 3) for x in xs) >= -1e-22)
    chk("B", f"[{nm}] kappa strictly decreasing (no elbow)",
        all(kap(xs[j]) > kap(xs[j + 1]) for j in range(len(xs) - 1)))

hdr("B''. and no bounded-spectrum frontier is divisible on its whole axis")
random.seed(11)
viol = 0
for _ in range(120):
    n = random.choice([2, 3, 4, 5])
    a = sorted(random.uniform(0, 4) for _ in range(n))
    w = [random.uniform(0.05, 1) for _ in range(n)]
    Z = sum(w)
    S, _ = finite(list(zip([v / Z for v in w], a)))
    if min(mp.diff(S, mp.mpf(j) / 5, 3) for j in range(-150, 300)) >= 0:
        viol += 1
chk("B", "120 random finite-atom spectra: every one has c_3 < 0 somewhere",
    viol == 0, f" exceptions={viol}")

# ----------------------------------------------------------- C: no test at op
hdr("C.  NO TEST AT THE OPERATING POINT.  c_3 >= kappa*s*(t-1) >= 0 at N_eff = 1/t")
random.seed(7)
bad = tested = 0
slack = []
for _ in range(120):
    n = random.choice([2, 2, 3, 4, 5])
    a = sorted(random.uniform(0, 4) for _ in range(n))
    if random.random() < 0.5:
        a[0] = 0.0
    w = [random.uniform(0.05, 1) for _ in range(n)]
    Z = sum(w)
    S, _ = finite(list(zip([v / Z for v in w], a)))
    for t in [1, 1.5, 2, 5, 20]:
        try:
            x = mp.findroot(lambda z: mp.diff(S, z, 1) ** 2 / (-mp.diff(S, z, 2)) - mp.mpf(1) / t,
                            mp.mpf(1))
        except Exception:
            continue
        if not (-40 < x < 120):
            continue
        tested += 1
        c3 = mp.diff(S, x, 3)
        bad += int(c3 < -1e-18)
        slack.append(float(c3 - (-mp.diff(S, x, 2)) * mp.diff(S, x, 1) * (t - 1)))
chk("C", f"c_3 >= 0 at every operating point ({tested} roots, 120 random spectra)",
    bad == 0, f" violations={bad}")
chk("C", "margin c_3 >= kappa*s*(t-1) holds", min(slack) > -1e-14, f" min slack={min(slack):.2e}")

hdr("C'. GRAIN LEVERAGE IS BOUNDED BY ONE BIT.  span = ln(2t/(1+t)) < ln 2")
for al in ["0.4", "1.3", "3.0"]:
    for einf in ["0.001", "0.05", "0.3"]:
        e0 = mp.mpf(einf)
        pts = [-mp.log(e0 * (1 + mp.mpf(1) / t)) for t in [1, 2, 10, 10 ** 6]]
        chk("C", f"  alpha={al} eps_inf={einf}", pts[-1] - pts[0] < mp.log(2),
            f" span={mp.nstr(pts[-1] - pts[0], 8)} (ln2={mp.nstr(mp.log(2), 8)})")

# ------------------------------------------------------------------- D: wall
hdr("D.  THE WALL REFUTES AT SECOND ORDER.  N_eff = rho-1 > ln rho = Delta_S")
for einf in ["0.001", "0.03", "0.2", "0.45"]:
    for al in [mp.mpf("0.4"), mp.mpf("1.3")]:
        e0 = mp.mpf(einf)
        S = lambda x: -mp.log(e0 + (1 - e0) * mp.e ** (-al * x))
        ok = True
        for x in [mp.mpf(j) / 5 for j in range(1, 60)]:
            s = mp.diff(S, x, 1)
            N = s ** 2 / (-mp.diff(S, x, 2))
            rho = mp.e ** (-S(x)) / e0
            if abs(N - (rho - 1)) > 1e-16 or N - mp.log(rho) <= 0:
                ok = False
        chk("D", f"  eps_inf={einf} alpha={al}: N_eff=rho-1 and the wall breaks at EVERY x", ok)

band_rows = []
for N in [mp.mpf(v) for v in ["0.05", "0.1", "0.2", "0.465", "0.75", "1", "2", "5"]]:
    band_rows.append(dict(N_eff=float(N),
                          tightening_factor=float((1 + N) * mp.e ** (-N)),
                          refutation_band_nats=float(N - mp.log(1 + N)),
                          axiom_bound_N_le=float(N), hypothesis_bound_dS_ge=float(mp.log(1 + N))))
print("\n  the archive's 'tightening factor' and the refutation band are one function:")
for r in band_rows:
    print(f"     N_eff={r['N_eff']:<7} tightening={r['tightening_factor']:.6f}"
          f"   band = N-ln(1+N) = {r['refutation_band_nats']:.5f} nats")
chk("D", "band -> 0 exactly as the tightening factor -> 1",
    abs((1 + mp.mpf("1e-6")) * mp.e ** (-mp.mpf("1e-6")) - 1) < 1e-11)

hdr("D'. divisible charts never violate the wall")
for nm, (S, L) in {"cure L=5": (lambda x: 5 * (1 - mp.e ** (-mp.mpf("0.45") * x)), 5),
                   "cure L=1.2": (lambda x: mp.mpf("1.2") * (1 - mp.e ** (-2 * x)), mp.mpf("1.2")),
                   "power-floor L=6": (lambda x: 6 - 3 * x ** mp.mpf("-0.7"), 6)}.items():
    worst = max(mp.diff(S, mp.mpf(j) / 10, 1) ** 2 / (-mp.diff(S, mp.mpf(j) / 10, 2))
                - (L - S(mp.mpf(j) / 10)) for j in range(5, 400))
    chk("D", f"  [{nm}] N_eff <= Delta_S everywhere", worst <= 1e-20,
        f" max excess={mp.nstr(worst, 4)}")

# --------------------------------------------------------------- E: exchange
hdr("E.  THE ATTEMPT/ITEM EXCHANGE RATE.  theta(k) = d ln S / d ln k in [0,1]")
NUS = {"two-atom g=.17/.95": [(0.55, 0.17), (0.45, 0.95)],
       "three-class census": [(0.5, 0.05), (0.35, 0.6), (0.15, 0.999)],
       "near-nested g=.999": [(0.25, 0.10), (0.75, 0.999)]}
for nm, nu in NUS.items():
    P = lambda k: mp.fsum([w * g ** k for w, g in nu])
    S = lambda k: -mp.log(P(k))
    th = lambda k: k * mp.diff(S, k, 1) / S(k)
    ks = [mp.mpf(j) / 4 for j in range(2, 80)]
    chk("E", f"[{nm}] theta in [0,1]", all(-1e-20 <= th(k) <= 1 + 1e-20 for k in ks))
    chk("E", f"[{nm}] ln D(t) = -int_0^{{ln t}} (1-theta) du",
        all(abs(mp.log(S(t) / (t * S(1))) + mp.quad(lambda u: 1 - th(mp.e ** u), [0, mp.log(t)]))
            < 1e-12 for t in [2, 3, 5, 10]))
    v = [th(k) for k in [mp.mpf("0.01"), 1, 5, 100, mp.mpf("1e5")]]
    chk("E", f"[{nm}] theta -> 1 at BOTH ends (the discount is a transient)",
        v[0] > 0.99 and v[-1] > 0.9 and min(v) < 0.6,
        " theta: " + " ".join(mp.nstr(z, 3) for z in v))

print("\n  panel read-off:  mean theta on [1,2] = 1 + ln D(2)/ln 2")
for fam, D2 in [("independent", mp.mpf("0.547")), ("nested", mp.mpf("0.522"))]:
    tb = 1 + mp.log(D2) / mp.log(2)
    print(f"     {fam:<12} D(2)={D2}  ->  theta_bar = {mp.nstr(tb, 4)}"
          f"   (an attempt is worth {mp.nstr(100 * tb, 3)}% of an item)")
print("     chain: beta = beta_clock * theta_task, both in [0,1]  =>  theta_task >= beta")

# ------------------------------------------------------------------- outputs
save(pd.DataFrame(ROWS), "table_derived_laws.csv")
save(pd.DataFrame(band_rows), "table_refutation_band.csv")
print("\n" + "=" * 78)
print(f"  {OK} checks passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
